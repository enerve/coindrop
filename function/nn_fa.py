'''
Created on 1 Mar 2019

@author: enerve
'''

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from .conv_net import Flatten

import logging
from function.value_function import ValueFunction
from function.net import Net
import util

class NN_FA(ValueFunction):
    '''
    A neural-network action-value function approximator
    '''

    def __init__(self,
                 alpha,
                 regularization_param,
                 batch_size,
                 max_iterations,
                 feature_eng):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Using NN FA")

        self.alpha = alpha
        self.regularization_param = regularization_param
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.feature_eng = feature_eng
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        num_inputs = feature_eng.num_inputs
        self.num_outputs = feature_eng.num_actions

        # Stats / debugging
        self.stat_error_cost = []
        self.stat_reg_cost = []
        self.stat_val_error_cost = []
                
        self.sids = self._sample_ids(3000, self.batch_size)
        self.last_loss = torch.zeros(self.batch_size, 7).cuda()
                
    def initialize_default_net(self):
        #         self.net = Net([num_inputs, 500, 500, 500, self.num_outputs])

        net = nn.Sequential(
            nn.Conv2d(2, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(30, 50, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(50*6*5, 500),
            nn.ReLU(),
            nn.Linear(500, self.num_outputs),
            nn.Sigmoid())

        self.init_net(net)

    def init_net(self, net):        
        net.cuda(self.device)

        #self.criterion = nn.MSELoss(reduce=False)
        self.criterion = nn.BCELoss(reduce=False)
        
        self.optimizer = optim.Adam(
            net.parameters(),
            lr=self.alpha,
            weight_decay=self.regularization_param,
            amsgrad=False)

        self.net = net
        
        self.logger.debug("Net:\n%s", self.net)
        
    def prefix(self):
        return 'neural_a%s_r%s_b%d_i%d_F%s_NN%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.feature_eng.prefix(),
                                     "convnet")

    def _value(self, state):
        X = self.feature_eng.x_adjust(state)
        with torch.no_grad():
            output = self.net(X.unsqueeze(0))
        return output[0] * 2 -1
    
    def _actions_mask(self, state):
        return self.feature_eng.valid_actions_mask(state)
        
    def value(self, state, action):
        output = self._value(state)
        return output[action].item()

    def best_action(self, state):
        V = self._value(state) + 1000 * self._actions_mask(state)
        i = torch.argmax(V).item()
        return i
    
    def random_action(self, state):
        return self.feature_eng.random_action(state)


    def update(self, data_collector):
        ''' Updates the value function model based on data collected since
            the last update '''

        data_collector.before_update()

        self.train(data_collector)
        self.test()

    def _prepare_data(self, steps_history_state, steps_history_action,
                      steps_history_target):
        steps_history_x = []
        steps_history_t = []
        steps_history_mask = []
        teye = torch.eye(self.num_outputs).to(self.device)

        #Sdict = {}
        #count_conflict = 0
        self.logger.debug("  Preparing for %d items", len(steps_history_state))
        
        for i, (S, a, t) in enumerate(zip(
                        steps_history_state,
                        steps_history_action,
                        steps_history_target)):
            if i == 250000:
                self.logger.debug("----------nuff----------")
                break
            
            t = (t+1.0) / 2
            for flip in [False, True]:
                if flip:
                    S = np.flip(S, axis=1).copy()
                    a = 6 - a
                x = self.feature_eng.x_adjust(S)
                m = teye[a].clone()
                
                steps_history_x.append(x)
                steps_history_t.append(t)
                steps_history_mask.append(m)

                #                 stS = '%s%d' % (np.array2string(S, separator=''),a)
                #                 if stS in Sdict:
                #                     if Sdict[stS] == 1:
                #                         count_conflict += 1
                #                     Sdict[stS] += 1
                #                 else:
                #                     Sdict[stS] = 1
                #                 if stS in Sdict:
                #                     if Sdict[stS] != t:
                #                         count_conflict += 1
                #                 Sdict[stS] = t
                
            if (i+1) % 10000 == 0:
                self.logger.debug("prepared %d*2", i+1)                
                #self.logger.debug("  conflict count: %d" % count_conflict)
            
                
        #util.hist(list(Sdict.values()), bins=100, range=(2,50))
        
        return steps_history_x, steps_history_t, steps_history_mask

    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self, data_collector):
        self.logger.debug("Preparing training data--")
        steps_history_x, steps_history_t, steps_history_m = \
            self._prepare_data(*data_collector.get_training_data())
        self.logger.debug("Preparing validation data--")
        val_steps_history_x, val_steps_history_t, val_steps_history_m = \
            self._prepare_data(*data_collector.get_validation_data())
        
        SHX = torch.stack(steps_history_x).to(self.device)
        SHT = torch.tensor(steps_history_t).to(self.device)
        SHM = torch.stack(steps_history_m).to(self.device)
        VSHX = torch.stack(val_steps_history_x).to(self.device)
        VSHT = torch.tensor(val_steps_history_t).to(self.device)
        VSHM = torch.stack(val_steps_history_m).to(self.device)
        
        self.logger.debug("Training with %d items...", len(steps_history_x))

#         with torch.no_grad():
#             X = SHX[self.sids]   # b x di
#             Y = SHT[self.sids]   # b
#             M = SHM[self.sids]   # b x do
#             Y = torch.unsqueeze(Y, 1)   # b x 1
#             
#             # forward
#             outputs = self.net(X)       # b x do
#             # loss
#             Y = Y * M  # b x do
#             loss = self.criterion(outputs, Y)  # b x do
#             with torch.no_grad():
#                 # Zero-out the computed losses for the other actions/outputs
#                 loss *= M   # b x do

        N = len(steps_history_t)
        
        # for stats
        preferred_samples = 1000
        period = self.max_iterations // preferred_samples
        period = max(period, 10)
        if period > self.max_iterations:
            self.logger.warning("max_iterations too small for period plotting")

        sum_error_cost = torch.zeros(self.num_outputs).to(self.device)
        sum_error_cost.detach()
        count_actions = torch.zeros(self.num_outputs).to(self.device)

        for i in range(self.max_iterations):
            self.optimizer.zero_grad()
            
            if self.batch_size == 0:
                # Do full-batch
                X = SHX   # N x di
                Y = SHT   # N
                M = SHM   # N x do
            else:
                ids = self._sample_ids(N, self.batch_size)
                X = SHX[ids]   # b x di
                Y = SHT[ids]   # b
                M = SHM[ids]   # b x do
            Y = torch.unsqueeze(Y, 1)   # b x 1
            
            # forward
            outputs = self.net(X)       # b x do
            # loss
            Y = Y * M  # b x do
            loss = self.criterion(outputs, Y)  # b x do
            with torch.no_grad():
                # Zero-out the computed losses for the other actions/outputs
                loss *= M   # b x do
            # backward
            onez = torch.ones(loss.shape).to(self.device) #TODO: move out?
            loss.backward(onez)
            
            # updated weights
            self.optimizer.step()
            
            # Stats
            with torch.no_grad():
                suml = torch.sum(loss, 0)
                countl = torch.sum(loss > 0, 0).float()
                
#                 ltz = (suml < 0).byte()
#                 if ltz.any():
#                     self.logger.debug("loss < 0")
#                 
#                 if i==0:
#                     self.logger.debug("Initial loss:\n  %s", suml / (countl + 0.0001))
#                 if i+1==self.max_iterations:
#                     self.logger.debug("   Last loss:\n  %s", suml / (countl + 0.0001))

                sum_error_cost.add_(suml)  # do
                count_actions.add_(countl)  # do

                if (i+1) % period == 0:
                    mean_error_cost = sum_error_cost / (count_actions + 0.01)
                    self.stat_error_cost.append(mean_error_cost.cpu().numpy())
    
                    #self.logger.debug("  loss=%0.2f", sum_error_cost.mean().item())
    
                    torch.zeros(self.num_outputs, out=sum_error_cost)
                    torch.zeros(self.num_outputs, out=count_actions)
                    
                    # Validation
                    X = VSHX
                    Y = torch.unsqueeze(VSHT, 1)
                    M = VSHM   # N x do
                    outputs = self.net(X)       # b x do
                    Y = Y * M  # b x do
                    loss = self.criterion(outputs, Y)  # b x do
                    loss *= M   # b x do
                    
                    suml = torch.sum(loss, 0)
                    countl = torch.sum(loss > 0, 0).float()
                    mean_error_cost = suml / (countl + 0.01)
                    self.stat_val_error_cost.append(mean_error_cost.cpu().numpy())

                    #self.live_stats()

            if (i+1) % 1000 == 0:
                self.logger.debug("   %d / %d", i+1, self.max_iterations)

        self.logger.debug("  trained \tN=%s \tE=%0.3f \tVE=%0.3f", N,
                          self.stat_error_cost[-1].mean().item(),
                          self.stat_val_error_cost[-1].mean().item())

        

    def test(self):
        pass

    def collect_stats(self, ep):
        pass
    
    def report_stats(self, pref=""):
        num = len(self.stat_error_cost[1:])

        n_cost = np.asarray(self.stat_error_cost[1:]).T
        labels = list(range(n_cost.shape[1]))        

        n_v_cost = np.asarray(self.stat_val_error_cost[1:]).T
        labels.extend(["val%d" % i for i in range(n_v_cost.shape[1])])
        cost = np.concatenate([n_cost, n_v_cost], axis=0)
        avgcost = np.stack([n_cost.mean(axis=0), n_v_cost.mean(axis=0)], axis=0)
        
        util.plot(cost,
                  range(num),
                  labels = labels,
                  title = "NN training/validation cost across actions",
                  pref=pref+"cost",
                  ylim=None)

        util.plot(avgcost,
                  range(num),
                  labels = ["training cost", "validation cost"],
                  title = "NN training/validation cost",
                  pref=pref+"avgcost",
                  ylim=None)

    
    def live_stats(self):
        num = len(self.stat_error_cost[1:])
        
        if num < 1:
            return

        n_cost = np.asarray(self.stat_error_cost[1:]).T
        n_v_cost = np.asarray(self.stat_val_error_cost[1:]).T
        avgcost =  np.stack([n_cost.mean(axis=0), n_v_cost.mean(axis=0)], axis=0)

        util.plot(avgcost,
                  range(num),
                  labels = ["training cost", "validation cost"],
                  title = "NN training/validation cost",
                  live=True)
        
    def save_model(self, pref=""):
        util.torch_save(self.net, pref)

    def load_model(self, fname, load_subdir):
        net = util.torch_load(fname, load_subdir)
        net.eval()
        self.init_net(net)
