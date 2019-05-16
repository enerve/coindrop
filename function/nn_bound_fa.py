'''
Created on 13 May 2019

@author: enerve
'''

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random

import logging
from function.value_function import ValueFunction
from function.net import Net
import util
from .conv_net import Flatten

class NN_Bound_FA(ValueFunction):
    '''
    A neural-network action-value function approximator for a single (bound)
    action.
    '''

    def __init__(self,
                 alpha,
                 regularization_param,
                 batch_size,
                 max_iterations,
                 model):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Using NN Bound FA")

        self.alpha = alpha
        self.regularization_param = regularization_param
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.model = model
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_outputs = 1

        # Stats / debugging
        self.stat_error_cost = []
        self.stat_reg_cost = []
        self.stat_val_error_cost = []
                
        self.sids = self._sample_ids(3000, self.batch_size)
        self.last_loss = torch.zeros(self.batch_size, 7).cuda()
                
    def initialize_default_net(self):
        net = nn.Sequential(
            nn.Conv2d(2, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(30, 50, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(50*6*5, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Sigmoid()) #TODO: tanh

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
        return 'neural_bound_a%s_r%s_b%d_i%d_F%s_NN%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.model.prefix(),
                                     "convnet")

    # ---------------- Run phase -------------------

    def _bind_action(self, S, action):
        ''' Applies the action to the state board and returns result '''
        B = np.copy(S)
        for h in range(6):
            if B[h, action] == 0:
                B[h, action] = 1
                return B
        
        self.logger.warning("Action on full column! %d on \n%s", action, S)
        return None        
        
    def _value(self, S, actions):
        ''' Gets values for all given actions '''
        x_list = []
        for action in actions:
            B = self._bind_action(S, action)
            x = self.model.feature(B)
            x_list.append(x)

        # Sends all bound actions as a batch to NN
        with torch.no_grad():
            XB = torch.stack(x_list).to(self.device)
            output = torch.t(self.net(XB))
        return output[0] * 2 -1 #TODO: tanh

#     def _value(self, S, actions):
#         ''' Gets values for all given actions '''
#         x_list = []
#         for action in actions:
#             B = self._bind_action(S, action)
#             x_list.append(torch.stack(
#                 [self.model.feature(B),
#                  self.model.feature(np.flip(B, axis=1).copy())]))
# 
#         # Sends all bound actions as a batch to NN
#         with torch.no_grad():
#             XB = torch.stack(x_list).to(self.device)
#             output = torch.t(self.net(XB))
#         return output[0] * 2 -1 #TODO: tanh

    def value(self, S, action):
        output = self._value(S, [action])
        return output[0].item()

    def _valid_actions(self, S):
        return np.nonzero(S[6-1] == 0)[0]

    def best_action(self, S):
        actions_list = self._valid_actions(S)
        with torch.no_grad():
            V = self._value(S, actions_list)
            i = torch.argmax(V).item()
            v = V[i].item()
        return actions_list[i], v 

    def random_action(self, S):
        actions_list = self._valid_actions(S)
        return random.choice(actions_list)
      
    # ---------------- Update phase -------------------
    
    def update(self, training_data_collector, validation_data_collector):
        ''' Updates the value function model based on data collected since
            the last update '''

        training_data_collector.before_update()

        self.train(training_data_collector, validation_data_collector)
        self.test()

    def _prepare_data(self, steps_history_state, steps_history_action,
                      steps_history_target):
        steps_history_x = []
        steps_history_t = []
        teye = torch.eye(self.num_outputs).to(self.device)

        self.logger.debug("  Preparing for %d items", len(steps_history_state))
        
        for i, (S, a, t) in enumerate(zip(
                        steps_history_state,
                        steps_history_action,
                        steps_history_target)):
            if i == 250000:
                self.logger.warning("------ too much to prepare ----------")
                break
            
            t = (t+1.0) / 2     #TODO: tanh
            for flip in [False, True]:
                B = self._bind_action(S, a) # TODO: move out
                if flip: B = np.flip(B, axis=1).copy()
                x = self.model.feature(B)
                
                steps_history_x.append(x)
                steps_history_t.append(t)
                
            if (i+1) % 10000 == 0:
                self.logger.debug("prepared %d*2", i+1)                
            
        return steps_history_x, steps_history_t

    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self, training_data_collector, validation_data_collector):
        self.logger.debug("Preparing training data--")
        steps_history_x, steps_history_t = \
            self._prepare_data(*training_data_collector.get_data())
        self.logger.debug("Preparing validation data--")
        val_steps_history_x, val_steps_history_t = \
            self._prepare_data(*validation_data_collector.get_data())
        
        SHX = torch.stack(steps_history_x).to(self.device)
        SHT = torch.tensor(steps_history_t).to(self.device)
        VSHX = torch.stack(val_steps_history_x).to(self.device)
        VSHT = torch.tensor(val_steps_history_t).to(self.device)
        
        self.logger.debug("Training with %d items...", len(steps_history_x))

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
            else:
                ids = self._sample_ids(N, self.batch_size)
                X = SHX[ids]   # b x di
                Y = SHT[ids]   # b
            Y = torch.unsqueeze(Y, 1)   # b x 1
            
            # forward
            outputs = self.net(X)       # b x 1
            # loss
            loss = self.criterion(outputs, Y)  # b x 1
            # backward
            onez = torch.ones(loss.shape).to(self.device) #TODO: move out?
            loss.backward(onez)
            
            # updated weights
            self.optimizer.step()
            
            # Stats
            with torch.no_grad():
                suml = torch.sum(loss, 0)
                countl = torch.sum(loss > 0, 0).float()

                sum_error_cost.add_(suml)  # 1
                count_actions.add_(countl)  # 1

                if (i+1) % period == 0:
                    mean_error_cost = sum_error_cost / (count_actions + 0.01)
                    self.stat_error_cost.append(mean_error_cost.cpu().numpy())
    
                    #self.logger.debug("  loss=%0.2f", sum_error_cost.mean().item())
    
                    torch.zeros(self.num_outputs, out=sum_error_cost)
                    torch.zeros(self.num_outputs, out=count_actions)
                    
                    # Validation
                    X = VSHX
                    Y = torch.unsqueeze(VSHT, 1)
                    outputs = self.net(X)       # b x 1
                    loss = self.criterion(outputs, Y)  # b x 1
                    
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
        util.torch_save(self.net, "boundNN_" + pref)

    def load_model(self, load_subdir, pref=""):
        fname = "boundNN_" + pref
        net = util.torch_load(fname, load_subdir)
        net.eval()
        self.init_net(net)
