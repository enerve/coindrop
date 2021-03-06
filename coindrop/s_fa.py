'''
Created on 20 Sep 2019

@author: enerve
'''

import collections
import logging
import numpy as np
import torch
import torch.nn as nn

from really.function import ValueFunction
from really import util
from really.function.conv_net import AllSequential, Flatten


class S_FA(ValueFunction):
    '''
    An action-value function approximator for Coindrop problem
    '''

    def __init__(self,
                 config,
                 model,
                 feature_eng):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Racecar S FA")

        self.num_columns = config.NUM_COLUMNS
        self.num_rows = config.NUM_ROWS

        self.model = model
        self.feature_eng = feature_eng
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')
        
    def prefix(self):
        return 'coindrop_M%s_F%s' % (self.model.prefix(),
                                    self.feature_eng.prefix())

    def init_default_model(self):
        A = 100
        B = 300
        C = 100
        D = 50
        E = 25
        net = AllSequential(collections.OrderedDict([
            ('1conv', nn.Conv2d(2, A, kernel_size=4, stride=1, padding=1)),
            #nn.Dropout2d(p=0.2),
            ('1relu', nn.LeakyReLU()),
            ('1bn', nn.BatchNorm2d(A)),
            ('2conv', nn.Conv2d(A, B, kernel_size=2, stride=1, padding=0)),
            #nn.Dropout2d(p=0.5),
            ('2relu', nn.LeakyReLU()),
            ('2bn', nn.BatchNorm2d(B)),
            ('3flatten', Flatten()),
            ('3lin', nn.Linear(B*5*4, C)),
            #nn.Dropout(p=0.5),
            ('3relu', nn.LeakyReLU()),
            #('3bn', nn.BatchNorm1d(C)),
            ('4lin', nn.Linear(C, D)),
            #nn.Dropout(p=0.2),
            ('4relu', nn.LeakyReLU()),
            #('4bn', nn.BatchNorm1d(D)),
            ('5lin', nn.Linear(D, E)),
            #nn.Dropout(p=0.2),
            ('5relu', nn.LeakyReLU()),
            #('5bn', nn.BatchNorm1d(E)),
            ('6lin', nn.Linear(E, self.num_outputs())),
            #('6sigmoid', nn.Sigmoid())
            #nn.Tanh()
            ]))
    
        self.model.init_net(net)

    def _value(self, state):
        X = self.feature_eng.x_adjust(state)
        output = self.model.value(X.unsqueeze(0))[0]
        return output * 2 - 1
    
    def value(self, state, action):
        ai = self.a_index(action)
        output = self._value(state)
        return output[ai].item()

    def _actions_mask(self, B):
        valid_actions_mask = (B[6-1] == 0) * 1
        return torch.from_numpy(valid_actions_mask).to(self.device).float()

    def best_action(self, state):
        V = self._value(state) - 1000 * (1 - self._actions_mask(state))
        i = torch.argmax(V).item()
        v = V[i].item()
        return self.action_from_index(i), v, V.tolist()

    def random_action(self, B):
        V = torch.rand(self.num_columns).to(self.device).float()
        V += 1000 * self._actions_mask(B)
        i = torch.argmax(V).item()
        return i

    def num_outputs(self):
        return self.num_columns
    
    def a_index(self, action):
        return action

    def action_from_index(self, a_index):
        return a_index


    # ------- Training --------

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
        steps_history_mask = []

        #Sdict = {}
        #count_conflict = 0
        self.logger.debug("  Preparing for %d items", len(steps_history_state))

        teye = torch.eye(self.num_outputs()).to(self.device)        
        for i, (S, a, target) in enumerate(zip(
                        steps_history_state,
                        steps_history_action,
                        steps_history_target)):
            if i == 250000:
                self.logger.warning("------ too much to prepare ----------")
                break


            for flip in [False, True]:
                if flip:
                    S = np.flip(S, axis=1).copy()
                    a = 6 - a
                x = self.feature_eng.x_adjust(S)
                t = (target + 1.0) / 2
                ai = self.a_index(a)
                m = teye[ai].clone()
            
                steps_history_x.append(x)
                steps_history_t.append(t)
                steps_history_mask.append(m)

            if (i+1) % 10000 == 0:
                self.logger.debug("prepared %d", i+1)
                #self.logger.debug("  conflict count: %d" % count_conflict)
            
        #util.hist(list(Sdict.values()), bins=100, range=(2,50))
        
        return steps_history_x, steps_history_t, steps_history_mask


    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self, training_data_collector, validation_data_collector):
        self.logger.debug("Preparing training data--")
        steps_history_x, steps_history_t, steps_history_m = \
            self._prepare_data(*training_data_collector.get_data())
        self.logger.debug("Preparing validation data--")
        val_steps_history_x, val_steps_history_t, val_steps_history_m = \
            self._prepare_data(*validation_data_collector.get_data())
        
        SHX = torch.stack(steps_history_x)
        SHT = torch.tensor(steps_history_t)
        SHM = torch.stack(steps_history_m)
        VSHX = torch.stack(val_steps_history_x)
        VSHT = torch.tensor(val_steps_history_t)
        VSHM = torch.stack(val_steps_history_m)

        self.model.train(SHX, SHT, SHM, VSHX, VSHT, VSHM)

    def test(self):
        pass

    def collect_stats(self, ep):
        self.model.collect_stats(ep)
    
    def collect_epoch_stats(self, epoch):
        self.model.collect_epoch_stats(epoch)
        self.feature_eng.collect_epoch_stats(epoch)
    
    def save_stats(self, pref=""):
        self.model.save_stats(pref)

    def load_stats(self, subdir, pref=""):
        self.model.load_stats(pref)

    def report_stats(self, pref=""):
        self.model.report_stats(pref)
    
    def live_stats(self):
        self.model.live_stats()
        
    def save_model(self, pref=""):
        self.model.save_model(pref)

    def load_model(self, load_subdir, pref=""):
        self.model.load_model(load_subdir, pref)
