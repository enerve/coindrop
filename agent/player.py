'''
Created on 24 Apr 2019

@author: enerve
'''

class Player(object):
    '''
    Base class for a connect-4 player
    '''

    def init_game(self, initial_state, initial_heights):
        ''' Initialize for a new game, and note player's start state
        '''
        self.S = initial_state
        
    def see_move(self, reward, new_state, h, moves=0):
        ''' Observe the effects on this agent of an action taken - possibly by
            another agent.
        '''
        self.S_ = new_state
         
    def next_move(self):
        ''' Agent's turn. Chooses the next move '''
        pass

    def game_over(self):
        ''' Wrap up game '''
        self.S_ = None

    def save_game(self):
        ''' Record game history for future purposes '''
        pass
    
    def learn_from_history(self):
        pass
    
    def process(self):
        ''' Process whatever happened so far
        '''
        pass

    def collect_stats(self, ep, num_episodes):
        pass
    
    def save_stats(self, pref=""):
        pass

    def load_stats(self, subdir, pref=""):
        pass

    def report_stats(self, pref):
        pass
    
    def live_stats(self):
        pass
