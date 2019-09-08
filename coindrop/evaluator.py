'''
Created on 23 May 2019

@author: enerve
'''

import logging

from game import Game

class Tester:
    
    def __init__(self, test_agent, opponent):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.test_agent = test_agent
        self.opponent = opponent
    
    def run_test(self, test_runs):
        # Test performance in actual games
        for start_first in [True, False]:
            self.logger.debug("-------- testing as %s player ----------",
                              "first" if start_first else "second")
            agent_sum_score = 0
            num_wins = num_losses = 0
            agent_sum_win_moves = 0
            agent_sum_lose_moves = 0
            if start_first:
                players = [self.test_agent, self.opponent] 
            else: 
                players = [self.opponent, self.test_agent]
            for tep in range(test_runs):
                game = Game(players)
                game.run()
                agent_sum_score += self.test_agent.game_performance()
                if self.test_agent.game_performance() > 0:
                    num_wins += 1 
                    agent_sum_win_moves += self.test_agent.moves
                elif self.test_agent.game_performance() < 0:
                    num_losses += 1
                    agent_sum_lose_moves += self.test_agent.moves
            self.logger.debug("FA agent R: %0.2f   (%d vs %d) / %d" % 
                              (agent_sum_score, num_wins, num_losses, test_runs))                
            if num_wins > 0:
                self.logger.debug("Avg #moves for win: %0.2f" % (agent_sum_win_moves/num_wins))
            if num_losses > 0:
                self.logger.debug("Avg #moves for loss: %0.2f" % (agent_sum_lose_moves/num_losses))
