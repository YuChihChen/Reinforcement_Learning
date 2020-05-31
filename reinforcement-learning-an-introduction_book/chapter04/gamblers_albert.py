import numpy as np
import matplotlib.pyplot as plt

from agent import AgentMDP
from enviroment import EnviromentMDP

""" 一維格子問題
情境：
    挑戰者一開始會處於 s in [1, 2, 3, ..., 99] 位置，
    挑戰者可以選擇動作 a in [0, 1, 2, ... min(s, 100-s)]
    有 ph 的機率到 s' = s+a ， 有 1-ph 的機率到 s' = s-a
    挑戰者到達 >= 99 則獲勝，到達 <=0 則失敗
目標：
    以最少步數到達

"""
class GamblerEnviroment(EnviromentMDP):
    def __init__(self, money_to_win, head_prob):
        self.money_to_win = money_to_win
        self.head_prob = head_prob
        super().__init__()
    
    def get_states(self):
        """Return list of states of the enviroment."""
        return list(range(self.money_to_win+1))
    
    def get_actions(self, state):
        """Return list of actions which are allowed by agent.
        
        a in [0, 1, 2, ..., min(s, money_to_win - s)]
        """
        if state == self.money_to_win or state == 0:
            return [0]
        max_a = min(state, self.money_to_win - state)
        return list(range(0, max_a+1))

    def get_transition_information(self, state, action):
        """Return list of [s', r, P(s',r|s,a)]."""
        if state == self.money_to_win:
            return [(self.money_to_win, 0, 1)]
        if state == 0:
            return [(0, 0, 1)]
        win_state = min(state+action, self.money_to_win)
        lose_state = max(0, state-action)
        
        is_final_win = (win_state == self.money_to_win) * 1
        is_final_lose = (lose_state == 0) * 0
        ti_win = (win_state, is_final_win, self.head_prob)
        ti_lose = (lose_state, -is_final_lose, 1-self.head_prob)
        return [ti_win, ti_lose]


class GamblerAgent(AgentMDP):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)


if __name__ == '__main__':
    env = GamblerEnviroment(100, 0.4)
    agent = GamblerAgent(env, 1 g)
    value_opt = agent.get_optimal_v_value_by_value_iteration(epsilon=0.000001)
    print(value_opt, '\n\n')
    plt.plot(value_opt.keys(), value_opt.values())
    plt.show()
    policy_opt = agent.extract_policy_from_values(value_opt)
    print(policy_opt)