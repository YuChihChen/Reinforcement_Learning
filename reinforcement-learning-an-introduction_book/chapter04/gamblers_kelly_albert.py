import math
import numpy as np
import matplotlib.pyplot as plt

from agent import AgentMDP
from enviroment import EnviromentMDP


class GamblerEnviroment(EnviromentMDP):
    def __init__(self, money_to_win, head_prob, win_factor, lose_factor):
        self.money_to_win = money_to_win
        self.head_prob = head_prob
        self.w = win_factor
        self.l = lose_factor
        super().__init__()
    
    def get_states(self):
        """Return list of states of the enviroment."""
        return list(range(self.money_to_win+1))
    
    def get_actions(self, state):
        """Return list of actions which are allowed by agent.
        
        a in [0, 1, 2, ..., state]
        """
        if state == self.money_to_win or state == 0:
            return [0]
        return list(range(0, state+1))

    def get_transition_information(self, state, action):
        """Return list of [s', r, P(s',r|s,a)]."""
        if state == self.money_to_win:
            return [(self.money_to_win, 0, 1)]
        if state == 0:
            return [(0, 0, 1)]

        win_reward = self.w * action
        win_state = state + win_reward
        if win_state >= self.money_to_win:
            ti_win = (self.money_to_win, win_reward, self.head_prob)
        else:
            ti_win = (win_state, win_reward, self.head_prob)
        
        lose_reward = - self.l * action
        lose_state = state + lose_reward
        if lose_state <= 0:
            ti_lose = (0, lose_reward, 1-self.head_prob)
        else:
            ti_lose = (lose_state, lose_reward, 1-self.head_prob)
        return [ti_win, ti_lose]


class GamblerAgent(AgentMDP):
    def __init__(self, env, gamma):
        super().__init__(env, gamma)


if __name__ == '__main__':
    
    # env = GamblerEnviroment(10, 0.5, 2, 1)
    # agent = GamblerAgent(env, 1.)
    # value_opt = agent.get_optimal_v_value_by_value_iteration(epsilon=1)
    # print(value_opt)
    # policy_opt = agent.extract_policy_from_values(value_opt)
    # print('\n\n')
    # print(policy_opt)


    # density of N!/k!(N-k)!
    # n! ~ sqrt(2*pi*n) * (n/e)^n
    def approximate_n_factorial(n):
        return np.sqrt(2 * math.pi * n) * ((n / math.e) ** n)
    
    def approximate_log_n_factorial(n):
        if n == 0:
            return 0
        a = np.log(np.sqrt(2 * math.pi * n))
        b = n * np.log(n / math.e)
        return  a + b

    def CN_k(N, k):
        log_Nf = approximate_log_n_factorial(N)
        log_kf = approximate_log_n_factorial(k)
        log_Nmkf = approximate_log_n_factorial(N-k)
        log_CN_k = log_Nf - log_kf - log_Nmkf
        return np.exp(log_CN_k)
    
    def CN_k_over_2_N(N, k):
        log_Nf = approximate_log_n_factorial(N)
        log_kf = approximate_log_n_factorial(k)
        log_Nmkf = approximate_log_n_factorial(N-k)
        log_CN_k = log_Nf - log_kf - log_Nmkf
        log_2_N = N * np.log(2)
        return np.exp(log_CN_k - log_2_N)

    N = 1000
    x = list(range(N+1))
    y = [CN_k_over_2_N(N, k) for k in x]
    plt.plot(x, y)
    plt.show()
    print(np.sum(y[400:600]))