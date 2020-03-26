import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


class MultiArmedBandit():
    """
    n_arm: int; the number of arms; default to be 10
    r_mu: list with length of k_arm; expended value of returns for each arm 
            if r_mu = None, it will be set randomly by standar normal 
    r_std: list with length of k_arm; stander deviation of returns for each arm
            if r_std = None, it will be set as 1.0 for each arm
    """
    def __init__(self, n_arm=10, r_mu=None, r_std=None):
        self.n_arm = n_arm
        self.r_mu = self.__get_r_mu(n_arm, r_mu)
        self.r_std = self.__get_r_std(n_arm, r_std)
        self.best_action = np.argmax(self.r_mu)
    
    @staticmethod
    def __get_r_mu(n_arm, r_mu):
        if r_mu is None:
            output = list(np.random.randn(n_arm))
        else:
            assert len(r_mu) == n_arm, f'len(r_mu)={len(r_mu)} != n_arm={n_arm}'
            output = [float(mu) for mu in r_mu]
        return output
    
    @staticmethod
    def __get_r_std(n_arm, r_std):
        if r_std is None:
            output = [1.] * n_arm
        else:
            assert len(r_std) == n_arm, f'len(r_std)={len(r_std)} != n_arm={n_arm}'
            output = [float(std) for std in r_std]
        return output

    def an_arm_be_pulled(self, k):
        assert k < self.n_arm, f'The bandit has no {k}th arm'
        r = float(self.r_std[k] * np.random.randn() + self.r_mu[k])
        return r
    

class Player():
    """
    q0: float, intitial setting of q values for each arm, default to be 0
    """
    def __init__(self, bandit, q0=0.):
        self.bandit = bandit
        self.n_arm = bandit.n_arm
        self.q0 = q0
        self.reset()
    
    def reset(self):
        self.N = 0
        self.r_ave = 0.
        self.rewards = dict()
        self.Na = np.zeros(self.n_arm)
        self.q_val = np.zeros(self.n_arm) + self.q0
        self.Ha = np.zeros(self.n_arm)
    
    def get_action(self, algo, **kwarg):
        if algo == 'epsilon_greedy':
            return self.epsilon_greedy(**kwarg)
        elif algo == 'UCB':
            return self.upper_confidence_bound(**kwarg)
        elif algo == 'policy_gradient':
            return self.policy_gradient()
        else:
            raise ValueError(f'algorithm {algo} is not avaliable')

    def epsilon_greedy(self, eps):
        if np.random.rand() < eps:
            return np.random.choice(range(self.n_arm))
        return np.argmax(self.q_val)

    def upper_confidence_bound(self, c):
        qu = self.q_val + c * ( np.log(self.N) / (self.Na + 1e-10) ) ** (1/2)
        return np.argmax(qu)

    def policy_gradient(self):
        expHa = np.exp(self.Ha)
        PIa = expHa / np.sum(expHa)
        return np.random.choice(range(self.n_arm), p=PIa)

    def pulling_an_arm(self, k):
        r = self.bandit.an_arm_be_pulled(k)
        if k not in self.rewards:
            self.rewards[k] = [r]
        else:
            self.rewards[k].append(r)
        self.Na[k] += 1
        self.N += 1
        self.r_ave = self.r_ave + (r - self.r_ave) / self.N
        return r

    def update_q(self, k, r, alpha=None):
        """ used after pulling an arm """
        assert k in range(self.n_arm)
        alpha = 1/self.Na[k] if alpha is None else alpha
        self.q_val[k] = self.q_val[k] + alpha * (r - self.q_val[k])

    def update_Ha(self, k, r, alpha):
        """ used after pulling an arm """
        assert k in range(self.n_arm)
        I_Ata = np.zeros(self.n_arm)
        I_Ata[k] += 1
        expHa = np.exp(self.Ha)
        PIa = expHa / np.sum(expHa)
        self.Ha = self.Ha + alpha * (r - self.r_ave) * (I_Ata - PIa)


def simulation(bandit, player, runs, times, algo, alpha_Ha=0.1, **kwarg):
    rewards = np.zeros((runs, times))
    hit_best_action = np.zeros((runs, times))
    I = tqdm(range(runs))
    for i in I:
        I.set_description(desc=f' run {i}')
        player.reset()
        for t in range(times):
            k = player.get_action(algo=algo, **kwarg)
            r = player.pulling_an_arm(k)
            player.update_q(k, r)
            if algo == 'policy_gradient':
                player.update_Ha(k, r, alpha_Ha)
            rewards[i, t] = r
            if k == bandit.best_action:
                hit_best_action[i, t] = 1

    return rewards, hit_best_action
    

if __name__ == '__main__':
    # r_mu = np.random.uniform(low=0, high=5, size=10)
    # bandit = MultiArmedBandit()
    # player = Player(bandit)
    # rewards, hit_best_action = simulation(bandit, player, 2000, 1000, 'epsilon_greedy', eps=0.1)

    # ts = list(range(rewards.shape[1]))
    # plt.hlines(y=bandit.r_mu[bandit.best_action], xmin=0, xmax=1000)
    # plt.plot(ts, rewards.mean(axis=0))
    # plt.show()

    # plt.hlines(y=1, xmin=0, xmax=1000)
    # plt.plot(ts, hit_best_action.mean(axis=0))
    # plt.show()


    a = 3
    
    


