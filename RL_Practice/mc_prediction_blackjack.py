import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

"""
env.action_space : Discrete(2) => 0 to Stand and 1 to Hit
env.observation_space : Tuple(Discrete(32), Discrete(11), Discrete(2))
    [0]: player's sum => min = 4 and max = 21 => total number = 31
    [1]: dealer's showing => min = 1 and max = 11 => total number = 11
    [2]: usable ace => False A=1, True A=11

"""
def find_min_max(env, iterations=1000):
    min0, max0, min1, max1 = (100, -1, 100, -1)
    for _ in range(iterations):
        obs = env.reset()
        if obs[0] < min0: min0 = obs[0]
        if obs[0] > max0: max0 = obs[0]
        if obs[1] < min1: min1 = obs[1]
        if obs[1] > max1: max1 = obs[1]
    print(f'player: min={min0}, max={max0}')
    print(f'dealer: min={min1}, max={max1}')


def policy_like21(obs):
    player, dealer, ace = obs
    return 0 if player >= 20 else 1


def one_game(policy, env):
    states, actions, rewards = [], [], []
    done = False
    obs = env.reset()
    while not done:
        states.append(obs)
        act = policy(obs)
        actions.append(act)
        obs, rew, done, info = env.step(act)
        rewards.append(rew)
    return states, actions, rewards


def first_visit_mc_prediction(policy, env, iterations):
    Vs = dict()
    Ns = dict()
    for i in range(iterations):
        states, actions, rewards = one_game(policy, env)
        size = len(states)
        R = 0
        for t in range(size-1, -1, -1):
            s = states[t]
            R += rewards[t]
            if s not in states[:t]:
                if s not in Vs:
                    Ns[s] = 1
                    Vs[s] = R
                else:
                    Ns[s] += 1
                    Vs[s] += (R - Vs[s]) / Ns[s]
        ps = (4, 2, False)
        if i % 1000 == 0 and ps in Vs:
            print(f'step:{i}, stae:{ps}, value:{Vs[ps]}') 
    return Vs
        

if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    Vs = first_visit_mc_prediction(policy_like21, env, 500000)