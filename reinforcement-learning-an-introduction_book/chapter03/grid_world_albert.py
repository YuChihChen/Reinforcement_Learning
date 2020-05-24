import numpy as np

from agent import AgentMDP
from enviroment import EnviromentMDP


class GridWordEnviroment(EnviromentMDP):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.A_position = (0, 1)
        self.A_prime_position = (4, 1)
        self.B_position = (0, 3)
        self.B_prime_position = (2, 3)
    
    def get_states(self):
        """Return list of states of the enviroment."""
        states = list()
        for i in range(grid_size):
            for j in range(grid_size):
                states.append((i, j))
        return states

    def get_actions(self, state):
        """Return list of actions which are allowed by agent.
        
        left: (0, -1), right: (0, +1), up: (-1, 0), down: (+1, 0)
        """
        return [(0, -1), (0, +1), (-1, 0), (+1, 0)]
    
    def get_transition_information(self, state, action):
        """Return list of [s', r, P(s',r|s,a)]."""
        if state == self.A_position:
            new_state = self.A_prime_position
            reward = 10.
        elif state == self.B_position:
            new_state = self.B_prime_position
            reward = 5.
        else:
            new_state = (state[0] + action[0], state[1] + action[1])
            if self._is_ouside_of_grid(new_state):
                new_state = state
                reward = -1.
            else:
                reward = 0.
        return [(new_state, reward, 1.)]
        

    def _is_ouside_of_grid(self, state):
        """Check if the state is outside the the grid world."""
        state_np = np.array(state)
        if (state_np < 0).any() or (state_np > grid_size-1).any():
            return True
        return False


class GridWordAgent(AgentMDP):
    def __init__(self, env, gamma):
        super().__init__()


if __name__ == '__main__':
    grid_size = 5
    env = GridWordEnviroment(grid_size=grid_size)
    agent = AgentMDP(env, gamma=0.9)
    Vvalues_random_policy = agent.policy_evaluation(policy=agent.random_policy, mode='Vp')
    Qvalues_random_policy = agent.policy_evaluation(policy=agent.random_policy, mode='Qp')
    for state in Vvalues_random_policy:
        Vp_s = 0.
        for action in Qvalues_random_policy[state]:
            Vp_s += Qvalues_random_policy[state][action] * 0.25
        diff = np.abs(Vvalues_random_policy[state] - Vp_s)
        print(f'state: {state}, diff: {diff}')