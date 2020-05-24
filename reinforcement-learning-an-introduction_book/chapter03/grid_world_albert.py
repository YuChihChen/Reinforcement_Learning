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
        super().__init__(env, gamma)
        self.grid_size = env.grid_size

    def fill_Vvalues_in_grid(self, values):
        grid = np.zeros((self.grid_size, self.grid_size))
        for state in values:
            grid[state[0], state[1]] = values[state]
        return grid
        

if __name__ == '__main__':
    grid_size = 5
    env = GridWordEnviroment(grid_size=grid_size)
    agent = GridWordAgent(env, gamma=0.9)

    Vvalues_random_policy = agent.get_v_value_by_policy_evaluation(policy=agent.random_policy)
    Qvalues_random_policy = agent.get_q_value_by_policy_evaluation(policy=agent.random_policy)
    for state in Vvalues_random_policy:
        Vp_s = 0.
        for action in Qvalues_random_policy[state]:
            Vp_s += Qvalues_random_policy[state][action] * 0.25
        diff = np.abs(Vvalues_random_policy[state] - Vp_s)
        print(f'state: {state}, diff: {diff}')
    
    Vvalues_optimal = agent.get_optimal_v_value_by_value_iteration()

    Vvalues_in_grid = agent.fill_Vvalues_in_grid(Vvalues_optimal)

    policy_optimal = agent.extract_policy_from_values(Vvalues_optimal)
    Vvalues_optimal_policy = agent.get_v_value_by_policy_evaluation(policy_optimal)
    optimal_Vvalues_in_grid = agent.fill_Vvalues_in_grid(Vvalues_optimal_policy)
    print(np.abs(Vvalues_in_grid - optimal_Vvalues_in_grid))
    
    policy_optimal2 = agent.get_optimal_policy_by_policy_iteration(max_inner_iteration=1)
    print(policy_optimal == policy_optimal2)