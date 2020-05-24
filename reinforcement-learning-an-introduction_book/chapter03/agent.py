import random
import numpy as np


class AgentMDP:
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.states = env.states
        self.random_policy = self._get_random_policy()
    
    def _get_random_policy(self):
        """Return an policy: {state: {action: P(a|s)}}."""
        policy = dict()
        for state in self.states:
            dict_action_prob = dict()
            actions = self.env.get_actions(state)
            num_actions = len(actions)
            for action in actions:
                dict_action_prob[action] = 1 / num_actions
            policy[state] = dict_action_prob
        return policy

    # =========================== Funcitons for Policy Evaluation ===========================
    def policy_evaluation(self, policy, mode='Vp', epsilon=0.01, max_iteration=100000):
        """Get values through value iteration with a give policy.
        
        Formula
            Qp(s,a) = sum_{s',r} P(s',r|s, a) * [ r + gamma * Vp(s') ]
                    = sum_{s',r} P(s',r|s, a) * [ r + gamma * sum_a' P(a'|s') * Qp(s') ]
            Vp(s) = sum_{a, s',r} P(a|s) * P(s',r|s, a) * [ r + gamma * Vp(s') ]
                  = sum_a P(a|s) * Qp(s,a)
        Inputs
            policy: dict, {state: {action: P(a|s)}}
            mode: 'Vp' for value calculation and 'Qp' for q-value calculation 
            epsilon: if all |Vp(s)_{k+1} - Vp(s)_{k}| < epsilon, then stop the while loop 
            max_iteration: if iteration > max_iteration, then stop the while loop
        """
        if mode not in ['Vp', 'Qp']:
            raise ValueError(f'mode {mode} is not avaliable')
        new_values = self._get_values_initialization(mode)
        max_diff = float('inf')
        num_iterations = 0
        while max_diff > epsilon and num_iterations < max_iteration:
            old_values = new_values.copy()
            max_diff = 0
            for state in self.states:
                new_values[state] = self._update_values(state, old_values, policy, mode)
                Vdiff = self._get_max_value_abs_difference(state, new_values, old_values, mode)
                if Vdiff > max_diff:
                    max_diff = Vdiff
            num_iterations += 1
            if num_iterations % 100 == 0:
                print(f'Running policy evaluation at step {num_iterations}, max_diff = {max_diff}')
            if num_iterations == max_iteration:
                print(f'Warning! Policy evaluation reach maximum iterations({max_iteration})')
        return new_values
    
    def get_Vp_from_Qp(self, state, Qvalues, policy):
        """Run Vp(s) = sum_a P(a|s) * Qp(s)."""
        Vp_s = 0.
        for action, Pa_s in policy[state].items():
            Vp_s += Pa_s * Qvalues[state][action]
        return Vp_s

    def _get_values_initialization(self, mode):
        values = dict()
        if mode == 'Vp':
            for state in self.states:
                values[state] = 0.
        elif mode == 'Qp':
            for state in self.states:
                values[state] = dict()
                for action in self.env.get_actions(state):
                    values[state][action] = 0.
        return values

    @staticmethod
    def _get_max_value_abs_difference(state, new_values, old_values, mode):
        if mode == 'Vp':
            max_diff = np.abs(new_values[state] - old_values[state])
        if mode == 'Qp':
            max_diff = 0.
            for action in new_values[state]:
                diff = np.abs(new_values[state][action] - old_values[state][action])
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    def _update_values(self, state, values, policy, mode):
        if mode == 'Vp':
            return self._update_Vp_s(state, values, policy)
        elif mode == 'Qp':
            return self._update_Qp_s_a(state, values, policy)

    def _update_Vp_s(self, state, values, policy):
        """Run Vp(s) = sum_{a, s',r} P(a|s) * P(s',r|s, a) * [ r + gamma * Vp(s') ]."""
        oVp = 0.
        for action, Pa_s in policy[state].items():
            for sp, r, Psr_sa in self.env.get_transition_information(state, action):
                oVp += Pa_s * Psr_sa * (r + self.gamma * values[sp])
        return oVp

    def _update_Qp_s_a(self, state, Qvalues, policy):
        """Run sum_{s',r} P(s',r|s, a) * [ r + gamma * sum_a' P(a'|s') * Qp(s') ]."""
        oQp = dict()
        for action in self.env.get_actions(state):
            oQp[action] = 0.
            for sp, r, Psr_sa in self.env.get_transition_information(state, action):
                Vp_sp = self.get_Vp_from_Qp(sp, Qvalues, policy)
                oQp[action] += Psr_sa * (r + self.gamma * Vp_sp)
        return oQp