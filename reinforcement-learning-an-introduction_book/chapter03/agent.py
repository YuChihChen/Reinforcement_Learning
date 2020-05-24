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

    # =========================== Bellman Iteration ===========================
    def _get_value_by_Bellman_equation(self, mode, inital_value, epsilon, max_iteration, update_fun, **kwargs):
        """Run value iteration for Bellman equation

        Inputs
            mode: ['v-value', 'q-value']
            inital_value: initial v-value or q-value
            epsilon: if all |new_value(s) - old_value(s)| < epsilon, then stop the while loop 
            max_iteration: if iteration > max_iteration, then stop the while loop
            update_fun: new_value[s] = update_fun(state, old_values, **kwargs)
        """
        if mode not in ['v-value', 'q-value']:
            raise ValueError(f'mode {mode} is not avaliable')
        new_values = inital_value
        max_diff = float('inf')
        num_iterations = 0
        while max_diff > epsilon and num_iterations < max_iteration:
            old_values = new_values.copy()
            max_diff = 0
            for state in self.states:
                new_values[state] = update_fun(state, old_values, **kwargs)
                Vdiff = self._get_max_value_abs_difference(state, new_values, old_values, mode)
                if Vdiff > max_diff:
                    max_diff = Vdiff
            num_iterations += 1
            if num_iterations % 1000 == 0:
                print(f'Step: {num_iterations}/{max_iteration}')
        return new_values

    def _get_values_initialization(self, mode):
        values = dict()
        if mode == 'v-value':
            for state in self.states:
                values[state] = 0.
        elif mode == 'q-value':
            for state in self.states:
                values[state] = dict()
                for action in self.env.get_actions(state):
                    values[state][action] = 0.
        return values

    @staticmethod
    def _get_max_value_abs_difference(state, new_values, old_values, mode):
        if mode == 'v-value':
            max_diff = np.abs(new_values[state] - old_values[state])
        if mode == 'q-value':
            max_diff = 0.
            for action in new_values[state]:
                diff = np.abs(new_values[state][action] - old_values[state][action])
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    # =========================== Funcitons for Evaluation ===========================
    def get_v_value_by_policy_evaluation(self, policy, initial_value=None, 
                                         epsilon=0.01, max_iteration=100000):
        """Get values through value iteration with a give policy.
        
        Formula
            Vp(s) = sum_{a, s',r} P(a|s) * P(s',r|s, a) * [ r + gamma * Vp(s') ]
        """
        if initial_value is None:
            initial_value = self._get_values_initialization('v-value')
        update_fun = self._update_Vp_s
        output_value = self._get_value_by_Bellman_equation('v-value', initial_value, epsilon, max_iteration, 
                                                     update_fun, policy=policy)
        return output_value
    
    def get_q_value_by_policy_evaluation(self, policy, initial_value=None, 
                                         epsilon=0.01, max_iteration=100000):
        """Get values through value iteration with a give policy.
        
        Formula
            Qp(s,a) := sum_{s',r} P(s',r|s, a) * [ r + gamma * Vp(s') ]
                     = sum_{s',r} P(s',r|s, a) * [ r + gamma * sum_a' P(a'|s') * Qp(s') ]
        """
        if initial_value is None:
            initial_value = self._get_values_initialization('q-value')
        update_fun = self._update_Qp_sa
        output_value = self._get_value_by_Bellman_equation('q-value', initial_value, epsilon, max_iteration, 
                                                     update_fun, policy=policy)
        return output_value

    def get_Vp_from_Qp(self, state, Qvalues, policy):
        """Run Vp(s) = sum_a P(a|s) * Qp(s)."""
        Vp_s = 0.
        for action, Pa_s in policy[state].items():
            Vp_s += Pa_s * Qvalues[state][action]
        return Vp_s

    def _update_Vp_s(self, state, values, policy):
        """Run Vp(s) = sum_{a, s',r} P(a|s) * P(s',r|s, a) * [ r + gamma * Vp(s') ]."""
        oVp = 0.
        for action, Pa_s in policy[state].items():
            for sp, r, Psr_sa in self.env.get_transition_information(state, action):
                oVp += Pa_s * Psr_sa * (r + self.gamma * values[sp])
        return oVp

    def _update_Qp_sa(self, state, Qvalues, policy):
        """Run sum_{s',r} P(s',r|s, a) * [ r + gamma * sum_a' P(a'|s') * Qp(s') ]."""
        oQp = dict()
        for action in self.env.get_actions(state):
            oQp[action] = 0.
            for sp, r, Psr_sa in self.env.get_transition_information(state, action):
                Vp_sp = self.get_Vp_from_Qp(sp, Qvalues, policy)
                oQp[action] += Psr_sa * (r + self.gamma * Vp_sp)
        return oQp
    
    # =========================== Optimization: Value Iteration ===========================
    def get_optimal_v_value_by_value_iteration(self, initial_value=None, 
                                               epsilon=0.01, max_iteration=100000):
        """Get optimal values through value iteration.
        
        Formula
            Vo(s) = max_a sum_{s',r} P(s',r|s, a) * [ r + gamma * Vo(s') ]
        """
        if initial_value is None:
            initial_value = self._get_values_initialization('v-value')
        update_fun = self._update_Vo_s
        output_value = self._get_value_by_Bellman_equation('v-value', initial_value, epsilon, max_iteration, 
                                                     update_fun)
        return output_value

    def extract_policy_from_values(self, values, sigle_action=False):
        """Return an optimal policy for a given values: {state: {action: P(a|s)}}."""
        policy = dict()
        for state in values:
            optimal_actions = list()
            maxQ = 0.
            for action in self.env.get_actions(state):
                Qv = 0.
                for sp, r, Psr_sa in self.env.get_transition_information(state, action):
                    Qv += Psr_sa * (r + self.gamma * values[sp])
                if Qv == maxQ:
                    optimal_actions.append(action)
                elif Qv > maxQ:
                    maxQ = Qv
                    optimal_actions = [action]
            num_actions = len(optimal_actions)
            policy[state] = dict()
            if sigle_action:
                for action in optimal_actions:
                    policy[state][action] = 1.
                    break
            else:
                for action in optimal_actions:
                    policy[state][action] = 1. / num_actions
        return policy

    def _update_Vo_s(self, state, values):
        """Run Vo(s) = max_a sum_{s',r} P(s',r|s, a) * [ r + gamma * Vo(s') ]."""
        Qvalues_list = list()
        for action in self.env.get_actions(state):
            Qv = 0.
            for sp, r, Psr_sa in self.env.get_transition_information(state, action):
                Qv += Psr_sa * (r + self.gamma * values[sp])
            Qvalues_list.append(Qv)
        return max(Qvalues_list)

    # =========================== Optimization: Policy Iteration ===========================
    def get_optimal_policy_by_policy_iteration(self, epsilon=0.01, max_inner_iteration=100, 
                                               max_outer_iteration=100000):
        """Get optimal policy throgh policy iteration

        Inputs
            epsilon: if all |new_value(s) - old_value(s)| < epsilon, then stop policy evaluation 
            max_inner_iteration: if iteration > max_inner_iteration, then stop policy evaluation
            max_outer_iteration: if iteration > max_outer_iteration, then stop the loop
        """
        Vvalues = self._get_values_initialization(mode='v-value')
        policy_old = self.extract_policy_from_values(Vvalues)
        policy_new = self.random_policy
        num_iterations = 0
        while policy_old != policy_new and num_iterations < max_outer_iteration:
            policy_old = policy_new.copy()
            Vvalues = self.get_v_value_by_policy_evaluation(policy_old, Vvalues, epsilon, max_inner_iteration)
            policy_new = self.extract_policy_from_values(Vvalues)
            num_iterations += 1
            if num_iterations % 1000 == 0:
                print(f'Step: {num_iterations}/{max_outer_iteration}')
        policy_new = self.extract_policy_from_values(Vvalues)   
        return policy_new