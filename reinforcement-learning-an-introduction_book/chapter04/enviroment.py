""" Markov Decision Process

--- Definition ---
For contructing a MDP problem, we need to define what is STAR:
    States(S): states of the system
    Transition Probability(T): P(s',r|s,a)
    Action(A): what actions can we take when we at state s
    Rewards(R): (s, a) -> s', what reward we can get

--- Formula ---
Probability to state s' while we take action a at state s, (s, a) -> s' 
    P^a_{s,s'} := P(s'|s,a) = sum_r P(s',r|s,a)

Expected reward getted after reaching state s' while we take action a at state s, (s, a) -> s'
    R^a_{s,s'} := R(s'|s,a) = sum_r r * P(r|s,a,s') = sum_r r * P(s',r|s,a) / P(s'|s,a)

Value function for a given policy
    Vp(s) := sum_{a, s',r} P(a, s',r|s) * [ r + gamma * Vp(s') ]
           = sum_{a, s',r} P(a|s) * P(s',r|s, a) * [ r + gamma * Vp(s') ]
           = sum_{a, s'} P(a|s) * P(s'|s,a) * [R^a_{s,s'} + gamma * Vp(s')]

Action-value function for a given policy
    Qp(s,a) := sum_{s',r} P(s',r|s, a) * [ r + gamma * Vp(s') ]
             = sum_{s',r} P(s',r|s, a) * [ r + gamma * sum_a' P(a'|s') * Qp(s') ]

Relation between Vp(s) and Qp(s,a)
    Vp(s) = sum_a P(a|s) * Qp(s)
"""

class EnviromentMDP:
    def __init__(self):
        self.states = self.get_states()

    def get_states(self):
        """Return list of states of the enviroment."""
        raise NotImplementedError

    def get_actions(self, state):
        """Return list of actions which are allowed by agent."""
        raise NotImplementedError

    def get_transition_information(self, state, action):
        """Return list of [s', r, P(s',r|s,a)]."""
        raise NotImplementedError