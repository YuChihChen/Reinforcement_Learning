import gym

def policy_evaluation(env, policy, gamma=1.0):
    values = [0] * env.observation_space.n
    iterations = 100000
    threshold = 1e-10
    for i in range(iterations):
        valuesOld = values.copy()
        diff = 0
        for state in range(env.observation_space.n):
            values[state] = 0
            for prob, nextS, reward, _  in env.P[state][policy[state]]:
                values[state] += prob * (reward + gamma * valuesOld[nextS])
            diff += abs(values[state] - valuesOld[state])
        if diff < threshold:
            print(f'policy evaluation converged at iterations {i}')
            break
    return values


def extract_policy(env, values, gamma=1.0):
    policy = [0] * env.observation_space.n
    for state in range(env.observation_space.n):
        optQ = float("-inf")
        for action in range(env.action_space.n):
            Qa = 0
            for prob, nextS, reward, _  in env.P[state][action]:
                Qa += prob * (reward + gamma * values[nextS])
            if Qa > optQ:
                optQ = Qa
                policy[state] = action
    return policy


def policy_iteration(env, gamma=1.0):
    policy = [0] * env.observation_space.n
    iterations = 100000
    for i in range(iterations):
        policyOld = policy.copy()
        vPI = policy_evaluation(env, policyOld, gamma)
        policy = extract_policy(env, vPI, gamma)
        if policy == policyOld:
            print(f'policy iteration converged at iterations {i}')
            break
    return policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    optimal_policy = policy_iteration(env=env, gamma=1.0)
    print(optimal_policy)