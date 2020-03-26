import gym

def value_iteration(env, gamma=1.0):
    values = [0] * env.observation_space.n
    iterations = 100000
    threshold = 1e-20
    for i in range(iterations):
        valuesOld = values.copy()
        diff = 0
        for state in range(env.observation_space.n):
            Qvalues = list()
            for action in range(env.action_space.n):
                Qa = 0
                for prob, nextS, reward, _  in env.P[state][action]:
                    Qa += prob * (reward + gamma * valuesOld[nextS])
                Qvalues.append(Qa)
            values[state] = max(Qvalues)
            diff += abs(values[state] - valuesOld[state])
        if diff < threshold:
            print(f'value iteration converged at iterations {i}')
            break
    return values
                    

def extract_policy(env, optValues, gamma=1.0):
    policy = [0] * env.observation_space.n
    for state in range(env.observation_space.n):
        optQ = float("-inf")
        for action in range(env.action_space.n):
            Qa = 0
            for prob, nextS, reward, _  in env.P[state][action]:
                Qa += prob * (reward + gamma * optValues[nextS])
            if Qa > optQ:
                optQ = Qa
                policy[state] = action
    return policy



if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    optimal_values = value_iteration(env=env, gamma=1.0)
    optimal_policy = extract_policy(env=env, optValues=optimal_values, gamma=1.0)
    print(optimal_policy)