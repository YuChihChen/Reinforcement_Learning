import numpy as np
from cvxopt import matrix, solvers
from cvxopt.solvers import qp

def min_val(alpha, S, p, x):
    S = np.array(S)
    p = np.array(p)
    x = np.array(x)
    return alpha * x.T.dot(S).dot(x) - x.T.dot(p)
     

"""
min alpha * x.T * S * x - x.T * p 
"""
alpha = 0.3
S = matrix([[ 0.09,  0.006, -0.075],
            [ 0.006,  0.01,  0.02],
            [-0.075,  0.02,  0.25]])
p = matrix([0.12, -0.15, 0.03])
ans1 = qp(2 * alpha * S, -p)
x1 = ans1['x']
print('========= solution 1 =========')
print(x1)


"""
min alpha * x.T * S * x + y
s.t. 
    - x.T * p - y <= 0
"""
alpha = 0.3
Se = matrix([ [0.0,    0.0,    0.0,    0.0],
              [0.0,   0.09,  0.006, -0.075],
              [0.0,  0.006,   0.01,   0.02],
              [0.0, -0.075,   0.02,   0.25]])
pe = matrix([1, 0.0, 0.0, 0.0])
G = matrix([[-1, -0.12, 0.15, -0.03]], (1,4))
h = matrix(0.0, (1,1))
ans2 = qp(2 * alpha * Se, pe, G, h)
x2 = ans2['x'][1:]
print('========= solution 2 =========')
print(x2)


"""
min y - x.T * p
s.t.
    alpha * x.T * S * x - y <= 0

S = K.T * K
"""
alpha = 0.3
c = matrix([1., -0.12, 0.15, -0.03])

K = (alpha ** 0.5) *np.linalg.cholesky(S).T
Ke = np.pad(K, ((1,0), (1,0)), 'constant', constant_values=(0, 0))


t0 = -0.5 * np.array([[ 1, 0, 0, 0, 1]])
t1 =  0.5 * np.array([[-1, 0, 0, 0, 1]])
t3 = np.pad(Ke, ((0,0), (0,1)), 'constant', constant_values=(0, 0))[1:,:]
T = np.r_[t0, t1, t3]


G = [ matrix( T[:, :-1] ) ]
h = [ matrix( -T[:, -1]) ]

ans3 = solvers.socp(c, Gq = G, hq = h)
x3 = ans3['x'][1:]
print('========= solution 3 =========')
print(x3)


