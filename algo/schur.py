import numpy as np
from scipy.linalg import schur

T = np.array([[ 0, 1, 2, 3],
              [-1, 0, 1, 2],
              [-2,-1, 0, 1],
              [-3,-2,-1, 0]])

C = np.array([[ 0, 1, 0,-1],
              [-1, 0, 1, 0],
              [ 0,-1, 0, 1],
              [-1, 0, 1, 0]])

K = np.array([[ 0, 1,-1],
              [-1, 0, 1],
              [ 1,-1, 0]])

a = schur(T)[1]
print(schur(C)[1])
d = schur(K)[1]

from matplotlib import pyplot as plt

print(a)
x, y = a[:,0].T, a[:,1].T
print(x, y)
plt.scatter(x, y)
plt.show()

print(d)
x, y = d[:,0].T, d[:,1].T
print(x, y)
plt.scatter(x, y)
plt.show()

N = 30
C = 60
labels = list(range(0, 3))
t = np.zeros((N, N))
c = np.random.rand(N, 3)
for i in range(N):
    c[i,np.random.randint(3)] = 0
    c[i,:] /= c[i,:].sum()
for i in range(N):
    for j in range(N):
        for _ in range(C):
            mi = np.random.choice(labels, p=c[i,:])
            mj = np.random.choice(labels, p=c[j,:])
            r = K[mi,mj]
            t[i,j] += r
            t[j,i] -= r

print(c)
print(t)

d = schur(t)[1]
print(d)
x, y = d[:,0].T, d[:,1].T
print(x, y)
plt.scatter(x, y)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()