# Restricted Boltzmann Machine

# length of V ... N
# length of H ... M

M, N = 4, 3

# visible units V
# weight matrix W
# hidden units H
# bias variable for visible units A
# bias variable for hidden units B

import numpy as np

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

def i2array(i, d):
  a = np.zeros((d))
  for j in range(d):
    if (i // (2 ** j)) % 2 == 1:
      a[j] = 1
  return a

def d2arrays(d):
  return [i2array(i, d) for i in range(2 ** d)]

def energy(V, H, p):
  return -np.dot(p['A'], V) - np.dot(p['B'], H) - np.dot(V, p['W'] @ H)

def energy_v(h, P):
  ''' compute energy on visible units '''
  return np.array([energy(v, h, P) for v in d2arrays(M)])

def energy_h(v, P):
  ''' compute energy on hidden units '''
  return np.array([energy(v, h, P) for h in d2arrays(N)])

def energy_vh(P):
  ''' compute energy on all units '''
  return np.array([[energy(v, h, P) for h in d2arrays(N)] for v in d2arrays(M)])

def probability_v(h, P):
  ''' compute probability distribution on visible units '''
  return softmax(-energy_v(h, P))

def probability_h(v, P):
  ''' compute probability distribution on hidden units '''
  return softmax(-energy_h(v, P))

def probability_vh(P):
  ''' compute probability distribution on all units '''
  return softmax(-energy_vh(P))

def train(samples, steps, lr, lr_decay):
  P = {'A':np.zeros((M)), 'B':np.zeros((N)), 'W':np.zeros((M, N))}
  for s in range(steps):
    eps = lr / (1 + s * lr_decay)
    v = samples[np.random.randint(len(samples))]
    # 1. calclate probabilities on H
    p_h = probability_h(v, P)
    #print('p_h = ', p_h)
    # 2. sample hidden unit over that distribution
    h = i2array(np.random.choice(np.arange(2 ** N), p=p_h), N)
    # 3. compute positive gradient
    pos_grad = np.outer(v, h)
    # 4. calcurate plobabilities on V
    p_v_prime = probability_v(h, P)
    #print('p_v_prime = ', p_v_prime)
    # 5. sample visible unit over that distribution
    v_prime = i2array(np.random.choice(np.arange(2 ** M), p=p_v_prime), M)
    # 6. calcurate probabilities on H
    p_h_prime = probability_h(v_prime, P)
    #print('p_h_prime = ', p_h_prime)
    # 7. sample hidden unit over that distribution
    h_prime = i2array(np.random.choice(np.arange(2 ** N), p=p_h_prime), N)
    # 8. compute positive gradient
    neg_grad = np.outer(v_prime, h_prime)
    # 9. update weight matrix W
    #print(pos_grad)
    #print(neg_grad)
    P['W'] += eps * (pos_grad - neg_grad)
    # 10. update bias vector A and B
    P['A'] += eps * (v - v_prime)
    P['B'] += eps * (h - h_prime)
  return P

print(i2array(2, 2))
print(d2arrays(3))

samples = [[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]]
P = train(samples, 100000, 1, 1e-3)

p_vh = probability_vh(P)
print((p_vh * 10000).astype(int))

#for h in d2arrays(N):
#  print((probability_v(h, P) * 10000).astype(int))

print((np.mean([probability_v(h, P) for h in d2arrays(N)], axis=0) * 10000).astype(int))

print((p_vh.sum(axis=0) * 10000).astype(int))
print((p_vh.sum(axis=1) * 10000).astype(int))
