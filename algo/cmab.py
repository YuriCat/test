import numpy as np

D = 2
A = 2

R = [[2, 2], [2.1, 0]]

def reward(a):
  r = R
  for d in range(D):
    r = r[a[d]]
  return r

def ucb1(n, n_a, q_a, p_a = 1):
  C = 2.0
  return q_a + C * ((np.log(n) / (n_a + 1)) ** 0.5) * p_a * np.exp(np.random.randn(D, A))

def pucb(n, n_a, q_a, p_a = 1):
  C = 2.0
  return q_a + C * p_a * (n ** 0.5) / (n_a + 1)

def thompson(n, n_a, q_a, p_a = None):
  C = 10.0
  return q_a + C / ((n_a + 1) ** 0.5) * np.random.randn(D, A)

def select(state, algo):
  ucb = algo(state['n'], state['n_a'], state['q_a'])
  #print(ucb)
  return np.argmax(ucb, axis=-1)

def bandit(algo, steps):
  avgv = np.mean(R)
  state = {
    'n':1, 'ev': avgv, 'n_a':np.zeros((D, A)), 'q_a':np.ones((D, A)) * avgv, 'p_a':None, 'v':None
  }
  for s in range(steps):
    a = select(state, algo)
    #print(a)
    r = reward(a)

    state['n'] += 1
    state['ev'] = (state['ev'] * (state['n'] - 1) + r) / state['n']
    for d in range(D):
      state['n_a'][d,a[d]] += 1
      state['q_a'][d,a[d]] = (state['q_a'][d,a[d]] * (state['n_a'][d,a[d]] - 1) + r) / state['n_a'][d,a[d]]
    
    if s & (s + 1) == 0:
      ba = np.argmax(state['n_a'], axis=-1)
      print(state['n_a'])
      print('step %d ev %f a %s qa %s' % (s + 1, state['ev'], ba, [state['q_a'][d,ba[d]] for d in range(D)]))

bandit(ucb1, 262144)
#bandit(pucb, 262144)
#bandit(thompson, 262144)