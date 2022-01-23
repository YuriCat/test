import time
import pickle
import bz2

import numpy as np

a = np.random.random((1000, 1000))
b = a.astype(np.float32)
c = a.astype(np.float16)
d = c.astype(np.float32)
print(a.dtype, b.dtype, c.dtype, d.dtype)

t = time.time()
k = bz2.compress(pickle.dumps(a))
tt = time.time()
pickle.loads(bz2.decompress(k))
print(len(k), tt - t, time.time() - tt)

t = time.time()
k = bz2.compress(pickle.dumps(b))
tt = time.time()
pickle.loads(bz2.decompress(k))
print(len(k), tt - t, time.time() - tt)

t = time.time()
k = bz2.compress(pickle.dumps(c))
tt = time.time()
pickle.loads(bz2.decompress(k))
print(len(k), tt - t, time.time() - tt)

t = time.time()
k = bz2.compress(pickle.dumps(d))
tt = time.time()
pickle.loads(bz2.decompress(k))
print(len(k), tt - t, time.time() - tt)
