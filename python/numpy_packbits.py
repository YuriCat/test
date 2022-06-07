import time, bz2, pickle
import numpy as np

a = np.random.randint(2, size=(1000000))
print(a)
print(a.dtype)

t = time.time()
b = np.packbits(a)
print(time.time() - t)
print(b)
print(b.dtype)
print(len(b))

t2 = time.time()
c = np.unpackbits(b)
print(time.time() - t)
print(c)
print(c.dtype)

for val in [a, b, c]:
    t1 = time.time()
    comp = bz2.compress(pickle.dumps(val))
    t2 = time.time()
    decomp = pickle.loads(bz2.decompress(comp))
    t3 = time.time()
    print(t2 - t1, t3 - t2, len(comp))
