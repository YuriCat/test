import time
import pickle
import bz2

import numpy as np

a = np.random.random((1000, 1000))
b = a.astype(np.float32)
c = a.astype(np.float16)
d = c.astype(np.float32)
e = (a * np.iinfo(np.uint64).max).astype(np.uint64)
f = (a * np.iinfo(np.uint32).max).astype(np.uint32)
g = (a * np.iinfo(np.uint16).max).astype(np.uint16)

for data in [a, b, c, d, e, f, g]:
    t = time.time()
    k = bz2.compress(pickle.dumps(data))
    tt = time.time()
    pickle.loads(bz2.decompress(k))
    print(data.dtype, len(k), tt - t, time.time() - tt)
