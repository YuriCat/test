# 注意: こういう場合にはキャッシュを使えないという例です

import time
import functools
import numpy as np

class Model:
    def __init__(self):
        self.x = None

    @functools.lru_cache(maxsize=10000)
    def sum(self):
        return self.x.sum()

m = Model()
for i in range(10):
    m.x = np.random.randn(1000000)
    for i in range(2):
        t = time.time()
        print(m.sum(), time.time() - t)
        print(m.x.sum())
