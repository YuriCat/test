
import numpy as np

# 一般化エントロピーの性質
# 一般化KL情報量の性質


for i in range(100):
    p = np.random.dirichlet([1] * 8)
    q = np.random.dirichlet([1] * 8)
    print(p, q)
    for alpha in [1, 1e-1, 1e-2, 1e-3]:
        ent = -((p ** alpha) * np.log(p)).sum()
        kl = ((p ** alpha) * (np.log(p) - np.log(q))).sum()
        print(ent, kl)
        assert ent > 0
        #assert kl > 0
       
