
# 多次元正規分布の平均距離

import numpy as np

for i in range(1, 11):
   rs = np.random.multivariate_normal(np.zeros(i), np.eye(i), 100000)

   dist = np.power(np.power(rs, 2).sum(-1), 0.5).mean()
   dist2 = np.power(np.power(rs, 2).sum(-1).mean(), 0.5)
   print(dist, dist2)

