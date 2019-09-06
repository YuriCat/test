
# Wasserstein distance の確認
# 一次元ならソートして距離の和を取るだけ

import numpy as np

N = 64
a = np.random.random(N)
b = np.random.random(N)

from scipy.stats import wasserstein_distance

print(wasserstein_distance(a, b))

a_sorted = np.sort(a)
b_sorted = np.sort(b)

print(np.abs(a_sorted - b_sorted).mean())