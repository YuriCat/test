
# 行列のコレスキー分解

import numpy as np
import scipy.linalg


cov = np.array([[1, 0], [0, 1]])
update_mat = np.array([[1, 0], [0, 1]])

chol_factor, lower = scipy.linalg.cho_factor(cov, lower=True, check_finite=False)

print(chol_factor)
print(lower)

kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(cov, update_mat.T).T,
            check_finite=False).T

print(kalman_gain)
