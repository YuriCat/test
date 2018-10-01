# 疎行列

import scipy.sparse as sp

mat = sp.lil_matrix((1, 16))

mat[0, 1] = 2
mat[0, 5] = 7