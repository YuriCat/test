
# https://github.com/Theelx/pybase122

import numpy as np
import pickle
import base64
import base122

a = np.random.random((1000, 1000))
a = pickle.dumps(a)

print(len(a))

print(len(base64.b64encode(a)))
print(len(base122.decode(base64.b64encode(a))))
