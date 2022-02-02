
import numpy as np
import pickle
import base64

a = np.random.random((1000, 1000))
a = pickle.dumps(a)

print(len(a))

print(len(base64.b16encode(a)))
print(len(base64.b32encode(a)))
print(len(base64.b64encode(a)))
print(len(base64.b85encode(a)))
print(len(base64.a85encode(a)))
