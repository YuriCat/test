# https://pytorch.org/docs/stable/onnx.html

import numpy as np
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('tmp.onnx')
tf_model = prepare(onnx_model)

x = {
    'a': np.random.randn(1, 4).astype(np.float32),
    'b': np.random.randn(1, 4).astype(np.float32),
}

outputs = tf_model.run(x)

print(outputs)
print([o.shape for o in outputs])