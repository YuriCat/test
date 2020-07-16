# https://pytorch.org/docs/stable/onnx.html

import numpy as np
import onnxruntime as ort

ort_session = ort.InferenceSession('tmp.onnx')

x = {
    'a': np.random.randn(1, 4).astype(np.float32),
    'b': np.random.randn(1, 4).astype(np.float32),
}

outputs = ort_session.run(None, x)

print(outputs)
print([o.shape for o in outputs])