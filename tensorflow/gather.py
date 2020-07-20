
import tensorflow as tf

# source


param = tf.constant([[[0], [1]], [[2], [3]]])
print(param.shape)

indices = tf.constant([[1,0,0,1],[1,0,1,0]])
print(indices.shape)

result = tf.gather(param, indices, axis=1)
print(result.shape)

print(result.numpy())
