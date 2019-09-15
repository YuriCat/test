import tensorflow as tf
import sonnet as snt

mlp = snt.Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(10),
])

logits = mlp(tf.random.normal([2, 4]))
print(logits)
all_variables = mlp.variables

class DRC(snt.Module):
    def __init__(self, d, n):
        super().__init__()
        self.n = n
        self.d = d
        self.blocks = []
        for _ in range(self.d):
            self.blocks.append(snt.Conv2DLSTM(
                input_shape=(5, 5, 16),
                output_channels=16,
                kernel_shape=(3, 3))
            )

    def initial_state(self, batch_size):
        hs = []
        for block in self.blocks:
            hs.append(block.initial_state(batch_size))
        return hs

    def __call__(self, ox, hs):
        hs = [h for h in hs]
        for _ in range(self.n):
            for i, block in enumerate(self.blocks):
                x, h = block(ox, hs[i])
                hs[i] = h
        return x, hs

#x = tf.random.normal([3, 5, 5, 4])
#h = net.initial_state(3)
#y = net(x, h)


net = DRC(3, 3)
x = tf.random.normal([3, 5, 5, 4])
h = net.initial_state(3)
for _ in range(10):
    y, h = net(x, h)
    print(y.shape)

#print(y)