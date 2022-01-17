import torch

prob = torch.FloatTensor([0.1, 0.2, 0.7])

for _ in range(5):
    index = prob.multinomial(num_samples=1, replacement=True)
    print(index)

counts = [0, 0, 0]
for _ in range(10000):
    index = prob.multinomial(num_samples=1, replacement=True)
    counts[index] += 1

print(counts)
