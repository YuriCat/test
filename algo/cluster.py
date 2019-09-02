import time
import numpy as np


M = 5

each_points = 1 + np.random.exponential(16.0, size=(M)).astype(np.int64)

N = each_points.sum()

print(each_points)

mean = np.random.random((M, 2))
labels = np.concatenate([np.ones(m, dtype=np.int64) * i for i, m in enumerate(each_points)])
x_org = [mean[i] + 0.1 * np.random.randn(m, 2) for i, m in enumerate(each_points)]
x = np.concatenate(x_org)
centers = np.array([xe.mean(axis=0) for xe in x_org])
print(mean)
print(labels)
print('centers = ')
print(centers)

import matplotlib.pyplot as plt
fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2,2,figsize=(10,6))
real = ax00
test = [ax01, ax10, ax11]
plt.suptitle('comparison of clustering algorithm')

for i, xe in enumerate(x_org):
    real.scatter(xe[:,0], xe[:, 1])
real.set_title('real')

def set_label(real_center, test_center):
    # 中心の平均が近くなるようにラベルの再設定
    labels = np.arange(len(test_center))
    loss = ((test_center - real_center) ** 2).sum()
    best_labels, best_loss = None, float('inf')
    for e in range(10000):
        i, j = np.random.randint(len(real_center), size=2)
        li, lj = labels[i], labels[j]
        d = 0
        d += ((test_center[i] - real_center[lj]) ** 2).sum()
        d += ((test_center[j] - real_center[li]) ** 2).sum()
        d -= ((test_center[i] - real_center[li]) ** 2).sum()
        d -= ((test_center[j] - real_center[lj]) ** 2).sum()
        if d < 0 or np.random.random() < 0.1:
            labels[i], labels[j] = lj, li
            loss += d
        #print(loss, best_loss, best_labels)
        if loss < best_loss:
            best_labels, best_loss = np.copy(labels), loss
    return best_labels

from sklearn.cluster import *

algorithms = {
    'KMeans': KMeans(M).fit,
    'Ward': AgglomerativeClustering(M, linkage='ward').fit,
    'Spectral': SpectralClustering(M).fit,
}

for c, data in enumerate(algorithms.items()):
    name, algo = data
    print(name)
    result = algo(x)
    print(result.labels_)
    cluster_centers = np.array([x[np.where(result.labels_ == i)].mean(axis=0) for i in range(M)])
    print(cluster_centers)
    nlabels = set_label(centers, cluster_centers)[result.labels_]
    print(nlabels)
    for i in range(M):
        xe = x[np.where(nlabels == i)]
        test[c].scatter(xe[:,0], xe[:,1])
    test[c].set_title(name)

plt.show()