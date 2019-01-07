from matplotlib import pyplot as plt
import pandas as pd

mat = pd.read_csv('out.csv').as_matrix()
plt.plot(mat[:,0])
plt.plot(mat[:,1])
plt.plot(mat[:,2])
plt.plot(mat[:,3])
plt.plot(mat[:,4])
plt.title('estimation of quantiles by P^2 Algorithm')
plt.show()
