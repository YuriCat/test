import numpy as np
from wildcat.network.local_endpoint import LocalEndpoint
from wildcat.util.matrix import random_symmetric_matrix
from wildcat.solver.ising_hamiltonian_solver import IsingHamiltonianSolver

#mat = random_symmetric_matrix(size=40)

# 自然数分割問題 [1, 1, 1, 1]
mat = np.array([[0,-1,-1,-1],
                [0, 0,-1,-1],
                [0, 0, 0,-1],
                [0, 0, 0, 0]])

# 自然数分割問題 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
integers = [1, 2, 3, 4, 5, 6, 7, 8, 10]
#integers = [3,2,6,9,2,5,7,3,3,6,7,3,5,3,2,2,2,6,8,4,6,3,3,6,4,3,3,2,2,5,8,9]
mat2 = np.zeros((len(integers), len(integers)))
for c, i in enumerate(integers):
    for d, j in enumerate(integers):
        if c < d:
            mat2[c, d] = -i * j
print(mat2)
mat2 /= 80.0
print(mat2)

solver = IsingHamiltonianSolver(ising_interactions=mat2)

def callback(arrangement):
    e = solver.hamiltonian_energy(arrangement)
    print("Energy: ", e)
    print("Spins: ", arrangement)
    print("sum = ", np.dot(arrangement, integers))

print(solver)
solver.solve(callback, endpoint=LocalEndpoint())