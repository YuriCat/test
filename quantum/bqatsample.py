from blueqat import Circuit
import math, copy

#number of qubit is not specified
c = Circuit()

#if you want to specified the number of qubit
c = Circuit(1)

c.x[0]
print(c)

d = copy.copy(c)
c.m[0]
c.run()
print(c.last_result())