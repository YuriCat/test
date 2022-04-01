# 量子テレポーテーション

from blueqat import Circuit

#量子テレポーテーション回路
a = Circuit().h[1].cx[1,2].cx[0,1].h[0].cx[1,2].cz[0,2].m[:]

print(a.run(shots=100))

print((Circuit().x[0] + a).run(shots=100))
