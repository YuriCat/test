
# 線形計画法で同時行動選択の最適確率を求める

# 普通のジャンケン

c = [-1] + [0] * 3  # 目的関数
A_eq = [
    [0, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 1, 1],
]
b_eq = [1] * 3
A_ub = [
    [1, -0, -1,  1],
    [1,  1, -0, -1],
    [1, -1,  1, -0],
]  # 決定変数の係数
b_ub = [0] * 3

bounds = [(None, None)] + [(0, None)] * 3  # 決定変数の下限、上限

from scipy.optimize import linprog
res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
print(res)


# ジャンケンが2つ合わさったゲーム

c = [-1] + [0] * 6  # 目的関数
A_eq = [
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
]
b_eq = [1] * 6
A_ub = [
    [1, -0, -1,  1, -0, -0, -0],
    [1,  1, -0, -1, -0, -0, -0],
    [1, -1,  1, -0, -0, -0, -0],
    [1, -0, -0, -0, -0, -1,  1],
    [1, -0, -0, -0,  1, -0, -1],
    [1, -0, -0, -0, -1,  1, -0],
]  # 決定変数の係数
b_ub = [0] * 6

bounds = [(None, None)] + [(0, None)] * 6  # 決定変数の下限、上限

from scipy.optimize import linprog
res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
print(res)

# 一意性の確認
tbounds = [(res.x[0], res.x[0])] + bounds[1:]

for i in range(6):
    tc = [0] * (1 + 6)
    tc[i + 1] = 1
    res_min = linprog(tc, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=tbounds)

    tc[i + 1] = -1
    res_max = linprog(tc, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=tbounds)

    print(res_min.x[i + 1], res_max.x[i + 1])

