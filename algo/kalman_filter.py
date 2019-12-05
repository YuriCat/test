
# カルマンフィルタ
# https://logics-of-blue.com/kalman-filter-concept/

import numpy as np

def model(ys):
    #m = 1.111
    m = 2
    if len(ys) == 0:
        return 0
    if len(ys) == 1:
        return ys[-1]
    elif len(ys) == 2:
        return ys[-1] + (ys[-1] - ys[-2]) / m
    else:
        return ys[-1] + (ys[-1] - ys[-2]) / m + ((ys[-1] - ys[-2]) - (ys[-2] - ys[-3])) / (m ** 3)


def kalman_filter(x, ys, p_prev, sigma_w, sigma_v):
    # x       現在の観測値
    # y_prev  以前の推定状態
    # p_prev  以前の状態の予測誤差の分散
    # sigma_w 状態方程式のノイズの分散
    # sigma_v 観測方程式のノイズの分散

    y_forcast = model(ys)
    p_forcast = p_prev + sigma_w

    kalman_gain = p_forcast / (p_forcast + sigma_v)

    y_filtered = y_forcast + kalman_gain * (x - y_forcast)
    p_filtered = (1 - kalman_gain) * p_forcast

    return y_filtered, p_filtered

xs = [np.sin(i * 0.05 + 1) + np.random.randn() * 0.1 for i in range(1000)]

sigma_w = 0.05
sigma_v = 1
ys = []
p = sigma_v

for x in xs:
    y, p = kalman_filter(x, ys, p, sigma_w, sigma_v)
    ys.append(y)


import matplotlib.pyplot as plt
plt.plot(xs, label='observation')
plt.plot(ys, label='filtered')
plt.legend()
plt.show()
