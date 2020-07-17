# 銀行家の丸め

# 参考　http://www.finetune.co.jp/~lyuka/technote/round/

import math

def banker_round(x, digit=0):
    k = 10 ** digit
    x *= k
    y = math.floor(x)
    t = x % 2
    if t <= 0.5 or 1 <= t < 1.5:
        return y / k
    else:
        return (y + 1) / k


# Pythonのround()関数はデフォルトで銀行家丸めを使っている

def compare(x, digit=0):
    print(round(x, digit), banker_round(x, digit))

compare(0.5)
compare(-0.5)
compare(1.5)
compare(-1.5)
compare(3.1415, 3)
compare(-3.1415, 3)