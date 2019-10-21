# forkによってプロセスを作成
# 参考: https://blog.chocolapod.net/momokan/entry/53

import os

def child():
    print('I am child.')

def parent():
    print('I am parent.')

pid = os.fork()
if pid == 0:
    child()
else:
    parent()
