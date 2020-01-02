
# Python 継承とデストラクタの勉強

class Hoge(object):
    def __del__(self):
        print('Hoge del')

class Kuma(object):
    def __del__(self):
        print('Kuma del')

class Fuga:#(Hoge):
    def __init__(self):
        self.kuma = Kuma()
    #def __del__(self):
    #    super().__del__()
    #    print('Fuga del')

a = Fuga()

import time
while True:
    time.sleep(1)

