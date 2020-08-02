
# スレッド付きのプログラムをCtrl+Cで正しく終了させる

import time
import multiprocessing
import threading


def hoge(name):
    # 自分自身がループするだけの関数
    cnt = 0
    while True:
        time.sleep(1)
        print('%s\'s loop %d' % (name, cnt))
        cnt += 1


def fuga(name):
    # スレッドを立てる関数
    t = threading.Thread(target=hoge, args=('%s-thread' % name,))
    t.daemon = True
    t.start()


def piyo(name):
    # スレッドを立てて自分自身もループする関数
    try:
        for i, f in enumerate([hoge, fuga]):
            t = threading.Thread(target=f, args=('%s-thread%d' % (name, i),))
            t.daemon = True
            t.start()
        hoge(name)
    finally:
        pass


if __name__ == '__main__':
    multiprocessing.Process(target=hoge, args=('process0',)).start()
    multiprocessing.Process(target=piyo, args=('process1',)).start()
    piyo('main')

