import multiprocessing
import time

def hoge():
  time.sleep(3)

if __name__ == '__main__':
  multiprocessing.Process(target=hoge).start()

