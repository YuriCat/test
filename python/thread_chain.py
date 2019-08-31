import time, threading
from concurrent.futures import ThreadPoolExecutor

def func(args):
    thread_id, n, k, ev_list = args
    while True:
        ev_list[thread_id].wait()
        ev_list[thread_id].clear()
        next_id = (thread_id + k) % n
        time.sleep(1)
        print(thread_id, '->', next_id)
        ev_list[next_id].set()

if __name__ == '__main__':
    n, k = 10, 3 # 総スレッド数, activeなスレッド数
    ev_list = [threading.Event() for _ in range(n)]
    for i in range(k):
        ev_list[i].set()
    with ThreadPoolExecutor(n) as executor:
        executor.map(func, [(i, n, k, ev_list) for i in range(n)])