import sys, time, multiprocessing

def func(args):
    index, m, w, n, k, qv_list = args
    while index < m:
        ev_list[index % n].wait()
        ev_list[index % n].clear()
        t = time.time()
        while time.time() - t < w:
            pass
        #time.sleep(w)
        index += n
        ev_list[(index + k) % n].set()

if __name__ == '__main__':
    m, w = int(sys.argv[1]), float(sys.argv[2])
    n, k = 10, 8
    manager = multiprocessing.Manager()
    ev_list = [manager.Event() for i in range(n)]
    for i in range(k):
        ev_list[i].set()
    st = time.time()
    with multiprocessing.Pool(n) as p:
        p.map(func, [(i, m, w, n, k, ev_list) for i in range(n)])
    print(time.time() - st)
