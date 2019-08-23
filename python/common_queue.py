import time, multiprocessing

def func(args):
    process_id, n, q = args
    while True:
        a = q.get()
        time.sleep(1)
        print(process_id, a)
        q.put(a + n)

if __name__ == '__main__':
    m = multiprocessing.Manager()
    q = m.Queue()
    n = 3
    for i in range(n):
        q.put(i)
    with multiprocessing.Pool(n) as p:
        p.map(func, [(i, n, q) for i in range(n)])