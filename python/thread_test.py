import time, threading, multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Pythonのスレッドの遅さ計測

M, N = 900, 3
args_list = [(i * (M//N), (i + 1) * (M//N)) for i in range(N)]

def func(m, n):
    print(m, n)
    k = 0
    for i in range(m, n):
        print(i)
        k += i
    print(k)

def func1(m, n):
    print(m, n)
    k = 0
    for i in range(m, n):
        print(i)
        k += i
    print(k)

def func2(m, n):
    print(m, n)
    k = 0
    for i in range(m, n):
        print(i)
        k += i
    print(k)

def func3(args):
    m, n = args
    print(m, n)
    k = 0
    for i in range(m, n):
        k += i
    print(k)

# 普通に計算

'''t = time.time()
func(0, M)
print(time.time() - t)

# マルチスレッドで計算
threads = [threading.Thread(target=func, args=args_list[i]) for i in range(N)]

t = time.time()
for th in threads: th.start()
for th in threads: th.join()
print(time.time() - t)'''

# マルチスレッドで別関数で計算
funcs = [func, func1, func2]
threads = [threading.Thread(target=funcs[i], args=args_list[i]) for i in range(N)]

t = time.time()
for th in threads: th.start()
#for th in threads: th.join()
#time.sleep(5)
print(time.time() - t)

# マルチプロセスで計算
'''t = time.time()
with multiprocessing.Pool(N) as pool:
    print(pool.map(func3, args_list))
print(time.time() - t)'''
