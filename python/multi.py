import time
from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing as multi
from more_itertools import flatten
import sys
import functools
import numpy as np


def process(i):
    print(str(i) + '¥n¥n¥n')
    l = []
    for j in range(1000):
        l.append({'id': j, 'sum': sum(range(i*j))})
    return l
#def pricess2(i):

def process(i):
    return np.random.random()


#def process(n):
#    return sum([i*n for i in range(100000)])


def usejoblib(job, num):
    result = Parallel(n_jobs=job)([delayed(process)(n) for n in range(num)])
    return result


def usemulti(job, num):
    p = Pool(multi.cpu_count() if job < 0 else job)
    result = p.map(process, range(num))
    p.close()
    return result

if __name__ == '__main__':
    argv = sys.argv
    total = 0
    n = 1

    #print(process(1))

    for i in range(n):
        s = time.time()
        if argv[1] == 'joblib':
            result = usejoblib(int(argv[2]),int(argv[3]))
        elif argv[1] == 'multi':
            result = usemulti(int(argv[2]),int(argv[3]))
        else:
            result = [process(j) for j in range(int(argv[3]))]
        elapsed = time.time()-s
        print('time: {0} [sec]'.format(elapsed))
        total += elapsed

    print('--------')
    print('average: {0} [sec]'.format(total/n))

    print(result)
    sums = functools.reduce(lambda x, y: x + y['sum'], list(flatten(result)), 0)
    print('total: {0}'.format(sums))
#    print('total: {0}'.format(sum(result)))
