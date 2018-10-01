import multiprocessing

def func(i):
    return i

print(multiprocessing.cpu_count())
for i in range(10):
    with multiprocessing.Pool(2) as p:
        result = p.map(func, list(range(10 + i)))
    print(result)
