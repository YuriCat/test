import copy, multiprocessing

global_var = None

class C:
    def __init__(self):
        self.i = 0
    def __str__(self):
        

def func(args):
    #global global_var
    print(args, global_var[args])

def main():
    global global_var
    global_var = copy.deepcopy([0, 1] * 100)
    n = 3
    #func(-1)
    with multiprocessing.Pool(n) as p:
        p.map(func, [i for i in range(n)])

if __name__ == '__main__':
    main()
    