def func(*args, **kwargs):
  print(args)
  print(kwargs)

print('args, kwargs')

func(a=0, b=1)
func(1, a=0, b=1)

args = {'a': 1, 'b': 3}

print('args as dicts')

func(*args)
func(**args)

print('function with orginal arguments')

def func2(a, b):
    print(a, b)

func(*args)
func2(**args)

print('instance method')

class A:
    def __init__(self, a, b):
        print(a, b)

a = A(**args)
