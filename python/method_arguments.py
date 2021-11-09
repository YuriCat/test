# 関数の引数を知る
# https://qiita.com/chankane/items/3909e9f2d1c5910cc60b

def func(a, b, *args, **kwargs):
    pass

class A:
    def func(self, c, d):
        pass

def get_arguments(f):
    # argcountで制限するとargsやkwargsは除かれる
    return f.__code__.co_varnames #[:f.__code__.co_argcount]

print(get_arguments(func))
print(get_arguments(A.func))
