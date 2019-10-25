# Python with構文の勉強

class Hoge:
    def __init__(self):
        print('init!')
    def __enter__(self):
        print('enter!')
        return self # asによってこの返り値を受ける
    def __exit__(self, *args):
        print('exit!')
    def __del__(self):
        print('del!')
    def fuga(self):
        print('fuga!')

with Hoge() as h:
    h.fuga()

# initはもっと前でも良い
k = Hoge()
with k as l:
    l.fuga()
