# 例外処理の勉強

class TestException(Exception):
    pass

def func():
    raise TestException()

try:
    func()
    print('ok')
except:
    print('bad')


try:
    func()
    print('ok')
except TestException:
    print('bad')

try:
    func()
    print('ok')
except KeyboardInterrupt:
    print('bad')
except:
    print('else')
