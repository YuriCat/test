
# multiprocessing使用時にpdbを用いたデバッグ
# 引用元: https://qiita.com/purple_jp/items/f1934a870dc5253cc9de

from multiprocessing import Pool
import pdb

#def f(x):
#    # 通常のステップ実行
#    pdb.set_trace()
#    return x*x

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        import sys
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def f(x):
    ForkedPdb().set_trace()
    return x * x

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [1, 2, 3, 4, 5]))
