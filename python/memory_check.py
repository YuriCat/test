# https://pg-chain.com/python-psutil

import psutil

# メモリ容量を取得
mem = psutil.virtual_memory() 
print(mem.total)
#[結果] 8492281856

# メモリ使用量を取得 
print(mem.used)
#[結果] 4748627968

# メモリ空き容量を取得 
print(mem.available)
#[結果] 3743653888
