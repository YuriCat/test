# https://blog.amedama.jp/entry/2018/07/23/080000

import time
from tqdm import tqdm

#for _ in tqdm(range(1000)):
#    time.sleep(1e-2)

from itertools import count

for _ in tqdm(count()):
    time.sleep(1e-2)
