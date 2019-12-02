
# 別プロセスを立てたスクリプトを落とす実験

import time
import multiprocessing
import subprocess
import signal
import socket
import sys

class Neko:
    def __del__(self):
        print('Neko del!')

def thread_target(*args):
    while True:
        time.sleep(1)

# 普通のクラス
a = Neko()
import threading
t = threading.Thread(target=thread_target)
t.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt as e:
    sys.exit(e)
