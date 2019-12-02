
# 別プロセスを立てたスクリプトを落とす実験

import time
import multiprocessing
import subprocess
import signal
import socket
import sys

class Neko:
    def __init__(self):
        import subprocess
        command = ['python', 'subprocess_server.py']
        self.proc = subprocess.Popen(command, start_new_session=True)
    def __del__(self):
        print('Neko del!')
        self.proc.send_signal(signal.SIGINT)
        self.proc.wait()
        print('wait subprocess')

def multiprocess_target(*args):
    #a = Neko()
    while True:
        time.sleep(1)

# 普通のクラス
a = Neko()
conn0, conn1 = multiprocessing.Pipe()
conn0.recv()

print('kk')

# multiprocessingで開いたターゲット
#multiprocessing.Process(target=multiprocess_target).start()

# connection starterで開いたターゲット
'''from connection_starter import ConnectionStarter
def cs_loop():
    cs = ConnectionStarter(multiprocess_target)
    cs.loop()
multiprocessing.Process(target=cs_loop).start()
time.sleep(1)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('', 9999))'''

while True:
    time.sleep(1)

