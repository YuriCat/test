import subprocess
import signal
import sys
import time
import os

if __name__ == '__main__':
    if len(sys.argv) > 1:
        while True:
            p = subprocess.Popen(['python3', os.path.join(os.getcwd(), 'send_sigint.py')])
            time.sleep(2)
            p.send_signal(signal.SIGINT)
            time.sleep(1)
    else:
        print('opened')
        while True:
            time.sleep(1)
