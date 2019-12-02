
# waiting sample

import time

print('subprocess start!')

class Kame:
    def __del__(self):
        print('Kame del!')

a = Kame()

while True:
    time.sleep(1)

print('subprocess end!')
