# select の使い方

import socket, select, time, random
import multiprocessing as mp

N = 3

def client(index):
    time.sleep(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('', 9876))
    while True:
        time.sleep(1)#random.random())
        sock.send(b'0000')
        sock.recv(1)

for i in range(N):
    mp.Process(target=client, args=(i,)).start()

ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ssock.bind(('', 9876))
ssock.listen(3)

conns = []
for _ in range(N):
    sock, _ = ssock.accept()
    print('accept')
    conns.append(sock)

print(conns)

while True:
    readable, _, _ = select.select(conns, [], [])
    print(len(readable))
    for conn in readable:
        conn.recv(4)
        print('received')
        conn.send(b'0')

