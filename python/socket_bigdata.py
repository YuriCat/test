import socket
import numpy as np
import multiprocessing as mp
from multiprocessing import connection

N = 3
L = 10000


def child(i, conns):
    ssock = socket.socket()
    while True:
        conns[np.random.randint(len(conns))].send(np.random.random(L))
        print('child %d sent' % i)
        readable_conns = mp.connection.wait(conns)
        for conn in readable_conns:
            conn.recv()
            print('child %d received' % i)
        

conns = {}

for i in range(N):
    for j in range(i + 1, N):
        conn0, conn1 = mp.connection.Pipe()
        conns[(i, j)] = conn0
        conns[(j, i)] = conn1

for i in range(N):
    mp.Process(target=child, args=(i, [conns[(i, j)] for j in range(N) if i != j])).start()

for conn in conns.values():
    conn.close()
