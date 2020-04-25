import numpy as np
import multiprocessing as mp
from multiprocessing import connection

N = 10
L = 10000000

def child(conn, i):
    while True:
        conn.send(np.ones(L))
        print('child %d sent' % i)
        conn.recv()
        print('child %d received' % i)

def server(conns):
    while True:
        ok_conns = mp.connection.wait(conns)
        for conn in ok_conns:
            conn.recv()
            conn.send(np.ones(L))


conns = []

for i in range(N):
    conn0, conn1 = mp.connection.Pipe()
    mp.Process(target=child, args=(conn1, i)).start()
    conn1.close()
    conns.append(conn0)

server(conns)
