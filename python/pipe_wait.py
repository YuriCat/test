import time, multiprocessing

def worker(i, conn):
    while True:
        time.sleep(i + 1)
        conn.send(i)

conns = []
for i in range(4):
    conn0, conn1 = multiprocessing.Pipe(duplex=False)
    multiprocessing.Process(target=worker, args=(i, conn1)).start()
    conns.append(conn0)

while True:
    conn_list = multiprocessing.connection.wait(conns)
    print([conn.recv() for conn in conn_list])
