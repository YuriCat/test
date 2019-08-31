import sys, time, multiprocessing

def func(process_id, n, k, m, w, recv_conn, send_conn):
    print(process_id)
    for i in range(process_id, m, n):
        if i >= k:
            _ = recv_conn.recv()
        st = time.time()
        #while time.time() - st < w:
        #    pass
        time.sleep(1)
        print(process_id, '->', ((process_id + k) % n))
        send_conn.send('')

if __name__ == '__main__':
    m = int(sys.argv[1])
    w = float(sys.argv[2])
    manager = multiprocessing.Manager()
    n, k = 501, 8 # 総プロセス数, activeなプロセス数
    parent_conn_map, child_conn_map = {}, {} 
    for i in range(n):
        parent_conn, child_conn = multiprocessing.Pipe()
        parent_conn_map[(i + k) % n] = parent_conn
        child_conn_map[i] = child_conn
        print('conn ', i)
    for i in range(n):
        p = multiprocessing.Process(target=func, args=(i, n, k, m, w, parent_conn_map[i], child_conn_map[i]))
        p.start()
