import sys, time, socket, multiprocessing
from pickled_connection import PickledConnection

# 接続を順番に待ち受ける

def server_func(index, conns):
    num = (index + 1) * 100
    while True:
        conns[0].send(num + 1)
        conns[1].send(num + 2)
        time.sleep(1)
        num += 2

def client_func(key, index, conn):
    while True:
        num = conn.recv()
        print('client %s-%d received %d' % (key, index, num))

def server(n):
    print('server')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 12345))
    sock.listen(n + 2) # 規定数受け付けてもさらに1つずつくるので+2
    conns = []
    accepted_conns = []
    index = 0

    while index < n:
        # 常に接続が二つ揃うまで待つ
        while len(conns) < 2:
            conn, _ = sock.accept()
            print('accepted')
            conn = PickledConnection(conn)
            conns.append(conn)

        conn = conns[0]
        accepted_conns.append(conn)
        conns = conns[1:]
        conn.send(index) # acceptを通知し、次の接続を待つ

        if len(accepted_conns) % 2 == 0:
            # サーバプロセスを立ち上げて通信を開始
            multiprocessing.Process(target=server_func, args=(index, accepted_conns[-2:])).start()
            index += 1

    for conn in accepted_conns:
        conn.close()

def client(key):
    print('client')
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('', 12345))
        conn = PickledConnection(sock)
        index = conn.recv() # accept連絡を待つ
        print('accepted for server %d' % index)

        # クライアントプロセスを切り離す
        multiprocessing.Process(target=client_func, args=(key, index, conn)).start()
        conn.close()


if __name__ == '__main__':
    if sys.argv[1] == 's':
        server(5)
    else:
        client(sys.argv[1])
