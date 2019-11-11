
# リクエストに対して別プロセスを立てて接続する

import io, time, struct, socket, pickle
import multiprocessing as mp


class PickledConnection:
    def __init__(self, conn):
        self.conn = conn

    def _recv(self, size):
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            n = len(chunk)
            buf.write(chunk)
            size -= n
        return buf

    def recv(self):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        buf = self._recv(size)
        return pickle.loads(buf.getvalue())

    def _send(self, buf):
        size = len(buf)
        while size > 0:
            n = self.conn.send(buf)
            size -= n
            buf = buf[n:]

    def send(self, msg):
        buf = pickle.dumps(msg)
        n = len(buf)
        header = struct.pack("!i", n)
        if n > 16384: chunks = [header, buf]
        elif n > 0: chunks = [header + buf]
        else: chunks = [header]
        for chunk in chunks:
            self._send(chunk)

class ConnectionStarter:
    def __init__(self, run_target, args):
        self.run_target = run_target
        self.args = args

    def loop(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(('', 9999))
        server_sock.listen(100)

        for i in range(100): # 限界数まで
            conn, _ = server_sock.accept()
            conn = PickledConnection(conn)
            print('server connected!')
            mp.Process(target=self.run_target, args=(self.args, conn, i), daemon=True).start()

        server_sock.close()


def server_target(args, conn, index):
    # 受け入れ側処理
    while True:
        msg = conn.recv()
        cnt = int(msg) + 1
        print('server %d counter %d' % (index, cnt))
        time.sleep(1)
        conn.send(cnt)

def client_target(args, conn, index):
    # リクエスト側処理
    print('client')
    cnt = 0
    while True:
        print('client %d counter %d' % (index, cnt))
        conn.send(cnt)
        msg = conn.recv()
        cnt = int(msg) + 1

def connection_server():
    # サーバ処理
    starter = ConnectionStarter(server_target, [])
    starter.loop()

def open_socket_connection():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('', 9999))
    return s

def request_starter():
    # リクエスト処理
    for i in range(4):
        conn = open_socket_connection()
        conn = PickledConnection(conn)
        print('client connected!')
        mp.Process(target=client_target, args=([], conn, i)).start()

if __name__ == '__main__':
    mp.Process(target=connection_server).start()
    time.sleep(1)
    request_starter()