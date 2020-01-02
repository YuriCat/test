
# 中央にサーバを置くタイプの接続

import sys
import time
import socket
import pickle
import io
import struct
import multiprocessing as mp

class PickledConnection:
    def __init__(self, conn):
        self.conn = conn

    def _recv(self, size):
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            size -= len(chunk)
            buf.write(chunk)
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

def data_consumer(q):
    while True:
        data = q.get()
        print(data)

def data_receiver(q, conn_client):
    while True:
        data = conn_client.recv()
        q.put(data)

def data_sender(conn_server):
    cnt = 0
    while True:
        time.sleep(1)
        conn_server.send(cnt)
        cnt += 1

def socket_server(q, port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(('', port))
    server_sock.listen(100)
    print('waiting connection...')

    while True:
        conn, _ = server_sock.accept()
        print('server connected!')
        mp.Process(target=data_receiver, args=(q, PickledConnection(conn))).start()

def start_server(q, port):
    mp.Process(target=data_consumer, args=(q,)).start()
    socket_server(q, port)

def start_client(port):
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(('', port))
    print('client connected!')

    data_sender(PickledConnection(client_sock))

if __name__ == '__main__':
    mode = sys.argv[1]
    port = int(sys.argv[2])

    if mode == 'c':
        start_client(port)
    else:
        #m = mp.Manager()
        q = mp.Queue()
        start_server(q, port)
