import io, struct, pickle

class PickledConnection:
    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        self.close()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def fileno(self):
        return self.conn.fileno()

    def _recv(self, size):
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            n = len(chunk)
            if n == 0:
                raise EOFError
            size -= n
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