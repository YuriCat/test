
import time
import queue
import threading
import multiprocessing as mp


class Worker:
    def __init__(self, worker_id, conn):
        self._worker_id = worker_id
        self._conn = conn
        print('Hello, I\'m %d.' % self._worker_id)

    def __del__(self):
        print('Goodbye from %d.' % self._worker_id)

    def run(self):
        while True:
            self._conn.send({'request': 'args'})
            args = self._conn.recv()
            role = args['role']

            if role == 'step':
                self._conn.send({'request': 'get'})
                number = self._conn.recv()
                time.sleep(0.5)
                print('Got %d' % number)
                self._conn.send({'request': 'set', 'number': number + 1})
                self._conn.recv()
            if role == 'quit':
                break


def run_worker(worker_id, conn):
    worker = Worker(worker_id, conn)
    try:
        worker.run()
    finally:
        conn.send({'request': 'quitok'})
        conn.close()


class Conductor:
    def __init__(self, num_workers):
        self._shutdown_flag = True
        self._num_workers = num_workers

    def __del__(self):
        self.goodbye()

    def hello(self):
        if not self._shutdown_flag:
            return

        self._shutdown_flag = False
        self._waiting_flag = False
        self._input_queue = queue.Queue(maxsize=self._num_workers)
        self._conns = []
        self._waiting_conns = []
        self._number = 0

        for i in range(self._num_workers):
            conn0, conn1 = mp.Pipe()
            mp.Process(target=run_worker, args=(i, conn1)).start()
            self._conns.append(conn0)
            conn1.close()

        self._thread_server = threading.Thread(target=self._server)
        self._thread_server.start()

    def goodbye(self):
        if self._shutdown_flag:
            return

        for _ in self._conns:
            self._append({'role': 'quit'})
        self.wait()
        self._shutdown_flag = True
        self._thread_server.join()

    def wait(self, timeout=0):
        self._waiting_flag = True
        elapsed = 0
        start = time.time()
        while not self._shutdown_flag and (timeout > 0 and elapsed < timeout):
            if self._input_queue.qsize() == 0 or len(self._waiting_conns) == len(self._conns):
                break
            time.sleep(0.1)
            elapsed = time.time() - start
        self._waiting_flag = False

    def step(self):
        self._append({'role': 'step'})

    def _append(self, args):
        while not self._shutdown_flag:
            try:
                self._input_queue.put(args, timeout=0.3)
                break
            except queue.Full:
                pass

    def _send(self, conn, data):
        conn.send(data)

    def _recv_conn(self):
        if self._waiting_flag or len(self._waiting_conns) == 0:
            self._waiting_conns += mp.connection.wait(self._conns, timeout=0.1)

        if len(self._waiting_conns) == 0:
            return None, None

        if self._waiting_flag:
            time.sleep(0.1)
            return None, None

        conn = self._waiting_conns[0]
        self._waiting_conns = self._waiting_conns[1:]

        try:
            data = conn.recv()
        except ConnectionResetError:
            self._conns.remove(conn)
            return None, None
        except EOFError:
            self._conns.remove(conn)
            return None, None

        return conn, data

    def _server(self):
        while len(self._conns) > 0:
            conn, data = self._recv_conn()
            if conn is None:
                continue

            request = data['request']

            if request == 'args':
                args = self._input_queue.get()
                self._send(conn, args)
            elif request == 'get':
                self._send(conn, self._number)
            elif request == 'set':
                self._number = data['number']
                self._send(conn, '')
            elif request == 'quitok':
                self._conns.remove(conn)

        while True:
            try:
                self._input_queue.get(timeout=0.1)
            except queue.Empty:
                break


if __name__ == '__main__':
    c = Conductor(num_workers=3)

    try:
        c.hello()
        for _ in range(30):
            c.step()
        c.wait()
    finally:
        c.goodbye()
