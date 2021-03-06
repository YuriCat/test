import time, threading, multiprocessing, queue

# Pythonの外部プロセスワーカ

def collatz(n):
    while n > 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    return True

def hoge(conn, index):
    print('worker %d' % index)
    while True:
        ret_list = []
        for _ in range(256):
            n = conn.recv()
            ret_list.append(collatz(n))
        conn.send((ret_list, len(ret_list)))

class MultiProcessWorkers:
    def __init__(self, func, send_generator, num, postprocess=None, buffer_length=512, num_receivers=1):
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.buffer_length = buffer_length
        self.num_receivers = num_receivers
        self.conns = []
        self.send_cnt = {}
        self.shutdown_flag = False
        self.lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=8)

        for i in range(num):
            conn0, conn1 = multiprocessing.Pipe(duplex=True)
            multiprocessing.Process(target=func, args=(conn1, i)).start()
            conn1.close()
            self.conns.append(conn0)
            self.send_cnt[conn0] = 0

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self.shutdown_flag = True

    def recv(self):
        return self.output_queue.get()

    def start(self):
        threading.Thread(target=self._sender).start()
        for i in range(self.num_receivers):
            threading.Thread(target=self._receiver, args=(i,)).start()

    def _sender(self):
        print('start sender')
        while not self.shutdown_flag:
            total_send_cnt = 0
            for conn, cnt in self.send_cnt.items():
                if cnt < self.buffer_length:
                    conn.send(next(self.send_generator))
                    self.lock.acquire()
                    self.send_cnt[conn] += 1
                    self.lock.release()
                    total_send_cnt += 1
            if total_send_cnt == 0:
                time.sleep(0.01)
        print('finished sender')

    def _receiver(self, index):
        print('start receiver %d' % index)
        conns = [conn for i, conn in enumerate(self.conns) if i % self.num_receivers == index]
        while not self.shutdown_flag:
            tmp_conns = multiprocessing.connection.wait(conns)
            for conn in tmp_conns:
                data, cnt = conn.recv()
                if self.postprocess is not None:
                    data = self.postprocess(data)
                while not self.shutdown_flag:
                    try:
                        self.output_queue.put(data, timeout=0.3)
                        self.lock.acquire()
                        self.send_cnt[conn] -= cnt
                        self.lock.release()
                        break
                    except queue.Full:
                        pass
        print('finished receiver %d' % index)


class MultiThreadWorkers:
    def __init__(self, func, send_generator, num, postprocess=None):
        self.func = func
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.num = num
        self.conns = []
        self.send_cnt = {}
        self.shutdown_flag = False
        self.lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=8)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self.shutdown_flag = True

    def recv(self):
        return self.output_queue.get()

    def start(self):
        class LocalConnection:
            def __init__(self, parent):
                self.parent = parent
            def send(self, recv_data):
                data, _ = recv_data
                if self.parent.postprocess is not None:
                    data = self.parent.postprocess(data)
                while not self.parent.shutdown_flag:
                    try:
                        self.parent.output_queue.put(data, timeout=0.3)
                        break
                    except queue.Full:
                        pass
            def recv(self):
                self.parent.lock.acquire()
                data = next(self.parent.send_generator)
                self.parent.lock.release()
                return data

        for i in range(self.num):
            conn = LocalConnection(self)
            threading.Thread(target=self.func, args=(conn, i)).start()


def sender():
    i = int(1e200)
    while True:
        if i % 1000 == 0:
            print(i)
        yield i
        i += 1


ws = MultiProcessWorkers(hoge, sender(), 4, postprocess=lambda x:x.count(True), num_receivers=4)
#ws = MultiThreadWorkers(hoge, sender(), 4, postprocess=lambda x:x.count(True))

ws.start()

while True:
    data = ws.recv()
    print(data)

