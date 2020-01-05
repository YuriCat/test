import time, threading, multiprocessing, queue

# Pythonの外部プロセスワーカ

def collatz(n):
    while n > 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    return True

def hoge(conn):
    while True:
        ret_list = []
        for _ in range(256):
            n = conn.recv()
            ret_list.append(collatz(n))
        conn.send((ret_list, len(ret_list)))

class MultiProccessWorkers:
    def __init__(self, func, send_generator, num, postprocess=None, buf_len=512, num_receivers=1):
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.buf_len = buf_len
        self.num_receivers = num_receivers
        self.conns = []
        self.send_cnt = {}
        self.shutdown_flag = False
        self.lock = threading.Lock()
        self.output_queue = queue.Queue()

        for _ in range(num):
            conn0, conn1 = multiprocessing.Pipe(duplex=True)
            multiprocessing.Process(target=func, args=(conn1,)).start()
            conn1.close()
            self.conns.append(conn0)
            self.send_cnt[conn0] = 0

        threading.Thread(target=self._sender).start()
        for i in range(self.num_receivers):
            threading.Thread(target=self._receiver, args=(i,)).start()

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self.shutdown_flag = True

    def recv(self):
        return self.output_queue.get()

    def _sender(self):
        print('start sender')
        while not self.shutdown_flag:
            total_send_cnt = 0
            for conn, cnt in self.send_cnt.items():
                if cnt < self.buf_len:
                    conn.send(next(self.send_generator))
                    self.lock.acquire()
                    self.send_cnt[conn] += 1
                    self.lock.release()
                    total_send_cnt += 1
            if total_send_cnt == 0:
                time.sleep(0.01)
        print('finished sender')

    def _receiver(self, index):
        print('start receiver')
        conns = [conn for i, conn in enumerate(self.conns) if i % self.num_receivers == index]
        while not self.shutdown_flag:
            tmp_conns = multiprocessing.connection.wait(conns)
            for conn in tmp_conns:
                data, cnt = conn.recv()
                if self.postprocess is not None:
                    data = self.postprocess(data)
                self.output_queue.put(data)
                self.lock.acquire()
                self.send_cnt[conn] -= cnt
                self.lock.release()
        print('finished receiver')


def sender():
    i = int(1e200)
    while True:
        if i % 1000 == 0:
            print(i)
        yield i
        i += 1


mpw = MultiProccessWorkers(hoge, sender(), 4, postprocess=lambda x:x.count(True), num_receivers=4)

while True:
    data = mpw.recv()
    print(data)


