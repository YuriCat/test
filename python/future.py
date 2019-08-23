from collections import deque
from time import sleep
from pykka import Future, ThreadingActor, ThreadingFuture


class FutureBuffer(ThreadingActor):
    use_daemon_thread = True

    def __init__(self, func, buffer_size):
        super().__init__()

        self._gen = func
        self._buffer_size = buffer_size
        self._buffer = deque([])

        self._fill_buffer()

    def _fill_buffer(self):
        while len(self._buffer) < self._buffer_size:
            a = Future()
            a.set_get_hook(self._gen)
            self._buffer.append(a)

    def pop(self):
        self._fill_buffer()
        return self._buffer.popleft()

def hoge(x):
    sleep(2)
    return 1

buffer = FutureBuffer(hoge, 2)

for _ in range(5):
  future = buffer.pop()
  print('pop')
  print(future.get())
