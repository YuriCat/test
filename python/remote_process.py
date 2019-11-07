
import os, sys, pickle, time

import remoto


class StandardConnection:
    def recv(self):
        rmsg = sys.stdin.readline().strip()
        return pickle.loads(rmsg)

    def send(self, smsg):
        sys.stdout.write(pickle.dumps(smsg) + os.linesep)

class RemoteProcess:
    def __init__(self, host, command_list):
        self.conn = remoto.connection.get('ssh')(host)
        for command in command_list:
            print(command)
            stdout = remoto.process.check(self.conn, command)[1]
            print(stdout)
            self.rmsg = stdout if len(stdout) == 0 else stdout[0] 
 

    def recv(self):
        return pickle.loads(self.rmsg)

    def send(self, smsg):
        stdout = remoto.process.check(self.conn, pickle.dumps(smsg))[1]
        print(stdout)
        print(stdout[0])
        self.rmsg = stdout[0]

def worker_loop(conn):
    i = 0
    while True:
        print('W->S: %d' % i)
        conn.send(i)
        i = conn.recv()
        print('W<-S: %d' % i)
        time.sleep(1)
        i = i - 1

def server_loop(conn):
    while True:
        i = conn.recv()
        print('S<-W: %d' % i)
        i = i + 1
        print('S->W: %d' % i)
        conn.send(i)

if __name__ == '__main__':
    if sys.argv[1] == 's':
        conn = RemoteProcess('localhost', [
           ['source', '.bashrc'],
           ['cd', os.getcwd()],
           ['python', str(__file__), 'w']
        ])
        server_loop(conn)
    else:
        conn = StandardConnection()
        worker_loop(conn)
