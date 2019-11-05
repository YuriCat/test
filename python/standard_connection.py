import os, sys

class StandardConnection:
    def recv(self):
        return sys.stdin.readline().strip()

    def send(self, msg):
        sys.stdout.write(msg + os.linesep)

if __name__ == '__main__':
    conn = StandardConnection()
    while True:
        rmsg = conn.recv()
        print('recv >> %s' % rmsg)
        smsg = rmsg + '*'
        print('send << %s' % smsg)
        conn.send(smsg)
