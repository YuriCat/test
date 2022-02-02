
from websocket import create_connection
import sys

host, port = sys.argv[1:3]

conn = create_connection('ws://%s:%s' % (host, port))
conn.send('kuma')
print(conn.recv())