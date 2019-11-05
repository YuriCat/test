# https://pypi.org/project/remoto/

import remoto
from remoto.process import check, run

conn = remoto.Connection('localhost')

print(check(conn, ['ls', '/nonexistent/path']))

conn = remoto.connection.get('ssh')('localhost')
run(conn, ['whoami'])
