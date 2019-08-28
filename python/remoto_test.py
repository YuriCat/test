# https://pypi.org/project/remoto/

import remoto
from remoto.process import check

conn = remoto.Connection('localhost')

print(check(conn, ['ls', '/nonexistent/path']))
