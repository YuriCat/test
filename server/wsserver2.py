
from websocket_server import WebsocketServer

def _new_client(client, server):
    print(client['address'])

def _message_received(client, server, message):
    server.send_message(client, 'received ' + message)

import sys

port = int(sys.argv[1])

server = WebsocketServer(port=port, host='0.0.0.0')
server.set_fn_new_client(_new_client)
server.set_fn_message_received(_message_received)
server.run_forever()
