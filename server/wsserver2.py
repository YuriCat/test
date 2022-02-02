
from websocket_server import WebsocketServer

def _new_client(client, server):
    print(client['address'])

def _message_received(client, server, message):
    server.send_message(client, 'received ' + message)


server = WebsocketServer(port=8080, host='127.0.0.1')
server.set_fn_new_client(_new_client)
server.set_fn_message_received(_message_received)
server.run_forever()
