# https://doitu.info/blog/5aabc92d31e68500964d9255

from bottle import request, Bottle, abort, static_file
from gevent.pywsgi import WSGIServer
from geventwebsocket import WebSocketError
from geventwebsocket.handler import WebSocketHandler, Client
from geventwebsocket.websocket import WebSocket
from typing import Any, Dict, List, Callable, Tuple, Type

app: Bottle = Bottle()
server: WSGIServer = WSGIServer(("0.0.0.0", 8888), app, handler_class=WebSocketHandler)

# WebSocketの処理
@app.route('/websocket')
def handle_websocket() -> Any:
    websocket: WebSocket = request.environ.get('wsgi.websocket')

    if not websocket:
        abort(400, 'Expected WebSocket request.')

    #予想だけど、WebSocketHandlerはクライアントごとに生成される？
    while True:
        try:
            handler: WebSocketHandler = websocket.handler
            message: str = websocket.receive()

            # 実際の動作に必要でないけど、以下のようなクラスのインスタンスが入ってるのが解る
            assert type(websocket.handler) == WebSocketHandler
            assert type(websocket.handler.server) == WSGIServer

            # WSGIServerには本来、clientというプロパティは無いけど、コレを継承しているWebSocketHandlerで動的に追加されてる。
            # なので、型ヒントが利用できない（？）のでignoreするように
            for client_tmp in handler.server.clients.values(): # type: ignore

                # for文では型ヒントが利用できないので、以下のように型ヒントを付けた変数に再代入している。
                client: Client = client_tmp

                if client.ws.environ:
                    print(client.ws.environ.get('HTTP_SEC_WEBSOCKET_KEY', ''))
                else:
                    print('空。ブラウザを閉じたり画面を更新したりすると、クライアントから「切断したよ」、という情報が飛んでくる。')

                client.ws.send('message: %s, (client_address :%s)' % (message, client.address))

        except WebSocketError:
            break

# WebSocketクライアントを返す
@app.route('/')
def top():
    return static_file('index.html', root='./')

server.serve_forever()
