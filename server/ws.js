var ws = new WebSocket('ws://echo.websocket.org');

ws.onopen = function () {
    console.log("Connecting is success!!");
    ws.send("Hello world!");
};
 
// メッセージを受け取った場合
ws.onmessage = function(e) {
    console.log(e.data);
    setTimeout(
        function() {
            ws.send('sample');
        },
        "1000"
    );
};

