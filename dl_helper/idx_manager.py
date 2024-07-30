import socket, time, sys

CODE = '0QYg9Ky17dWnN4eK'
def get_idx(train_title):
    # 定义服务器地址和端口
    HOST = '168.138.158.156'
    PORT = 12345

    # 创建一个 TCP 套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        # 发送消息给服务器
        message = f'{CODE}_{train_title}'
        client_socket.sendall(message.encode())
        # 接收服务器的响应
        response = client_socket.recv(1024).decode()
        return int(response)
    finally:
        # 关闭套接字
        client_socket.close()

    raise Exception('get idx error')

def run_idx_manager():
    # 定义服务器地址和端口
    HOST = '0.0.0.0'
    PORT = 12345

    # 创建一个 TCP 套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定地址和端口
    server_socket.bind((HOST, PORT))

    # 开始监听连接
    server_socket.listen(5)
    print(f"Server is listening on {HOST}:{PORT}")

    titles = {}
    latest_timestamp = 0

    while True:
        # 等待客户端连接
        client_socket, client_address = server_socket.accept()
        print(f"Connected to {client_address}")

        # 检查重置 titles
        if time.time() - latest_timestamp > 30*60:
            titles = {}
        latest_timestamp = time.time()

        # 接收客户端消息
        data = client_socket.recv(1024)
        if data:
            _code, train_title = data.decode().split('_', maxsplit=1)
            if _code == CODE:
                if train_title not in titles:
                    titles[train_title] = -1
                titles[train_title] += 1

                # 发送idx回客户端
                client_socket.sendall(f'{titles[train_title]}'.encode())

        # 关闭与客户端的连接
        client_socket.close()

if "__main__" == __name__:
    run_idx_manager()
    # if len(sys.argv) > 1 and sys.argv[1] == "server":
    #     run_idx_manager()
    # else:
    #     print(f'idx: {get_idx("test")}')