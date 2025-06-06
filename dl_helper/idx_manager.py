"""
接收客户端数据:
    CODE_title

会根据title返回最新的索引
    - 默认从0开始, 1小时后重置索引为0
    - 如果 title 中包含 _b@1000 则从1000开始
    - 如果 title 以 once 开头 则不重置
    - 如果 title 以 time 开头 则直接返回当前时间戳ms

"""


import socket, time, sys, os, re

CODE = '0QYg9Ky17dWnN4eK'
def get_idx(train_title):
    # 定义服务器地址和端口
    HOST = '146.235.33.108'
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

record_file = os.path.join(os.path.expanduser('~'), 'block_ips.txt')
def read_block_ips():

    if not os.path.exists(record_file):
        return []

    with open(record_file, 'r') as f:
        ips = f.read().splitlines()
        return ips

def update_block_ips(ips):
    with open(record_file, 'w') as f:
        f.write('\n'.join(ips))

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
    onece_titles = {}

    latest_timestamp = 0

    # 读取ban ip
    block_ips = read_block_ips()

    while True:
        # 等待客户端连接
        client_socket, client_address = server_socket.accept()
        client_ip = client_address[0]

        if client_ip in block_ips:
            print(f"Blocked connection from {client_ip}")
            # 关闭与客户端的连接
            client_socket.close()
            continue

        print(f"Connected to {client_address}")

        # 检查重置 titles
        if time.time() - latest_timestamp > 1*3600:
            print(f'重置titles字典 {time.time()} {latest_timestamp} diff:{time.time() - latest_timestamp}')
            titles = {}
        latest_timestamp = time.time()

        # 接收客户端消息
        need_block = False
        try:
            data = client_socket.recv(1024)
            if data:
                try:
                    data_str = data.decode()
                except:
                    need_block = True
                    data_str = ''
                    
                if '_' in data_str:
                    _code, train_title = data_str.split('_', maxsplit=1)

                    begin_idx = 0
                    # 使用正则表达式匹配 'b@' 后面的数字部分
                    # 起始的 索引为 0
                    result = re.search(r'_b@(\d+)_', train_title)
                    if result:
                        begin_idx = int(result.group(1))

                    if _code == CODE:
                        idx = 0
                        if train_title.startswith('once'):
                            if train_title not in onece_titles:
                                onece_titles[train_title] = begin_idx
                            idx = onece_titles[train_title]
                            onece_titles[train_title] += 1
                        elif train_title.startswith('time'):
                            idx = int(time.time() * 1000)
                        else:
                            if train_title not in titles:
                                titles[train_title] = begin_idx
                            idx = titles[train_title]
                            titles[train_title] += 1

                        print(f'{train_title} {idx}')

                        # 发送idx回客户端
                        client_socket.sendall(f'{idx}'.encode())
                    else:
                        need_block = True
                else:
                    need_block = True

            else:
                need_block = True

        except ConnectionResetError as e:
            pass

        if need_block:
            block_ips.append(client_ip)
            update_block_ips(block_ips)
            print(f'block ip: {client_ip}')

        # 关闭与客户端的连接
        client_socket.close()

if "__main__" == __name__:
    # print(f'idx: {get_idx("test")}')
    # run_idx_manager()
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_idx_manager()
    else:
        print(f'idx: {get_idx("test")}')