# -*- coding: utf-8 -*-
import asyncio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from py_ext.tool import log, init_logger, get_exception_msg
from py_ext.wechat import send_wx

from datetime import datetime, timezone, timedelta
import socket, time, sys, os, re
import pickle
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import uvloop
from typing import Optional
import signal

from dl_helper.rl.socket_base import CODE, PORT, send_msg, recv_msg, tune_tcp_socket
from dl_helper.rl.socket_base import async_recv_msg, async_send_msg
from dl_helper.rl.rl_utils import read_train_title_item
from dl_helper.rl.param_keeper import ExperimentHandler

init_logger(f'net_center_{datetime.now().strftime("%Y%m%d")}', home='net_center', enqueue=True, timestamp=False)

class BlockIPs:
    """管理block ip的类"""
    def __init__(self):
        self.record_file = os.path.join(os.path.expanduser('~'), 'block_ips_net_center.txt')
        self.ips = self.read()

    def read(self):
        """读取block ip列表"""
        if not os.path.exists(self.record_file):
            return []
        with open(self.record_file, 'r') as f:
            ips = f.read().splitlines()
            return list(set(ips)) # 去重

    def _update(self, ips):
        """更新block ip列表"""
        with open(self.record_file, 'w') as f:
            f.write('\n'.join(sorted(set(ips)))) # 排序并去重

    def add(self, ip):
        """添加ip到block ip列表"""
        if ip not in self.ips: # 避免重复添加
            log(f'add block ip: {ip}')
            self.ips.append(ip)
            self._update(self.ips)

    def is_blocked(self, ip):
        """检查ip是否被block"""
        return ip in self.ips

class AsyncSocketServer:
    def __init__(self, backlog: int = 1000):
        self.host = '0.0.0.0'
        self.port = PORT
        self.backlog = backlog
        self.server: Optional[asyncio.Server] = None
        self.clients = {}

        # 超时检查
        self.last_activity = 0
        self.timeout = 300  # 5分钟超时
        self.check_interval = 60  # 每分钟检查一次
        
        # 设置socket选项
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tune_tcp_socket(self.sock)

        # 初始化block ip
        self.block_ips = BlockIPs()

        # 初始化实验处理器      
        self.handlers = {}
        train_dict = read_train_title_item()
        for title, config in train_dict.items():
            log(f'{title} init')
            self.handlers[title] = ExperimentHandler(title, config)

    async def check_timeout(self):
        """检查是否超时"""
        log(f"check timeout begin")
        while True:
            await asyncio.sleep(self.check_interval)

            log(f"check timeout, clients: {len(self.clients)}")
    
            if len(self.clients) > 0:
                self.last_activity = time.time()
                log(f"server has activity, last activity: {self.last_activity}")

            if self.last_activity == 0:
                continue

            if time.time() - self.last_activity > self.timeout:
                log("server has not activity for 5 minutes, shutdown...")
                await self.shutdown()
                break

    async def handle_client(self, reader: asyncio.StreamReader, 
                          writer: asyncio.StreamWriter):
        peer = writer.get_extra_info('peername')
        client_ip, client_port = peer

        log(f'handle client {client_port}')
        # 内网穿透后不需要block
        # # 0. 检查是否被block
        # if self.block_ips.is_blocked(client_ip):
        #     log(f"Blocked connection from {client_ip}")
        #     writer.close()  # 对于被block的连接，直接关闭即可
        #     return

        disconnect_means_client_dead = False
        try:
            """
            1. 接收 CODE_ip
            2. 接收指令数据 
            3. 处理指令
            4. 关闭连接
            """
            # 1. 接收 CODE_ip
            res = await async_recv_msg(reader, timeout=3)
            _code, client_ip = res.decode().split('_', maxsplit=1)
            log(f'recv code: {_code}, ip: {client_ip}')
            if _code != CODE:
                log(f'code error: {_code}')
                raise Exception(f'code error: {_code}')
            
            # 记录 ip 连接
            if client_ip not in self.clients:
                self.clients[client_ip] = set()
            self.clients[client_ip].add(writer)
            msg_header = f'[{client_ip:<15} {client_port:<5}]'
            log(f"{msg_header} connected")

            # 2.1 接收指令数据
            data = await async_recv_msg(reader, timeout=3)
            a = data.decode()
            log(f'{msg_header} recv data: {a}')

            # 2.2 分解指令
            train_title, cmd = a.split(':', maxsplit=1)
            msg_header += f'[{train_title}]'

            # 判断是否是关键连接
            if cmd in ['wait_params', 'update_gradients']:
                log(f'{msg_header} {cmd} is key connection')
                disconnect_means_client_dead = True

            # 2.3 获取处理器
            if train_title == 'test':
                handler = list(self.handlers.values())[0]
            else:
                if train_title not in self.handlers:
                    log(f'{train_title} not found')
                    return
                handler = self.handlers[train_title]

            # 3 处理指令
            await handler.async_handle_request(client_ip, msg_header, cmd, writer, reader)

            # 4. 关闭连接
            return
               
        except ConnectionError as e:
            log(f"Connection error from client {peer}\n{get_exception_msg()}")
        except Exception as e:
            log(f"Error handling client {peer}\n{get_exception_msg()}")
            # 内网穿透后不需要block
            # if not safe_connect:
            #     self.block_ips.add(client_ip)
        finally:
            # 如果关键连接断开，则关闭该ip的所有连接
            if disconnect_means_client_dead:
                if client_ip == '127.0.0.1':
                    # 若ip解析就异常了，直接关闭
                    if not writer.is_closing():
                        writer.close()  # 如果连接还未关闭，则关闭它
                        await writer.wait_closed()
                else:
                    log(f'{msg_header} client {client_ip} is dead, close all connections from ip:{client_ip}')
                    for writer in self.clients[client_ip]:
                        if not writer.is_closing():
                            writer.close()  # 如果连接还未关闭，则关闭它
                            await writer.wait_closed()
                    del self.clients[client_ip]
            else:
                self.clients[client_ip].remove(writer)
                if not writer.is_closing():
                    writer.close()  # 如果连接还未关闭，则关闭它
                    await writer.wait_closed()
                if len(self.clients[client_ip]) == 0:
                    del self.clients[client_ip]

            log(f"Client disconnected from {client_ip} {client_port}")

    async def start(self):
        self.sock.bind((self.host, self.port))
        
        # 使用start_server创建服务器
        self.server = await asyncio.start_server(
            self.handle_client,
            sock=self.sock,
            backlog=self.backlog,
        )
        
        log(f"Server started on {self.host}:{self.port}")

        # 启动超时检查
        timeout_task = asyncio.create_task(self.check_timeout())
            
        try:
            async with self.server:
                await self.server.serve_forever()
        except asyncio.CancelledError:
            log("Server task cancelled")
        finally:
            timeout_task.cancel()
            await self.shutdown()

    async def shutdown(self):
        log("Shutting down server...")
        
        # 关闭所有客户端连接
        for client_ip in self.clients:
            for writer in self.clients[client_ip]:
                writer.close()
                await writer.wait_closed()
            
        # 停止服务器
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        log("Server shutdown complete")
        # 强制退出程序
        os._exit(0)

def main():
    # 使用uvloop替换默认事件循环
    uvloop.install()
    
    # 创建服务器实例
    server = AsyncSocketServer()

    try:
        # 运行服务器
        asyncio.run(server.start(), debug=True)
    except KeyboardInterrupt:
        log("Received keyboard interrupt")
    except Exception as e:
        log(f'ERROR:\n{get_exception_msg()}')
    finally:
        log("Server stopped")

if __name__ == '__main__':
    # 异步启动
    main()

