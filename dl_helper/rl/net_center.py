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

init_logger(f'net_center_{datetime.now().strftime("%Y%m%d")}', enqueue=True, timestamp=False)

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
        self.clients = set()

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
        while True:
            await asyncio.sleep(self.check_interval)

            if len(self.clients) > 0:
                self.last_activity = time.time()

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

        # 内网穿透后不需要block
        # # 0. 检查是否被block
        # if self.block_ips.is_blocked(client_ip):
        #     log(f"Blocked connection from {client_ip}")
        #     writer.close()  # 对于被block的连接，直接关闭即可
        #     return

        self.clients.add(writer)
        msg_header = f'[{client_ip:<15} {client_port:<5}]'
        log(f"{msg_header} connected")
        msg_header_add_title = False

        try:
            """
            1. 接收 CODE (验证)
            2. 接收指令数据 
            3. 处理指令
            4. 关闭连接
            """
            # 1. 接收 CODE
            res = await async_recv_msg(reader, timeout=3)
            _code = res.decode()
            log(f'{msg_header} recv code: {_code}')
            if _code != CODE:
                log(f'code error: {_code}')
                raise Exception(f'code error: {_code}')

            # 2.1 接收指令数据
            data = await async_recv_msg(reader, timeout=3)
            a = data.decode()
            log(f'{msg_header} recv data: {a}')

            # 2.2 分解指令
            train_title, cmd = a.split(':', maxsplit=1)
            if not msg_header_add_title:
                msg_header += f'[{train_title}]'
                msg_header_add_title = True

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
            self.clients.remove(writer)
            if not writer.is_closing():
                writer.close()  # 如果连接还未关闭，则关闭它
            log(f"Client disconnected from {peer}")

    async def start(self):
        self.sock.bind((self.host, self.port))
        
        # 使用start_server创建服务器
        self.server = await asyncio.start_server(
            self.handle_client,
            sock=self.sock,
            backlog=self.backlog,
        )
        
        log(f"Server started on {self.host}:{self.port}")
        
        # 启动所有处理器
        loop = asyncio.get_event_loop()
        for handler in self.handlers.values():
            handler.start(loop)

        # 启动超时检查
        timeout_task = asyncio.create_task(self.check_timeout())
        
        # 设置信号处理
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda: asyncio.create_task(self.shutdown())
            )
            
        async with self.server:
            try:
                await self.server.serve_forever()
            finally:
                timeout_task.cancel()

    async def shutdown(self):
        log("Shutting down server...")
        
        # 关闭所有客户端连接
        for writer in self.clients:
            writer.close()
            await writer.wait_closed()
            
        # 停止服务器
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # 停止事件循环
        asyncio.get_event_loop().stop()
        log("Server shutdown complete")

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
    # 初始化实验处理器
    # run_param_center()

    # 异步启动
    asyncio.run(main())

