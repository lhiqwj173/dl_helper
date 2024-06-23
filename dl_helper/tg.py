# copied from https://github.com/tulir/mautrix-telegram/blob/master/mautrix_telegram/util/parallel_file_transfer.py
# Copyright (C) 2021 Tulir Asokan
import asyncio
import threading
import hashlib
import inspect
import math
import os, time
from dl_helper.train_param import logger
from collections import defaultdict
from typing import Optional, List, AsyncGenerator, Union, Awaitable, DefaultDict, Tuple, BinaryIO

from telethon import errors
from telethon import utils, helpers, TelegramClient
from telethon.crypto import AuthKey
from telethon.sessions import StringSession
from telethon.network import MTProtoSender
from telethon.tl.alltlobjects import LAYER
from telethon.tl.functions import InvokeWithLayerRequest
from telethon.tl.functions.auth import ExportAuthorizationRequest, ImportAuthorizationRequest
from telethon.tl.functions.upload import (GetFileRequest, SaveFilePartRequest,
                                          SaveBigFilePartRequest)
from telethon.tl.types import (Document, InputFileLocation, InputDocumentFileLocation,
                               InputPhotoFileLocation, InputPeerPhotoFileLocation, TypeInputFile,
                               InputFileBig, InputFile)

from py_ext.lzma import decompress

try:
    from mautrix.crypto.attachments import async_encrypt_attachment
except ImportError:
    async_encrypt_attachment = None

TypeLocation = Union[Document, InputDocumentFileLocation, InputPeerPhotoFileLocation,
                     InputFileLocation, InputPhotoFileLocation]


class DownloadSender:
    client: TelegramClient
    sender: MTProtoSender
    request: GetFileRequest
    remaining: int
    stride: int

    def __init__(self, client: TelegramClient, sender: MTProtoSender, file: TypeLocation, offset: int, limit: int,
                 stride: int, count: int) -> None:
        self.sender = sender
        self.client = client
        self.request = GetFileRequest(file, offset=offset, limit=limit)
        self.stride = stride
        self.remaining = count

    async def next(self) -> Optional[bytes]:
        if not self.remaining:
            return None
        result = await self.client._call(self.sender, self.request)
        self.remaining -= 1
        self.request.offset += self.stride
        return result.bytes

    def disconnect(self) -> Awaitable[None]:
        return self.sender.disconnect()


class UploadSender:
    client: TelegramClient
    sender: MTProtoSender
    request: Union[SaveFilePartRequest, SaveBigFilePartRequest]
    part_count: int
    stride: int
    previous: Optional[asyncio.Task]
    loop: asyncio.AbstractEventLoop

    def __init__(self, client: TelegramClient, sender: MTProtoSender, file_id: int, part_count: int, big: bool,
                 index: int,
                 stride: int, loop: asyncio.AbstractEventLoop) -> None:
        self.client = client
        self.sender = sender
        self.part_count = part_count
        if big:
            self.request = SaveBigFilePartRequest(file_id, index, part_count, b"")
        else:
            self.request = SaveFilePartRequest(file_id, index, b"")
        self.stride = stride
        self.previous = None
        self.loop = loop

    async def next(self, data: bytes) -> None:
        if self.previous:
            await self.previous
        self.previous = self.loop.create_task(self._next(data))

    async def _next(self, data: bytes) -> None:
        self.request.bytes = data
        await self.client._call(self.sender, self.request)
        self.request.file_part += self.stride

    async def disconnect(self) -> None:
        if self.previous:
            await self.previous
        return await self.sender.disconnect()


class ParallelTransferrer:
    client: TelegramClient
    loop: asyncio.AbstractEventLoop
    dc_id: int
    senders: Optional[List[Union[DownloadSender, UploadSender]]]
    auth_key: AuthKey
    upload_ticker: int

    def __init__(self, client: TelegramClient, dc_id: Optional[int] = None) -> None:
        self.client = client
        self.loop = self.client.loop
        self.dc_id = dc_id or self.client.session.dc_id
        self.auth_key = (None if dc_id and self.client.session.dc_id != dc_id
                         else self.client.session.auth_key)
        self.senders = None
        self.upload_ticker = 0

    async def _cleanup(self) -> None:
        await asyncio.gather(*[sender.disconnect() for sender in self.senders])
        self.senders = None

    @staticmethod
    def _get_connection_count(file_size: int, max_count: int = 20,
                              full_size: int = 100 * 1024 * 1024) -> int:
        if file_size > full_size:
            return max_count
        return math.ceil((file_size / full_size) * max_count)

    async def _init_download(self, connections: int, file: TypeLocation, part_count: int,
                             part_size: int) -> None:
        minimum, remainder = divmod(part_count, connections)

        def get_part_count() -> int:
            nonlocal remainder
            if remainder > 0:
                remainder -= 1
                return minimum + 1
            return minimum

        # The first cross-DC sender will export+import the authorization, so we always create it
        # before creating any other senders.
        self.senders = [
            await self._create_download_sender(file, 0, part_size, connections * part_size,
                                               get_part_count()),
            *await asyncio.gather(
                *[self._create_download_sender(file, i, part_size, connections * part_size,
                                               get_part_count())
                  for i in range(1, connections)])
        ]

    async def _create_download_sender(self, file: TypeLocation, index: int, part_size: int,
                                      stride: int,
                                      part_count: int) -> DownloadSender:
        return DownloadSender(self.client, await self._create_sender(), file, index * part_size, part_size,
                              stride, part_count)

    async def _init_upload(self, connections: int, file_id: int, part_count: int, big: bool
                           ) -> None:
        self.senders = [
            await self._create_upload_sender(file_id, part_count, big, 0, connections),
            *await asyncio.gather(
                *[self._create_upload_sender(file_id, part_count, big, i, connections)
                  for i in range(1, connections)])
        ]

    async def _create_upload_sender(self, file_id: int, part_count: int, big: bool, index: int,
                                    stride: int) -> UploadSender:
        return UploadSender(self.client, await self._create_sender(), file_id, part_count, big, index, stride,
                            loop=self.loop)

    async def _create_sender(self) -> MTProtoSender:
        dc = await self.client._get_dc(self.dc_id)
        sender = MTProtoSender(self.auth_key, loggers=self.client._log)
        await sender.connect(self.client._connection(dc.ip_address, dc.port, dc.id,
                                                     loggers=self.client._log,
                                                     proxy=self.client._proxy))
        if not self.auth_key:
            # logger.debug(f"Exporting auth to DC {self.dc_id}")
            auth = await self.client(ExportAuthorizationRequest(self.dc_id))
            self.client._init_request.query = ImportAuthorizationRequest(id=auth.id,
                                                                         bytes=auth.bytes)
            req = InvokeWithLayerRequest(LAYER, self.client._init_request)
            await sender.send(req)
            self.auth_key = sender.auth_key
        return sender

    async def init_upload(self, file_id: int, file_size: int, part_size_kb: Optional[float] = None,
                          connection_count: Optional[int] = None) -> Tuple[int, int, bool]:
        connection_count = connection_count or self._get_connection_count(file_size)
        part_size = (part_size_kb or utils.get_appropriated_part_size(file_size)) * 1024
        part_count = (file_size + part_size - 1) // part_size
        is_large = file_size > 10 * 1024 * 1024
        await self._init_upload(connection_count, file_id, part_count, is_large)
        return part_size, part_count, is_large

    async def upload(self, part: bytes) -> None:
        await self.senders[self.upload_ticker].next(part)
        self.upload_ticker = (self.upload_ticker + 1) % len(self.senders)

    async def finish_upload(self) -> None:
        await self._cleanup()

    async def download(self, file: TypeLocation, file_size: int,
                       part_size_kb: Optional[float] = None,
                       connection_count: Optional[int] = None) -> AsyncGenerator[bytes, None]:
        connection_count = connection_count or self._get_connection_count(file_size)
        part_size = (part_size_kb or utils.get_appropriated_part_size(file_size)) * 1024
        part_count = math.ceil(file_size / part_size)
        # logger.debug("Starting parallel download: "
        #           f"{connection_count} {part_size} {part_count} {file!s}")
        await self._init_download(connection_count, file, part_count, part_size)

        part = 0
        while part < part_count:
            tasks = []
            for sender in self.senders:
                tasks.append(self.loop.create_task(sender.next()))
            for task in tasks:
                data = await task
                if not data:
                    break
                yield data
                part += 1
                # logger.debug(f"Part {part} downloaded")

        # logger.debug("Parallel download finished, cleaning up connections")
        await self._cleanup()


parallel_transfer_locks: DefaultDict[int, asyncio.Lock] = defaultdict(lambda: asyncio.Lock())


def stream_file(file_to_stream: BinaryIO, chunk_size=1024):
    while True:
        data_read = file_to_stream.read(chunk_size)
        if not data_read:
            break
        yield data_read


async def _internal_transfer_to_telegram(client: TelegramClient,
                                         response: BinaryIO,
                                         file_name,
                                         progress_callback: callable
                                         ) -> Tuple[TypeInputFile, int]:
    file_id = helpers.generate_random_long()
    file_size = os.path.getsize(response.name)

    hash_md5 = hashlib.md5()
    uploader = ParallelTransferrer(client)
    part_size, part_count, is_large = await uploader.init_upload(file_id, file_size)
    buffer = bytearray()
    for data in stream_file(response):
        if progress_callback:
            r = progress_callback(response.tell(), file_size)
            if inspect.isawaitable(r):
                await r
        if not is_large:
            hash_md5.update(data)
        if len(buffer) == 0 and len(data) == part_size:
            await uploader.upload(data)
            continue
        new_len = len(buffer) + len(data)
        if new_len >= part_size:
            cutoff = part_size - len(buffer)
            buffer.extend(data[:cutoff])
            await uploader.upload(bytes(buffer))
            buffer.clear()
            buffer.extend(data[cutoff:])
        else:
            buffer.extend(data)
    if len(buffer) > 0:
        await uploader.upload(bytes(buffer))
    await uploader.finish_upload()
    if is_large:
        return InputFileBig(file_id, part_count, file_name), file_size
    else:
        return InputFile(file_id, part_count, file_name, hash_md5.hexdigest()), file_size


async def download_file(client: TelegramClient,
                        location: TypeLocation,
                        out: BinaryIO,
                        progress_callback: callable = None
                        ) -> BinaryIO:
    size = location.size
    dc_id, location = utils.get_input_location(location)
    # We lock the transfers because telegram has connection count limits
    downloader = ParallelTransferrer(client, dc_id)
    downloaded = downloader.download(location, size)
    async for x in downloaded:
        out.write(x)
        if progress_callback:
            r = progress_callback(out.tell(), size)
            if inspect.isawaitable(r):
                await r

    return out


async def upload_file(client: TelegramClient,
                      file: BinaryIO,
                      file_name,
                      progress_callback: callable = None,

                      ) -> TypeInputFile:
    res = (await _internal_transfer_to_telegram(client, file, file_name, progress_callback))[0]
    return res

t = 0
count = 0
def progress_cb(current, total):
    global t, count
    count += 1
    if t == 0:
        t = time.time()
        return

    if count % 100 != 0:
        return

    cur_mb = current / 1024 / 1024
    cur_cost = time.time() - t
    if cur_cost == 0:
        return 

    speed = cur_mb / cur_cost
    pct = current / total
    remain = (total/ 1024 / 1024 - cur_mb) / speed
    print(f'done: {pct:.2%}, speed: {speed:.2f} MB/s remain: {remain:.2f}s', end='\r')

    if current == total:
        t = 0

async def get_channel_entity(client, name):
    name = 'dl_dataset'
    
    # 匹配channel
    async for dialog in client.iter_dialogs():
        if dialog.name == name:
            # 频道文件列表
            channel_username = dialog.id  # 替换为频道的用户名或 ID
            return await client.get_entity(channel_username)  # 获取频道实体对象
    
    return None

async def _upload_file(client, filepath, channel_name):
    entity = await get_channel_entity(client, channel_name)

    with open(filepath, "rb") as out:
        media = await upload_file(client, out, filepath.split('/')[-1], progress_callback=progress_cb)
        await client.send_file(entity, media)

async def _handle_massage(client, massage_name, handle_func, channel_name):
    """ 返回成功/失败 """
    entity = await get_channel_entity(client, channel_name)

    messages = client.iter_messages(entity, reverse=True)
    files = []
    async for message in messages:
        if not (message.file and message.file.name):
            continue

        files.append(message.file.name)
        if message.file.name != massage_name:
            continue

        # 处理数据
        await handle_func(client, message)
        return True

    print(f"Files: {files}")
    print(f"no match: {massage_name}")
    return False

async def _del_file(client, filename, channel_name):
    async def del_func(client, message):
        await client.delete_messages(message.peer_id, message)
    return await _handle_massage(client, filename, del_func, channel_name)

async def _download_dataset(client, dataset_name, dst_folder, channel_name):
    """ 返回成功/失败 """
    async def download_func(client, message):
        print(f"File Name: {message.file.name}")
        print(f"File Size: {message.file.size}")

        # 下载文件
        dst = './data' if not dst_folder else dst_folder
        os.makedirs(dst, exist_ok=True)
        _file = os.path.join(dst, message.file.name)
        print(f"download: {_file}")
        with open(_file, "wb") as out:
            await download_file(client, message.document, out, progress_callback=progress_cb)

        # 尝试解压文件
        try:
            decompress(_file)
        except:
            pass

    return await _handle_massage(client, dataset_name, download_func, channel_name)

async def tg_download_async(session, dataset_name, dst_folder='', channel_name = 'dl_dataset' ):
    """ 返回成功/失败 """
    # 创建客户端
    client = TelegramClient(StringSession(session), 1, '1')
    async with client:
        return await _download_dataset(client, dataset_name, dst_folder, channel_name)

async def tg_del_file_async(session, file_name, channel_name = 'dl_dataset' ):
    # 创建客户端
    client = TelegramClient(StringSession(session), 1, '1')
    async with client:
        return await _del_file(client, file_name, channel_name)

async def tg_upload_async(session, filepath, channel_name = 'dl_dataset' ):
    filepath = filepath.replace('\\', '/')

    # 创建客户端
    client = TelegramClient(StringSession(session), 1, '1')
    async with client:
        await _upload_file(client, filepath, channel_name)

def thread_run_async_func(func, rets, *args):
    for i in range(3):
        try:
            rets.append(asyncio.run(func(*args)))
            return

        except errors.FloodWaitError as e:
            print('Have to sleep', e.seconds, 'seconds')
            time.sleep(e.seconds)
        except Exception as e:
            print(e)
            time.sleep(10)

def run_async_func(func, *args, **kwargs):
    # try:
    #     loop = asyncio.get_running_loop()
    # except RuntimeError:  # 'RuntimeError: There is no current event loop...'
    #     loop = None

    # if loop and loop.is_running():
    #     tsk = loop.create_task(func(*args, **kwargs))
    #     return tsk
    # else:
    #     print('Starting new event loop')
    #     return asyncio.run(func(*args, **kwargs))
    rets = []
    t = threading.Thread(target=thread_run_async_func, args=(func, rets, *args))
    t.start()
    t.join()

    return rets[0]

def tg_download(session, dataset_name, dst_folder='', channel_name = 'dl_dataset' ):
    return run_async_func(tg_download_async, session, dataset_name, dst_folder, channel_name)

def tg_del_file(session, file_name, channel_name = 'dl_dataset' ):
    return run_async_func(tg_del_file_async, session, file_name, channel_name)

def tg_upload(session, filepath, channel_name = 'dl_dataset' ):
    return run_async_func(tg_upload_async, session, filepath, channel_name)

if __name__ == '__main__':
    ses = '1BVtsOKABu6pKio99jf7uqjfe5FMXfzPbEDzB1N5DFaXkEu5Og5dJre4xg4rbXdjRQB7HpWw7g-fADK6AVDnw7nZ1ykiC5hfq-IjDVPsMhD7Sffuv0lTGa4-1Dz2MktHs3e_mXpL1hNMFgNm5512K1BWQvij3xkoiHGKDqXLYzbzeVMr5e230JY7yozEZRylDB_AuFeBGDjLcwattWnuX2mnTZWgs-lS1A_kZWomGl3HqV84UsoJlk9b-GAbzH-jBunsckkjUijri6OBscvzpIWO7Kgq0YzxJvZe_a1N8SFG3Gbuq0mIOkN3JNKGTmYLjTClQd2PIJuFSxzYFPQJwXIWZlFg0O2U='
    
    # res = tg_download(ses, 'tdx-a.7z', r'C:\Users\lh\Desktop\temp')
    # res = tg_download(ses, 'tdx-a.7z')

    res = tg_del_file(ses, r"tdx-a.7z")
    # res = tg_upload(ses, r"C:\Users\lh\Downloads\tdx-a.7z")

    print(f'res: {res}')