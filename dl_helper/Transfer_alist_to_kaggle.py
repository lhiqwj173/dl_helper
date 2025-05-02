import os
from py_ext.alist import alist
from py_ext.lzma import decompress, compress_folder

def main():
    """
    1. 下载 alist 文件到本地
    2. 遍历解压文件
    3. 压缩成一个单独的压缩包
    """
    # 下载压缩文件
    alist_folder = r'/bc_train_data/'
    local_folder = r'bc_train_data'
    alist_client = alist(os.environ['ALIST_USER'], os.environ['ALIST_PWD'])
    files = alist_client.listdir(alist_folder)
    alist_client.download([os.path.join(alist_folder, i['name']) for i in files], local_folder)

    # 解压文件
    files = os.listdir(local_folder)
    for file in files:
        decompress(os.path.join(local_folder, file))

    # 压缩成一个单独的压缩包
    compress_folder(local_folder, local_folder+'.7z', level=9, inplace=False)

if __name__ == '__main__':
    main()
