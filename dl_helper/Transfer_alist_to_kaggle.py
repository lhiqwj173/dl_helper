import os, shutil
from pathlib import Path
from py_ext.alist import alist
from py_ext.lzma import decompress, compress_folder

def main_0():
    """
    1. 下载 alist 文件到本地
    2. 遍历解压文件
    3. 压缩成一个单独的压缩包
    """
    # 下载压缩文件
    alist_folder = r'/bc_train_data_wait/'
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

def batch_tar_and_remove(local_folder, batch_size=20):
    """
    每{batch_size}个文件打包一个.tar文件，打包后删除原文件
    
    参数:
        local_folder: 要打包的源文件夹路径
        batch_size: 每批打包的文件数量(默认20)
    """
    # 获取文件夹下所有文件（不包括子目录）
    all_files = [f for f in Path(local_folder).iterdir() if f.is_file()]
    
    # 按批次处理
    for batch_num, i in enumerate(range(0, len(all_files), batch_size), 1):
        batch_files = all_files[i:i + batch_size]
        
        # 创建临时文件夹存放当前批次文件
        temp_dir = Path(local_folder) / f"temp_batch_{batch_num}"
        temp_dir.mkdir(exist_ok=True)
        
        # 移动文件到临时文件夹
        for file in batch_files:
            shutil.move(str(file), str(temp_dir / file.name))
        
        # 打包临时文件夹
        tar_name = f"bc_train_data_batch_{batch_num}"
        shutil.make_archive(
            base_name=str(Path(local_folder) / tar_name),
            format="tar",
            root_dir=temp_dir,
        )
        
        # 删除临时文件夹及其内容
        shutil.rmtree(temp_dir)
        
        print(f"已打包: {tar_name}.tar (包含{len(batch_files)}个文件)")

def main():
    """
    1. 下载 alist 文件到本地
    2. 打包成一个文件
    """
    # 下载压缩文件
    alist_folder = r'/bc_train_data_wait/'
    local_folder = r'bc_train_data'
    alist_client = alist(os.environ['ALIST_USER'], os.environ['ALIST_PWD'])
    files = alist_client.listdir(alist_folder)
    alist_client.download([os.path.join(alist_folder, i['name']) for i in files], local_folder)

    # 打包文件夹（不压缩，生成 .tar 文件）
    batch_tar_and_remove(local_folder)

if __name__ == '__main__':
    main()
