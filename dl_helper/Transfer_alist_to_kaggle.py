import os, shutil, sys, subprocess
from pathlib import Path
import time
import threading
from tqdm import tqdm
import tarfile

from py_ext.alist import alist
from py_ext.lzma import decompress, compress_folder

CORE_COUNT = os.cpu_count()  # 获取 CPU 核心数
LIMIT = int(80 * CORE_COUNT)  # 例如 4 核时 limit=320

# 定义支持的视频文件扩展名
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# 文件大小限制（2GB，单位：字节）
SIZE_LIMIT = 2 * 1024 * 1024 * 1024  # 2GB in bytes

def gpu_available():
    try:
        output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        return "GPU is available" in output or "NVIDIA" in output
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_file_size(file_path):
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)

def get_video_duration(file_path):
    """获取视频总时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def calculate_target_bitrate(file_path, target_size):
    """根据目标文件大小估算码率（kbps）"""
    duration = get_video_duration(file_path)
    audio_bitrate = 128  # kbps
    target_bitrate = ((target_size * 8) / duration - audio_bitrate * 1000) / 1000  # kbps
    return max(target_bitrate, 1000)  # 最低码率设为1000kbps

def get_original_bitrate(file_path):
    """获取视频文件的原始码率（kbps）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'stream=bit_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    bitrates = result.stdout.strip().split('\n')
    video_bitrate = next((int(b) for b in bitrates if b.isdigit()), 0)
    return video_bitrate / 1000  # 转换为kbps

def read_stream(stream, name, total_duration):
    """线程函数：实时读取并打印流，同时计算进度和剩余时间"""
    begin_time = time.time()

    for line in iter(stream.readline, ''):
        if "out_time=" in line:
            try:
                time_str = line.split('=')[1].strip()
                h, m, s = time_str.split(':')
                current_time = float(h) * 3600 + float(m) * 60 + float(s)
                remaining_time = total_duration - current_time
                progress_percent = (current_time / total_duration) * 100
                speed = current_time / (time.time() - begin_time)
                if speed > 0:
                    left_time = int(remaining_time / speed)
                    hours = int(left_time // 3600)
                    minutes = int((left_time % 3600) // 60)
                    print(f"进度: {progress_percent:.2f}% | 剩余时间: {hours}小时{minutes}分", end='\r')
                else:
                    print(f"进度: {progress_percent:.2f}%", end='\r')
            except:
                pass

def compress_video_0(file_path, target_bitrate=None):
    """使用 FFmpeg 压缩视频文件，根据 GPU 可用性选择加速方式并实时显示输出"""
    temp_output = file_path + '.temp.mp4'
    
    # 如果未指定目标码率，则动态计算
    if target_bitrate is None:
        target_bitrate = calculate_target_bitrate(file_path, SIZE_LIMIT)
    else:
        target_bitrate = max(target_bitrate, 1000)
    
    # 根据 GPU 可用性选择编码器
    if gpu_available():
        encoder = 'h264_nvenc'  # GPU 加速
        device = 'GPU'
        cmd = ['ffmpeg', '-i', file_path, '-c:v', encoder, '-b:v', f'{target_bitrate}k', '-c:a', 'aac', '-b:a', '128k', '-progress', '-', '-y', temp_output]
    else:
        encoder = 'libx264'     # CPU
        device = 'CPU'
        cmd = ['ffmpeg', '-i', file_path, '-c:v', encoder, '-b:v', f'{target_bitrate}k', '-c:a', 'aac', '-b:a', '128k', '-progress', '-', '-y', temp_output]
    
    try:
        # 获取视频总时长
        total_duration = get_video_duration(file_path)
        
        # 启动 FFmpeg 进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,  # 行缓冲
        )
        
        # 如果使用 CPU，限制 CPU 使用率到 80%
        if not gpu_available():
            pid = process.pid
            cpulimit_cmd = ['cpulimit', '-p', str(pid), '-l', str(LIMIT)]
            cpulimit_process = subprocess.Popen(cpulimit_cmd)
        
        # 创建线程读取 stdout 和 stderr
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT", total_duration))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR", total_duration))
        
        stdout_thread.start()
        stderr_thread.start()
        
        # 等待 FFmpeg 进程结束
        process.wait()
        
        # 如果使用了 cpulimit，终止它
        if not gpu_available():
            cpulimit_process.terminate()
        
        # 等待线程完成
        stdout_thread.join()
        stderr_thread.join()
        
        # 检查 FFmpeg 是否成功执行
        if process.returncode == 0:
            shutil.move(temp_output, file_path)
            print(f"\n成功压缩并覆盖: {file_path} (目标码率: {target_bitrate}kbps, 使用 {device})")
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except subprocess.CalledProcessError as e:
        print(f"压缩失败: {file_path}, 错误: {e}")
        if os.path.exists(temp_output):
            os.remove(temp_output)

def compress_video_1(file_path, target_bitrate=None, codec='h264'): 
    """使用 FFmpeg 压缩视频文件，根据 GPU 和 codec 参数选择编码器"""
    temp_output = file_path + '.temp.mp4'

    # 如果未指定目标码率，则动态计算
    if target_bitrate is None:
        target_bitrate = calculate_target_bitrate(file_path, SIZE_LIMIT)
    else:
        target_bitrate = max(target_bitrate, 1000)

    # 获取视频总时长
    total_duration = get_video_duration(file_path)

    # 选择编码器
    device = 'CPU'
    if codec == 'h264':
        encoder = 'h264_nvenc' if gpu_available() else 'libx264'
        device = 'GPU' if 'nvenc' in encoder else 'CPU'
    elif codec == 'h265':
        encoder = 'hevc_nvenc' if gpu_available() else 'libx265'
        device = 'GPU' if 'nvenc' in encoder else 'CPU'
    elif codec == 'av1':
        encoder = 'libaom-av1'  # AV1 仅 CPU 编码
        device = 'CPU'
    else:
        raise ValueError(f"不支持的编码格式: {codec}")

    # 构建 FFmpeg 命令
    cmd = [
        'ffmpeg', '-i', file_path,
        '-c:v', encoder,
        '-b:v', f'{target_bitrate}k',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-progress', '-', '-y', temp_output
    ]

    try:
        # 启动 FFmpeg 进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )

        # 如果使用 CPU，限制 CPU 使用率到 80%
        if device == 'CPU':
            pid = process.pid
            cpulimit_cmd = ['cpulimit', '-p', str(pid), '-l', str(LIMIT)]
            cpulimit_process = subprocess.Popen(cpulimit_cmd)

        # 启动读取线程
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "STDOUT", total_duration))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "STDERR", total_duration))

        stdout_thread.start()
        stderr_thread.start()

        process.wait()

        if device == 'CPU':
            cpulimit_process.terminate()

        stdout_thread.join()
        stderr_thread.join()

        if process.returncode == 0:
            shutil.move(temp_output, file_path)
            print(f"\n成功压缩并覆盖: {file_path} (目标码率: {target_bitrate}kbps, 使用 {device}, 编码器: {codec})")
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"压缩失败: {file_path}, 错误: {e}")
        if os.path.exists(temp_output):
            os.remove(temp_output)

def compress_video(file_path, target_size_gb=1.95, audio_bitrate_kbps=128):
    """
    压缩视频到指定大小（默认1.95GB）并保留最大质量（H.265双遍编码）
    使用 subprocess.Popen 调用 ffmpeg，压缩完成后替换原文件
    """
    # 转换大小
    target_size_bytes = int(target_size_gb * 1024 ** 3)
    temp_output = file_path + ".temp_compressed.mp4"
    passlog_file = "ffmpeg2pass"

    # 获取视频时长（单位：秒）
    cmd_duration = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    duration_result = subprocess.run(cmd_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        duration = float(duration_result.stdout.strip())
    except ValueError:
        raise RuntimeError("无法获取视频时长")

    # 计算目标总码率（bps）
    total_bitrate = target_size_bytes * 8 / duration  # bits per second
    audio_bitrate = audio_bitrate_kbps * 1000  # audio bitrate in bps
    video_bitrate = int(total_bitrate - audio_bitrate)

    print(f"⏱ Duration: {duration:.2f}s")
    print(f"🎯 Target Size: {target_size_gb} GB")
    print(f"🎥 Video Bitrate: {video_bitrate / 1000:.2f} kbps")
    print(f"🔊 Audio Bitrate: {audio_bitrate / 1000:.2f} kbps")

    # 第1遍（无音频）
    cmd_pass1 = [
        "ffmpeg", "-y", "-i", file_path,
        "-c:v", "libx265", "-b:v", str(video_bitrate), "-pass", "1",
        "-preset", "slow", "-x265-params", "aq-mode=3",
        "-an", "-f", "null", os.devnull
    ]
    subprocess.Popen(cmd_pass1).wait()

    # 第2遍（带音频）
    cmd_pass2 = [
        "ffmpeg", "-y", "-i", file_path,
        "-c:v", "libx265", "-b:v", str(video_bitrate), "-pass", "2",
        "-preset", "slow", "-x265-params", "aq-mode=3",
        "-c:a", "aac", "-b:a", f"{audio_bitrate_kbps}k",
        temp_output
    ]
    subprocess.Popen(cmd_pass2).wait()

    # 清理2-pass日志文件
    for f in [passlog_file + ext for ext in ["-0.log", "-0.log.mbtree"]]:
        if os.path.exists(f):
            os.remove(f)

    # 检查输出文件是否存在
    if not os.path.exists(temp_output):
        raise RuntimeError("压缩失败，未生成输出文件")

    # 替换原始文件
    backup_path = file_path + ".bak"
    shutil.move(file_path, backup_path)
    shutil.move(temp_output, file_path)
    print(f"✅ 压缩完成，原文件已替换（备份: {backup_path}）")

def process_folder_0(folder_path, target_bitrate=None):
    """递归遍历文件夹并处理视频文件"""
    # 收集所有视频文件路径
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if '.temp.mp4' in file:
                continue
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1].lower() in VIDEO_EXTENSIONS:
                video_files.append(file_path)
    
    if not video_files:
        print(f"在 {folder_path} 中未找到视频文件")
        return
    
    print(f"共找到 {len(video_files)} 个视频文件")
    for file_path in tqdm(video_files):
        size = get_file_size(file_path)
        original_bitrate = get_original_bitrate(file_path)
        if target_bitrate is not None:
            if original_bitrate > target_bitrate * 1.1:  # 允许10%的误差
                print(f"发现需要压缩的视频文件: {file_path} ({size / 1024 / 1024:.2f}MB, {original_bitrate}kbps)")
                compress_video(file_path, target_bitrate)
            else:
                print(f"跳过文件（码率已足够低）: {file_path}")
        else:
            if size > SIZE_LIMIT or original_bitrate > 5000:
                print(f"发现需要压缩的视频文件: {file_path} ({size / 1024 / 1024:.2f}MB, {original_bitrate}kbps)")
                compress_video(file_path)
            else:
                print(f"跳过文件: {file_path}")

def process_folder(folder_path, target_bitrate=None, codec='h264'):
    """递归遍历文件夹并处理视频文件"""
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if '.temp.mp4' in file:
                continue
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1].lower() in VIDEO_EXTENSIONS:
                video_files.append(file_path)

    if not video_files:
        print(f"在 {folder_path} 中未找到视频文件")
        return

    print(f"共找到 {len(video_files)} 个视频文件")
    for file_path in tqdm(video_files):
        size = get_file_size(file_path)
        original_bitrate = get_original_bitrate(file_path)
        if target_bitrate is not None:
            if original_bitrate > target_bitrate * 1.1:
                compress_video(file_path, target_bitrate, codec)
            else:
                print(f"跳过文件（码率已足够低）: {file_path}")
        else:
            if size > SIZE_LIMIT or original_bitrate > 5000:
                compress_video(file_path, codec=codec)
            else:
                print(f"跳过文件: {file_path}")

def process_folder(folder_path, codec='h264'):
    """递归遍历文件夹并处理视频文件"""
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if '.temp.mp4' in file:
                continue
            file_path = os.path.join(root, file)
            if os.path.splitext(file)[1].lower() in VIDEO_EXTENSIONS:
                video_files.append(file_path)

    if not video_files:
        print(f"在 {folder_path} 中未找到视频文件")
        return

    print(f"共找到 {len(video_files)} 个视频文件")
    for file_path in tqdm(video_files):
        size = get_file_size(file_path)
        if size > SIZE_LIMIT:
            compress_video(file_path)
        else:
            print(f"跳过文件: {file_path}")

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

def bc_train_data_wait():
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
    batch_tar_and_remove(local_folder, 40)

def only_transfer():
    """
    1. 下载 alist 文件到本地
    """
    # 下载压缩文件
    alist_folder = r'/only_transfer/'
    local_folder = r'transfer'
    alist_client = alist(os.environ['ALIST_USER'], os.environ['ALIST_PWD'], host='http://168.138.158.156:5244')
    files = alist_client.listdir(alist_folder)

    for file in files:
        print(f'开始下载 {file["name"]}')
        alist_client.download(os.path.join(alist_folder, file['name']), local_folder)
        print(f'下载完成 {file["name"]}')

def bt_transfer():
    """
    1. 下载 bt下载主机 alist 文件到本地
    """
    # 下载压缩文件
    alist_folder = r'/completed/'
    local_folder = r'/completed'
    output_folder = r'completed'
    os.makedirs(output_folder, exist_ok=True)
    alist_client = alist(os.environ['ALIST_USER'], os.environ['ALIST_PWD'], host='http://168.138.158.156:5244')

    files = alist_client.listdir(alist_folder)
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
    for file in files:
        # 判断文件是否为视频文件
        if not any(file['name'].lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
            continue

        print(f'开始下载 {file["name"]}')
        alist_client.download(os.path.join(alist_folder, file['name']), local_folder)
        print(f'下载完成 {file["name"]}')

        # 若是视频文件，执行一边压缩脚本
        process_folder(local_folder, 2500, codec)

        # 移动到 output_folder
        # 将视频后缀的文件再增加后缀 'file'
        # 避免kaggle识别到视频播放
        for file in os.listdir(local_folder):
            shutil.move(os.path.join(local_folder, file), os.path.join(output_folder, file))
            with tarfile.open(os.path.join(output_folder, file + '.tar'), "w") as tar:
                tar.add(os.path.join(output_folder, file), arcname=file)
            # 删除原文件
            os.remove(os.path.join(output_folder, file))

if __name__ == '__main__':
    # h264/h265/av1
    codec = 'h264'
    for arg in sys.argv[1:]:
        if arg == 'bc_train_data_wait':
            bc_train_data_wait()
        elif arg == 'only_transfer':
            only_transfer()
        elif arg == 'bt_transfer':
            bt_transfer()
        elif arg.startswith('codec='):
            codec = arg.split('=')[1]
            print(f'使用编码器: {codec}')