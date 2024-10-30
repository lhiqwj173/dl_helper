import psutil, pickle, torch, os
from py_ext.wechat import wx
from py_ext.tool import debug, log
from dl_helper.train_param import logger, match_num_processes
if match_num_processes() ==8:
    import torch_xla.core.xla_model as xm

from torchstat import stat
from torchinfo import summary
from torch.nn.utils import parameters_to_vector
import df2img

def save_df_pic(title, df, fig_size=(500, 140)):
    fig = df2img.plot_dataframe(df, row_fill_color=("#ffffff", "#f2f2f2"), fig_size=fig_size)
    df2img.save_dataframe(fig=fig, filename=title)

def model_params_num(model):
    # 将所有参数转换为一个向量
    vector = parameters_to_vector(model.parameters())
    # 获取参数数量
    num_params = vector.numel()
    return num_params

def _check_nan(output):
    has_nan = torch.isnan(output).any(dim=-1)  # 检查是否有 NaN
    has_inf = torch.isinf(output).any(dim=-1)  # 检查是否有 inf

    # 找出包含 NaN 或 inf 值的批次索引
    batch_indices = torch.nonzero(has_nan | has_inf, as_tuple=True)[0]
    return batch_indices

def check_nan(output, ids):
    # 找出包含 NaN 或 inf 值的批次索引
    batch_indices = _check_nan(output)
    if batch_indices.numel() > 0:
        bad_ids = []
        for i in list(batch_indices):
            bad_ids.append(ids[i])
        msg = f'训练数据异常 nan/inf ids:{bad_ids}'
        wx.send_message(msg)
        raise Exception(msg)

def stop_all_python_processes():
    current_pid = os.getpid()

    # 获取当前正在运行的所有进程
    all_processes = psutil.process_iter()

    # 遍历所有进程并停止 Python 进程
    for process in all_processes:
        try:
            process_info = process.as_dict(attrs=['pid', 'name'])
            pid = process_info['pid']
            name = process_info['name']

            # 如果进程是 Python 进程且不是当前进程，则终止该进程
            if name.lower() == 'python' and pid != current_pid:
                process.terminate()
                print(f"Terminated Python process: {pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 忽略无法访问或不存在的进程
            pass

def report_memory_usage(msg=''):
    memory_usage = psutil.virtual_memory()
    print(f"{msg} CPU 内存占用：{memory_usage.percent}% ({memory_usage.used/1024**3:.3f}GB/{memory_usage.total/1024**3:.3f}GB)")
    # tpu_mem_info = xm.get_memory_info(xm.xla_device())
    # print(tpu_mem_info)
    # tpu_used = tpu_mem_info["kb_total"] - tpu_mem_info["kb_free"]
    # print(f"{msg} TPU 内存占用：{tpu_used/1024**3:.3f}GB/{tpu_mem_info['kb_total']/1024**3:.3f}GB")

    # # 获取当前进程ID
    # pid = psutil.Process().pid
    # # 获取当前进程对象
    # process = psutil.Process(pid)
    # # 获取当前进程占用的内存信息
    # memory_info = process.memory_info()
    # # 将字节大小转换为GB
    # memory_gb = memory_info.rss / (1024 ** 3)
    # # 打印内存大小
    # logger.debug(f"{msg} 占用的内存：{memory_gb:.3f}GB")


    # # 获取当前进程ID
    # current_pid = os.getpid()

    # # 获取当前进程对象
    # current_process = psutil.Process(current_pid)

    # # 获取当前进程占用的内存信息
    # memory_info = current_process.memory_info()

    # # 将字节大小转换为GB
    # memory_gb = memory_info.rss / (1024 ** 3)

    # # 打印当前进程的内存大小
    # # print(f"当前进程ID: {current_pid}, 当前进程占用的内存：{memory_gb:.3f}GB")

    # # 统计所有子进程的内存使用情况
    # total_memory_gb = memory_gb

    # # 获取所有进程ID
    # pids = psutil.pids()

    # for pid in pids:
    #     try:
    #         # 获取进程对象
    #         process = psutil.Process(pid)

    #         # 获取进程的父进程ID
    #         parent_pid = process.ppid()

    #         # 如果父进程ID与当前进程ID相同，则属于当前程序的子进程
    #         if parent_pid == current_pid:
    #             # 获取进程占用的内存信息
    #             memory_info = process.memory_info()

    #             # 将字节大小转换为GB
    #             memory_gb = memory_info.rss / (1024 ** 3)

    #             # 累加子进程的内存大小到总内存大小
    #             total_memory_gb += memory_gb
    #             # 打印子进程的内存大小
    #             # print(f"子进程ID: {pid}, 子进程占用的内存：{memory_gb:.3f}GB")
    #     except (psutil.NoSuchProcess, psutil.AccessDenied):
    #         # 跳过无法访问的进程
    #         continue
    
    # # # 打印合并统计后的内存大小
    # msg = '合并统计后的内存大小' if msg == '' else f'{msg}'
    # print(f"{msg}: {total_memory_gb:.3f}GB")