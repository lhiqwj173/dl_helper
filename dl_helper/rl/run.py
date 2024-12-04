import os
import torch
import torch.multiprocessing as mp
from dl_helper.train_param import match_num_processes, get_gpu_info

def train_func_device(rank, num_processes, a, b, c):
    # 根据环境获取对应设备
    _run_device = get_gpu_info()
    if _run_device == 'TPU':  # 如果是TPU环境
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    elif _run_device in ['T4x2', 'P100']:
        device = torch.device(f'cuda:{rank}' if num_processes > 1 else 'cuda')
    else:
        device = torch.device('cpu')

    print(rank, num_processes)
    print(a, b, c)

def train_func_device_tpu(rank, *args):
    """
    @param rank: 当前进程的序号 (0 到 num_processes-1)
    @param args: 第一个参数是num_processes，之后是用户的位置参数
    @param kwargs: 关键字参数（从最后一个位置参数解包得到）
    """
    train_func_device = args[0]  # 第一个参数是 train_func_device
    user_args = args[1:-1]   # 中间的是用户的位置参数
    user_kwargs = args[-1]   # 最后一个参数是kwargs字典
    
    train_func_device(rank, *user_args, **user_kwargs)

def run_client_learning(train_func_device, args, kwargs={}):
    num_processes = match_num_processes()

    if num_processes == 1:
        # 单设备
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # 使用设备运行
        train_func_device(0, 1, *args, **kwargs)
    else:
        # 多设备
        if get_gpu_info() == "TPU":
            pass

            # # TPU多设备训练
            # import torch_xla.distributed.xla_multiprocessing as xmp
            
            # # 清理TPU环境变量,避免冲突
            # try:
            #     os.environ.pop('TPU_PROCESS_ADDRESSES')
            #     os.environ.pop('CLOUD_TPU_TASK_ID')
            #     # 添加必要的环境变量
            #     os.environ['TPU_NUM_DEVICES'] = str(num_processes)
            #     os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;localhost:51011'  # TPU worker配置
            # except:
            #     pass
                
            # # 将kwargs打包到args中传递
            # combined_args = (train_func_device, num_processes, *args, kwargs)  # 把kwargs作为最后一个位置参数
            
            # # 使用XLA的多进程启动器
            # xmp.spawn(
            #     train_func_device_tpu,
            #     args=combined_args,
            #     nprocs=num_processes,
            #     start_method='fork'  # 显式指定启动方法
            # )

        else:
            print('GPU/CPU多进程训练')
            mp.set_start_method('spawn', force=True)
            processes = []
            
            for rank in range(num_processes):
                p = mp.Process(
                    target=train_func_device,
                    args=(rank, num_processes, *args),
                    kwargs=kwargs
                )
                p.start()
                processes.append(p)
                
            for p in processes:
                p.join()