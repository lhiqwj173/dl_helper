import os
import torch
import torch.multiprocessing as mp

from py_ext.tool import log    

from dl_helper.rl.rl_env.lob_env import data_producer, LOB_trade_env
from dl_helper.rl.rl_env.breakout_env import BreakoutEnv
from dl_helper.rl.rl_env.cartpole_env import CartPoleEnv
from dl_helper.train_param import match_num_processes, get_gpu_info

def run_val_test(val_test, rank, agent, env):
    """根据val_test类型和rank决定是否执行及执行类型
    
    Args:
        val_test: 验证/测试类型 'val'/'test'/'all'
        rank: 进程rank
        agent: 智能体
        env: 环境
    """
    should_run = False
    test_type = None
    
    if val_test in ['val', 'test']:
        # val或test模式只在rank0执行
        if rank == 0:
            should_run = True
            test_type = val_test
    elif val_test == 'all':
        # all模式下rank0执行val,rank1执行test
        if rank == 0:
            should_run = True
            test_type = 'val'
        elif rank == 1:
            should_run = True
            test_type = 'test'
            
    if should_run:
        i = 0
        while True:
            log(f'{rank} {i} test {test_type} dataset...')
            i += 1

            # 同步最新参数
            # 拉取服务器的最新参数并更新
            agent.update_params_from_server(env)

            log(f'{rank} {i} wait metrics for {test_type}')
            t = time.time()
            metrics = agent.val_test(env, data_type=test_type)
            log(f'{rank} {i} metrics: {metrics}, cost: {time.time() - t:.2f}s')
            # 发送验证结果给服务器
            send_val_test_data(agent.train_title, test_type, metrics)

def _run_client_learning_device(init_env_func, rank, num_processes, *args, **kwargs):
    # 根据环境获取对应设备
    _run_device = get_gpu_info()
    if _run_device == 'TPU':  # 如果是TPU环境
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    elif _run_device in ['T4x2', 'P100']:
        device = torch.device(f'cuda:{rank}' if num_processes > 1 else 'cuda')
    else:
        device = torch.device('cpu')
    log(f'rank: {rank}, num_processes: {num_processes} device: {device}, run...')
    
    # 移动到设备
    agent,num_episodes  = args[:2]
    args = args[2:]
    agent.to(device)
    agent.tracker.set_rank(rank)
    
    # 初始化环境
    env = init_env_func()

    # 开始训练
    if kwargs.get('val_test', False):
        # 验证/测试
        run_val_test(kwargs['val_test'], rank, agent, env)
    else:
        log(f'{rank} learn...')
        agent.learn(env, 5 if kwargs.get('enable_profiling', False) else num_episodes, *args)

def run_client_learning_device_lob(rank, num_processes, *args, **kwargs):
    # 初始化环境 
    def init_env():
        dp = data_producer(simple_test=kwargs.get('simple_test', False), file_num=15 if kwargs.get('enable_profiling', False) else 0)
        return LOB_trade_env(data_producer=dp)
    return _run_client_learning_device(init_env, rank, num_processes, *args, **kwargs)

def run_client_learning_device_breakout(rank, num_processes, *args, **kwargs):
    def init_env():
        return BreakoutEnv()
    return _run_client_learning_device(init_env, rank, num_processes, *args, **kwargs)

def run_client_learning_device_carpole(rank, num_processes, *args, **kwargs):
    def init_env():
        return CartPoleEnv()
    return _run_client_learning_device(init_env, rank, num_processes, *args, **kwargs)

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