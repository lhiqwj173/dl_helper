import threading
import multiprocessing
import sys, os, time
# os.environ["RAY_DEDUP_LOGS"] = "0"
import matplotlib.pyplot as plt
from ray.tune.registry import get_trainable_cls, register_env
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log

train_folder = 'cartpole'
init_logger('20250130_cartpole', home=train_folder, timestamp=False)

if __name__ == "__main__":
    is_server = False
    # 获取参数
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "server":
                is_server = True

    train_title = f'20250130_cartpole'

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes()

    # 训练配置
    config = (
        ClientPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("CartPole-v1")
        .env_runners(num_env_runners=1)# 4核cpu，暂时选择1个环境运行器
        .extra_config(
            learner_group_class=ClientLearnerGroup,
            learner_group_kwargs={
                'train_folder': train_folder,
                "train_title": train_title,
            },
        )
    )

    if is_server:
        # dump config
        add_train_title_item(train_title, config)

    else:
        # 客户端配置
        config = config.training(
            learner_class=ClientPPOTorchLearner,# 分布式客户端 学习者
        )

        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )

        # 询问服务器，本机是否需要验证环节
        need_val = request_need_val(train_title)
        log(f"need_val: {need_val}")
        if need_val:
            config = config.evaluation(
                evaluation_interval=10,
                evaluation_duration=3,
            )

        # 客户端运行
        # 构建算法
        algo = config.build()

        out_file = os.path.join(train_folder, 'out.csv')

        begin_time = time.time()
        # 训练循环
        # 标准训练 30 527.9s 
        rounds = 30
        # rounds = 5
        for i in range(rounds):
            log(f"\nTraining iteration {i+1}/{rounds}")
            result = algo.train()
            simplify_rllib_metrics(result, out_func=log, out_file=out_file)
        log(f'all rounds done')

        # 停止学习者额外的事件进程
        algo.learner_group.stop_extra_process()
        log(f"learner_group.stop_extra_process done")

        # 停止算法
        algo.stop()
        log(f"algo.stop done")

        if need_val:
            # 绘制训练曲线
            plot_training_curve(train_folder, out_file, time.time() - begin_time, y_axis_max=500)
            log(f"plot_training_curve done")

        log(f'{train_title} client all done')

        # 强制结束所有非守护线程
        for thread in threading.enumerate():
            if thread != threading.current_thread() and not thread.daemon:
                try:
                    log(f"尝试关闭线程 {thread.name}")
                    thread.join(timeout=1.0)
                except Exception as e:
                    log(f"无法关闭线程 {thread.name}: {e}")
        
        # 确保所有子进程被终止
        for process in multiprocessing.active_children():
            try:
                log(f"尝试终止进程 {process.name}")
                process.terminate()
                process.join(timeout=1.0)
            except Exception as e:
                log(f"无法终止进程 {process.name}: {e}")

        log("所有资源清理完成")