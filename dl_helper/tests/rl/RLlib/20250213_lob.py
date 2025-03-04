import sys, os, time, pickle
import pandas as pd
# os.environ["RAY_DEDUP_LOGS"] = "0"
import matplotlib.pyplot as plt
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from dl_helper.rl.costum_rllib_module.ppoconfig import ClientPPOConfig
from dl_helper.rl.costum_rllib_module.client_learner import ClientPPOTorchLearner
from dl_helper.rl.costum_rllib_module.client_learner import ClientLearnerGroup
from dl_helper.rl.costum_rllib_module.lob_model import lob_PPOCatalog, LobCallbacks
from dl_helper.rl.easy_helper import *
from dl_helper.train_param import match_num_processes
from dl_helper.rl.rl_utils import add_train_title_item, plot_training_curve, simplify_rllib_metrics, BaseCustomPlotter
from dl_helper.rl.socket_base import request_need_val
from py_ext.tool import init_logger, log
from py_ext.datetime import beijing_time

from dl_helper.train_folder_manager import TrainFolderManager

train_folder = 'lob'
log_name = f'20250213_lob_{beijing_time().strftime("%Y%m%d")}'
init_logger(log_name, home=train_folder, timestamp=False)

class Plotter(BaseCustomPlotter):
    def get_additional_plot_count(self):
        """
        返回需要额外绘制的图表数量
        custom_metrics_illegal_ratio,
        custom_metrics_trade_num,
        custom_metrics_win_ratio,
        custom_metrics_profit_loss_ratio,
        custom_metrics_sharpe_ratio,
        custom_metrics_max_drawdown,
        custom_metrics_trade_return,
        custom_metrics_hold_length,
        custom_metrics_excess_return

        custom_metrics_val_illegal_ratio,
        custom_metrics_val_trade_num,
        custom_metrics_val_win_ratio,
        custom_metrics_val_profit_loss_ratio,
        custom_metrics_val_sharpe_ratio,
        custom_metrics_val_max_drawdown,
        custom_metrics_val_trade_return,
        custom_metrics_val_hold_length,
        custom_metrics_val_excess_return
        """
        return 8
    def plot(self, out_data, axes_list):
        """
        子类必须实现
        绘制额外图表
        """
        datetime = pd.to_datetime(out_data['datetime'])

        # 1. illegal_ratio 和 win_ratio
        ax = axes_list[0]
        illegal_ratio_max = max(out_data['custom_metrics_illegal_ratio'])
        win_ratio_max = max(out_data['custom_metrics_win_ratio'])
        val_illegal_ratio_max = max(out_data['custom_metrics_val_illegal_ratio'])
        val_win_ratio_max = max(out_data['custom_metrics_val_win_ratio'])
        ax.plot(datetime, out_data['custom_metrics_illegal_ratio'], 'r-', label=f'illegal_ratio({illegal_ratio_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_win_ratio'], 'g-', label=f'win_ratio({win_ratio_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_illegal_ratio'], 'r-', label=f'val_illegal_ratio({val_illegal_ratio_max:.2f})')
        ax.plot(datetime, out_data['custom_metrics_val_win_ratio'], 'g-', label=f'val_win_ratio({val_win_ratio_max:.2f})')
        ax.set_title('Illegal Ratio & Win Ratio')
        ax.legend()

        # 2. trade_num
        ax = axes_list[1]
        trade_num_max = max(out_data['custom_metrics_trade_num'])
        val_trade_num_max = max(out_data['custom_metrics_val_trade_num'])
        ax.plot(datetime, out_data['custom_metrics_trade_num'], 'b-', label=f'trade_num({trade_num_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_trade_num'], 'b-', label=f'val_trade_num({val_trade_num_max:.2f})')
        ax.set_title('Trade Number')
        ax.legend()

        # 3. profit_loss_ratio
        ax = axes_list[2]
        profit_loss_ratio_max = max(out_data['custom_metrics_profit_loss_ratio'])
        val_profit_loss_ratio_max = max(out_data['custom_metrics_val_profit_loss_ratio'])
        ax.plot(datetime, out_data['custom_metrics_profit_loss_ratio'], 'g-', label=f'profit_loss_ratio({profit_loss_ratio_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_profit_loss_ratio'], 'g-', label=f'val_profit_loss_ratio({val_profit_loss_ratio_max:.2f})')
        ax.set_title('Profit Loss Ratio')
        ax.legend()

        # 4. sharpe_ratio
        ax = axes_list[3]
        sharpe_ratio_max = max(out_data['custom_metrics_sharpe_ratio'])
        val_sharpe_ratio_max = max(out_data['custom_metrics_val_sharpe_ratio'])
        ax.plot(datetime, out_data['custom_metrics_sharpe_ratio'], 'r-', label=f'sharpe_ratio({sharpe_ratio_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_sharpe_ratio'], 'r-', label=f'val_sharpe_ratio({val_sharpe_ratio_max:.2f})')
        ax.set_title('Sharpe Ratio')
        ax.legend()

        # 5. max_drawdown
        ax = axes_list[4]
        max_drawdown_max = max(out_data['custom_metrics_max_drawdown'])
        val_max_drawdown_max = max(out_data['custom_metrics_val_max_drawdown'])
        ax.plot(datetime, out_data['custom_metrics_max_drawdown'], 'r-', label=f'max_drawdown({max_drawdown_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_max_drawdown'], 'r-', label=f'val_max_drawdown({val_max_drawdown_max:.2f})')
        ax.set_title('Max Drawdown')
        ax.legend()

        # 6. trade_return
        ax = axes_list[5]
        trade_return_max = max(out_data['custom_metrics_trade_return'])
        val_trade_return_max = max(out_data['custom_metrics_val_trade_return'])
        ax.plot(datetime, out_data['custom_metrics_trade_return'], 'g-', label=f'trade_return({trade_return_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_trade_return'], 'g-', label=f'val_trade_return({val_trade_return_max:.2f})')
        ax.set_title('Trade Return')
        ax.legend()

        # 7. hold_length
        ax = axes_list[6]
        hold_length_max = max(out_data['custom_metrics_hold_length'])
        val_hold_length_max = max(out_data['custom_metrics_val_hold_length'])
        ax.plot(datetime, out_data['custom_metrics_hold_length'], 'b-', label=f'hold_length({hold_length_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_hold_length'], 'b-', label=f'val_hold_length({val_hold_length_max:.2f})')
        ax.set_title('Hold Length')
        ax.legend()

        # 8. excess_return
        ax = axes_list[7]
        excess_return_max = max(out_data['custom_metrics_excess_return'])
        val_excess_return_max = max(out_data['custom_metrics_val_excess_return'])
        ax.plot(datetime, out_data['custom_metrics_excess_return'], 'g-', label=f'excess_return({excess_return_max:.2f})', alpha=0.4)
        ax.plot(datetime, out_data['custom_metrics_val_excess_return'], 'g-', label=f'val_excess_return({val_excess_return_max:.2f})')
        ax.set_title('Excess Return')
        ax.legend()

if __name__ == "__main__":

    # plot_training_curve(r"C:\Users\lh\Desktop\temp", out_file=r"C:\Users\lh\Downloads\out_20250304.csv", custom_plotter=Plotter())
    # import sys
    # sys.exit()

    run_type = 'self'

    # 获取参数
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "server":
                run_type = 'server'
            elif arg == "client":
                run_type = 'client'

    train_title = f'20250213_lob'

    # 根据设备gpu数量选择 num_learners
    num_learners = match_num_processes()

    # 训练配置
    config = (
        ClientPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            "lob",
            # 环境配置
            env_config={
                'data_type': 'train',# 训练/测试
                'his_len': 10,# 每个样本的 历史数据长度
                'file_num': 10,# 数据生产器 限制最多使用的文件（日期）数量
                'need_cols': [item for i in range(5) for item in [f'BASE卖{i+1}价', f'BASE卖{i+1}量', f'BASE买{i+1}价', f'BASE买{i+1}量']],
                'use_symbols': ['513050'],

                'train_folder' :train_folder,
                'log_name': log_name,
            }
        )# 直接使用
        .env_runners(num_env_runners=0)# 4核cpu，暂时选择1个环境运行器
        # 自定义模型
        .rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=lob_PPOCatalog),# 使用自定义配置
            model_config={
                # 自定义编码器参数
                'input_dims' : (20, 10),
                'extra_input_dims' : 4,
                'ds' : (20, 40, 40, 3),
                'ts' : (10, 6, 3, 1),
            },
        )
        .callbacks(LobCallbacks)
    )

    if run_type == 'server':
        config = config.extra_config(
            learner_group_class=ClientLearnerGroup,
            learner_group_kwargs={
                'train_folder': train_folder,
                "train_title": train_title,
            },
        )

        # dump config
        add_train_title_item(train_title, config)

    elif run_type == 'client':
        # 客户端配置
        config = config.training(
            learner_class=ClientPPOTorchLearner,# 分布式客户端 学习者
        )

        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )

        config = config.extra_config(
            learner_group_class=ClientLearnerGroup,
            learner_group_kwargs={
                'train_folder': train_folder,
                "train_title": train_title,
            },
        )

        # 询问服务器，本机是否需要验证环节
        need_val = request_need_val(train_title)
        log(f"need_val: {need_val}")
        if need_val:
            config = config.evaluation(
                evaluation_interval=15,
                evaluation_duration=3,
            )

        # 客户端运行
        # 构建算法
        algo = config.build()

        begin_time = time.time()
        # 训练循环
        # rounds = 2000
        rounds = 100
        # rounds = 3
        for i in range(rounds):
            log(f"\nTraining iteration {i+1}/{rounds}")
            result = algo.train()
            simplify_rllib_metrics(result, out_func=log)
        
        # 停止学习者额外的事件进程
        algo.learner_group.stop_extra_process()
        log(f"learner_group.stop_extra_process done")

        if need_val:
            # 绘制训练曲线
            plot_training_curve(train_folder, time.time() - begin_time, y_axis_max=30)

        # 停止算法
        algo.stop()
        log(f"algo.stop done")

    else:
        # 单机运行
        config = config.learners(    
            num_learners=num_learners,
            num_gpus_per_learner=1,
        )
        config = config.evaluation(
            evaluation_interval=15,
            evaluation_duration=3,
        )

        # 构建算法
        algo = config.build()
        # print(algo.learner_group._learner.module._rl_modules['default_policy'])

        # 训练文件夹管理
        train_folder_manager = TrainFolderManager(train_folder)
        if train_folder_manager.exists():
            log(f"restore from {train_folder_manager.checkpoint_folder}")
            algo.restore_from_path(train_folder_manager.checkpoint_folder)

        begin_time = time.time()
        # rounds = 2000
        rounds = 30
        for i in range(rounds):
            log(f"Training iteration {i+1}/{rounds}")
            result = algo.train()
            # 保存result
            result_file = os.path.join(train_folder, f'result_{i}.pkl')
            with open(result_file, 'wb') as f:
                pickle.dump(result, f)

            out_file = os.path.join(train_folder, f'out_{beijing_time().strftime("%Y%m%d")}.csv')
            simplify_rllib_metrics(result, out_func=log, out_file=out_file)

            if i>0 and (i % 10 == 0 or i == rounds - 1):
                # 保存检查点
                checkpoint_dir = algo.save_to_path(train_folder_manager.checkpoint_folder)
                log(f"Checkpoint saved in directory {checkpoint_dir}")
                # 绘制训练曲线
                plot_training_curve(train_folder, out_file, time.time() - begin_time, custom_plotter=Plotter())
                # 压缩并上传
                train_folder_manager.push()

        # 停止算法
        algo.stop()
        log(f"algo.stop done")
