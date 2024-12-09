import numpy as np
import os
import pickle
from collections import defaultdict, deque

from py_ext.tool import log, get_log_file

class DQNTracker:
    def __init__(self, title, n_days, num_actions, rank=0):
        """
        DQN学习过程跟踪器,用于记录和统计DQN训练过程中的各项指标

        Args:
            title: 标题,用于标识跟踪器
            n_days: 统计周期（天数）,用于设定保留多少天的历史数据进行统计分析
                   比如设置为7,则只保留最近7天的数据用于计算移动平均等指标
            num_actions: 动作空间大小,即智能体可以采取的动作数量
                        比如对于买卖股票:
                        - 0表示买入
                        - 1表示卖出
                        - 2表示持有
                        则num_actions=3
        
            rank: 排名,用于标识跟踪器

        统计指标包含:
            1. 奖励相关:
               - 每天的奖励总和
               - 每天的平均奖励
               - N天的移动平均奖励
            2. 动作分布:
               - 每个动作的选择次数
               - 每个动作的选择频率
            3. 损失函数:
               - TD误差均值
               - Q网络损失值
            4. 性能指标:
               - 非法动作率
               - 胜率
               - 败率
        """
        self.title = title 
        self.rank = rank
        self.n_days = n_days
        self.num_actions = num_actions
        
        # 奖励相关
        self.daily_rewards = []  # 每天的奖励列表
        self.episode_rewards = []  # 当天的episode奖励
        
        # 动作分布相关
        self.action_counts = defaultdict(int)  # 当天的动作计数
        self.daily_action_counts = []  # 每天的动作分布
        
        # 损失相关
        self.td_errors = []  # 当天的TD误差
        self.losses = []     # 当天的损失值
        self.daily_td_errors = []    # 每天的TD误差和
        self.daily_losses = []       # 每天的损失值和
        
        # 比率相关
        self.illegal_counts = 0      # 当天的非法动作次数
        self.total_counts = 0        # 当天的总动作次数
        self.win_counts = 0          # 当天的胜利次数
        self.loss_counts = 0         # 当天的失败次数
        self.daily_ratios = []       # 每天的各种比率

        # 额外的评价指标(从 lob_env step返回的info中获取)
        # sortino_ratio
        # sharpe_ratio
        # max_drawdown
        # total_return
        # sortino_ratio_bm
        # sharpe_ratio_bm
        # max_drawdown_bm
        # total_return_bm
        self.extra_metrics = {}
        self.daily_extra_metrics = {}

    def set_rank(self, rank):
        self.rank = rank

    def update_extra_metrics(self, k, v):
        """更新额外的评价指标"""
        if k not in self.extra_metrics:
            self.extra_metrics[k] = []
        self.extra_metrics[k].append(v)

    def update_reward(self, episode_reward):
        """更新奖励"""
        self.episode_rewards.append(episode_reward)
    
    def update_action(self, action):
        """更新动作统计"""
        self.action_counts[action] += 1
        self.total_counts += 1
    
    def update_td_error(self, td_error):
        """更新TD误差"""
        self.td_errors.append(td_error)
    
    def update_loss_value(self, loss):
        """更新损失值"""
        self.losses.append(loss)
    
    def update_illegal(self):
        """更新非法动作计数"""
        self.illegal_counts += 1
    
    def update_win(self):
        """更新胜利计数"""
        self.win_counts += 1
    
    def update_loss_count(self):
        """更新失败计数"""
        self.loss_counts += 1
    
    def day_end(self):
        """一天结束时的统计"""
        # 奖励统计
        daily_total_reward = sum(self.episode_rewards)
        self.daily_rewards.append(daily_total_reward)
        
        # 动作分布统计
        self.daily_action_counts.append(dict(self.action_counts))
        
        # 损失相关统计
        self.daily_td_errors.append(sum(self.td_errors))
        self.daily_losses.append(sum(self.losses))
        
        # 比率统计
        self.daily_ratios.append({
            'illegal_ratio': self.illegal_counts / max(1, self.total_counts),
            'win_ratio': self.win_counts / max(1, self.win_counts + self.loss_counts),
            'loss_ratio': self.loss_counts / max(1, self.win_counts + self.loss_counts)
        })

        # 更新额外的评价指标
        for k, v in self.extra_metrics.items():
            if k not in self.daily_extra_metrics:
                self.daily_extra_metrics[k] = []
            self.daily_extra_metrics[k].append(np.mean(v))

        # 重置当天统计
        self._reset_daily_stats()
        
        # 保持N天的数据
        self._maintain_n_days_data()
    
    def save(self):
        """保存所有统计数据到文件"""
        data = {
            'daily_rewards': self.daily_rewards,
            'daily_action_counts': self.daily_action_counts,
            'daily_td_errors': self.daily_td_errors,
            'daily_losses': self.daily_losses,
            'daily_ratios': self.daily_ratios,
            'daily_extra_metrics': self.daily_extra_metrics,
            'episode_rewards': self.episode_rewards,
            'action_counts': self.action_counts,
            'td_errors': self.td_errors,
            'losses': self.losses,
            'illegal_counts': self.illegal_counts,
            'win_counts': self.win_counts,
            'loss_counts': self.loss_counts,
            'total_counts': self.total_counts,
            'extra_metrics': self.extra_metrics
        }
        log(f'save to tracker_{self.title}_{self.rank}.pkl')
        # 保存到文件
        with open(f'tracker_{self.title}_{self.rank}.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        """加载统计数据"""
        if os.path.exists(f'tracker_{self.title}_{self.rank}.pkl'):
            log(f'load from tracker_{self.title}_{self.rank}.pkl')  
            with open(f'tracker_{self.title}_{self.rank}.pkl', 'rb') as f:
                data = pickle.load(f)
            self.daily_rewards = data['daily_rewards']
            self.daily_action_counts = data['daily_action_counts']
            self.daily_td_errors = data['daily_td_errors']
            self.daily_losses = data['daily_losses']
            self.daily_ratios = data['daily_ratios']
            self.daily_extra_metrics = data['daily_extra_metrics']
            self.episode_rewards = data['episode_rewards']
            self.action_counts = data['action_counts']
            self.td_errors = data['td_errors']
            self.losses = data['losses']
            self.illegal_counts = data['illegal_counts']
            self.win_counts = data['win_counts']
            self.loss_counts = data['loss_counts']
            self.total_counts = data['total_counts']
            self.extra_metrics = data['extra_metrics']
        else:
            log(f'tracker_{self.title}_{self.rank}.pkl not found')

    def get_metrics(self):
        """
        获取统计指标
        
        强化学习评价指标
        - total_reward: 总奖励
        - average_reward: 平均奖励
        - moving_average_reward: 移动平均奖励
        - total_td_error: 总TD误差
        - total_loss: 总损失值
        - average_illegal_ratio: 平均非法动作率
        - average_win_ratio: 平均胜率
        - average_loss_ratio: 平均败率
        - action_{k}_ratio k: 0-2
        
        交易评价指标
        - sortino_ratio
        - sharpe_ratio
        - max_drawdown
        - total_return
        - sortino_ratio_bm
        - sharpe_ratio_bm
        - max_drawdown_bm
        - total_return_bm
        """
        # if not self.daily_rewards or len(self.daily_rewards) < self.n_days:
        #     return {}
        
        self.save()
        metrics = {
            # 奖励相关
            'total_reward': sum(self.daily_rewards),
            'average_reward': np.mean(self.daily_rewards),
            'moving_average_reward': self._calculate_moving_average(self.daily_rewards),
            
            # 损失相关
            'total_td_error': sum(self.daily_td_errors),
            'total_loss': sum(self.daily_losses),
            
            # 比率相关
            'average_illegal_ratio': np.mean([r['illegal_ratio'] for r in self.daily_ratios]),
            'average_win_ratio': np.mean([r['win_ratio'] for r in self.daily_ratios]),
            'average_loss_ratio': np.mean([r['loss_ratio'] for r in self.daily_ratios])
        }

        # 动作分布
        # action_{k}_ratio k: 0-2
        action_distribution = self._get_action_distribution()
        for k, v in action_distribution.items():
            metrics[f'action_{k}_ratio'] = v
        
        # 额外的评价指标
        # sortino_ratio
        # sharpe_ratio
        # max_drawdown
        # total_return
        # sortino_ratio_bm
        # sharpe_ratio_bm
        # max_drawdown_bm
        # total_return_bm
        for k, v in self.daily_extra_metrics.items():
            metrics[k] = np.mean(v)

        return metrics
    
    def _reset_daily_stats(self):
        """重置当天的统计数据"""
        self.episode_rewards = []
        self.action_counts.clear()
        self.td_errors = []
        self.losses = []
        self.illegal_counts = 0
        self.total_counts = 0
        self.win_counts = 0
        self.loss_counts = 0
        self.extra_metrics.clear()
    
    def _maintain_n_days_data(self):
        """维护N天的数据"""
        self.daily_rewards = self.daily_rewards[-self.n_days:]
        self.daily_action_counts = self.daily_action_counts[-self.n_days:]
        self.daily_td_errors = self.daily_td_errors[-self.n_days:]
        self.daily_losses = self.daily_losses[-self.n_days:]
        self.daily_ratios = self.daily_ratios[-self.n_days:]

        for k in list(self.daily_extra_metrics.keys()):
            self.daily_extra_metrics[k] = self.daily_extra_metrics[k][-self.n_days:]

    def _calculate_moving_average(self, data, window=3):
        """计算移动平均"""
        return np.convolve(data, np.ones(window)/window, mode='valid')[-1]
    
    def _get_action_distribution(self):
        """获取动作分布"""
        total_actions = defaultdict(int)
        for daily_counts in self.daily_action_counts[-self.n_days:]:
            for action, count in daily_counts.items():
                total_actions[action] += count
        
        total_count = sum(total_actions.values())
        if not total_count:
            return {}
            
        # 按照键的顺序排序
        return {k: total_actions[k]/total_count 
               for k in sorted(total_actions.keys())}


if __name__ == '__main__':
    # 测试代码
    tracker = DQNTracker(n_days=5, num_actions=3)
    
    # 测试添加每日数据
    for i in range(10):
        # 模拟一天的数据
        tracker.update_reward(0.5)  # 更合理的奖励值
        tracker.update_action(np.random.randint(0,3))  # 随机动作 0-2
        tracker.update_td_error(0.1)  # 更小的TD误差
        tracker.update_loss_value(0.05)  # 更小的损失值
        
        if np.random.random() < 0.1:  # 10%概率非法动作
            tracker.update_illegal()
            
        if np.random.random() < 0.5:  # 50%胜率
            tracker.update_win()
        else:
            tracker.update_loss_count()
        
        # 结束当天
        tracker.day_end()
        
    # 获取统计指标
    metrics = tracker.get_metrics()
    print("统计指标:")
    # for k, v in metrics.items():
    #     print(f"{k}: {v}")
    print(metrics)

    # 测试动作分布
    action_dist = tracker._get_action_distribution() 
    print("\n动作分布:")
    print(action_dist)
