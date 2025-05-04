from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR

class OneCycleLR(_OneCycleLR):
    def load_state_dict(self, state_dict):
        """加载调度器状态，并适应新的total_steps。
        
        计算原始相对位置，然后根据新的total_steps重新计算last_epoch和schedule_phases。
        这样可以在total_steps变化时保持学习率调度的连续性。
        
        Args:
            state_dict (dict): 调度器状态。应该是从:meth:`state_dict`调用返回的对象。
        """
        # 保存当前的total_steps
        current_total_steps = self.total_steps
        
        # 从旧的state_dict中获取信息
        old_last_epoch = state_dict.get('last_epoch', 0)
        old_total_steps = state_dict.get('total_steps', 1)  # 防止除零
        
        # 计算相对位置（完成百分比）
        relative_position = float(old_last_epoch) / float(old_total_steps)
        
        # 加载状态字典中的其他数据
        self.__dict__.update(state_dict)
        
        # 恢复当前的total_steps（防止被旧值覆盖）
        self.total_steps = current_total_steps
        
        # 根据相对位置和新的total_steps计算新的last_epoch
        self.last_epoch = int(relative_position * self.total_steps)
        
        # 重建schedule_phases以适应新的total_steps
        if len(self._schedule_phases) == 3:  # 三阶段模式
            # 从旧的阶段计算pct_start
            pct_start = (float(self._schedule_phases[0]['end_step']) + 1) / old_total_steps
            
            # 使用相同的pct_start重建阶段
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": float(2 * pct_start * self.total_steps) - 2,
                    "start_lr": "max_lr",
                    "end_lr": "initial_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "max_momentum",
                },
            ]
        else:  # 两阶段模式（默认）
            # 从旧的阶段计算pct_start
            pct_start = (float(self._schedule_phases[0]['end_step']) + 1) / old_total_steps
            
            # 使用相同的pct_start重建阶段
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": "max_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
            ]

