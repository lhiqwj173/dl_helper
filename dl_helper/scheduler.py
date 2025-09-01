from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import torch
from py_ext.tool import debug, log
from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from dl_helper.train_param import tpu_available
if tpu_available():
    import torch_xla.core.xla_model as xm

class BaseScheduler:
    """
    学习率调度器基类，提供通用的接口。

    该类定义了学习率调度器的通用接口，所有自定义调度器都应继承该类。

    Methods:
        batch_step(): 在每个batch/step后调用，默认空实现。子类可覆写此方法。
    """
    def batch_step(self):
        """默认空实现，在每个batch后调用"""
        pass

def lr_lambda(x, min_lr, max_lr, total_iters):
    """
    计算指数学习率衰减函数。

    该函数实现了一个指数衰减的学习率调度，从 max_lr 开始逐渐衰减到 min_lr。
    随着迭代次数 x 的增加，学习率将从 max_lr 指数递减到 min_lr。

    Args:
        x (int): 当前迭代次数，从 1 开始计数。
        min_lr (float): 最终学习率的下限。
        max_lr (float): 初始学习率的上限。
        total_iters (int): 总迭代次数。

    Returns:
        float: 当前迭代的学习率。

    Examples:
        >>> lr_lambda(1, 1e-5, 1e-1, 100)
        0.001
        >>> lr_lambda(100, 1e-5, 1e-1, 100)
        1e-05
    """
    return min_lr * (max_lr / min_lr) ** (x / total_iters)

class LRFinder(BaseScheduler):
    """
    学习率查找器，用于自动寻找最优学习率。

    该类实现了学习率范围测试 (LR Range Test) 的功能，它在一个定义的学习率范围内
    逐渐增加学习率，同时记录每个学习率对应的损失值。用户可以通过分析学习率-损失
    曲线来选择合适的初始学习率用于训练。

    该方法基于 Leslie N. Smith 的论文《Cyclical Learning Rates for Training Neural Networks》
    中的思路，通过在训练的早期快速测试不同学习率，找到损失下降最快或振荡前的最优点。

    Attributes:
        optimizer (torch.optim.Optimizer): 需要被测试的优化器。
        min_lr (float): 学习率测试的下限，默认为 1e-7。
        max_lr (float): 学习率测试的上限，默认为 1。
        total_iters (int): 总的测试迭代次数，默认为 60。
        iteration (int): 当前迭代计数。
        history (dict): 记录学习率和损失的历史数据，包含 'lr' 和 'loss' 两个键。

    Args:
        optimizer (torch.optim.Optimizer): 优化器实例。
        total_iters (int, optional): 测试总迭代次数。默认 60。
        min_lr (float, optional): 最小学习率。默认 1e-7。
        max_lr (float, optional): 最大学习率。默认 1。
        *args: 额外的位置参数（保留接口兼容性）。
        **kwargs: 额外的关键字参数（保留接口兼容性）。

    Methods:
        step(loss_array): 执行一步学习率调整并记录损失。
        state_dict(): 返回状态字典，排除 optimizer。
        load_state_dict(state_dict): 从状态字典恢复状态。
        use_lr(lr): 手动设置验证的最优学习率。

    Example:
        >>> import torch
        >>> from torch import nn
        >>> model = nn.Linear(10, 1)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> finder = LRFinder(optimizer, total_iters=50)
        >>> # 在训练循环中调用 step
        >>> # finder.step([current_loss])
    """
    def __init__(self, optimizer, *args, total_iters: int=60, min_lr: float=1e-7, max_lr: float=1, **kwargs):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iters = total_iters
        # self.lr_lambda = lambda x: self.min_lr * (self.max_lr / self.min_lr) ** (x / self.total_iters)
        self.iteration = 0
        self.history = {'lr': [], 'loss': []}
        
        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = min_lr

    def step(self, loss_array, *args, **kwargs):
        loss = loss_array[-1]

        self.iteration += 1

        # lr = self.lr_lambda(self.iteration)
        lr = lr_lambda(self.iteration, self.min_lr, self.max_lr, self.total_iters)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.history['lr'].append(lr)
        self.history['loss'].append(loss)
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def use_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr

class ConstantLRScheduler(_LRScheduler, BaseScheduler):
    """
    恒定学习率调度器。

    该调度器在整个训练过程中保持学习率不变，不进行任何调整。
    这对于某些特定场景非常有用，例如：
    - 当学习率已经通过其他方式预设为最优值的场景。
    - 简单的训练任务，不需要复杂的学习率调度策略。
    - 与其他学习率调整机制（如基于梯度的调整）结合使用的场景。

    该类继承自 PyTorch 的 _LRScheduler，提供了完整的调度器接口支持。

    Args:
        optimizer (torch.optim.Optimizer): 被包装的优化器。其参数组中的学习率将被保持不变。
        last_epoch (int, optional): 最后一个 epoch 的索引，用于恢复训练状态。默认值: -1。

    Attributes:
        base_lrs (list): 优化器中每个参数组的初始学习率。

    Methods:
        get_lr(): 返回每个参数组当前的恒定学习率。
        _get_closed_form_lr(): 返回恒定学习率（与 get_lr() 相同）。

    Example:
        >>> import torch
        >>> from torch import nn
        >>> model = nn.Linear(10, 1)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> scheduler = ConstantLRScheduler(optimizer)
        >>> # 学习率将一直保持为 0.01
        >>> for epoch in range(10):
        ...     # training loop
        ...     scheduler.step()
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """返回每个参数组的恒定学习率。"""
        return [base_lr for base_lr in self.base_lrs]
    
    def _get_closed_form_lr(self):
        """返回恒定学习率(与get_lr相同)。"""
        return self.base_lrs

class ReduceLROnPlateau(_ReduceLROnPlateau, BaseScheduler):
    """
    基于验证损失的 ReduceLROnPlateau 学习率调度器扩展。

    该类继承自 PyTorch 的 ReduceLROnPlateau 调度器，并添加了自定义的功能。
    当验证损失不再下降时，会减少学习率以帮助模型收敛。

    主要特性：
    - 支持接收损失数组（数组中的最后一个值作为当前损失）
    - 提供手动设置学习率的功能 (use_lr)
    - 完全兼容父类 ReduceLROnPlateau 的所有参数和行为
    
    Args:
        optimizer (Optimizer): 被包装的优化器。
        mode (str, optional): 'min' 或 'max'。默认: 'min'。
        factor (float, optional): 学习率降低的倍数。默认: 0.5。
        patience (int, optional): 等待的 epoch 数。默认: 10。
        verbose (bool, optional): 是否打印信息。默认: False。
        min_lr (float or list, optional): 学习率的下限。默认: 0。
        
    Methods:
        step(loss_array): 执行一步调度，使用 loss_array[-1] 作为当前损失。
        use_lr(lr): 手动设置指定学习率【注：此方法为扩展功能】。
    """
    def step(self, loss_array, *args, **kwargs):
        # 从数组中提取最后一个损失值
        loss = loss_array[-1]
        super().step(loss)

    def use_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class WarmupReduceLROnPlateau(ReduceLROnPlateau, BaseScheduler):
    """
    一个结合了“逐-step预热”和“ReduceLROnPlateau”策略的学习率调度器。

    该调度器将业界标准的“逐-step预热”与经典的“学习率平坦期下降”策略相结合，
    旨在实现更稳定、更高效的模型训练。

    工作流程：
    1.  **预热阶段 (Warmup Phase)**：在训练开始的前 `warmup_steps` 个训练步（step/batch）中，
        学习率从 0 开始线性增加到初始学习率。此阶段通过在每个训练步后调用 `scheduler.step_step()` 来执行。
    2.  **平坦期下降阶段 (Plateau Phase)**：预热结束后，调度器切换到 `ReduceLROnPlateau` 模式。
        此时，需要在每个 epoch 结束后，使用验证集指标调用 `scheduler.step(metrics=...)`。
        如果指标在 `patience` 个 epoch 内没有改善，学习率将会降低。

    Args:
        optimizer (Optimizer): 被包装的优化器。
        warmup_steps (int): 预热阶段持续的总训练步数（steps/batches）。
        **kwargs: 传递给父类 `ReduceLROnPlateau` 的所有参数。

    正确的使用方式:
        >>> train_loader = ... # 你的训练数据加载器
        >>> warmup_epochs = 5
        >>> # 根据 epoch 数和 dataloader 长度计算总的 warmup steps
        >>> warmup_steps = warmup_epochs * len(train_loader)
        >>> scheduler = WarmupReduceLROnPlateau(optimizer, warmup_steps=warmup_steps, patience=10)
        >>>
        >>> for epoch in range(num_epochs):
        ...     # --- 训练循环 ---
        ...     for batch in train_loader:
        ...         optimizer.zero_grad()
        ...         loss = ...
        ...         loss.backward()
        ...         optimizer.step()
        ...         # 在每个训练 step 后调用，用于更新 warmup
        ...         scheduler.step_step()  # <--- 使用 step_step()
        ...
        ...     # --- 验证循环 ---
        ...     validation_loss = ... # 计算验证损失
        ...
        ...     # 在每个 epoch 结束后调用，并传入验证指标
        ...     # 这将在 warmup 结束后触发 ReduceLROnPlateau 逻辑
        ...     scheduler.step([validation_loss]) # <--- 使用 step()
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int, **kwargs):
        # 保存优化器中设置的初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # 调用父类构造函数
        super().__init__(optimizer, **kwargs)

        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # 在预热开始前，将学习率设置为 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0

    def batch_step(self):
        """
        在每个训练步（batch）后调用。
        
        此方法专门用于处理预热阶段的学习率线性增长。
        预热结束后，调用此方法将不再改变学习率。
        """
        # --- 预热阶段逻辑 ---
        if self.current_step < self.warmup_steps:
            # 计算学习率的缩放比例
            scale = (self.current_step + 1) / self.warmup_steps
            
            # 为每个参数组应用预热学习率
            for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                param_group['lr'] = base_lr * scale
        
        # --- 预热后阶段 ---
        # 确保在预热刚结束时，学习率被精确设置为基础学习率
        elif self.current_step == self.warmup_steps:
            for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                param_group['lr'] = base_lr
            
        self.current_step += 1

    def step(self, loss_array, *args, **kwargs):
        """
        在每个 epoch 结束并完成验证后调用。
        
        此方法在预热阶段结束后，会触发 `ReduceLROnPlateau` 的核心逻辑，
        根据传入的验证指标（如损失）来判断是否需要降低学习率。

        Args:
            loss_array (list or tuple): 包含验证损失的数组，将使用最后一个值。
        """
        # 只有在 warmup 结束后，才执行 ReduceLROnPlateau 的逻辑
        if self.current_step > self.warmup_steps:
            super().step(loss_array=loss_array)


    def state_dict(self):
        """返回调度器的状态字典，以便保存和恢复。"""
        parent_state = super().state_dict()
        parent_state['current_step'] = self.current_step
        parent_state['base_lrs'] = self.base_lrs
        return parent_state

    def load_state_dict(self, state_dict):
        """从状态字典加载状态。"""
        self.current_step = state_dict.pop('current_step')
        self.base_lrs = state_dict.pop('base_lrs')
        super().load_state_dict(state_dict)

class OneCycle(BaseScheduler):
    """
    单周期学习率调度器。

    该调度器实现了一个三角学习率策略（Triangular LR Rate Finder），学习率按照单周期的模式变化：
    首先增加到最大值，然后逐渐降低到最小值。这种策略被证明能在较少的 epoch 内达到更好的性能。

    学习率变化模式：
    - 阶段 1 (0 到 42.5%)：从 min_lr 线性增加到 max_lr
    - 阶段 2 (42.5% 到 85%)：从 max_lr 线性降低到 min_lr
    - 阶段 3 (85% 到 100%)：从 min_lr 继续降低到 min_lr / 500

    该方法基于 Leslie Smith 的论文《Cyclical Learning Rates for Training Neural Networks》。

    Attributes:
        optimizer (torch.optim.Optimizer): 被管理的优化器。
        min_lr (float): 最小学习率（最低点）。
        max_lr (float): 最大学习率（高峰点）。
        total_iters (int): 总迭代次数。
        max_lr_epoch_idx (int): 达到最大学习率的迭代索引。
        final_epoch_idx (int): 第二个阶段结束的迭代索引。
        each_diff_lr (float): 第一阶段每步的学习率增量。
        each_diff_lr_final (float): 最终阶段每步的学习率减量。
        iteration (int): 当前迭代计数。

    Args:
        optimizer (torch.optim.Optimizer): 需要管理的优化器。
        total_iters (int): 总的训练迭代次数 (epoch 或 batch)。
        min_lr (float): 最小学习率。会自动调整为至少 500e-7。
        max_lr (float): 最大学习率。
        *args: 保留的额外位置参数。
        **kwargs: 保留的额外关键字参数。

    Raises:
        AssertionError: 当 max_lr 不大于 min_lr 时抛出。

    Methods:
        step(loss_array): 执行一步学习率调度。
        state_dict(): 返回状态字典，用于保存和恢复状态。
        load_state_dict(state_dict): 从状态字典恢复状态。
        use_lr(lr): 手动设置指定学习率。

    Example:
        >>> import torch
        >>> from torch import nn
        >>> model = nn.Linear(10, 1)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = OneCycle(optimizer, total_iters=100, min_lr=1e-6, max_lr=1e-2)
        >>> for iter in range(100):
        ...     # training step
        ...     scheduler.step([loss])  # 注意：loss_array[-1] 被使用

    Note:
        - 建议在学习率范围内使用多个 epoch 而不是单个 epoch
        - min_lr 会自动调整为至少 500e-7，以避免数值问题
        - 当学习率下降到 1e-7 以下时，调度停止以防止过错误的学习
    """

    def __init__(self, optimizer, total_iters: int, min_lr: float, max_lr: float, *args, **kwargs):
        self.optimizer = optimizer
        self.min_lr = max(min_lr, 500 * 1e-7 )
        self.max_lr = max_lr
        assert self.max_lr > self.min_lr, f'max_lr must be greater than min_lr, {self.max_lr} < {self.min_lr}'
        self.total_iters = total_iters

        one_cycle_epochs = int(total_iters * 0.85)
        self.max_lr_epoch_idx = one_cycle_epochs // 2
        self.final_epoch_idx = self.max_lr_epoch_idx * 2

        # 每次调整的学习率
        self.each_diff_lr = (self.max_lr - self.min_lr) / self.max_lr_epoch_idx
        self.each_diff_lr_final = (self.min_lr - self.min_lr / 500) / (total_iters - self.final_epoch_idx)

        self.iteration = 0
        
        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def step(self, *args, **kwargs):
        self.iteration += 1

        # 计算新的学习率（基于第一个参数组的当前学习率）
        cur_lr = self.optimizer.param_groups[0]["lr"]
        if self.iteration <= self.max_lr_epoch_idx:
            new_lr = cur_lr + self.each_diff_lr
        elif self.iteration <= self.final_epoch_idx:
            new_lr = cur_lr - self.each_diff_lr
        else:
            # 最终阶段
            new_lr = cur_lr - self.each_diff_lr_final

        if new_lr <= 1e-7:
            return  # 学习率已经最小

        # 应用新的学习率到所有参数组
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def use_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr

class OneCycle_fast(OneCycle, BaseScheduler):
    """
    增强版单周期学习率调度器，支持自适应调整。

    该类继承自 OneCycle 基础调度器，并添加了智能的自适应机制：
    当训练损失连续多个批次上升时，会自动提前进入降学习率阶段，
    从而更快地调整到合适的学习率范围。

    智能调整机制：
    - **自动检测**: 检测训练损失连续 4 次上升 (loss_array[-1] > -2 > -3 > -4)
    - **自适应调整**: 提前进入学习率降低阶段，缩短第一阶段时间
    - **性能优化**: 避免在过高的学习率下训练造成损失震荡

    学习率曲线：
    - 标准模式: 遵循 OneCycle 的三角形模式 (上升 -> 下降 -> 最终阶段)
    - 自适应模式: 当损失上升时自动缩短上升阶段，延长稳定阶段

    Attributes:
        train_loss_bad_appear (bool): 是否已经检测到训练损失连续上升。
        （继承自 OneCycle 的所有属性）

    Args:
        optimizer (torch.optim.Optimizer): 需要管理的优化器。
        total_iters (int, optional): 总的训练迭代次数。默认: 30。
        min_lr (float, optional): 最小学习率。默认: 500e-7。
        max_lr (float, optional): 最大学习率。默认: 1e-2。
        *args: 额外的位置参数。
        **kwargs: 额外的关键字参数。

    Methods:
        step(loss_array): 执行一步学习率调度，支持自适应调整逻辑。
        （继承自 OneCycle 的所有方法）

    Example:
        >>> import torch
        >>> from torch import nn
        >>> model = nn.Linear(10, 1)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = OneCycle_fast(optimizer, total_iters=50, min_lr=1e-6, max_lr=1e-2)
        >>> for iter in range(50):
        ...     # training step
        ...     scheduler.step([loss])
        # 如果损失连续上升，调度器会自动调整学习率下降策略

    Note:
        - 当检测到训练损失连续 4 次上升时，自动提前进入降学习率阶段
        - 这有助于避免在过高学习率下的训练震荡
        - 相比基础 OneCycle，在动荡的训练曲线下可能有更好的表现
        - 适合用于数据分布不均匀或模型初始化不够好的场景
    """

    def __init__(self, optimizer, total_iters: int=30, min_lr: float = 500 * 1e-7, max_lr: float = 1e-2, *args, **kwargs):
        super().__init__(optimizer, total_iters, min_lr, max_lr, *args, **kwargs)
        self.train_loss_bad_appear = False
        
    def step(self, loss_array, *args, **kwargs):
        """
        train loss 连续 3 次上升, 则进入减低学习率阶段，
        学习率上升阶段与 OneCycle 一致，学习率上限设置倾向于更大
        """
        # 检查是否 连续 4 次上升
        if len(loss_array) >= 4 and not self.train_loss_bad_appear:
            if loss_array[-1] > loss_array[-2] > loss_array[-3] > loss_array[-4]:
                self.train_loss_bad_appear = True
                # 更改 各个调整区域的idx
                diff = self.max_lr_epoch_idx - self.iteration
                if diff > 0:
                    self.max_lr_epoch_idx -= diff
                    self.each_diff_lr = (self.each_diff_lr * self.max_lr_epoch_idx) /  (self.max_lr_epoch_idx + diff * 2)# 改成延长第二段的时间
                    # self.final_epoch_idx -=diff*2

        loss = loss_array[-1]

        self.iteration += 1
        # lr = self.lr_lambda(self.iteration)
        cur_lr = self.optimizer.param_groups[0]["lr"]
        if self.iteration <= self.max_lr_epoch_idx:
            lr = cur_lr + self.each_diff_lr
        elif self.iteration <= self.final_epoch_idx:
            lr = cur_lr - self.each_diff_lr
        else:
            # 最终阶段
            lr = cur_lr - self.each_diff_lr_final
        
        if lr <= 1e-7:
            return # 学习率已经最小

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 当训练的损失序列区域平缓时，减低学习率
class ReduceLR_slow_loss(BaseScheduler):
    """
    基于损失变化率的智能学习率调度器。

    该调度器通过计算训练损失的移动平均值变化率来动态调整学习率。
    当损失平缓下降 (变动率低于阈值) 时，认为训练已进入稳定阶段，
    此时降低学习率可以帮助模型更好地收敛。

    工作原理：
    1. 计算最近 patience 个损失值的移动平均线
    2. 计算两相邻移动平均值的百分比变动率 (pct_change)
    3. 当变动率 >= min_pct 时，认为损失下降过慢，触发学习率降低
    4. 降低学习率后，需等待 patience 个步骤再开始下一次检测

    这在训练后期特别有用，因为当损失变化很小 (平缓) 时，继续使用较高学习率
    可能会导致参数空间徘徊，无法达到更优的最小值。通过降低学习率，
    可以让模型在残差空间里找到更精确的最优点。

    Attributes:
        optimizer (torch.optim.Optimizer): 被管理的优化器。
        min_pct (float): 触发学习率降低的变动率阈值。负值表示损失减少百分比。
        patience (int): 计算移动平均线的时间窗口大小。也用作调节后的等待周期。
        factor (float): 学习率降低倍数。new_lr = old_lr * factor。
        min_lr (float): 学习率的下限。
        eps (float): 学习率变化的最小阈值，用于避免过小变动。
        wait (int): 等待计数器，防止连续触发降低。
        debug (bool): 是否启用调试输出。

    Args:
        optimizer (torch.optim.Optimizer): 需要管理的优化器。
        min_pct (float, optional): 触发阈值。默认: -0.00005 (-0.005%)。
        patience (int, optional): 耐心值和窗口大小。默认: 20。
        factor (float, optional): 降低倍数。默认: 0.1。
        min_lr (float, optional): 最小学习率。默认: 0。
        eps (float, optional): 最小变化阈值。默认: 1e-8。
        debug (bool, optional): 调试模式。默认: False。

    Methods:
        step(array_loss): 执行一步检查，必要时降低学习率。
        _reduce_lr(): 内部方法，执行学习率降低。
        state_dict(): 返回状态字典，排除优化器。
        load_state_dict(state_dict): 从状态字典恢复状态。
        use_lr(lr): 手动设置指定学习率。

    Example:
        >>> import torch
        >>> from torch import nn
        >>> import torch.nn.functional as F
        >>> model = nn.Linear(10, 1)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> scheduler = ReduceLR_slow_loss(optimizer, patience=10, factor=0.5)
        >>>
        >>> for epoch in range(100):
        ...     # training
        ...     losses = []
        ...     for batch in train_loader:
        ...         loss = F.mse_loss(model(batch[0]), batch[1])
        ...         losses.append(loss.item())
        ...         # 其他训练代码...
        ...     scheduler.step(losses)
        ...     print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")

    Note:
        - 使用 PyTorch 张量操作进行高效计算
        - 在 TPU 环境中自动处理步标记
        - patience 影响敏感度：较大的 patience 更稳定但响应较慢
        - min_pct 的负值表示百分比减少，例如 -0.00005 表示 0.005%的增加
        - 当差点降低发生时，会等待 patience 个步骤后再进行新的检测

    References:
        - 损失平缓阶段检测基于移动平均线分析
        - 通过变动率阈值实现的自适应学习率调整算法
    """

    def __init__(self, optimizer, min_pct=-0.00005, patience=20, factor=0.1, min_lr=0, eps=1e-8, debug=False):
        if optimizer is None:
            raise ValueError("optimizer cannot be None")

        if patience <= 0:
            raise ValueError("patience must be a positive integer")

        if factor <= 0 or factor >= 1:
            raise ValueError("factor must be in range (0, 1)")

        if min_lr < 0:
            raise ValueError("min_lr cannot be negative")

        if eps < 0:
            raise ValueError("eps cannot be negative")

        self.optimizer = optimizer
        self.min_pct = min_pct
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self.wait = 0
        self.debug = debug

    def step(self, array_loss, *args, **kwargs):
        # print('step')
        if self.wait > 0:
            self.wait -= 1
            return

        # # 计算损失均线，ma=self.patience
        # # 均线变动率 大于 min_pct 则减少学习率
        # loss = pd.DataFrame({'loss': array_loss}).dropna()
        # if len(loss) < self.patience+1:
        #     return
        # loss['ma'] = loss['loss'].rolling(self.patience).mean()
        # loss['pct'] = loss['ma'].pct_change()
        # loss['match'] = loss['pct']>=self.min_pct
        # if loss.iloc[-1]['match']:
        #     self._reduce_lr()
        # elif self.debug:
        #     print('pass')

        # 改用torch
        # print(array_loss.shape)
        if array_loss.shape[0] < self.patience+1:
            return

        # 计算损失均线
        # print(array_loss)
        loss_ma = array_loss.unfold(dimension=0, size=self.patience, step=1).mean(dim=1)
        # print(loss_ma)

        # 计算均线变动率，确保不会出现除零错误
        prev_ma = loss_ma[:-1]
        curr_ma = loss_ma[1:]
        # 避免除零：如果前一个移动平均值为0，使用一个小的正数
        prev_ma_safe = torch.where(prev_ma == 0, torch.tensor(1e-8, device=prev_ma.device), prev_ma)
        loss_pct_change = curr_ma / prev_ma_safe - 1
        # print(loss_pct_change)
        # 判断是否满足减少学习率条件
        match = (loss_pct_change >= self.min_pct)
        # print(match)

        if tpu_available():
            xm.mark_step()
        if match[-1].item():
            self._reduce_lr()
        elif self.debug:
            print('pass')

    def _reduce_lr(self):
        self.wait = self.patience
        if self.debug:
            print('reduce_lr')
        if self.optimizer is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
    
    def use_lr(self, lr):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr   

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


