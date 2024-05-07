"""
初始化参数

通用参数
train_title: 项目标题
root: 项目路径

训练超参数
epochs: 训练轮数
batch_size: 批次大小
learning_rate: 学习率
warm_up_epochs: warm up 数
no_better_stop: 早停参数
random_mask: 随机遮蔽
random_mask_row: 随机遮蔽行
amp: 是否使用混合精度
amp_ratio: 混合精度比例
label_smoothing: 标签平滑
weight_decay: 权重衰减
init_learning_ratio: 测试最优学习率
increase_ratio: 测试最优学习率增加比例
data_set: 数据集
model: 模型
"""
import torch
from datetime import datetime
import multiprocessing
import os



# 日志
from loguru import logger as _logger
class logger:
  @staticmethod
  def debug(*args):
    _logger.debug(*args)

  @staticmethod
  def error(*args):
    _logger.error(*args)

  @staticmethod
  def add(*args):
    _logger.add(*args)



class Params:
  #############################
  # 通用参数
  #############################
  train_title = ''

  describe = ''

  # 项目路径
  root = ''

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  workers = int(multiprocessing.cpu_count())

  #############################
  # 训练超参数
  #############################
  epochs = 0

  batch_size = 0

  # learning_rate
  learning_rate = 0.001

  # warm up 数
  warm_up_epochs = 3

  # 早停参数
  no_better_stop = 15

  # 随机遮蔽
  random_mask = 0
  random_mask_row = 0

  # 是否使用混合精度
  amp = False
  amp_ratio = 2

  label_smoothing=0.1

  weight_decay=0.01

  # 测试最优学习率
  # 非训练使用
  init_learning_ratio=0
  increase_ratio=0.2

  data_set = ''

  # 模型
  model = None

params = Params()

def init_param(
    _train_title, _root,
    _epochs, _batch_size, _learning_rate, _warm_up_epochs, _no_better_stop, _random_mask, _random_mask_row, _amp, _label_smoothing, _weight_decay, _init_learning_ratio, _increase_ratio, _data_set, _model,
    _describe=''

):
    global params

    params.train_title = _train_title
    params.root = _root

    params.epochs = _epochs
    params.batch_size = _batch_size
    params.learning_rate = _learning_rate
    params.warm_up_epochs = _warm_up_epochs
    params.no_better_stop = _no_better_stop
    params.random_mask = _random_mask
    params.random_mask_row = _random_mask_row
    params.amp = _amp
    params.label_smoothing = _label_smoothing
    params.weight_decay = _weight_decay
    params.init_learning_ratio = _init_learning_ratio
    params.increase_ratio = _increase_ratio
    params.data_set = _data_set
    params.model = _model
    params.describe = _describe

    # 运行变量
    os.makedirs(os.path.join(params.root, 'var'), exist_ok=True)

    # 储存数据
    os.makedirs(os.path.join(params.root, 'data'), exist_ok=True)

    # log
    os.makedirs(os.path.join(params.root, 'log'), exist_ok=True)
    logger.add(os.path.join(params.root, 'log', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))

    # 检查是否有可用的GPU
    logger.debug(params.device)

    # cpu核数
    logger.debug(f'workers: {params.workers}')