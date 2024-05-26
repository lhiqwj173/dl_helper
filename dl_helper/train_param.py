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

def data_parm2str(parm):
    # return f"pred_{parm['predict_n']}_pass_{parm['pass_n']}_y_{parm['y_n']}_bd_{parm['begin_date'].replace('-', '_')}_dr_{'@'.join([str(i) for i in parm['data_rate']])}_th_{parm['total_hours']}_s_{parm['symbols']}_t_{parm['target'].replace(' ', '')}"
    parmstr = f"pred_{'@'.join([str(i) for i in parm['predict_n']])}_pass_{parm['pass_n']}_y_{parm['y_n']}_bd_{parm['begin_date'].replace('-', '_')}_dr_{'@'.join([str(i) for i in parm['data_rate']])}_th_{parm['total_hours']}_s_{parm['symbols']}_t_{parm['taget'].replace(' ', '')}"

    # 新增加数据参数，为了匹配之前的数据名称，默认4h，不进行编码
    if 'std_mode' in parm and parm['std_mode'] != '4h':
        parmstr += f"_std_{parm['std_mode']}"

    return parmstr

def data_str2parm(s):
    s = s.split('.')[0]
    p = s.split('_')
    return {
        'predict_n': int(p[1]) if '@' not in p[1] else [int(i) for i in p[1].split('@')],
        'pass_n': int(p[3]),
        'y_n': int(p[5]),
        'begin_date': f'{p[7]}-{p[8]}-{p[9]}',
        'data_rate': tuple([int(i) for i in p[11].split('@')]),
        'total_hours': int(p[13]),
        'symbols': p[15],
        'target': p[17],
        'std_mode': p[19]  # 4h/1d/5d
    }

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
  # workers = int(multiprocessing.cpu_count())
  workers = 0

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
  random_scale = 0

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
  data_parm = None
  y_n = 0
  use_trade = False
  use_pk = True

  regress_y_idx = -1
  classify_y_idx = -1
  classify_func = None

  # 模型
  model = None

  data_folder = ''

params = Params()

def init_param(
    train_title, root, model, data_set,

    # 训练参数
    learning_rate, batch_size, 
    epochs=100, warm_up_epochs=3, 
    no_better_stop=15,amp=False, label_smoothing=0.1, weight_decay=0.01, 

    # 数据增强
    random_mask=0, random_scale=0, random_mask_row=0, 

    # 测试最优学习率
    init_learning_ratio = 0, increase_ratio = 0.2, 

    # 使用回归数据集参数
    y_n=1,regress_y_idx=-1,classify_y_idx=-1,classify_func=None,
    
    # 数据集路径
    data_folder = '',

    # 数据使用部分
    use_pk = True, use_trade = False,

    describe=''

):
    global params

    params.train_title = train_title
    params.root = root

    params.epochs = epochs
    params.batch_size = batch_size
    params.learning_rate = learning_rate
    params.warm_up_epochs = warm_up_epochs
    params.no_better_stop = no_better_stop
    params.random_mask = random_mask
    params.random_scale = random_scale
    params.random_mask_row = random_mask_row
    params.amp = amp
    params.label_smoothing = label_smoothing
    params.weight_decay = weight_decay
    params.init_learning_ratio = init_learning_ratio
    params.increase_ratio = increase_ratio
    params.data_set = data_set
    params.data_parm = data_str2parm(data_set)
    params.regress_y_idx = regress_y_idx
    params.classify_y_idx = classify_y_idx
    params.classify_func = classify_func
    params.model = model
    params.describe = describe
    params.y_n = y_n
    params.use_pk = use_pk
    params.use_trade = use_trade

    params.data_folder = data_folder if data_folder else './data'

    # 运行变量
    os.makedirs(os.path.join(params.root, 'var'), exist_ok=True)

    # log
    os.makedirs(os.path.join(params.root, 'log'), exist_ok=True)
    logger.add(os.path.join(params.root, 'log', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))

    # 检查是否有可用的GPU
    logger.debug(params.device)

    # cpu核数
    logger.debug(f'workers: {params.workers}')