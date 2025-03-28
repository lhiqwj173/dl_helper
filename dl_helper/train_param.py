"""
初始化参数

通用参数
train_title: 项目标题
root: 项目路径

训练超参数
epochs: 训练轮数
batch_size: 批次大小
learning_rate: 学习率
abs_learning_rate: # 绝对学习率，如果设置，则无视learning_rate，也不会基于设备数量再进行调整
lr_scheduler_class: 学习率调度类
learning_rate_scheduler_patience: 学习率衰减延迟
no_better_stop: 早停参数
random_mask: 随机遮蔽
random_mask_row: 随机遮蔽行
amp: 是否使用混合精度: fp16/bf16/no
label_smoothing: 标签平滑
weight_decay: 权重衰减
init_learning_ratio: 测试最优学习率
increase_ratio: 测试最优学习率增加比例
data_set: 数据集
"""
import torch
from datetime import datetime
import multiprocessing
import subprocess, os, sys

tpu = False
in_kaggle = False
in_colab = False

def is_kaggle():
    return in_kaggle

def is_colab():
    return in_colab

def cloud_platform():
    global in_kaggle, in_colab
    if any(key.startswith("KAGGLE") for key in os.environ.keys()):
        in_kaggle = True
    elif "IPython" in sys.modules:
        in_colab = "google.colab" in str(sys.modules["IPython"].get_ipython())
cloud_platform()

def get_gpu_info():
    if 'TPU_WORKER_ID' in os.environ:
        global tpu
        tpu = True
        _run_device = 'TPU'

        for i in ['CLOUD_TPU_TASK_ID', 'TPU_PROCESS_ADDRESSES']:
            try:
                os.environ.pop(i)
            except:
                pass

    # elif 'CUDA_VERSION' in os.environ:
    elif os.environ.get('CUDA_HOME', ''):
        # 执行 nvidia-smi 命令，并捕获输出
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 解析输出，去掉标题行
        gpu_info = result.stdout.split('\n')[1].strip()
        if 'T4' in gpu_info:
            _run_device = 'T4x2'
        elif 'P100' in gpu_info:
            _run_device = 'P100'
        
    else:
        _run_device = 'CPU'

    return _run_device
get_gpu_info()

def match_num_processes():
    device = get_gpu_info()
    if device == 'TPU':
        return 8
    elif device == 'T4x2':
        return 2
    elif device == 'P100':
        return 1
    else:
        return 1

def tpu_available():
    return tpu

def data_parm2str(parm):
    # return f"pred_{parm['predict_n']}_pass_{parm['pass_n']}_y_{parm['y_n']}_bd_{parm['begin_date'].replace('-', '_')}_dr_{'@'.join([str(i) for i in parm['data_rate']])}_th_{parm['total_hours']}_s_{parm['symbols']}_t_{parm['target'].replace(' ', '')}"
    parmstr = f"pred_{'@'.join([str(i) for i in parm['predict_n']])}_pass_{parm['pass_n']}_y_{parm['y_n']}_bd_{parm['begin_date'].replace('-', '_')}_dr_{'@'.join([str(i) for i in parm['data_rate']])}_th_{parm['total_hours']}_s_{parm['symbols']}_t_{parm['target'].replace(' ', '')}"

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
  train_title = 'train_title'

  describe = 'describe'

  # 项目路径
  root = './train_title'

  alist_upload_folder = 'train_data'

  workers = 0

  debug = False

  seed = 42

  k_fold_idx = 0
  k_fold_k = 0
  k_fold_ratio = (30,2,1)
  #############################
  # 训练超参数
  #############################
  epochs = 100

  batch_size = 64

  # learning_rate
  learning_rate = 0.001
  learning_rate_scheduler_patience = 20
  abs_learning_rate = 0

  # 早停参数
  no_better_stop = 15

  # 缓存数据间隔
  checkpointing_steps =  15

  # 随机遮蔽
  random_mask = 0
  random_mask_row = 0
  random_scale = 0

  # 数据降采样
  down_freq = 0

  # 是否使用混合精度
  amp = ''

  label_smoothing=0.1

  weight_decay=0.01

  # 弃用!!!
  # 测试最优学习率
  # 非训练使用
  init_learning_ratio=0
  increase_ratio=0.2

  data_set = ''
  y_n = 0

  regress_y_idx = -1
  classify_y_idx = -1
  y_func = None
  y_filter = None
  classify = False

  data_folder = ''

  # 模型类型 
  cnn=False

  test=False

  # 用于模型融合 
  need_meta_output=True

  def __init__(
      self,
      train_title, root, data_set,

      # 训练参数
      batch_size, 
      learning_rate = 0, 
      abs_learning_rate = 0,# 绝对学习率，如果设置，则无视learning_rate，也不会基于设备数量再进行调整
      epochs=100, 
      no_better_stop=15, checkpointing_steps=15, label_smoothing=0.1, weight_decay=0.01, workers=0,
      
      alist_upload_folder = 'train_data',

      # 数据增强
      random_mask=0, random_scale=0, random_mask_row=0, down_freq=0,

      # 测试最优学习率
      init_learning_ratio = 0, increase_ratio = 0.2, 

      # 学习率衰退延迟
      learning_rate_scheduler_patience = 20,

      # 使用回归数据集参数
      classify = False,cnn=False,
      y_n=1,regress_y_idx=-1,classify_y_idx=-1,y_func=None,y_filter=None,
      
      # 数据集路径
      data_folder = '',

      # 交叉验证
      k_fold_idx=0,k_fold_k=0,k_fold_ratio=(30,2,1),

      describe='',

      debug = False,seed = 42,amp='no',

      # 测试运行
      test=False,

      # 模型融合
      need_meta_output=False,



  ):
      # 添加训练后缀 (训练设备/混合精度)
      run_device = get_gpu_info()
      self.train_title = f'{train_title}_{run_device}'
      self.root = f'{root}_{run_device}'

      self.alist_upload_folder = alist_upload_folder

      if amp not in ['no', 'fp8', 'fp16', 'bf16']:
        self.amp = 'no'
      else:
        self.amp = amp

      if self.amp in ['fp8', 'fp16', 'bf16']:
          self.train_title = f'{self.train_title }_{self.amp}'
          self.root = f'{self.root }_{self.amp}'
        
      self.epochs = epochs
      self.batch_size = batch_size
      self.learning_rate = learning_rate
      self.abs_learning_rate = abs_learning_rate
      self.no_better_stop = no_better_stop
      self.checkpointing_steps = checkpointing_steps
      self.random_mask = random_mask
      self.random_scale = random_scale
      self.random_mask_row = random_mask_row
      self.down_freq = down_freq
      self.label_smoothing = label_smoothing
      self.weight_decay = weight_decay
      self.init_learning_ratio = init_learning_ratio
      self.increase_ratio = increase_ratio

      self.data_set = data_set

      self.classify = classify
      self.y_n = y_n
      self.regress_y_idx = regress_y_idx
      self.classify_y_idx = classify_y_idx
      self.y_func = y_func
      self.y_filter = y_filter

      self.describe = describe
      self.workers = workers

      self.data_folder = data_folder if data_folder else './data'

      self.debug = debug
      self.test = test
      self.seed = seed
      self.need_meta_output = need_meta_output

      self.k_fold_idx = k_fold_idx
      self.k_fold_k = k_fold_k
      self.k_fold_ratio = k_fold_ratio

      # # log
      # os.makedirs(os.path.join(self.root, 'log'), exist_ok=True)
      # logger.add(os.path.join(self.root, 'log', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))