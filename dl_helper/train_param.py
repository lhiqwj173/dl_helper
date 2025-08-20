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
no_better_stop: 早停参数
amp: 是否使用混合精度: fp16/bf16/no
label_smoothing: 标签平滑
weight_decay: 权重衰减
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

  # 项目路径
  root = './train_title'

  alist_upload_folder = 'train_data'

  debug = False

  seed = 42

  #############################
  # 训练超参数
  #############################
  epochs = 100

  batch_size = 32
  batch_n = 1

  # learning_rate
  learning_rate = 0.001
  abs_learning_rate = 0

  # 早停参数
  no_better_stop = 15

  # 缓存数据间隔
  checkpointing_steps =  15

  # 是否使用混合精度
  amp = ''

  label_smoothing=0.1

  weight_decay=0.01

  y_n = 0

  classify = False

  # 模型类型 
  cnn=False

  test=False



  def __init__(
      self,
      train_title='test', 

      # 训练参数
      batch_size=64, 
      batch_n=1,
      learning_rate = 3e-4, 
      abs_learning_rate = 0,# 绝对学习率，如果设置，则无视learning_rate，也不会基于设备数量再进行调整
      epochs=100, 
      no_better_stop=15, checkpointing_steps=15, label_smoothing=0.1, weight_decay=0,
      
      alist_upload_folder = 'train_data',

      # 使用回归数据集参数
      classify = False,cnn=False,
      y_n=1,
      
      debug = False,seed = 42,amp='no',

      # 测试运行
      test=False,



      **kwargs
  ):
      # 添加训练后缀 (训练设备/混合精度)
      self.train_title = f'{train_title}'
      self.root = self.train_title

      self.alist_upload_folder = alist_upload_folder

      if amp not in ['no', 'fp8', 'fp16', 'bf16']:
        self.amp = 'no'
      else:
        self.amp = amp

      if self.amp in ['fp8', 'fp16', 'bf16']:
          self.train_title = f'{self.train_title }_{self.amp}'
          self.root = f'{self.root}_{self.amp}'
        
      self.epochs = int(epochs)
      self.batch_n = int(batch_n)
      self.batch_size = int(batch_size) * self.batch_n
      self.learning_rate = float(learning_rate) * self.batch_n
      self.abs_learning_rate = int(abs_learning_rate)
      self.no_better_stop = int(no_better_stop)
      self.checkpointing_steps = int(checkpointing_steps)
      self.label_smoothing = float(label_smoothing)
      self.weight_decay = float(weight_decay)

      self.classify = bool(classify)
      self.y_n = int(y_n)

      self.debug = bool(debug)
      self.test = bool(test)
      self.seed = int(seed)


      # # log
      # os.makedirs(os.path.join(self.root, 'log'), exist_ok=True)
      # logger.add(os.path.join(self.root, 'log', f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))