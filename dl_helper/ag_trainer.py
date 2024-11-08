from autogluon.tabular import TabularDataset, TabularPredictor
from dl_helper.tests.autogluon.metrics import mean_class_f1_scorer
import pandas as pd
import numpy as np
import os,pickle,subprocess
import time
import threading

def get_gpu_name():
    if 'CUDA_VERSION' in os.environ:
        # 执行 nvidia-smi 命令，并捕获输出
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 解析输出，去掉标题行
        gpu_info = result.stdout.split('\n')[1].strip()
        if 'T4' in gpu_info:
            return 'T4x2'
        elif 'P100' in gpu_info:
            return 'P100'
    else:
        return ''

def get_gpu_num():
    gpu_name = get_gpu_name()
    if gpu_name:
        if 'T4' in gpu_name:
            return 2
        elif 'P100' in gpu_name:
            return 1
    else:
        return 'auto'

def compress_update(title, root):
    # 打包文件夹 并 上传到alist
    zip_file = f'{title}.7z'
    compress_folder(root, zip_file, 9, inplace=False)
    print('compress_folder done')

    # 上传更新到alist
    client = alist(os.environ.get('ALIST_USER'), os.environ.get('ALIST_PWD'))
    client.upload(zip_file, '/ag_train_data/')
    print('upload done')

# 共享的标志变量
stop_flag = False

def keep_update(title, root):
    while not stop_flag:
        # 执行更新操作
        compress_update(title, root)
        # 等待一段时间后再次执行
        time.sleep(60 * 3)  # 10分钟
        # time.sleep(600)  # 10分钟


kaggle = any(key.startswith("KAGGLE") for key in os.environ.keys())

os.environ['ALIST_USER'] = 'admin'
os.environ['ALIST_PWD'] = 'LHss6632673'

def autogluon_train_func(title='', id='id', label='label', use_length=500000, yfunc=None, train_data_folder=''):
    """
    title : 训练标题,用于存储的文件夹名, 默认为空, 为空时使用时间戳
    id, label : id/标签列名
    yfunc : 标签预处理函数
    train_data_folder : 训练数据路径
    """
    if kaggle:
        train_data_folder = '/kaggle/input/' + os.listdir('/kaggle/input')[0]

    time_info = time.strftime("%Y%m%d_%H%M_", time.localtime())
    if '' == title:
        title = os.path.basename(train_data_folder).replace('lh_q_t0_', '')
    title = time_info + title

    # gpu后缀
    gpu_name = get_gpu_name()
    if gpu_name:
        title += '_' + gpu_name
        
    root=f'/ag_train_data/{title}'

    # id, label = 'id', 'label'
    os.makedirs(root, exist_ok=True)

    # 启动报告线程
    global stop_flag
    stop_flag = False
    thread = threading.Thread(target=keep_update, args=(title, root))
    thread.start()

    # 读取训练数据
    # train_data = TabularDataset(pickle.load(open(os.path.join(train_data_folder, 'train.pkl'), 'rb')))
    train_data = TabularDataset(pd.read_pickle(open(os.path.join(train_data_folder, 'train.pkl'), 'rb')))
    if use_length > 0:
        train_data = train_data.iloc[-int(min(use_length, len(train_data))):].reset_index(drop=True)
    print(f'训练数据长度: {len(train_data)}')
    # 预处理标签
    if not None is yfunc:
        train_data[label] = train_data[label].apply(yfunc)

    # 训练模型
    predictor = TabularPredictor(label=label, eval_metric=mean_class_f1_scorer, verbosity=3, log_file_path=os.path.join(root, 'log.txt'))
    clear_train_data = train_data.drop(columns = [id])
    predictor.fit(clear_train_data, num_gpus=get_gpu_num())

    # 读取测试数据
    # test_data = TabularDataset(pickle.load(open(os.path.join(train_data_folder, 'test.pkl'), 'rb')))
    test_data = TabularDataset(pd.read_pickle(open(os.path.join(train_data_folder, 'test.pkl'), 'rb')))
    if use_length > 0:
        test_data = test_data.iloc[-int(min(use_length, len(test_data))):].reset_index(drop=True)
    print(f'测试数据长度: {len(test_data)}')
    # 预处理标签
    if not None is yfunc:
        test_data[label] = test_data[label].apply(yfunc)

    # 预测 储存
    test_to_save = test_data.loc[:, [id, label]].copy()
    predict_proba = predictor.predict_proba(test_data.drop(columns=[id, label]))
    test_to_save[[str(i) for i in range(len(list(predict_proba)))]] = predict_proba.values
    # 重命名 label -> target
    test_to_save.rename(columns={label: 'target'}, inplace=True)
    symbol, test_begin = test_data[id].iloc[0].split('_')
    test_end = test_data[id].iloc[-1].split('_')[-1]
    test_to_save.to_csv(os.path.join(root, f'{symbol}_{test_begin}_{test_end}.csv'), index=False)

    # 评估
    predictor.evaluate(test_data, silent=True)

    leaderboard = predictor.leaderboard(test_data)
    cols = list(leaderboard)
    leaderboard['rank'] = leaderboard['score_val'].rank(ascending=False)
    leaderboard = leaderboard.loc[:, ['rank'] + cols]
    leaderboard.to_csv(os.path.join(root, f'leaderboard.csv'), index=False)

    importance = predictor.feature_importance(test_data.drop(columns = [id]))
    importance.to_csv(os.path.join(root, f'feature_importance.csv'), index=False)

    # 停止报告线程
    stop_flag = True
    thread.join()

    # 压缩更新
    compress_update(title, root)




