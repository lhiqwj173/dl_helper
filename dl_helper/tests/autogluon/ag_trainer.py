from autogluon.tabular import TabularDataset, TabularPredictor
from metrics import mean_class_f1_scorer
import pandas as pd
import numpy as np
import os,pickle,subprocess

def get_gpu_num():
    if 'CUDA_VERSION' in os.environ:
        # 执行 nvidia-smi 命令，并捕获输出
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 解析输出，去掉标题行
        gpu_info = result.stdout.split('\n')[1].strip()
        if 'T4' in gpu_info:
            return 2
        elif 'P100' in gpu_info:
            return 1
    else:
        return 'auto'

kaggle = any(key.startswith("KAGGLE") for key in os.environ.keys())

def autogluon_train_func(id, label, y_len, root='/train_result', yfunc=None, train_data_folder=''):
    """
    id, label : id/标签列名
    y_len : 标签类别数
    root : 训练结果保存路径
    yfunc : 标签预处理函数
    train_data_folder : 训练数据路径
    """

    # id, label = 'id', 'label'
    # y_len = 3
    os.makedirs(root, exist_ok=True)

    if kaggle:
        train_data_folder = '/kaggle/input/' + os.listdir('/kaggle/input')[0]

    # 读取训练数据
    # train_data = TabularDataset(pickle.load(open(os.path.join(train_data_folder, 'train.pkl'), 'rb')))
    train_data = TabularDataset(pd.read_pickle(open(os.path.join(train_data_folder, 'train.pkl'), 'rb')))
    print(f'训练数据长度: {len(train_data)}')
    # 预处理标签
    if not None is yfunc:
        train_data[label] = train_data[label].apply(yfunc)

    # 训练模型
    predictor = TabularPredictor(label=label, eval_metric=mean_class_f1_scorer, verbosity=4, log_file_path=os.path.join(root, 'log.txt'))
    clear_train_data = train_data.drop(columns = [id])
    predictor.fit(clear_train_data, num_gpus=get_gpu_num())

    # 读取测试数据
    # test_data = TabularDataset(pickle.load(open(os.path.join(train_data_folder, 'test.pkl'), 'rb')))
    test_data = TabularDataset(pd.read_pickle(open(os.path.join(train_data_folder, 'test.pkl'), 'rb')))
    # 预处理标签
    if not None is yfunc:
        test_data[label] = test_data[label].apply(yfunc)
    print(f'测试数据长度: {len(test_data)}')

    # 预测 储存
    test_to_save = test_data.loc[:, [id, label]].copy()
    test_to_save[[str(i) for i in range(y_len)]] = predictor.predict_proba(test_data.drop(columns=[id, label])).values
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

    




