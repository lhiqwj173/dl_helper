"""
用于追踪训练过程评价指标
"""
import time, math, os, copy, pickle, datetime
import itertools
import traceback
# import dataframe_image as dfi

import pandas as pd

from datetime import timedelta
from datetime import datetime
import torch
import torch.nn.functional as F
from torchmetrics import F1Score, R2Score
from sklearn.metrics import r2_score as sk_r2_score
from sklearn.metrics import matthews_corrcoef

from py_ext.tool import debug, log

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

from accelerate.utils import broadcast, gather_object

from dl_helper.scheduler import ReduceLR_slow_loss, ReduceLROnPlateau, WarmupReduceLROnPlateau, LRFinder
from dl_helper.tool import save_df_pic
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dl_helper.train_param import tpu_available, data_str2parm
if tpu_available():
    import torch_xla.core.xla_model as xm

# 允许的记录类型:
# train / val / train_final / train_best / val_final / val_best / test_final / test_best / test_dummy

MODEL_TYPES=['model_final', 'model_best', 'model_dummy']
MODEL_FINAL, MODEL_BEST, MODEL_DUMMY = MODEL_TYPES

TEST_TYPES=['test_final', 'test_best', 'test_dummy']
TEST_FINAL, TEST_BEST, TEST_DUMMY = TEST_TYPES

TYPES_NO_NEED_SYMBOL_SCORE = ['train', 'val']

TYPES_NEED_LABEL_COUNT = ['train', 'val', 'test_final']
TYPES_NEED_CAL_THRESHOLD = ['test_final', 'test_best']
TYPES_NEED_OUTPUT = ['train_final', 'train_best', 'val_final', 'val_best', 'test_final', 'test_best', 'test_dummy']# 用于模型融合 基特征

def last_value(data):
    """返回最后一个非nan值"""
    for i in range(len(data)-1, -1, -1):
        if not math.isnan(data[i]):
            return data[i]
    raise ValueError("没有找到非nan值")


def cal_balance_acc(y_pred, y_true, y_n):
    unique_labels = [torch.tensor(i, device=y_pred.device) for i in range(y_n)]
    recall_values = []

    for label in unique_labels:
        true_positives = torch.sum((y_true == label) & (y_pred == label))
        false_negatives = torch.sum((y_true == label) & (y_pred != label))
        recall = true_positives / (true_positives + false_negatives)
        recall_values.append(recall)

    # 计算均衡 ACC
    balanced_acc = torch.mean(torch.stack(recall_values))
    if math.isnan(balanced_acc):
        pickle.dump((y_pred, y_true, y_n), open(f'nan_acc_data.pkl', 'wb'))
        raise Exception("acc nan")
    
    return balanced_acc

def class_accuracy(y_pred, y_true, y_n):
    class_correct = [0] * y_n
    class_total = [0] * y_n
    
    for i in range(y_n):
        class_correct[i] = torch.logical_and(y_pred == i, y_pred == y_true).sum().item()
        class_total[i] = (y_true == i).sum().item()
    
    class_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(y_n)]
    
    return torch.tensor(class_acc, device=y_pred.device)

def class_acc_score(y_pred, y_true, y_n):
    """
    计算每个类别的准确率。
    
    参数:
    y_pred (array-like): 预测的类别标签
    y_true (array-like): 真实的类别标签
    y_n (int): 类别数量
    
    返回:
    list: 每个类别的准确率列表
    """
    # 将输入转换为 NumPy 数组，便于处理
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    
    # 检查 y_pred 和 y_true 的长度是否一致
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred 和 y_true 的长度必须相同")
    
    # 初始化存储每个类别准确率的列表
    accuracies = []
    
    # 遍历每个类别
    for class_label in range(y_n):
        # 找出真实标签中属于当前类别的索引
        class_indices = np.where(y_true == class_label)[0]
        
        # 如果该类别没有样本，准确率为 0
        if len(class_indices) == 0:
            accuracies.append(0.0)
        else:
            # 计算该类别中预测正确的样本数
            correct_predictions = np.sum(y_pred[class_indices] == y_true[class_indices])
            # 计算准确率：正确预测数 / 该类别总样本数
            accuracy = correct_predictions / len(class_indices)
            accuracies.append(accuracy)
    
    return torch.tensor(accuracies)

def class_mcc_score(y_pred, y_true, y_n):
    # 转换为numpy计算
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    # 计算每个类别的 MCC
    mcc_values = []
    for i in range(y_n):
        # 创建二元分类问题: 类别 i vs. 其他类别
        y_true_binary = np.where(y_true_np == i, 1, 0)
        y_pred_binary = np.where(y_pred_np == i, 1, 0)

        # 计算 MCC
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        mcc_values.append(mcc)
    
    return torch.tensor(mcc_values, device=y_pred.device)

def class_f1_score(y_pred, y_true, y_n):
    # 计算每个类别的 F1 分数
    f1_score = F1Score(num_classes=y_n, average='none', task='multiclass').to(y_pred.device)  # 设置 average='none' 以计算每个类别的 F1 分数
    # print('F1Score', flush=True)
    # 计算 F1 Score
    class_f1 = f1_score(y_pred, y_true)
    # print('compute', flush=True)
    return class_f1

def r2_score(y_pred, y_true):
    # 计算方差加权 R2
    _r2_score = R2Score(multioutput='variance_weighted').to(y_pred.device)
    variance_weighted_r2 = _r2_score(y_pred, y_true)
    return variance_weighted_r2

def sklearn_r2_score(y_pred, y_true):
    # 转换为numpy计算
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
    r2 = sk_r2_score(y_true_np, y_pred_np, multioutput='variance_weighted')
    return torch.tensor(r2, device=y_pred.device, dtype=torch.float32)

def class_f1_score_each_code(_type, symbol_score, codes, y_pred, y_true, y_n, root):
    # train/val/test_final/test_best/test_dummy/final_train/final_val/best_train/best_val
    total_codes = set(codes)
    for _code in total_codes:
        match_ids = [i for i in range(len(codes)) if codes[i] == _code]
        match_y_pred = y_pred[match_ids]
        match_y_true = y_true[match_ids]

        class_f1 = class_f1_score(match_y_pred, match_y_true, y_n)

        if _code not in symbol_score:
            symbol_score[_code] = {}

        sum_score = 0
        for i in range(y_n - 1):
            sum_score += class_f1[i].cpu().item()

        symbol_score[_code][f'{_type}_class_f1'] = sum_score / (y_n - 1)

    df_all = pd.DataFrame(symbol_score).T
    print(list(df_all))
    for model_type in ['final', 'best']:
        # 保存到pic
        need_cols = [i for i in list(df_all) if model_type in i]
        test_col = [i for i in need_cols if 'test' in i]
        if len(test_col) == 0:
            continue

        test_col = test_col[0]
        train_col = f'train_{model_type}_class_f1'
        val_col = f'val_{model_type}_class_f1'

        df = df_all.loc[:,need_cols].copy().sort_values(test_col, ascending=False)

        for col in list(df):
            df[col] = df[col].apply(lambda x: '{:.4f}'.format(x))
        idx = range(len(df))

        for col in [train_col, val_col]:
            if col not in list(df):
                continue

            rank = df[col].rank(method='first', ascending=False) - 1
            rank_change = (idx - rank).astype(int)
            df[col] = df[col] + rank_change.apply(lambda x: '(' + ('+' + str(x) if x > 0 else str(x)) + ')' if x != 0 else '')

        save_df_pic(os.path.join(root, f'{model_type}_symbol_f1_rank.png'),df,(500, 140))
        # dfi.export(df, os.path.join(root, 'symbol_f1_rank.png'))

def r2_score_each_code(_type, symbol_score, codes, y_pred, y_true, y_n, root):
    # train/val/test_final/test_best/test_dummy/final_train/final_val/best_train/best_val
    total_codes = set(codes)
    for _code in total_codes:
        match_ids = [i for i in range(len(codes)) if codes[i] == _code]
        match_y_pred = y_pred[match_ids]
        match_y_true = y_true[match_ids]

        score = sklearn_r2_score(match_y_pred, match_y_true)

        if _code not in symbol_score:
            symbol_score[_code] = {}

        symbol_score[_code][f'{_type}_r2'] = score

    df_all = pd.DataFrame(symbol_score).T
    print(list(df_all))
    for model_type in ['final', 'best']:
        # 保存到pic
        need_cols = [i for i in list(df_all) if model_type in i]
        test_col = [i for i in need_cols if 'test' in i]
        if len(test_col) == 0:
            continue

        test_col = test_col[0]
        train_col = f'train_{model_type}_r2'
        val_col = f'val_{model_type}_r2'

        df = df_all.loc[:,need_cols].copy().sort_values(test_col, ascending=False)

        for col in list(df):
            df[col] = df[col].apply(lambda x: '{:.4f}'.format(x))
        idx = range(len(df))

        for col in [train_col, val_col]:
            if col not in list(df):
                continue

            rank = df[col].rank(method='first', ascending=False) - 1
            rank_change = (idx - rank).astype(int)
            df[col] = df[col] + rank_change.apply(lambda x: '(' + ('+' + str(x) if x > 0 else str(x)) + ')' if x != 0 else '')

        save_df_pic(os.path.join(root, f'{model_type}_symbol_r2.png'),df,(500, 140))

def f1_score(y_true, y_pred, y_n):
    # 计算加权 F1 分数
    f1_score = F1Score(num_classes=y_n, average='weighted', task='multiclass').to(y_pred.device)
    return f1_score(y_pred, y_true).unsqueeze(0)

def cal_recall(y_true, y_pred, y_n):
    """
    计算多分类任务的 recall。
    
    参数:
        y_true (torch.Tensor): 真实标签，形状为 (N,)，值为 0 到 y_n-1
        y_pred (torch.Tensor): 预测标签，形状为 (N,)，值为 0 到 y_n-1
        y_n (int): 类别数
    
    返回:
        dict: 包含每个类别的 recall 和宏平均 recall
    """
    # 确保输入是张量
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    
    # 初始化 TP 和 FN 的计数
    recalls = []
    for c in range(y_n):
        # 真正例 (TP): 预测为 c 且真实标签为 c
        true_positive = torch.sum((y_pred == c) & (y_true == c)).float()
        # 真实正例总数 (TP + FN): 真实标签为 c 的样本
        true_positive_plus_false_negative = torch.sum(y_true == c).float()
        
        # 计算 recall
        if true_positive_plus_false_negative > 0:
            recall_c = true_positive / true_positive_plus_false_negative
        else:
            recall_c = torch.tensor(0.0)  # 如果该类别没有样本，recall 为 0
        recalls.append(recall_c)
    
    # 转换为张量
    recalls = torch.tensor(recalls)
    
    # 计算宏平均 recall
    macro_recall = recalls.mean() if y_n > 0 else torch.tensor(0.0)
    
    # 返回每个类别的 recall 和宏平均 recall
    result = {f'class_recall_{c}': recalls[c].item() for c in range(y_n)}
    result['recall'] = macro_recall.item()
    
    return result

def plot_roc_curve_0(y_true, y_score, file_path):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().numpy()

    plt.figure(figsize=(15, 5))

    # 计算每个类别的 ROC 曲线并绘制子图
    best_thresholds = []
    for i in range(y_score.shape[1]):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])

        # 计算使 F1 分数最优的阈值
        precision, recall, thresholds = precision_recall_curve(y_true == i, y_score[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # f1_scores nan -> 备份数据
        if math.isnan(np.max(f1_scores)):
            pickle.dump((y_true, y_score), open(file_path.replace('ROC_curve.png', 'y_true_y_score_dump.pkl'), 'wb'))

        best_threshold = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_threshold)

        plt.subplot(1, 3, i+1)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {i} {np.max(f1_scores):.3f}')
        plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], c='red', marker='x', label=f'Best Threshold: {best_threshold:.3f}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)

    threshold_file = file_path.replace('ROC_curve.png', 'threshold.txt')
    with open(threshold_file, 'w') as f:
        d = ",".join([f'{i:.3f}' for i in best_thresholds])
        f.write(f'{d}\n')

def plot_roc_curve(y_true, y_score, file_path):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().numpy()

    n_classes = y_score.shape[1]  # 获取类别数
    # 动态计算子图的行数和列数
    cols = min(3, n_classes)  # 每行最多 3 列
    rows = (n_classes + cols - 1) // cols  # 计算所需行数

    plt.figure(figsize=(15, 5 * rows))  # 动态调整画布高度

    best_thresholds = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])

        # 计算使 F1 分数最优的阈值
        precision, recall, thresholds = precision_recall_curve(y_true == i, y_score[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # f1_scores nan -> 备份数据
        if math.isnan(np.max(f1_scores)):
            pickle.dump((y_true, y_score), open(file_path.replace('ROC_curve.png', 'y_true_y_score_dump.pkl'), 'wb'))

        best_threshold = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_threshold)

        # 使用动态的子图索引
        plt.subplot(rows, cols, i + 1)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {i} {np.max(f1_scores):.3f}')
        plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], c='red', marker='x', label=f'Best Threshold: {best_threshold:.3f}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()  # 关闭画布，防止内存泄漏

    threshold_file = file_path.replace('ROC_curve.png', 'threshold.txt')
    with open(threshold_file, 'w') as f:
        d = ",".join([f'{i:.3f}' for i in best_thresholds])
        f.write(f'{d}\n')

def max_downward_slope(numbers):
    """
    返回向下的最大斜率的idx
    """
    slopes = []
    for i in range(1, len(numbers)):
        slope = (numbers[i] - numbers[i-1]) / 1  # 假设间隔为 1
        slopes.append(slope)
    return slopes.index(min(slopes))

def move_numbered_columns_to_end(df):
    # 获取所有列名
    cols = df.columns.tolist()
    
    # 筛选以_数字结尾的列
    numbered_cols = [col for col in cols if col.split('_')[-1].isdigit()]
    
    # 保持非编号列的原始顺序
    non_numbered_cols = [col for col in cols if col not in numbered_cols]
    
    # 按原始顺序组合：非编号列 + 编号列
    new_order = non_numbered_cols + numbered_cols
    
    # 重新排列DataFrame的列
    return df[new_order]

class Tracker_None():
    def __init__(self, *args, **kwargs):
        self.epoch_count = 0
        self.step_in_epoch = 0
        self.step_count = 0
        self.need_test = False

    def plot(self):
        pass

    def update(self):
        pass

    def track(self, *args, **kwargs):
        pass

class Tracker():
    def __init__(self, model_name, params, accelerator, scheduler, num_processes, printer, data_id_getter_func=None):
        self.model_name = model_name
        # 时间统计
        self.begin_time = time.time()
        self.notebook_begin_time = time.time()
        self.cost_hour = 0# 之前的notebook训练耗时
        self.cur_notebook_cost_hour = 0# 当前notebook耗时
        self.each_epoch_time_cost = 0
        self.epoch_count = 0
        self.mini_epoch_count = 0
        self.step_count = 0
        self.best_model_epoch = 0
        # 每个epoch训练中的阶段
        # 0: 训练 1: 验证
        self.step_in_epoch = 0
        self.run_limit_hour = 12 if  num_processes != 8 else 9
        self.need_test = False
        # self.need_test = True

        # 统计训练集的标签分布
        self.label_counts = {}
        self.label_count_done = {}

        self.params = params
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.printer = printer

        # 储存参数
        self.no_need_save_parm = [
            'notebook_begin_time',
            'need_test',
            'params',
            'accelerator',
            'scheduler',
            'printer',
        ]

        # 用于output输出
        # 格式为 [(id, target, predict), ...]
        self.output_datas=[]

        # 最终数据
        self.data = {}

        # 训练学习率
        self.data['lr'] = []

        # 跟新类别
        self.track_update = ''

        # 计算变量
        self.temp = {}

        self.reset_temp()

        # 储存各个标的的各过程的 0, 1 类f1 score
        self.symbol_score = {}

        # 获取数据 id 的函数
        # 参数为 batch data，返回 batch id tesnsor格式
        # 可以用于 模型融合输出
        self.data_id_getter_func = data_id_getter_func

    def record_best_model_epoch(self):
        self.best_model_epoch = self.epoch_count - 1

    def update_mini_batch(self):
        self.mini_epoch_count += 1

    def cal_threshold_f1score(self):
        if self.params.y_n > 3:
            # 类别大于3，不计算threshold
            # 计算量太大
            return

        # pickle.dump((self.temp, self.params), open('debug_data.pkl', 'wb'))

        folder = self.track_update.replace('test', 'model')
        folder = os.path.join(self.params.root, folder)

        # 按照不同的 threshold 计算均衡f1 score
        # 读取 threshold
        threshold_file = os.path.join(folder, 'threshold.txt')
        with open(threshold_file, 'r')as f:
            thresholds = f.readline().strip().split(',')
            thresholds = [float(i) for i in thresholds]

        thresholds = torch.tensor(thresholds,device=self.temp['softmax_predictions'].device)
        # print(f"thresholds: {thresholds}")
        categories = [i for i in range(self.params.y_n)]

        combinations = []
        for r in range(1, len(categories) + 1):
            combinations.extend(itertools.permutations(categories, r))
        # print(f"combinations: {combinations}")

        combinations = [i for i in combinations if len(i) == len(categories)]
        combinations = [torch.tensor(i,device=self.temp['softmax_predictions'].device) for i in combinations]
        # print(f"combinations: {combinations}")

        thresholded_predictions = self.temp['softmax_predictions'] > thresholds
        thresholded_predictions_int = thresholded_predictions.int()
        # print(f"thresholded_predictions_int")
        rets = []
        for comb in combinations:
            # print(f"comb: {comb}")
            # 预测类别
            y_pred = torch.argmax(thresholded_predictions_int[:, comb], dim=1)

            # print(y_pred.shape)
            # print(self.temp['_y_true'].shape)
            balance_acc = cal_balance_acc(
                y_pred, self.temp['_y_true'], self.params.y_n
            ).unsqueeze(0)
            # self.printer.print('balance_acc')
            
            # 计算加权 F1 分数
            f1_score = F1Score(num_classes=self.params.y_n, average='weighted', task='multiclass').to(y_pred.device)
            weighted_f1 = f1_score(y_pred, self.temp['_y_true']).unsqueeze(0)
        
            rets.append((comb, balance_acc, weighted_f1))

        # print(f"rets: {rets}")

        # 按照 weighted_f1 排序
        rets = sorted(rets, key=lambda x: x[2])
        # print(f"sorted rets: {rets}")

        with open(threshold_file, 'a')as f:
            f.write('comb,balance_acc,weighted_f1\n')
            for i, (comb, balance_acc, weighted_f1) in enumerate(rets):
                comb_str = '_'.join([str(i.item()) for i in comb])
                f.write(f"{comb_str},{balance_acc.item()},{weighted_f1.item()}\n")

    def update(self):
        # 标记label分布统计完成
        if self.params.classify and self.accelerator.is_main_process and self.track_update not in self.label_count_done and self.track_update in TYPES_NEED_LABEL_COUNT:
            self.label_count_done[self.track_update] = True

        self.printer.print(f'update {self.track_update}')

        # 模型 output 输出，用于模型融合训练
        # > train/val/test > date_file
        # id,target,0,1,2
        if self.track_update in TYPES_NEED_OUTPUT and self.params.need_meta_output:

            dataset_type, model_type = self.track_update.split('_')
            save_folder = os.path.join(self.params.root, f"model_{model_type}")
            assert dataset_type in ['train', 'val', 'test'], f'error dataset_type:{dataset_type}'

            out_folder = os.path.join(save_folder, dataset_type)
            os.makedirs(out_folder, exist_ok=True)

            self.printer.print(f'输出模型output: {model_type} {dataset_type} 共{len(self.output_datas)}条')

            # id 排序
            # 格式为 [(id, target, predict), ...]
            self.output_datas = sorted(self.output_datas, key=lambda x: x[0])

            # self.printer.print(f'输出模型output: {model_type} {dataset_type} {_id}')
            with open(os.path.join(out_folder, f'{model_type}_{dataset_type}.csv'), 'w') as f:
                # 输出 列名
                f.write('id')
                # 获取 target 维度
                # 支持多个输出
                target_length = 1 if len(self.output_datas[0][1].shape)==0 else self.output_datas[0][1].shape[0]
                if target_length > 1:
                    for i in range(target_length):
                        f.write(f',target{i}')
                else:
                    f.write(f',target')
                # 预测类别
                for i in range(self.params.y_n):
                    f.write(f',{i}')
                f.write('\n')

                # 储存数据
                for _id, target, predict in self.output_datas:
                    predict_str = ','.join([str(float(i)) for i in predict])
                    target_str = ','.join([str(float(i)) for i in target])
                    f.write(f'{_id},{target_str},{predict_str}\n')

            self.output_datas = []# 重置
            self.printer.print(f'输出模型output: {model_type} {dataset_type} 完成')

        # 等待同步
        self.accelerator.wait_for_everyone()

        # 计算变量 -> data
        # 主进程计算data
        if self.accelerator.is_main_process:
            
            # 更新训练时间记录
            self.cur_notebook_cost_hour = (time.time() - self.notebook_begin_time) / 3600

            # 计算数据
            _loss = torch.mean(self.temp['_loss']).unsqueeze(0).cpu()
            print(f"_loss {_loss.device}")

            if self.params.classify:
                self.temp['softmax_predictions'] = self.temp['_y_pred']

                if self.track_update in TYPES_NEED_CAL_THRESHOLD:
                    self.cal_threshold_f1score()

                _, self.temp['_y_pred'] = torch.max(self.temp['softmax_predictions'], dim=1)

                # 改用 Balanced Accuracy
                balance_acc = cal_balance_acc(
                    self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n
                ).unsqueeze(0).cpu()
                # self.printer.print('balance_acc')
                
                # 计算加权 F1 分数
                weighted_f1 = f1_score(self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n).cpu()
                # self.printer.print('weighted_f1')

                # 计算 recall / macro_0/1/2
                recall_dict = cal_recall(self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n)

                # 计算各个类别 f1 score
                class_f1 = class_f1_score(self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n).cpu()
                # print('class_f1', flush=True)

                # 计算各个类别 mcc
                class_mcc = class_mcc_score(self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n).cpu()
                # print('class_mcc', flush=True)

                # 计算各个类别的 acc
                class_acc = class_acc_score(self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n).cpu()

                # # 各个类别按照 code 分类计数 f1 score
                # # train/val 不需要计算
                # if self.track_update not in TYPES_NO_NEED_SYMBOL_SCORE:
                #     class_f1_score_each_code(self.track_update, self.symbol_score, self.temp['_codes'], self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n, self.params.root)

                # self.printer.print('class_f1_each_code')

            else:
                # 计算方差加权 R2
                variance_weighted_r2 = sklearn_r2_score(self.temp['_y_pred'], self.temp['_y_true']).cpu()

                # # 各个类别按照 code 分类计数 r2 score
                # # train/val 不需要计算
                # if self.track_update not in TYPES_NO_NEED_SYMBOL_SCORE:
                #     r2_score_each_code(self.track_update, self.symbol_score, self.temp['_codes'], self.temp['_y_pred'], self.temp['_y_true'], self.params.y_n, self.params.root)
                # print('r2_each_code', flush=True)
        
            # self.printer.print(f'_loss: {_loss.shape}')
            # self.printer.print(f'balance_acc: {balance_acc.shape}')
            # self.printer.print(f'weighted_f1: {weighted_f1.shape}')

            # 记录数据
            if f'{self.track_update}_loss' not in self.data:
                if _loss >0:
                    # 否则不需要记录，正常情况loss>0 -> 该loss是伪造的，实际不需要记录
                    self.data[f'{self.track_update}_loss'] = _loss

                if self.params.classify:
                    self.data[f'{self.track_update}_acc'] = balance_acc
                    self.data[f'{self.track_update}_f1'] = weighted_f1
                    self.data[f'{self.track_update}_recall'] = torch.tensor([recall_dict['recall']])
                    for i in range(len(class_f1)):
                        self.data[f'{self.track_update}_class_acc_{i}'] = class_acc[i].unsqueeze(0)
                        self.data[f'{self.track_update}_class_mcc_{i}'] = class_mcc[i].unsqueeze(0)
                        self.data[f'{self.track_update}_class_f1_{i}'] = class_f1[i].unsqueeze(0)
                        self.data[f'{self.track_update}_class_recall_{i}'] = torch.tensor([recall_dict[f'class_recall_{i}']])

                else:
                    self.data[f'{self.track_update}_r2'] = variance_weighted_r2.unsqueeze(0)

            else:
                if _loss >0:
                    # name = f"self.data[f'{self.track_update}_loss']"
                    # print(f"{name} {self.data[f'{self.track_update}_loss'].device}")
                    # print(f"_loss {_loss.device}")
                    self.data[f'{self.track_update}_loss'] = torch.cat([self.data[f'{self.track_update}_loss'], _loss])

                if self.params.classify:
                    self.data[f'{self.track_update}_acc'] = torch.cat([self.data[f'{self.track_update}_acc'], balance_acc])
                    self.data[f'{self.track_update}_f1'] = torch.cat([self.data[f'{self.track_update}_f1'], weighted_f1])
                    self.data[f'{self.track_update}_recall'] = torch.cat([self.data[f'{self.track_update}_recall'], torch.tensor([recall_dict['recall']])])
                    for i in range(len(class_f1)):
                        self.data[f'{self.track_update}_class_acc_{i}'] = torch.cat([self.data[f'{self.track_update}_class_acc_{i}'], class_acc[i].unsqueeze(0)])
                        self.data[f'{self.track_update}_class_mcc_{i}'] = torch.cat([self.data[f'{self.track_update}_class_mcc_{i}'], class_mcc[i].unsqueeze(0)])
                        self.data[f'{self.track_update}_class_f1_{i}'] = torch.cat([self.data[f'{self.track_update}_class_f1_{i}'], class_f1[i].unsqueeze(0)])
                        self.data[f'{self.track_update}_class_recall_{i}'] = torch.cat([self.data[f'{self.track_update}_class_recall_{i}'], torch.tensor([recall_dict[f'class_recall_{i}']])])

                else:
                    self.data[f'{self.track_update}_r2'] = torch.cat([self.data[f'{self.track_update}_r2'], variance_weighted_r2.unsqueeze(0)])

        # self.printer.print('update tracker...')
        if 'train' == self.track_update:
            # self.printer.print('update train round')
            # train 结束，指向验证阶段
            self.step_in_epoch = 1

            # 学习率调整记录
            lr_change = torch.tensor(0, device=self.accelerator.device)
            if self.accelerator.is_main_process:
                # self.printer.print('scheduler.step')

                # 记录学习率
                self.data['lr'].append(self.scheduler.optimizer.param_groups[0]["lr"])
                # self.printer.print('append lr')

                # 更新 学习率
                self.scheduler.step(self.data['train_loss'])

                # self.printer.print('step done')
                if self.data['lr'][-1] != self.scheduler.optimizer.param_groups[0]["lr"]:
                    lr_change += 1
            # self.printer.print('step done')

            # 同步学习率
            self.accelerator.wait_for_everyone()
            lr_change = broadcast(lr_change)
            # self.printer.print('lr_change')

            if tpu_available():
                xm.mark_step()

            if lr_change.item() == 1:
                # self.printer.print('broadcast lr')
                cur_lr = torch.tensor(self.scheduler.optimizer.param_groups[0]["lr"], device=self.accelerator.device)

                self.accelerator.wait_for_everyone()
                cur_lr = broadcast(cur_lr)

                # 在其他设备上应用学习率
                # self.printer.print(f'apply not main lr -> {cur_lr}')
                if not self.accelerator.is_main_process:
                    self.scheduler.use_lr(cur_lr)

        if 'val' == self.track_update:
            # val 结束，重置为训练阶段
            # self.printer.print('update val round, step_in_epoch -> 0')
            self.step_in_epoch = 0
            # self.printer.print(f'step_in_epoch :{self.step_in_epoch}')
            self.epoch_count += 1

            # 绘制roc 曲线
            if self.params.classify and self.accelerator.is_main_process:# and not isinstance(self.scheduler, LRFinder):
                pic_file = os.path.join(self.params.root, MODEL_FINAL, f"ROC_curve.png")
                plot_roc_curve(self.temp['_y_true'], self.temp['softmax_predictions'], pic_file)

        if 'val' == self.track_update and not self.need_test:
            need_test_temp = torch.tensor(0, device=self.accelerator.device)
            if self.accelerator.is_main_process:
                # 判断是否需要储存 训练数据
                self.each_epoch_time_cost = (self.cost_hour + self.cur_notebook_cost_hour) / (self.epoch_count if self.epoch_count > 0 else 1)
                free_time = self.run_limit_hour - self.cur_notebook_cost_hour
                if free_time < self.each_epoch_time_cost * 1.1:
                    self.printer.print(f'each_epoch_time_cost:{self.each_epoch_time_cost}h, free_time:{free_time}h, run time out, need test/predict')
                    need_test_temp +=1
            # 同步到其他设备
            self.accelerator.wait_for_everyone() 
            need_test_temp = broadcast(need_test_temp)
            self.need_test = need_test_temp.item() == 1

        self.reset_temp()

    def reset_temp(self):
        # 重置计算变量
        self.track_update = ''
        self.temp = {}

        self.temp['_ids'] = []

        self.temp['_loss'] = None
        self.temp['_num'] = 0
        self.temp['_codes'] = []
        self.temp['_y_true'] = None
        self.temp['_y_pred'] = None

        self.step_count = 0

    def print_state(self):
        self.printer.print(f"------------tracker data------------")
        """
        # 时间统计
        self.begin_time = time.time()
        self.epoch_count = 0
        self.step_count = 0
        # 每个epoch训练中的阶段
        # 0: 训练 1: 验证
        self.step_in_epoch = 0
        self.run_limit_hour = 12 if  num_processes != 8 else 9
        self.need_test = False
        """
        self.printer.print(f'[train state]')
        self.printer.print(f'begin time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.begin_time))}')
        self.printer.print(f'pre cost hour: {self.cost_hour}')
        self.printer.print(f'epoch done: {self.epoch_count}')
        self.printer.print(f'step in epoch: {"train" if self.step_in_epoch == 0 else "val"}')
        self.printer.print(f'step done: {self.step_count}')
        self.printer.print(f'run limit hour: {self.run_limit_hour}')
        self.printer.print(f'need save: {self.need_test}')

        self.printer.print(f'')
        self.printer.print(f'[train temp]')
        for i in self.temp:
            if 'y' in i:
                self.printer.print(f"{i}: {len(self.temp[i]) if not None is self.temp[i] else None}")
            else:
                self.printer.print(f"{i}: {self.temp[i]}")

        self.printer.print(f'')
        self.printer.print(f'[train data]')
        for i in self.data:
            self.printer.print(f"{i}: {self.data[i]}")

        for _code in self.symbol_score:
            for k in self.symbol_score[_code]:
                self.printer.print(f"[symbol_score] {_code} {k}: {self.symbol_score[_code][k]}")

        self.printer.print(f"------------tracker data------------")

    def record(self, key, value):
        """记录每个epoch的数据"""
        if key not in self.data:
            self.data[key] = []
        
        if isinstance(value, torch.Tensor):
            value = value.cpu().item()

        self.data[key].append(value)

    def track(self, _type, output, data, target, loss=None):
        # assert _type in ['train', 'val', 'test'], f'error: _type({_type}) should in [train, val, test]'
        # self.printer.print(self.temp[f'{_type}_y_true'], main=False)
        # self.printer.print(self.temp[f'{_type}_y_pred'], main=False)
        self.track_update = _type

        # epoch内的迭代次数
        self.step_count += 1

        # TODO 区分 2分类 与 多分类
        if self.params.classify:
            predict = F.softmax(output, dim=1)
        else:
            predict = output

        # 模型 output 输出，用于模型融合训练
        if self.data_id_getter_func:
            all_ids = self.data_id_getter_func(data)
            if self.track_update in TYPES_NEED_OUTPUT and self.params.need_meta_output:
                # 按日期分类输出数据
                for i in range(predict.shape[0]):
                    self.output_datas.append((all_ids[i], target[i], predict[i]))

        # 汇总所有设备上的数据
        # self.printer.print('sync track...')
        self.accelerator.wait_for_everyone()
        
        if None is loss:
            # 伪造一个loss
            loss = torch.zeros(1, device=self.accelerator.device)
        # self.printer.print(f"loss,{loss.shape}", main=False)
        # self.printer.print(f"target,{target.shape}", main=False)
        # self.printer.print(f"predict,{predict.shape}", main=False)
        _loss, _y_true, _y_pred = self.accelerator.gather_for_metrics((loss, target, predict))
        # self.printer.print(f"gather_for_metrics done", main=False)

        # self.printer.print('gather loss, y_true, y_pred done')
        if _type in TYPES_NO_NEED_SYMBOL_SCORE:
            # train/val 不需要, 避免浪费计算
            pass
        elif _type in TYPES_NEED_OUTPUT and self.data_id_getter_func:
            _ids = gather_object(all_ids)
        else:
            pass

        # self.printer.print('_ids done', main=False)

        # 记录label分布
        # test_best / test_dummy 不需要记录
        if self.params.classify and self.accelerator.is_main_process and _type not in self.label_count_done and _type in TYPES_NEED_LABEL_COUNT:
            # debug('统计 label_counts')
            if _type not in self.label_counts:
                self.label_counts[_type] = torch.bincount(_y_true, minlength=self.params.y_n)
            else:
                self.label_counts[_type] += torch.bincount(_y_true, minlength=self.params.y_n)
            # debug('统计 label_counts done')

        if len(_loss.shape) == 0:
            _loss = _loss.unsqueeze(0)

        # self.printer.print('main cal track...')
        if self.accelerator.is_main_process:
            if None is self.temp['_y_true']:
                self.temp['_y_true'] = _y_true
                self.temp['_y_pred'] = _y_pred
                self.temp['_loss'] = _loss
            else:
                self.temp['_y_true'] = torch.cat([self.temp['_y_true'], _y_true])
                self.temp['_y_pred'] = torch.cat([self.temp['_y_pred'], _y_pred])
                self.temp['_loss'] = torch.cat([self.temp['_loss'], _loss])
            # self.printer.print('temp data done')

            if _type in TYPES_NEED_OUTPUT and self.data_id_getter_func:
                self.temp['_ids'] += _ids

            # self.temp['_codes'] += codes
            self.temp['_num'] += _y_true.shape[0]

        # self.printer.print(f"track done", main=False)

    def get_mean_socre_important(self):
        if self.accelerator.is_main_process:
            if self.params.classify:
                return self.data[f'val_loss'].cpu().numpy().tolist(), min
                return self.data[f'val_f1'].cpu().numpy().tolist(), max
            else:
                # 返回r2列表
                return self.data[f'val_r2'].cpu().numpy().tolist(), max

        else:
            return [], None

    def save_result(self):
        self._plot()
        self._save_result()

    def _plot(self):
        if self.accelerator.is_main_process:
            params = self.params

            # 总耗时
            cost_hour = self.cost_hour + self.cur_notebook_cost_hour

            # x 数量
            epochs = self.params.epochs

            # 标准化数据，nan补气数据
            data = {}
            for i in self.data:
                data[i] = [] if None is self.data[i] else [self.data[i]] if isinstance(self.data[i], (float, int, str)) else copy.deepcopy(self.data[i]) if isinstance(self.data[i], list) else self.data[i].cpu().tolist()

                print(f"{i}: {data[i]}")

                if 'test' in i:
                    data[i] = [data[i][-1]] * epochs if len(data[i]) else []
                else:
                    data[i] = data[i] + (epochs - len(data[i])) * [np.nan]

            # 更改键名 test_final -> test
            # 选择 TEST_FINAL 进行绘图
            old_keys = [i for i in data if TEST_FINAL in i]
            new_keys = [i.replace(TEST_FINAL, 'test') for i in old_keys]
            print(f'old_keys:\n{old_keys}')
            print(f'new_keys:\n{new_keys}')
            for old_key, new_key in zip(old_keys, new_keys):
                data[new_key] = data.pop(old_key)

            # print(data)
            # 将 data 作为 csv 全量保存
            df = move_numbered_columns_to_end(pd.DataFrame(data))
            # 删除以 mean_class_f1 结尾的列名
            df = df.drop([col for col in df.columns if col.endswith('mean_class_f1')], axis=1)
            df.to_csv(os.path.join(params.root, f'all_data.csv'), index=False)

            # 创建图形和坐标轴
            fig, axs = None, None
            ax1 = None
            if params.classify:
                # 分类模型
                fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [7, 3]})
            else:
                fig, axs = plt.subplots(figsize=(15, 10))
                axs = [axs]# 转成可迭代对象，统一管理
            
            # 绘制最佳模型指示线
            for _ax in axs:
                _ax.axvline(x=self.best_model_epoch, color='r', linewidth=5, alpha=0.3)
            
            ax1 = axs[0]

            # 用于添加图例
            ax1_handles = []

            # 测试集损失
            if not isinstance(self.scheduler, LRFinder) and 'test_loss' in data:
                ax1_handles.append(ax1.plot(list(range(epochs)), data['test_loss'], label=f"test loss {last_value(data['test_loss']):.4f}", c='b', linestyle='--')[0])
            # 绘制loss曲线
            ax1_handles.append(ax1.plot(list(range(epochs)), data['train_loss'], label=f"train loss {last_value(data['train_loss']):.4f}", c='#7070FF')[0])
            if not isinstance(self.scheduler, LRFinder):
                # 计算误差最低点
                min_train_loss = min(data['train_loss'])
                min_test_loss = min(data['val_loss'])
                min_train_x = data['train_loss'].index(min_train_loss)
                min_test_x = data['val_loss'].index(min_test_loss)

                ax1_handles.append(ax1.plot(list(range(epochs)), data['val_loss'], label=f"validation loss {last_value(data['val_loss']):.4f}", c='b')[0])
                # 标记损失最低点
                ax1_handles.append(ax1.scatter(min_train_x, min_train_loss, c='#7070FF',label=f'train loss min: {min_train_loss:.4f}'))
                ax1_handles.append(ax1.scatter(min_test_x, min_test_loss, c='b',label=f'validation loss min: {min_test_loss:.4f}'))
                # self.printer.print(f'plot loss')
            else:
                # 标记向下最大斜率点
                down_max_idx = max_downward_slope(data['train_loss'])
                ax1_handles.append(ax1.scatter(down_max_idx, data['train_loss'][down_max_idx], c='#7070FF',label=f'max downward slope lr: {data["lr"][down_max_idx]:.3e}'))
                # 标记最小损失处的学习率
                min_train_loss = min(data['train_loss'])
                min_train_x = data['train_loss'].index(min_train_loss)
                min_train_loss_lr = data['lr'][min_train_x]
                ax1_handles.append(ax1.scatter(min_train_x, min_train_loss, c='#7070FF',label=f'train loss min: {min_train_loss:.4f}, lr: {min_train_loss_lr:.3e}'))

            if not isinstance(self.scheduler, LRFinder):
                if params.classify:
                    # 分类模型
                    # 计算acc最高点
                    max_train_acc = max(data['train_acc'])
                    max_test_acc = max(data['val_acc'])
                    max_train_acc_x = data['train_acc'].index(max_train_acc)
                    max_test_acc_x = data['val_acc'].index(max_test_acc)
                    # 测试集准确率
                    if 'test_acc' in data:
                        ax1_handles.append(ax1.plot(list(range(epochs)), data['test_acc'], label=f"test acc {last_value(data['test_acc']):.4f}", c='r', linestyle='--')[0]) 
                    # 绘制acc曲线
                    ax1_handles.append(ax1.plot(list(range(epochs)), data['train_acc'], label=f"train acc {last_value(data['train_acc']):.4f}", c='#FF7E7E')[0])
                    ax1_handles.append(ax1.plot(list(range(epochs)), data['val_acc'], label=f"validation acc {last_value(data['val_acc']):.4f}", c='r')[0])
                    # 标记准确率最高点
                    ax1_handles.append(ax1.scatter(max_train_acc_x, max_train_acc, c='#FF7E7E',label=f'train acc max: {max_train_acc:.4f}'))
                    ax1_handles.append(ax1.scatter(max_test_acc_x, max_test_acc, c='r',label=f'validation acc max: {max_test_acc:.4f}'))
                    # self.printer.print(f'plot acc')

                else:
                    # 回归模型
                    # 计算r2最高点
                    max_train_r2 = max(data['train_r2'])
                    max_test_r2 = max(data['val_r2'])
                    max_train_r2_x = data['train_r2'].index(max_train_r2)
                    max_test_r2_x = data['val_r2'].index(max_test_r2)
                    # 测试集r2
                    if 'test_r2' in data:
                        ax1_handles.append(ax1.plot(list(range(epochs)), data['test_r2'], label=f"test r2 {last_value(data['test_r2']):.4f}", c='r', linestyle='--')[0])
                    # 绘制r2曲线
                    ax1_handles.append(ax1.plot(list(range(epochs)), data['train_r2'], label=f"train r2 {last_value(data['train_r2']):.4f}", c='#FF7E7E')[0])
                    ax1_handles.append(ax1.plot(list(range(epochs)), data['val_r2'], label=f"validation r2 {last_value(data['val_r2']):.4f}", c='r')[0])
                    # 标记r2最高点
                    ax1_handles.append(ax1.scatter(max_train_r2_x, max_train_r2, c='#FF7E7E',label=f'train r2 max: {max_train_r2:.4f}'))
                    ax1_handles.append(ax1.scatter(max_test_r2_x, max_test_r2, c='r',label=f'validation r2 max: {max_test_r2:.4f}'))
                    # self.printer.print(f'plot r2')

            # 创建右侧坐标轴
            ax2 = ax1.twinx()

            # 绘制学习率
            line_lr, = ax2.plot(list(range(epochs)), data['lr'], label='lr', c='#87CEFF',linewidth=2,alpha =0.5)
            # self.printer.print(f'plot lr')

            # 添加图例
            ax1.legend(handles=ax1_handles)

            # 启用次刻度
            ax1.minorticks_on()

            # 启用网格线，并设置淡显的风格
            ax1.grid(True, which='both', alpha=0.3)
            ax1.set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值

            # 图2
            if params.classify:
                # 分类模型
                t2_handles = []

                # 计算f1最高点
                max_train_f1 = max(data["train_f1"])
                max_test_f1 = max(data["val_f1"])
                max_train_class_f1s = []
                max_val_class_f1s = []
                for i in range(min(3, params.y_n)):
                    max_train_class_f1 = max(data[f"train_class_f1_{i}"])
                    max_val_class_f1 = max(data[f"val_class_f1_{i}"])
                    max_train_class_f1s.append(max_train_class_f1)
                    max_val_class_f1s.append(max_val_class_f1)

                max_train_f1_x = data["train_f1"].index(max_train_f1)
                max_test_f1_x = data["val_f1"].index(max_test_f1)
                max_train_class_f1_xs = []
                max_val_class_f1_xs = []
                for i in range(min(3, params.y_n)):
                    max_train_class_f1_x = data[f"train_class_f1_{i}"].index(max_train_class_f1s[i])
                    max_val_class_f1_x = data[f"val_class_f1_{i}"].index(max_val_class_f1s[i])
                    max_train_class_f1_xs.append(max_train_class_f1_x)
                    max_val_class_f1_xs.append(max_val_class_f1_x)

                colors = [
                    ('#57C838', '#8DE874'),  # 均衡 f1 (green)
                    ('#fc5454', '#feb9b9'),  # 类别 0 (red)
                    ('#b03ef9', '#dba5fd'),  # 类别 1 (purple)
                    ('#1E90FF', '#ADD8E6'),  # 类别 2 (blue)
                    ('#FFA500', '#FFDAB9'),  # 类别 3 (orange)
                    ('#32CD32', '#98FB98'),  # 类别 4 (lime green)
                    ('#FF69B4', '#FFB6C1'),  # 类别 5 (pink)
                    ('#00CED1', '#AFEEEE'),  # 类别 6 (turquoise)
                    ('#FFD700', '#FFFACD'),  # 类别 7 (yellow)
                    ('#6A5ACD', '#B0C4DE'),  # 类别 8 (slate blue)
                    ('#FF4500', '#FFA07A')   # 类别 9 (orange-red)
                ]

                # 测试集f1
                if 'test_f1' in data:
                    t2_handles.append(axs[1].plot(list(range(epochs)), data["test_f1"], label=f'test f1 {last_value(data["test_f1"]):.4f}', c=colors[0][0], linestyle='--')[0])
                    for i in range(min(3, params.y_n)):
                        t2_handles.append(axs[1].plot(list(range(epochs)), data[f"test_class_f1_{i}"], label=f'test class {i} f1 {last_value(data[f"test_class_f1_{i}"]):.4f}', c=colors[i+1][0], linestyle='--')[0])

                # 绘制f1曲线
                t2_handles.append(axs[1].plot(list(range(epochs)), data["train_f1"], c=colors[0][1])[0])
                t2_handles.append(axs[1].plot(list(range(epochs)), data["val_f1"], label=f'val f1 {last_value(data["val_f1"]):.4f} ({last_value(data["train_f1"]):.4f})', c=colors[0][0])[0])
                for i in range(min(3, params.y_n)):
                    t2_handles.append(axs[1].plot(list(range(epochs)), data[f"train_class_f1_{i}"], c=colors[i+1][1])[0])
                    t2_handles.append(axs[1].plot(list(range(epochs)), data[f"val_class_f1_{i}"], label=f'val class {i} f1 {last_value(data[f"val_class_f1_{i}"]):.4f} ({last_value(data[f"train_class_f1_{i}"]):.4f})', c=colors[i+1][0])[0])

                # 标记f1最高点
                t2_handles.append(axs[1].scatter(max_train_f1_x, max_train_f1, c=colors[0][1]))
                t2_handles.append(axs[1].scatter(max_test_f1_x, max_test_f1, c=colors[0][0],label=f'val f1 max: {max_test_f1:.4f} ({max_train_f1:.4f})'))
                for i in range(min(3, params.y_n)):
                    t2_handles.append(axs[1].scatter(max_train_class_f1_xs[i], max_train_class_f1s[i], c=colors[i+1][1]))
                    t2_handles.append(axs[1].scatter(max_val_class_f1_xs[i], max_val_class_f1s[i], c=colors[i+1][0],label=f'val class {i} f1 max: {max_val_class_f1s[i]:.4f} ({max_train_class_f1s[i]:.4f})'))

                # 启用次刻度
                axs[1].minorticks_on()

                # 启用网格线，并设置淡显的风格
                axs[1].grid(True, which='both', alpha=0.3)
                # 取消y轴的次刻度
                axs[1].yaxis.set_minor_locator(plt.NullLocator())
                axs[1].set_xlim(-1, epochs+1)  # 设置 x 轴显示范围从 0 开始到最大值
                axs[1].legend(handles=t2_handles)
                # self.printer.print(f'plot f1 score')

            title = f'{params.train_title}'
            title+= f' | {datetime.now().strftime("%Y%m%d")}              cost:{cost_hour:.2f}H'
            if self.each_epoch_time_cost:
                # 单epoch耗时, 预计等待时间, 下次重启 北京时间
                next_restart_time = datetime.fromtimestamp(self.notebook_begin_time + self.run_limit_hour * 3600) + timedelta(hours=8)
                title += f'({self.each_epoch_time_cost:.2f}h/e, wait {(self.each_epoch_time_cost*(self.params.epochs - self.epoch_count)):.2f}h, next restart: {str(next_restart_time)[:16]})'

            plt.title(title)

            pic_file = os.path.join(params.root, f"{params.train_title}.png")
            plt.savefig(pic_file)
            # self.printer.print(f'plot done: {pic_file}')

            # 绘制对比柱状图 train/val/best/final/dummy
            # 'test_final_loss': 0.9020195007324219,
            # 'test_final_acc': 0.6873999834060669,
            # 'test_final_f1': 0.5510613322257996,
            # 'test_final_class_f1_0': 0.2587284445762634,
            # 'test_final_class_f1_1': 0.2538251578807831,
            score_data = {}
            data_lack = False
            for _type in ['train', 'val', 'test_best', 'test_final', 'test_dummy']:
                if f'{_type}_loss' not in self.data:
                    data_lack = True
                    break

                score_data[f'{_type}_loss'] = self.data[f'{_type}_loss'][-1].cpu().item()

                if self.params.classify:
                    if f'{_type}_acc' not in self.data or f'{_type}_f1' not in self.data:
                        data_lack = True
                        break

                    score_data[f'{_type}_acc'] = self.data[f'{_type}_acc'][-1].cpu().item()
                    score_data[f'{_type}_f1'] = self.data[f'{_type}_f1'][-1].cpu().item()

                    for i in range(min(3, params.y_n)):
                        if f'{_type}_class_f1_{i}' not in self.data:
                            data_lack = True
                            break
                        score_data[f'{_type}_class_f1_{i}'] = self.data[f'{_type}_class_f1_{i}'][-1].cpu().item()

                    self.data[f'{_type}_mean_class_f1'] = sum([score_data[f'{_type}_class_f1_{i}'] for i in range(min(3, params.y_n))]) / (min(3, params.y_n))
                else:
                    if f'{_type}_r2' not in self.data:
                        data_lack = True
                        break

                    score_data[f'{_type}_r2'] = self.data[f'{_type}_r2'][-1].cpu().item()

            if not data_lack:
                # 计算增强说明文字
                tag_texts = {}
                for _type in ['train', 'val', 'test_best', 'test_final']:
                    if self.params.classify:
                        self.data[f'{_type}_mean_class_f1_enhanced_pct'] = 100 * (self.data[f'{_type}_mean_class_f1'] - self.data['test_dummy_mean_class_f1']) / self.data['test_dummy_mean_class_f1']
                        tag_texts[_type] = f"{self.data[f'{_type}_mean_class_f1']:.2f}({self.data[f'{_type}_mean_class_f1_enhanced_pct']:.2f}%)"
                    else:
                        self.data[f'{_type}_r2_enhanced_pct'] = 100 * (score_data[f'{_type}_r2'] - self.data['test_dummy_r2']) / self.data['test_dummy_r2']
                        tag_texts[_type] = f"{score_data[f'{_type}_r2']:.2f}({self.data[f'{_type}_r2_enhanced_pct'].cpu().item():.2f}%)"
                
                # loss分组
                loss_score_data = [score_data[i] for i in score_data if 'loss' in i]
                if self.params.classify:
                    tag_texts['test_dummy'] = f"{self.data['test_dummy_mean_class_f1']:.2f}"

                    # 按照不同指标分组
                    acc_score_data = [score_data[i] for i in score_data if 'acc' in i]
                    f1_score_data = [score_data[i] for i in score_data if 'f1' in i and 'class' not in i]
                    
                    class_f1_score_datas = []
                    for j in range(min(3, params.y_n)):
                        class_f1_score_datas.append([score_data[i] for i in score_data if f'class_f1_{j}' in i])

                else:
                    tag_texts['test_dummy'] = f"{self.data['test_dummy_r2'].cpu().item():.2f}"
                    # 按照不同指标分组
                    r2_score_data = [score_data[i] for i in score_data if 'r2' in i]

                labels = ['Train', 'Val', 'Best', 'Final', 'Dummy']

                # 自定义颜色
                colors = [
                    '#4B1C62', '#7B618B', '#B1AABF', '#EBE8EC', '#F46537',
                    '#2E8B57', '#DAA520', '#483D8B', '#FF8C69', '#E6E6FA',
                    '#5A2A7A', '#8A70A0', '#C0B8D4', '#F2F0F5', '#F5774A',
                    '#3A9C68', '#E1B036', '#554A9C', '#FF9A7A', '#ECEBFF',
                    '#68288E', '#987FB5', '#CCC4E0', '#F8F6FA', '#F88A5E',
                    '#46AD79', '#E8BB4C', '#6257AD', '#FFA88B', '#F2F0FF',
                    '#7636A2', '#A68ECA', '#D8D0EC', '#FDFCFF', '#FB9C72',
                    '#52BE8A', '#EFC662', '#6F64BE', '#FFB69C', '#F8F5FF',
                    '#8344B6', '#B49DDF', '#E4DCF8', '#FFFFFF', '#FDAE86',
                    '#5ECF9B', '#F6D178', '#7C71CF', '#FFC4AD', '#FEFAFF'
                ]

                # 绘制柱状图
                fig, ax = plt.subplots()
                width = 0.15
                alpha = 0.6

                bar1 = ax.bar(labels, loss_score_data, width, color=colors[0], label='Loss', alpha=alpha)
                text_bar = None
                if self.params.classify:
                    bar2 = ax.bar([i + width for i in range(5)], acc_score_data, width, color=colors[1], label='Accuracy', alpha=alpha)
                    bar3 = ax.bar([i + 2*width for i in range(5)], f1_score_data, width, color=colors[2], label='F1', alpha=alpha)

                    bars = []
                    for idx, class_f1_score_data in enumerate(class_f1_score_datas):
                        bars.append(ax.bar([i + (3 + idx)*width for i in range(5)], class_f1_score_data, width, color=colors[3+idx], label=f'Class F1 ({idx})', alpha=alpha))
                
                    text_bar = bars[0]
                else:
                    bar2 = ax.bar([i + width for i in range(5)], r2_score_data, width, color=colors[1], label='R2', alpha=alpha)
                    text_bar = bar2
                
                # 添加增强说明文字
                tag_type = ['train', 'val', 'test_best', 'test_final', 'test_dummy']
                for i, bar in enumerate(text_bar):
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval, tag_texts[tag_type[i]], ha='center', va='bottom')

                ax.set_ylabel('Scores')
                ax.set_title('Scores by group')
                ax.legend()

                pic_file = os.path.join(params.root, f"Scores_by_group.png")
                plt.savefig(pic_file)

        self.accelerator.wait_for_everyone()
        # debug('plot done')

    def _format_value(self, value):
        """格式化值以写入CSV"""
        if isinstance(value, (list, tuple)):
            return "@".join([str(i) for i in value])
        elif isinstance(value, float):
            return f"{value:.6f}"
        elif isinstance(value, torch.Tensor):
            return f"{value.item():.6f}" if value.numel() == 1 else "@".join([str(i) for i in value.tolist()])
        elif ',' in str(value):
            return str(value).replace(',', '_')
        return str(value)

    def _save_result(self):
        """将结果写入CSV文件"""
        if self.accelerator.is_main_process:
            ## 记录结果
            result_file = os.path.join(self.params.root, 'result.csv')

            # # 初始化列名
            # with open(result_file, 'w') as f:
            #     # 训练参数
            #     for key in self.params.__dict__:
            #         if key != 'y_func':
            #             f.write(f'{key},')
            #     # 数据标签分布
            #     for i in self.label_counts:
            #         f.write(f'label_{i},')
            #     # 模型
            #     f.write('model,')
            #     # 训练结果
            #     for i in self.data:
            #         if i == 'lr':
            #             continue
            #         f.write(f'{i},')
            #     f.write('each_epoch_cost,cost\n')

            # def write_values(f, values):
            #     if isinstance(values, list) or isinstance(values, tuple):
            #         f.write(f'{"@".join([str(i) for i in values])},')

            #     elif ',' in str(values):
            #         f.write(f'{values},'.replace(',', '_'))
                    
            #     else:
            #         f.write(f'{values},')

            # # 写入结果
            # with open(result_file, 'a') as f:
            #     # 训练参数
            #     for key in self.params.__dict__:
            #         if key != 'y_func':
            #             write_values(f, self.params.__dict__[key])

            #     # 数据标签分布
            #     for i in self.label_counts:
            #         # debug(self.label_counts[i])
            #         label_pct = (self.label_counts[i] / self.label_counts[i].sum()) * 100
            #         label_pct /= torch.min(label_pct)
            #         label_counts = self.label_counts[i].to('cpu').tolist()
            #         strs = [f'{int(i)}' for i in label_pct.to('cpu').tolist()]
            #         strs = [f'{strs[i]}({label_counts[i]})' for i in range(len(strs))]
            #         write_values(f, strs)
            #     # 模型
            #     f.write(f'{self.model_name},')
            #     # 训练结果
            #     # 选择val_loss 最小的点
            #     best_idx = torch.where(self.data['val_loss'] == min(self.data['val_loss']))[0]
            #     if best_idx.shape[0] > 1:
            #         best_idx = best_idx[-1]
            #     # log(f'loss {self.data["val_loss"]}')
            #     # log(f'min {min(self.data["val_loss"])}')
            #     # log(f'min {min(self.data["val_loss"]).shape}')
            #     # log(f'best_idx {best_idx.shape}')
            #     # log(f'best_idx {best_idx}')
            #     for i in self.data:
            #         print(f'{i} {type(self.data[i])}')
            #         print(f'{self.data[i]}')
            #         if i == 'lr':
            #             continue
            #         if not None is self.data[i]:
            #             if (isinstance(self.data[i], (list, tuple, torch.Tensor))):
            #                 if len(self.data[i]) >= best_idx+1:
            #                     d = self.data[i][best_idx]
            #                 else:
            #                     d = self.data[i][-1]
            #             else:
            #                 d = self.data[i]

            #             if isinstance(d, torch.Tensor):
            #                 d = d.item()

            #             if isinstance(d, float):
            #                 f.write(f'{d:.4f},')
            #             else:
            #                 f.write(f'{d},')
            #         else:
            #             f.write(f',')
            #     f.write(f"{self.each_epoch_time_cost:.2f}h,{(self.cost_hour + self.cur_notebook_cost_hour):.2f}h\n")
        
            # 初始化结果字典
            result_dict = {}

            # 训练参数
            for key in self.params.__dict__:
                if key != 'y_func':
                    result_dict[key] = self._format_value(self.params.__dict__[key])

            # 数据标签分布
            for i in self.label_counts:
                label_pct = (self.label_counts[i] / self.label_counts[i].sum()) * 100
                label_pct /= torch.min(label_pct)
                label_counts = self.label_counts[i].to('cpu').tolist()
                strs = [f'{int(pct)}({count})' for pct, count in zip(label_pct.to('cpu').tolist(), label_counts)]
                result_dict[f'label_{i}'] = self._format_value(strs)

            # 模型名称
            result_dict['model'] = self.model_name

            # 训练结果（选择val_loss最小的点）
            best_idx = torch.where(self.data['val_loss'] == min(self.data['val_loss']))[0]
            if best_idx.shape[0] > 1:
                best_idx = best_idx[-1]

            for key in self.data:
                if key == 'lr':
                    continue
                if self.data[key] is not None:
                    if isinstance(self.data[key], (list, tuple, torch.Tensor)):
                        data_len = len(self.data[key]) if isinstance(self.data[key], (list, tuple)) else self.data[key].shape[0]
                        if 'test_' not in key and data_len > best_idx:
                            result_dict[key + '_best'] = self._format_value(self.data[key][best_idx])
                        result_dict[key] = self._format_value(self.data[key][-1])
                    else:
                        result_dict[key] = self._format_value(self.data[key])
                else:
                    result_dict[key] = ''

            # 时间成本
            result_dict['each_epoch_cost'] = f"{self.each_epoch_time_cost:.2f}h"
            result_dict['cost'] = f"{(self.cost_hour + self.cur_notebook_cost_hour):.2f}h"

            # 写入CSV
            with open(result_file, 'w') as f:
                f.write(','.join(result_dict.keys()) + '\n')
                f.write(','.join(str(result_dict[key]) for key in result_dict) + '\n')
        
        self.accelerator.wait_for_everyone()
        # debug('save_result done')

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in self.no_need_save_parm}

    def load_state_dict(self, state_dict):
        for i in self.no_need_save_parm:
            if i in state_dict: 
                del state_dict[i]
        self.__dict__.update(state_dict)

        # 延续训练 更新耗时记录
        # cur_notebook_cost_hour -> cost_hour
        self.cost_hour += self.cur_notebook_cost_hour
        self.cur_notebook_cost_hour = 0

        if '_ids' not in self.temp:
            self.temp['_ids'] = []

        for i in self.label_counts:
            if isinstance(self.label_counts[i], torch.Tensor):
                self.label_counts[i] = self.label_counts[i].to(self.accelerator.device)
        
        for i in self.temp:
            if isinstance(self.temp[i], torch.Tensor):
                self.temp[i] = self.temp[i].to(self.accelerator.device)

        for i in self.data:
            if isinstance(self.data[i], torch.Tensor):
                self.data[i] = self.data[i]

if __name__ == '__main__':
    import torch
    from accelerate import Accelerator
    import multiprocessing as mp

    from dl_helper.trainer import printer
    from dl_helper.scheduler import ReduceLR_slow_loss

    class p:
        classify = True
        # classify = False
        epochs = 20
        train_title = 'test_title'
        describe = 'test_describe'
        root = './'

    def random_classify_data():
        output = torch.randn(10, 3)
        target = torch.randint(0, 3, (10,))
        loss = torch.nn.CrossEntropyLoss()(output, target)
        return output, target, loss
    
    def random_regress_data():
        output = torch.randn(10, 1)
        target = torch.randn(10, 1)
        loss = torch.nn.MSELoss()(output, target)
        return output, target, loss
    
    state_dict = None

    params = p()
    accelerator = Accelerator()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = ReduceLR_slow_loss(optimizer)
    lock = mp.Manager().Lock()
    p = printer(lock, accelerator)
    t = Tracker(params, accelerator, scheduler, 1, p)
    for i in range(15):
        for _type in ['train', 'val']:
            for j in range(10):
                if params.classify:
                    output, target, loss = random_classify_data()
                else:
                    output, target, loss = random_regress_data()
                t.track(output, target, loss, _type)
        
        if i%10 == 0:
            state_dict = t.state_dict()

        t.update()
        
    for j in range(10):
        if params.classify:
            output, target, loss = random_classify_data()
        else:
            output, target, loss = random_regress_data()
        t.track(output, target, loss, 'test')

    t.update()
    t.plot()

    t.load_state_dict(state_dict)
    t.update()