from autogluon.core.metrics import make_scorer
from sklearn.metrics import f1_score
import numpy as np


def class_f1_score_sklearn(y_true, y_pred):
    """
    计算每个类别的F1 score
    
    参数:
    y_true: 真实标签,numpy数组或列表
    y_pred: 预测标签,numpy数组或列表
    
    返回:
    numpy数组,包含每个类别的F1 score
    """
    # 确保输入是numpy数组
    if isinstance(y_true, (list, tuple)):
        y_true = np.array(y_true)
    if isinstance(y_pred, (list, tuple)):
        y_pred = np.array(y_pred)
        
    # 如果输入是概率,转换为类别
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # 计算每个类别的F1 score
    class_f1 = f1_score(y_true, y_pred, average=None)
    
    return class_f1

def mean_class_f1(y_true, y_pred):
    class_f1 = class_f1_score_sklearn(y_true, y_pred)
    return (sum(class_f1) - class_f1[-1]) / (len(class_f1) -1)


mean_class_f1_scorer = make_scorer(name='mean_class_f1',
                                 score_func=mean_class_f1,
                                 optimum=1,
                                 greater_is_better=True,
                                 needs_class=True)