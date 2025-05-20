from torchmetrics.classification import BinaryF1Score
from torchmetrics import F1Score, R2Score

import torch

# 假设 y_pred 是正类的概率，形状为 [batch_size]
# y_true 是二值标签，形状为 [batch_size]
def compute_binary_f1(y_pred, y_true):
    # f1_score = F1Score(num_classes=2, average='weighted', task='multiclass').to(y_pred.device)
    f1_score = F1Score(num_classes=2, average='weighted', task='binary').to(y_pred.device)
    return f1_score(y_pred, y_true).unsqueeze(0)

# 示例数据
y_pred = torch.tensor([0.9, 0.2, 0.8, 0.4])  # 正类的概率
y_true = torch.tensor([1, 0, 1, 0])           # 真实标签
f1 = compute_binary_f1(y_pred, y_true)
print(f1)  # 输出加权 F1 分数