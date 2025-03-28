import torch
import torch.nn as nn
import math
from dl_helper.models.tabl import BL_layer, TABL_layer

from dl_helper.train_param import tpu_available
if tpu_available():
  import torch_xla.core.xla_model as xm

class BiN(nn.Module):
  def __init__(self, d1, t1):
        super().__init__()
        self.t1 = t1
        self.d1 = d1

        bias1 = torch.Tensor(t1,1)
        self.B1 = nn.Parameter(bias1)
        nn.init.constant_(self.B1, 0)

        l1 = torch.Tensor(t1,1)
        self.l1 = nn.Parameter(l1)
        nn.init.xavier_normal_(self.l1)     

        bias2 = torch.Tensor(d1,1)
        self.B2 = nn.Parameter(bias2)
        nn.init.constant_(self.B2, 0)

        l2 = torch.Tensor(d1,1)
        self.l2 = nn.Parameter(l2)
        nn.init.xavier_normal_(self.l2)      

        y1 = torch.Tensor(1,)
        self.y1 = nn.Parameter(y1)
        nn.init.constant_(self.y1, 0.5)

        y2 = torch.Tensor(1,)
        self.y2 = nn.Parameter(y2)
        nn.init.constant_(self.y2, 0.5)

  def forward(self, x):
    #if the two scalars are negative then we setting them to 0 
    # if tpu_available():
    #   xm.mark_step()
    # xm.mark_step()
    # if (self.y1[0] < 0): 
    #     y1 = torch.Tensor(1,).to(x.device)
    #     self.y1 = nn.Parameter(y1)
    #     nn.init.constant_(self.y1, 0.01)

    # xm.mark_step()
    # if (self.y2[0] < 0): 
    #     y2 = torch.Tensor(1,).to(x.device)
    #     self.y2 = nn.Parameter(y2)
    #     nn.init.constant_(self.y2, 0.01)

    self.y1.data = torch.where(self.y1.data < 0, torch.tensor([0.01], device=x.device), self.y1.data)
    self.y2.data = torch.where(self.y2.data < 0, torch.tensor([0.01], device=x.device), self.y2.data)

    #normalization along the temporal dimensione 
    T2 = torch.ones([self.t1, 1], dtype=x.dtype).to(x.device)
    x2 = torch.mean(x, axis=2).to(x.device) 
    x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1))

    std = torch.std(x, axis=2)
    std = torch.reshape(std, (std.shape[0], std.shape[1], 1))

    #it can be possible that the std of some temporal slices is 0, and this produces inf values, so we have to set them to one
    # std[std < 1e-4] = 1            
    std = torch.where(std < 1e-4, torch.ones_like(std), std)

    diff = x - (x2@(T2.T))
    Z2 = diff / (std@(T2.T))

    X2 = self.l2 @ T2.T
    X2 = X2 * Z2
    X2 = X2 + (self.B2 @ T2.T)   

    #normalization along the feature dimension
    T1 = torch.ones([self.d1, 1]).to(x.device)
    x1 = torch.mean(x, axis=1) 
    x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1))

    std = torch.std(x, axis=1)
    std = torch.reshape(std, (std.shape[0], std.shape[1], 1))

    op1 = x1@T1.T
    op1 = torch.permute(op1, (0, 2, 1))

    op2 = std@T1.T
    op2 = torch.permute(op2, (0, 2, 1))
    
    z1 = (x - op1) / (op2)
    X1 = (T1 @ self.l1.T)
    X1 = X1 * z1 
    X1 = X1 + (T1 @ self.B1.T)

    #weighing the imporance of temporal and feature normalization
    x = self.y1*X1 + self.y2*X2
    
    return x

class m_bin_btabl(nn.Module):
  @classmethod
  def model_name(cls):
      return "binbtabl"

  def __init__(self, d2, d1, t1, t2, d3, t3):
    super().__init__()

    self.BiN = BiN(d1, t1)
    self.BL = BL_layer(d2, d1, t1, t2)
    self.TABL = TABL_layer(d3, d2, t2, t3)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    #first of all we pass the input to the BiN layer, then we use the B(TABL) architecture
    x = self.BiN(x)

    self.max_norm_(self.BL.W1.data)
    self.max_norm_(self.BL.W2.data)
    x = self.BL(x)
    x = self.dropout(x)

    self.max_norm_(self.TABL.W1.data)
    self.max_norm_(self.TABL.W.data)
    self.max_norm_(self.TABL.W2.data)
    x = self.TABL(x)
    
    # x = torch.squeeze(x)
    x = torch.squeeze(x,dim=2)# 保留batch维度

    # # 不应该在这里 softmax
    # x = torch.softmax(x, 1)

    return x

  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm))

class m_bin_ctabl(nn.Module):
  @classmethod
  def model_name(cls):
      return "binctabl"

  def __init__(self, d2, d1, t1, t2, d3, t3, d4, t4):
    super().__init__()

    self.BiN = BiN(d1, t1)
    self.BL = BL_layer(d2, d1, t1, t2)
    self.BL2 = BL_layer(d3, d2, t2, t3)
    self.TABL = TABL_layer(d4, d3, t3, t4)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    #first of all we pass the input to the BiN layer, then we use the C(TABL) architecture
    x = self.BiN(x)

    # if tpu_available():
    #   xm.mark_step()
    # xm.mark_step()
    # with torch.no_grad():
    #   self.max_norm_(self.BL.W1.data)
    #   self.max_norm_(self.BL.W2.data)
    #   self.max_norm_(self.BL2.W1.data)
    #   self.max_norm_(self.BL2.W2.data)
    #   self.max_norm_(self.TABL.W1.data)
    #   self.max_norm_(self.TABL.W.data)
    #   self.max_norm_(self.TABL.W2.data)

    with torch.no_grad():
      self.max_norm_(self.BL.W1)
      self.max_norm_(self.BL.W2)
      self.max_norm_(self.BL2.W1)
      self.max_norm_(self.BL2.W2)
      self.max_norm_(self.TABL.W1)
      self.max_norm_(self.TABL.W)
      self.max_norm_(self.TABL.W2)

    x = self.BL(x)
    x = self.dropout(x)

    x = self.BL2(x)
    x = self.dropout(x)

    x = self.TABL(x)

    # in: (1, 40, 100)
    # 1 torch.Size([1, 3, 1])
    # 2 torch.Size([3]) -》 torch.Size([1, 3])
    # 
    # in: (2, 40, 100)
    # 1 torch.Size([2, 3, 1])
    # 2 torch.Size([2, 3])
    # x = torch.squeeze(x)
    x = torch.squeeze(x,dim=2)# 保留batch维度

    # # 不应该在这里 softmax
    # x = torch.softmax(x, 1)
    
    return x

  # def max_norm_(self, w):
  #   if (torch.linalg.matrix_norm(w) > 10.0):
  #     norm = torch.linalg.matrix_norm(w)
  #     desired = torch.clamp(norm, min=0.0, max=10.0)
  #     w *= (desired / (1e-8 + norm))   
  def max_norm_(self, p):
    norm = torch.linalg.matrix_norm(p.data)
    desired = torch.clamp(norm, min=0.0, max=10.0)
    p.data.mul_(torch.where(norm > 10.0, desired / (1e-8 + norm), torch.tensor(1., device=p.data.device)) )
  
class m_bin_ctabl_fix_shape(m_bin_ctabl):
  def forward(self, x):
    x = torch.transpose(x, 1, 2)
    return super().forward(x)
  
if __name__ == "__main__":
    
    from torchinfo import summary   
    from thop import profile
    from thop import clever_format

    device = 'cpu'

    # model = m_bin_btabl(120, 40, 10, 5, 3, 1)
    # model = m_bin_ctabl(60, 40, 10, 10, 120, 5, 3, 1)
    model = m_bin_ctabl(60, 40, 100, 40, 120, 12, 3, 1)
    print(model)

    summary(model, (1, 40, 100), device=device)

    model = model.to(device)
    input = torch.randn((1, 40, 100)).to(device)

    # 导出模型为ONNX格式
    onnx_path = "C:/Users/lh/Desktop/temp/bin_ctabl.onnx"
    torch.onnx.export(model, input, onnx_path, do_constant_folding=False,
                input_names=['input'], output_names=['output'])

    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params])
    # print(f"FLOPs: {flops} Params: {params}")

    # out = model(input)
    # print(out.shape)# torch.Size([2, 3])

