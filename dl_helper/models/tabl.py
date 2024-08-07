import torch
import torch.nn as nn

from dl_helper.train_param import tpu_available
if tpu_available():
  import torch_xla.core.xla_model as xm

class TABL_layer(nn.Module):

    def __init__(self, d2, d1, t1, t2):
        super().__init__()
        self.t1 = t1

        weight = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')
        
        weight2 = torch.Tensor(t1, t1)
        self.W = nn.Parameter(weight2)
        nn.init.constant_(self.W, 1/t1)
 
        weight3 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight3)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        bias1 = torch.Tensor(d2, t2)
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        l = torch.Tensor(1,)
        self.l = nn.Parameter(l)
        nn.init.constant_(self.l, 0.5)

        self.activation = nn.ReLU()

    def forward(self, X):
        
        #maintaining the weight parameter between 0 and 1.
        # if tpu_available():
        #   xm.mark_step()
        # xm.mark_step()
        # if (self.l[0] < 0):
        #   l = torch.Tensor(1,)
        #   self.l = nn.Parameter(l)
        #   nn.init.constant_(self.l, 0.0)
        # xm.mark_step()
        # if (self.l[0] > 1):
        #   l = torch.Tensor(1,)
        #   self.l = nn.Parameter(l)
        #   nn.init.constant_(self.l, 1.0)
        self.l.data = torch.where(self.l[0] < 0, torch.tensor([0.0],device=X.device), self.l.data)
        self.l.data = torch.where(self.l[0] > 1, torch.tensor([1.0],device=X.device), self.l.data)

        #modelling the dependence along the first mode of X while keeping the temporal order intact (7)
        X = self.W1 @ X

        #enforcing constant (1) on the diagonal
        W = self.W -self.W *torch.eye(self.t1,dtype=torch.float32).to(X.device)+torch.eye(self.t1,dtype=torch.float32).to(X.device)/self.t1

        #attention, the aim of the second step is to learn how important the temporal instances are to each other (8)
        E = X @ W

        #computing the attention mask  (9)
        A = torch.softmax(E, dim=-1)

        #applying a soft attention mechanism  (10)
        #he attention mask A obtained from the third step is used to zero out the effect of unimportant elements
        X = self.l[0] * (X) + (1.0 - self.l[0])*X*A

        #the final step of the proposed layer estimates the temporal mapping W2, after the bias shift (11)
        y = X @ self.W2 + self.B
        return y

class BL_layer(nn.Module):
  def __init__(self, d2, d1, t1, t2):
        super().__init__()
        weight1 = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight1)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')

        weight2 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight2)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        bias1 = torch.zeros((d2, t2))
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        self.activation = nn.ReLU()

  def forward(self, x):

    x = self.activation(self.W1 @ x @ self.W2 + self.B)

    return x

class m_btabl(nn.Module):
  @classmethod
  def model_name(cls):
      return "btabl"

  def __init__(self, d2=120, d1=40, t1=10, t2=5, d3=3, t3=1):
    super().__init__()

    self.BL = BL_layer(d2, d1, t1, t2)
    self.TABL = TABL_layer(d3, d2, t2, t3)
    self.dropout = nn.Dropout(0.1)
    
  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm))
        
  def forward(self, x):

    self.max_norm_(self.BL.W1.data)
    self.max_norm_(self.BL.W2.data)
    x = self.BL(x)
    x = self.dropout(x)

    self.max_norm_(self.TABL.W1.data)
    self.max_norm_(self.TABL.W.data)
    self.max_norm_(self.TABL.W2.data)
    x = self.TABL(x)
    x = torch.squeeze(x)
    x = torch.softmax(x, 1)
    return x

class m_ctabl(nn.Module):
  @classmethod
  def model_name(cls):
      return "ctabl"

  def __init__(self, d2=60, d1=40, t1=10, t2=10, d3=120, t3=5, d4=3, t4=1):
    super().__init__()
    
    self.BL = BL_layer(d2, d1, t1, t2)
    self.BL2 = BL_layer(d3, d2, t2, t3)
    self.TABL = TABL_layer(d4, d3, t3, t4)
    self.dropout = nn.Dropout(0.1)

  def max_norm_(self, w):
    with torch.no_grad():
      if (torch.linalg.matrix_norm(w) > 10.0):
        norm = torch.linalg.matrix_norm(w)
        desired = torch.clamp(norm, min=0.0, max=10.0)
        w *= (desired / (1e-8 + norm))
        
  def forward(self, x):
 
    self.max_norm_(self.BL.W1.data)
    self.max_norm_(self.BL.W2.data)
    x = self.BL(x)
    x = self.dropout(x)

    self.max_norm_(self.BL2.W1.data)
    self.max_norm_(self.BL2.W2.data)
    x = self.BL2(x)
    x = self.dropout(x)

    self.max_norm_(self.TABL.W1.data)
    self.max_norm_(self.TABL.W.data)
    self.max_norm_(self.TABL.W2.data)
    x = self.TABL(x)
    x = torch.squeeze(x)
    x = torch.softmax(x, 1)
    return x
    

if __name__ == "__main__":
    
    from torchinfo import summary   
    from thop import profile
    from thop import clever_format

    device = 'cuda'

    #model = m_btabl(120, 40, 10, 5, 3, 1)
    model = m_ctabl(60, 40, 100, 40, 120, 12, 3, 1)
    print(model)

    summary(model, (10, 40, 100), device=device)

    model = model.to(device)
    input = torch.randn((2, 40, 100)).to(device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params])
    print(f"FLOPs: {flops} Params: {params}")

    out = model(input)
    print(out.shape)# torch.Size([2, 3])