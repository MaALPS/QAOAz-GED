import cirq
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class MetaQAOA(nn.Module):
    def __init__(self, p, n=9):
        super(MetaQAOA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * n * n, 128)
        self.fc2 = nn.Linear(128, 2 * p)  # 输出 QAOA 变分参数

    def forward(self, Q):
        x = Q.unsqueeze(1)  # 变成 (batch, 1, n, n)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 输出 2p 维参数

# 初始化模型
p = 3  # QAOA 层数
model = MetaQAOA(p=p, n=9)
model.load_state_dict(torch.load("meta_qaoa_weights.pth"))
model.eval()
def test_meta_qaoa(model, Q):
    model.eval()
    with torch.no_grad():
        theta_pred = model(Q.unsqueeze(0))  # 预测 θ
    return theta_pred

# 生成新 QUBO 问题
Q_test = generate_QUBO(n=9)
theta_pred = test_meta_qaoa(model, Q_test)

print("Predicted QAOA Parameters:", theta_pred)