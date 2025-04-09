import os
import pickle

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


def compute_energy(Q, params, depth = 3, num = 3, factor = 1000):
    """QAOA 计算能量"""
    n = num*num
    qubits = [cirq.GridQubit(0, i) for i in range(0, num)]
    rep = 1000
    A = 0.001
    def xy_initialization(qubits):
        block_size = int(np.sqrt(len(qubits)))
        # 遍历每个块
        for i in range(block_size):
            yield cirq.X.on(qubits[i * block_size + i])

    def coef1(i, j):
        a = 1 / 2 * Q[i][j]
        return a

    def coef2(n, i):
        a = 0
        for j in range(n):
            a += Q[i][j]
        return a
    # print(params)
    circuit = cirq.Circuit()
    # circuit.append(initialization(qubits))
    qubits_init = [cirq.GridQubit(0, i) for i in range(0, num)]
    circuit.append(xy_initialization(qubits_init))
    for d in range(0, depth):

        for i in range(0, num - 1):
            for j in range(i + 1, num):
                circuit += cirq.CNOT(qubits[i], qubits[j])
                circuit += cirq.rz(coef1(i, j) * params[int(2 * d)] * factor)(qubits[j])
                circuit += cirq.CNOT(qubits[i], qubits[j])
        for i in range(0, num):
            circuit += cirq.rz(coef2(n, i) * params[int(2 * d)] * factor)(qubits[i])

        # for k in range(0,num):
        #      circuit+=cirq.rx(params[int(2*d+1)]*factor)(qubits[k])

        block_size = int(np.sqrt(num))
        beta = params[int(2 * d + 1)] * factor
        # 组内环形
        for block_start in range(0, num, block_size):
            # 取出当前块的量子比特
            block_qubits = qubits[block_start:block_start + block_size]

            # 每个块内部的环形XY混合器
            for i in range(len(block_qubits) - 1):  # 遍历相邻比特对
                beta = params[int(2 * d + 1)] * factor

                # 添加 X_i X_j 项
                circuit += cirq.XXPowGate(exponent=2 * beta / np.pi)(block_qubits[i], block_qubits[i + 1])

                # 添加 Y_i Y_j 项
                circuit += cirq.YYPowGate(exponent=2 * beta / np.pi)(block_qubits[i], block_qubits[i + 1])
            # 形成环形
            circuit += cirq.XXPowGate(exponent=2 * beta / np.pi)(block_qubits[block_size - 1], block_qubits[0])
            circuit += cirq.YYPowGate(exponent=2 * beta / np.pi)(block_qubits[block_size - 1], block_qubits[0])
        # 组间环形
        for block_start in range(0, num, block_size):
            # 取出当前块的量子比特
            block_qubits = qubits[block_start:block_start + block_size]
            # 块之间的完全图XX和YY门连接
            if block_start + block_size < num:
                next_block_qubits = qubits[block_start + block_size:block_start + 2 * block_size]
                for i in range(len(block_qubits)):
                    # 添加XX门
                    circuit += cirq.XXPowGate(exponent=2 * A * beta / np.pi)(block_qubits[i], next_block_qubits[i])
                    # 添加YY门
                    circuit += cirq.YYPowGate(exponent=2 * A * beta / np.pi)(block_qubits[i], next_block_qubits[i])
        # 形成环形
        last_block_qubits = qubits[num - block_size:num]
        first_block_qubits = qubits[0:block_size]
        for i in range(len(last_block_qubits)):
            # 添加XX门
            circuit += cirq.XXPowGate(exponent=2 * beta / np.pi)(last_block_qubits[i], first_block_qubits[i])
            # 添加YY门
            circuit += cirq.YYPowGate(exponent=2 * beta / np.pi)(last_block_qubits[i], first_block_qubits[i])

    circuit.append(cirq.measure(*qubits, key='x'))
    simulator = cirq.Simulator()
    results = simulator.run(circuit, repetitions=rep)
    results = str(results)[2:].split(", ")
    newresult = []
    for i in range(0, rep):
        hold = []
        for j in range(0, num):
            hold.append(int(results[j][i]))
        newresult.append(hold)
    gr = newresult
    total_cost = 0
    for r in range(0, len(gr)):
        for i in range(n):
            for j in range(n):
                total_cost += gr[r][i] * gr[r][j] * Q[i][j]
    total_cost = float(total_cost) / rep
    return total_cost


def maml_train(model, tasks, inner_steps=5, alpha=0.01, beta=0.001, epochs=50):
    """MAML 训练过程"""
    meta_optim = torch.optim.Adam(model.parameters(), lr=beta)

    for epoch in range(epochs):
        meta_loss = 0

        for Q, theta_star in tasks:
            Q = Q.unsqueeze(0)  # 加入 batch 维度
            theta_initial = model(Q)
            theta = theta_initial.clone().requires_grad_()

            # 内循环（Adaptation）
            for _ in range(inner_steps):
                energy = compute_energy(Q, theta)
                theta = theta - alpha * torch.autograd.grad(energy, theta)[0]

            # 计算元损失
            meta_loss += F.mse_loss(theta, theta_star)  # 计算 \|θ - θ^*\|^2

        # 外循环优化
        meta_optim.zero_grad()
        meta_loss.backward()
        meta_optim.step()

        print(f"Epoch {epoch + 1}, Meta Loss: {meta_loss.item()}")
        torch.save(model.state_dict(), "meta_qaoa_weights.pth")



# 初始化模型
p = 3  # QAOA 层数
model = MetaQAOA(p=p, n=9)

task_data = []
data_dir = "data/"  # 你的数据目录

# 遍历所有 .pkl 文件
for filename in sorted(os.listdir(data_dir)):
    if filename.endswith(".pkl"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "rb") as f:
            Q, theta_star = pickle.load(f)  # 读取每个文件的数据
            task_data.append((Q, theta_star))

print(f"加载了 {len(task_data)} 组任务数据")

print(task_data)
# 训练
#maml_train(model, task_data)



