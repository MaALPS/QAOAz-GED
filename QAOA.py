import os
import statistics
import time

import cirq
import numpy as np
import pandas as pd
import random
import pickle
from scipy.optimize import minimize
from util import generate_cost_matrix, draw_graph, generate_topology



node=3
p=0.1
n=node*node
num = n
matrix_penalty=n+1
depth =3
rep = 1000
factor = 1000
def load_graphs_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

# r=1
# [graph1,graph2] = load_graphs_from_pickle(f"dataset/graph_n{node}_p{p}_r{r}.pkl")
#
# Q = generate_cost_matrix(graph1,graph2,penalty=matrix_penalty)

def process(cov_matrix,penalty,node):
    n=m=node
    k=node*node
    linear_coeffs = np.zeros(k)  # 初始化线性系数矩阵c
    constant_term = 2 * penalty * node  # 初始化常数项

    # 提取C部分的二次项系数
    quadratic_coeffs = np.copy(cov_matrix)

    # 提取lambda_1和lambda_2部分的二次项和线性项系数
    for i in range(n):
        for j in range(m):
            linear_coeffs[i * m + j] += -2 * penalty
            for k in range(m):
                quadratic_coeffs[i * m + j][i * m + k] += penalty

    for j in range(m):
        for i in range(n):
            linear_coeffs[j + i * m] += -2 * penalty
            for l in range(n):
                quadratic_coeffs[j + i * m][j + l * m] += penalty

    # 将矩阵变成下三角矩阵
    d = np.copy(np.diagonal(quadratic_coeffs))
    quadratic_coeffs += quadratic_coeffs.T

    quadratic_coeffs -= np.diag(d)

    quadratic_coeffs = np.tril(quadratic_coeffs)

    return quadratic_coeffs, linear_coeffs

def coef1(i,j):
    a = 1/2*quadratic_coeffs[i][j]
    return a
def coef2(n,i):
    a=linear_coeffs[i]
    for j in range(n):
        a += Q[i][j]
    return a

qubits = [cirq.GridQubit(0, i) for i in range(0, num)]


def xy_initialization(qubits):
    block_size = int(np.sqrt(len(qubits)))
    # 遍历每个块
    for i in range(block_size):
        yield cirq.X.on(qubits[i*block_size+i])


def create_circuit(params):
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

        for k in range(0,num):
             circuit+=cirq.rx(params[int(2*d+1)]*factor)(qubits[k])



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

    return newresult

def cost_function(params):
    gr = create_circuit(params)
    total_cost = 0
    for r in range(0, len(gr)):
        for i in range(n):
            for j in range(n):
                total_cost+=gr[r][i]*gr[r][j]*Q[i][j]
    total_cost = float(total_cost) / rep
    return total_cost


def verify(matrix):
    cov = matrix
    a = []
    for i in range(2 ** n):
        bb = bin(i)[2:]
        if len(bb) < n:
            bb = (n - len(bb)) * '0' + bb
        a.append(bb)
    h = []
    for i in a:
        if i.count("1") == node:
            h.append(i)
    b = h

    c = []
    for t in b:
        s1 = 0
        for i in range(n):
            for j in range(n):
                g = list(t)
                g[i] = int(g[i])
                g[j] = int(g[j])
                s1 += g[i] * Q[i][j] * g[j]
        c.append(s1)

    # 1. 找到最小的代价
    min_cost = min(c)

    # 2. 获取所有对应的解
    min_cost_solutions = []
    for idx, cost in enumerate(c):
        if cost == min_cost:
            min_cost_solutions.append(b[idx])

    # 输出最小代价及其对应的解集合
    print("最小代价:", min_cost)
    print("最小代价的解集合:", min_cost_solutions)
    return min_cost, min_cost_solutions


# f[r][i] 表示第 r 个近似解的第 i 个字符串
# min_cost_solutions[i] 表示第 i 个最优解的字符串
def compute_opt_gap(f, min_cost_solutions):
    n = len(f[0])  # 每个解的长度
    k = len(min_cost_solutions)  # 最优解集合的大小
    total_gap = 0.0  # 用于累加所有近似解的最优间隙

    # 遍历所有近似解
    for r in range(len(f)):
        min_gap = float(n)  # 初始化为一个很大的值

        # 对于每个近似解，遍历所有最优解
        for i in range(k):
            # 计算近似解 f[r] 与最优解 min_cost_solutions[i] 之间的 Hamming 距离
            distance = 0.0
            for j in range(n):
                if f[r][j] != int(min_cost_solutions[i][j]):
                    distance += 1
            # 计算 Hamming 距离的比例
            gap = float(distance / n)
            min_gap = min(min_gap, gap)  # 选择最小的间隙

        total_gap += min_gap  # 将每个近似解的最优间隙加到总和中
    # 计算所有近似解的平均最优间隙
    avg_opt_gap = float(total_gap / len(f))
    return avg_opt_gap

def optimize():
    init = [float(random.randint(0, 628)) / float(100) for i in range(0, 2 * depth)]
    # print(init)
    out = minimize(cost_function, x0=init, method="cobyla", options={'maxiter': 800})
    # print(out)
    optimal_params = out['x']
    f = create_circuit(optimal_params)
    # print(f)
    min_cost, min_cost_solutions = verify(Q)
    avg_opt_gap = compute_opt_gap(f, min_cost_solutions)
    # print(f)
    print('avg_opt_gap:', avg_opt_gap)

    with open("output4.txt", "a") as file:  # 打开文件以追加模式
        file.write(f"\nStarting EXP on GED-QAOAz depth={depth} n={node} p={p}>>\n")
        nums = []
        freq = []
        for i in range(0, len(f)):
            number = 0
            for j in range(0, len(f[i])):
                number += 2 ** (len(f[i]) - j - 1) * f[i][j]
            if (number in nums):
                freq[nums.index(number)] = freq[nums.index(number)] + 1
            else:
                nums.append(number)
                freq.append(1)

        freq = [s / sum(freq) for s in freq]

        print('nums:', nums)
        print('freq:', freq)
        file.write(f"avg_opt_gap:{avg_opt_gap}\n")
        file.write(f"nums: {nums}\n")
        file.write(f"freq: {freq}\n")
        freq_b = freq.copy()
        freq.sort(reverse=True)
        t3 = freq[:3]
        result = list(np.where(np.array(freq_b) == t3[0])[0])

        res = []
        for i in result:
            if i not in res:
                res.append(i)

        bb0 = bin(nums[res[0]])[2:]

        if len(bb0) < n:
            bb0 = (n - len(bb0)) * '0' + bb0

        print(bb0)
        print(t3[0])
        # print(bb1)
        # print(t3[1])
        # print(bb2)
        # print(t3[2])
        file.write(f"best_string: {bb0}\n")
        file.write(f"frequency: {t3[0]}\n")
        file.write("<<END EXP\n")
        x = range(0, 2 ** num)
        y = []
        for i in range(0, len(x)):
            if (i in nums):
                y.append(freq_b[nums.index(i)])
                # print(nums.index(i))
            else:
                y.append(0)
        if bb0 in min_cost_solutions:
            acc=1
        else:
            acc=0
        print("acc:",acc)
        file.write(f"acc: {acc}\n")
    return [x, y, bb0, avg_opt_gap, acc]

def evaluate(sols):
    a = sols
    s1 = 0
    for i in range(n):
        s1 = 0
        for j in range(n):
            g = list(a)
            g[i] = int(g[i])
            g[j] = int(g[j])
            s1 += g[i] * Q[i][j] * g[j]

    return s1
    # with open("output_qaoaz.txt", "a") as file:  # 打开文件以追加模式
    #     file.write(f"\nAvg:{sum(c) / len(c)}, \nMax:{max(c)}, \nMin:{min(c)}, \nStd:{std_deviation}\n")


with open("output4.txt", "a") as file:  # 打开文件以追加模式
    file.write("\n---------------------------------------------------------------\n")

# 定义存放pickle文件的目录
dataset_dir = 'dataset'

# 获取dataset目录下所有文件名
file_names = os.listdir(dataset_dir)

# 过滤出所有pickle文件并将其保存到data_list
data_list = [f for f in file_names if f'n{node}_p{p}' in f and f.endswith('.pkl')]

# 打印结果
print(data_list)
avg_opt_gaps = []
best_solutions = []
accs=0.0
run_time = []
for _ in range(len(data_list)):
    with open("output4.txt", "a") as file:  # 打开文件以追加模式
        file.write(f"data:{data_list[_]}:")

    [graph1, graph2] = load_graphs_from_pickle(f"dataset/{data_list[_]}")

    Q = generate_cost_matrix(graph1, graph2, penalty=0)
    quadratic_coeffs, linear_coeffs = process(Q,penalty=matrix_penalty,node=node)
    # %%


    for run in range(1):
        print(f"run {_+1}……")
        start_time = time.time()
        x, y, best_solution, avg_opt_gap, acc = optimize()
        best_solutions.append(best_solution)
        avg_opt_gaps.append(avg_opt_gap)
        run_time.append(time.time() - start_time)
        accs+=acc

    print(best_solutions)

with open("output_qaoa.txt", "a") as file:  # 打开文件以追加模式
    file.write(f"\ndata:{data_list}:\ndepth:{depth}\n")
    file.write(f"avg_opt_gaps:{avg_opt_gaps}\naccs:{float(accs/len(data_list))}\navg_opt_gap:{float(sum(avg_opt_gaps)/len(data_list))}\navg_time:{float(sum(run_time)/len(run_time))}\n")

