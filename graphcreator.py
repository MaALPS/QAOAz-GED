import networkx as nx
import random
import pickle
import os
from matplotlib import pyplot as plt
# 创建保存图文件的目录
if not os.path.exists('dataset'):
    os.makedirs('dataset')
def generate_random_graphs(node_count, p):
    # 生成随机图
    graph1 = nx.fast_gnp_random_graph(node_count, p)
    graph2 = nx.fast_gnp_random_graph(node_count, p)

    # 随机分配节点形状
    for node in graph1.nodes:
        graph1.nodes[node]['shape'] = random.choice(['circle', 'rectangle'])

    for node in graph2.nodes:
        graph2.nodes[node]['shape'] = random.choice(['circle', 'rectangle'])

    # 随机分配边的权重
    for u, v in graph1.edges:
        graph1[u][v]['weight'] = random.randint(1, 3)

    for u, v in graph2.edges:
        graph2[u][v]['weight'] = random.randint(1, 3)

    return [graph1, graph2]

def save_graphs_to_pickle(graphs, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(graphs, f)

def load_graphs_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

def spawn_problem(node, p, rp):
    graphs = generate_random_graphs(node, p)  # Adjust p (probability) as needed
    file_name = f"dataset/graph_n{node}_p{p}_r{rp}.pkl"
    save_graphs_to_pickle(graphs, file_name)
    return file_name

for node in [3,4,5]:
    for p in [0.1,0.33,0.66,0.9]:
        for rp in range(10):
            file_name = spawn_problem(node, p, rp+1)
            print(f"Graphs saved to: {file_name}")


# 读取图
# graphs = load_graphs_from_pickle("dataset/graph_3_p0.5.pkl")
# print(f"Graphs loaded from pickle: {graphs}")

