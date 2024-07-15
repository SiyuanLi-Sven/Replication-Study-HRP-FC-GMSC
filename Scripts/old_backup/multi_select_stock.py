import framework_portfolio_weight_performace as portfolio_weight_performace
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os 


def draw_mst(matrix, k, ax, title, if_draw=False):
    '''
    功能: 绘制FC矩阵的最小生成树. 标记最核心的一个资产, 和最边缘的k个资产.

    matrix: 处理过的相关系数矩阵, 如FC和GMSC
    ax: 绘图位置, plt对象
    title: 绘图标题
    k: 选择的最远的资产的数量
    '''

    # 计算距离矩阵
    distance_matrix = np.sqrt(2 * (1 - matrix.values))
    
    # 创建图
    G = nx.Graph()
    
    # 添加带有权重的边和节点标签
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            weight = np.round(distance_matrix[i][j], 4)
            G.add_edge(matrix.index[i], matrix.index[j], weight=weight)
    
    # 构建最小生成树
    mst = nx.minimum_spanning_tree(G)
    # 获取每个节点的度
    degrees = dict(mst.degree())
    # 找到度最高的节点（最紧密的节点）
    most_connected_node = max(degrees, key=degrees.get)
    # 按度排序节点，选择度最低的k个节点（最稀疏的节点）
    sparsest_nodes = sorted(degrees, key=degrees.get)[:k]

    if if_draw:
        # 使用Kamada-Kawai布局
        pos = nx.kamada_kawai_layout(mst)
        # 绘制所有节点
        nx.draw_networkx_nodes(mst, pos, node_color='lightgrey', node_size=400, ax=ax)
        # 突出显示中心节点
        nx.draw_networkx_nodes(mst, pos, nodelist=[most_connected_node], node_color='red', node_size=600, ax=ax)
        # 突出显示边缘节点
        nx.draw_networkx_nodes(mst, pos, nodelist=sparsest_nodes, node_color='green', node_size=600, ax=ax)
        # 绘制边
        nx.draw_networkx_edges(mst, pos, width=2, ax=ax, edge_color='black')
        # 节点标签
        nx.draw_networkx_labels(mst, pos, ax=ax)
        # 关闭坐标轴
        ax.set_title(title)
        ax.axis('off')  

    return sparsest_nodes


def gen_mst_select(k, window_size, return_data):
    # 创建子目录
    directory = 'selected_stocks'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 取前window_size行数据计算相关性矩阵
    if window_size > len(return_data):
        print(f"Window size {window_size} is greater than available data length.")
        return

    windowed_data = return_data.iloc[:window_size]
    correlation_matrix = windowed_data.corr()

    # 实例化处理对象
    instance = portfolio_weight_performace.CorrProcess(correlation_matrix)
    FC_matrix, GMSC_matrix = instance.FC_GMSC()

    # 对FC和GMSC两种类型分别处理
    for matrix_type, matrix in [('FC', FC_matrix), ('GMSC', GMSC_matrix)]:
        
        # 生成MST并获取最边缘的k个节点
        sparsest_nodes = draw_mst(matrix, k, None, f'{matrix_type} MST')

        # 保存到文本文件
        filename = f'{directory}/k_{k}_w_{window_size}_{matrix_type}.txt'
        with open(filename, 'w') as file:
            file.write(", ".join(sparsest_nodes) + "\n")



def main():
    print("请保证当前工作路径在Scripts文件夹下")
    print("当前工作路径:{}".format(os.getcwd()))

    print('读取\清洗数据(30s内)')
    price_data = portfolio_weight_performace.readWindSP500()
    return_data = price_data.pct_change().dropna()

    # ks和window_sizes的定义
    # ks = [10, 30, 50, 100, 150, 200, 250, 300, 500]
    # window_sizes = [10, 30, 50, 100, 150, 200, 250, 300, 500]
    ks = [10, 30, 50, 100, 150, 200]
    window_sizes = [30, 50, 100, 200, 300, 500]

    # 创建参数组合
    params = [(k, window_size, return_data) for k in ks for window_size in window_sizes]

    # 使用Pool并行执行
    print('开始多线程选股')
    with Pool(processes=8) as pool:
        useless_results = tqdm(pool.starmap(gen_mst_select, params))




if __name__ == '__main__':
    main()
    print("所有文件处理完成。")
