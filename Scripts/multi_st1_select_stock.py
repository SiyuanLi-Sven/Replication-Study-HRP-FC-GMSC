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

import warnings
warnings.filterwarnings('ignore')


def draw_mst(matrix, k=10, ax=0, title='', if_draw=False):
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
    all_nodes = sorted(degrees, key=degrees.get)

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

    return sparsest_nodes, all_nodes


def gen_mst_select(window_size, price_data):
    # 创建子目录
    # 初始化一些列表来保存所有的实验结果
    lis_weights_HRP_FC = []
    lis_weights_HRP_GMSC = []

    lis_test_date = []
    lis_FC_ranked = []
    lis_GMSC_ranked = []

    count = 0
    # 迭代DataFrame，创建每个实验期的切片
    for start_day in tqdm(range(len(price_data) - window_size - 1)):

        end_train_day = start_day + window_size

        # 实验期的切片
        experiment_slice = price_data.iloc[start_day:end_train_day]

        # 测试期的切片, 从实验期开始后120
        test_date = end_train_day + 1
        lis_test_date.append(price_data.index[test_date])
        returns_train = experiment_slice.pct_change().dropna()

        # 记录MST顺序
        instance0 = portfolio_weight_performace.CorrProcess(returns_train)
        FC_matrix , GMSC_matrix = instance0.FC_GMSC()

        _, all_nodes_FC = draw_mst(FC_matrix, k=100, ax=1, title='MST of FC Matrix', if_draw=False)
        _, all_nodes_GMSC = draw_mst(GMSC_matrix, k=100, ax=1, title='MST of GMSC Matrix', if_draw=False)

        lis_FC_ranked.append(all_nodes_FC)
        lis_GMSC_ranked.append(all_nodes_GMSC)


        # HRP FC
        HRP_FC = portfolio_weight_performace.GenPortfolioWeight(returns_train, special_corr=FC_matrix, shrinkage=False)
        weights_FC = HRP_FC.genHRPweights()
        lis_weights_HRP_FC.append(weights_FC)


        # HRP GMSC
        HRP_GMSC = portfolio_weight_performace.GenPortfolioWeight(returns_train, special_corr=GMSC_matrix, shrinkage=False)
        weights_GMSC = HRP_GMSC.genHRPweightsSpecialCorr()
        lis_weights_HRP_GMSC.append(weights_GMSC)


    lis_weights_ND = []
    for series in lis_weights_HRP_FC:
        count = len(series)
        equal_weight = 1 / count
        # 创建一个新的Series，索引与原Series相同，每个元素的值都是等权重
        equal_weight_series = pd.Series(equal_weight, index=series.index)
        lis_weights_ND.append(equal_weight_series)
    
    df_weight_HRP_FC = pd.DataFrame(lis_weights_HRP_FC, index=lis_test_date)
    df_weight_HRP_GMSC = pd.DataFrame(lis_weights_HRP_GMSC, index=lis_test_date)
    df_weight_ND = pd.DataFrame(lis_weights_ND, index=lis_test_date)  
    df_FC_ranked = pd.DataFrame(lis_FC_ranked, index=lis_test_date)
    df_GMSC_ranked = pd.DataFrame(lis_GMSC_ranked, index=lis_test_date)

    with pd.ExcelWriter(r'selected_stocks/all_nodes_weights_w_{}.xlsx'.format(window_size)) as writer:
        print('保存权重、收益率和累计收益率到一个Excel文件, Excel文件已经打开')
        # pd.DataFrame(price_data.columns).to_excel(writer, sheet_name='selected_assets')
        df_weight_HRP_FC.to_excel(writer, sheet_name='df_weight_HRP_FC')
        df_weight_HRP_GMSC.to_excel(writer, sheet_name='df_weight_HRP_GMSC')
        df_weight_ND.to_excel(writer, sheet_name='df_weight_ND')
        df_FC_ranked.to_excel(writer, sheet_name='df_FC_ranked')
        df_GMSC_ranked.to_excel(writer, sheet_name='df_GMSC_ranked')


def main():
    print("请保证当前工作路径在Scripts文件夹下")
    print("当前工作路径:{}".format(os.getcwd()))

    print('读取\清洗数据(30s内)')
    price_data = portfolio_weight_performace.readWindSP500()
    window_sizes = [30, 50, 100, 200, 300, 500]

    # 创建参数组合
    params = [(window_size, price_data) for window_size in window_sizes]

    # 使用Pool并行执行
    print('开始多线程选股')
    with Pool(processes=6) as pool:
        useless_results = pool.starmap(gen_mst_select, params)


if __name__ == '__main__':
    main()
    print("所有文件处理完成。")
