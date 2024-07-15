'''
这个程序是original_method_paper.ipynb的py版本, 如果ipynb出现路径问题, 可以尝试运行这个版本.

'''

import framework_portfolio_weight_performace as portfolio_weight_performace
from itertools import combinations
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import warnings

warnings.filterwarnings('ignore')
print("请保证当前工作路径在Scripts文件夹下\n当前工作路径: {} \n路径检查: {}".format(os.getcwd(), "正确" if os.getcwd().endswith("Scripts") else "错误"))


'''
0.mst选股函数
'''
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


price_data = portfolio_weight_performace.readWindSP500()
eturn_data = price_data.pct_change().dropna()
SP500_df = pd.read_excel(r'..\WashedData\SP500indexFromWind.xlsx', index_col=0, header=0)
cumulative_return_SP500 = (1 + SP500_df.pct_change().dropna()).cumprod()


'''
1.创建基准组合
预计耗时8h
'''
print('开始创建基准组合, 预计耗时8h')
w = 120
window_size = int(w)    # 窗口大小

# 初始化一些列表来保存所有的实验结果
# lis_weights_HRP = []
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

    '''# HRP
    HRP_instance = portfolio_weight_performace.GenPortfolioWeight(returns_train, shrinkage=False)
    weights_HRP = HRP_instance.genHRPweights()
    lis_weights_HRP.append(weights_HRP)'''

    # 记录MST顺序
    instance0 = portfolio_weight_performace.CorrProcess(returns_train)
    FC_matrix , GMSC_matrix = instance0.FC_GMSC()

    sparsest_nodes_FC, all_nodes_FC = draw_mst(FC_matrix, k=100, ax=1, title='MST of FC Matrix', if_draw=False)
    sparsest_nodes_GMSC, all_nodes_GMSC = draw_mst(GMSC_matrix, k=100, ax=1, title='MST of GMSC Matrix', if_draw=False)

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

with pd.ExcelWriter(r'original_alpha/all_nodes_weights.xlsx') as writer:
    print('保存权重、收益率和累计收益率到一个Excel文件, Excel文件已经打开')
    # pd.DataFrame(price_data.columns).to_excel(writer, sheet_name='selected_assets')
    df_weight_HRP_FC.to_excel(writer, sheet_name='df_weight_HRP_FC')
    df_weight_HRP_GMSC.to_excel(writer, sheet_name='df_weight_HRP_GMSC')
    df_weight_ND.to_excel(writer, sheet_name='df_weight_ND')
    df_FC_ranked.to_excel(writer, sheet_name='df_FC_ranked')
    df_GMSC_ranked.to_excel(writer, sheet_name='df_GMSC_ranked')

portfolio_HRP_FC = portfolio_weight_performace.PortfolioPerformance(lis_weights = df_weight_HRP_FC,
                                                        price_df=price_data,
                                                        lis_test_date=lis_test_date,
                                                        title='HRP_FC')
portfolio_HRP_GMSC = portfolio_weight_performace.PortfolioPerformance(lis_weights = df_weight_HRP_GMSC,
                                                        price_df=price_data,
                                                        lis_test_date=lis_test_date,
                                                        title='HRP_GMSC')

weight_df_HRP, daily_return_HRP, cumulative_return_HRP = portfolio_HRP_FC.plot_charts(if_show_pic=False)
weight_df_HRP, daily_return_HRP, cumulative_return_HRP = portfolio_HRP_GMSC.plot_charts(if_show_pic=False)
df_FC_ranked = pd.DataFrame(lis_FC_ranked,index=lis_test_date)
df_GMSC_ranked = pd.DataFrame(lis_GMSC_ranked,index=lis_test_date)

with pd.ExcelWriter(r'original_alpha/original_weights_HRP_FC.xlsx') as writer:
    weight_df_HRP.to_excel(writer, sheet_name='weight_HRP_FC')
    daily_return_HRP.to_excel(writer, sheet_name='daily_return_HRP_FC')
    cumulative_return_HRP.to_excel(writer, sheet_name='cumulative_return_HRP_FC')
    df_FC_ranked.to_excel(writer, sheet_name='FC_ranked')

with pd.ExcelWriter(r'original_alpha/original_weights_HRP_GMSC.xlsx') as writer:
    weight_df_HRP.to_excel(writer, sheet_name='weight_HRP_GMSC')
    daily_return_HRP.to_excel(writer, sheet_name='daily_return_HRP_GMSC')
    cumulative_return_HRP.to_excel(writer, sheet_name='cumulative_return_HRP_GMSC')
    df_GMSC_ranked.to_excel(writer, sheet_name='GMSC_ranked')



'''
2.根据基准HRP组合创建FC和GMSC的不同k组合
'''
print('根据基准HRP组合创建FC和GMSC的不同k组合')
# 读取Excel文件
file_path_FC = 'original_alpha/original_weights_HRP_FC.xlsx'
sheets_dict_FC = pd.read_excel(file_path_FC, sheet_name=None, index_col=0)
file_path_GMSC = 'original_alpha/original_weights_HRP_GMSC.xlsx'
sheets_dict_GMSC = pd.read_excel(file_path_GMSC, sheet_name=None, index_col=0)

# 处理各个sheet: FC
weight_df_HRP_FC = sheets_dict_FC['weight_HRP_FC']
daily_return_HRP_FC = pd.read_excel(file_path_FC, sheet_name='daily_return_HRP_FC', index_col=0, header=0)
df_FC_ranked = pd.read_excel(file_path_FC, sheet_name='FC_ranked', index_col=0, header=0)

# 处理各个sheet: GMSC
weight_df_HRP_GMSC = sheets_dict_GMSC['weight_HRP_GMSC']
daily_return_HRP_GMSC = pd.read_excel(file_path_GMSC, sheet_name='daily_return_HRP_GMSC', index_col=0, header=0)
df_GMSC_ranked = pd.read_excel(file_path_GMSC, sheet_name='GMSC_ranked', index_col=0, header=0)

def select_and_normalize_weights(k, weight_df_HRP, df_ranked, output_path, FC_GMSC='FC'):
    # 初始化存储选择后权重的DataFrame
    selected_weight = pd.DataFrame(0, index=weight_df_HRP.index, columns=weight_df_HRP.columns)
    
    # 对于每天的数据
    for date in weight_df_HRP.index:
        # 获取当天边缘程度排名最低的k个证券
        securities = df_ranked.loc[date].iloc[:k].values
        
        # 在HRP权重中检索这些证券的权重，并赋值到新的DataFrame
        selected_weight.loc[date, securities] = weight_df_HRP.loc[date, securities]
        
        # 归一化权重，使得每行的权重总和为1
        total_weight = selected_weight.loc[date].sum()
        if total_weight > 0:
            selected_weight.loc[date] /= total_weight

    # 保存DataFrame为CSV
    selected_weight.to_csv(f'{output_path}/k_{k}_selected_weight_{FC_GMSC}.csv')


output_path = 'original_alpha'

for k in tqdm([5,10,20,30,50,100,150,200,250,300,350]):
    select_and_normalize_weights(k, weight_df_HRP_FC, df_FC_ranked, output_path, FC_GMSC='FC')
    select_and_normalize_weights(k, weight_df_HRP_GMSC, df_GMSC_ranked, output_path, FC_GMSC='GMSC')

'''
3.读取不同k组合的权重, 评估表现
'''
print('读取不同k组合的权重, 评估表现')

def read_weights(k, path='original_alpha'):
    weights_GMSC = pd.read_csv(f'{path}/k_{k}_selected_weight_GMSC.csv', index_col=0)
    weights_FC = pd.read_csv(f'{path}/k_{k}_selected_weight_FC.csv', index_col=0)
    
    return {
        f'k_{k}_GMSC_df': weights_GMSC,
        f'k_{k}_FC_df': weights_FC
    }

k_values = [5, 10, 20, 30, 50, 100, 200, 300, 350]
weights_dict = {}
for k in k_values:
    weights = read_weights(k)
    weights_dict.update(weights)


price_data_global = portfolio_weight_performace.readWindSP500()


performance_objects = {}

for k in tqdm(k_values):
    gmsc_key = f'k_{k}_GMSC_df'
    fc_key = f'k_{k}_FC_df'

    # 创建性能对象并存储
    performance_objects[f'portfolio_{fc_key}'] = portfolio_weight_performace.PortfolioPerformance(
        lis_weights=weights_dict[fc_key],
        price_df=price_data_global,
        lis_test_date=weights_dict[fc_key].index.tolist(),
        title=f'HRP_FC_{k}'
    )

    performance_objects[f'portfolio_{gmsc_key}'] = portfolio_weight_performace.PortfolioPerformance(
        lis_weights=weights_dict[gmsc_key],
        price_df=price_data_global,
        lis_test_date=weights_dict[gmsc_key].index.tolist(),
        title=f'HRP_GMSC_{k}'
    )

import os
from tqdm import tqdm
folder = 'original_alpha/performance'
if not os.path.exists(folder):
    os.makedirs(folder)


for key, obj in tqdm(performance_objects.items()):
    
    weight_df, daily_return, cumulative_return = obj.plot_charts(if_show_pic=False)
    turnover_rates= obj.turnover_rates
    feefree_cumulative_return= obj.feefree_cumulative_return

    with pd.ExcelWriter(f'original_alpha/performance/weights_returns_{key}.xlsx') as writer:
        weight_df.to_excel(writer, sheet_name='weight')
        daily_return.to_excel(writer, sheet_name='daily_return')
        cumulative_return.to_excel(writer, sheet_name='cumulative_return')
        turnover_rates.to_excel(writer, sheet_name='turnover_rates')
        feefree_cumulative_return.to_excel(writer, sheet_name='feefree_cumulative_return')


# 设定文件夹路径
folder_path = 'original_alpha/performance'

# 列出符合条件的文件
files = [f for f in os.listdir(folder_path) if f.startswith('weights_returns') and f.endswith('.xlsx')]

data_list = []
for filename in tqdm(files):
    # 解析文件名以获取参数
    parts = filename.replace('weights_returns_portfolio_', '').replace('.xlsx', '').split('_')
    k = parts[1]
    strategy_type = parts[2]
    
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, filename)

    # 读取 Excel 文件中的数据
    daily_return = pd.read_excel(file_path, sheet_name='daily_return')
    daily_return.columns = ['date', 'daily_return']
    daily_return.set_index('date', inplace=True)

    # 将解析的参数和数据存储
    data_list.append({
        'k': k,
        'strategy_type': strategy_type,
        'daily_return': daily_return,
    })



def calculate_financial_metrics(df, risk_free_rate=0.03):
    """
    Calculate annualized return, annualized standard deviation, Sharpe ratio, and maximum drawdown.

    Args:
    df (DataFrame): DataFrame with a 'daily_return' column and datetime index.
    risk_free_rate (float): Annual risk-free rate, default is 3%.

    Returns:
    tuple: (annualized return, annualized standard deviation, Sharpe ratio, maximum drawdown)
    """
    # 确保'daily_return'列存在
    if 'daily_return' not in df.columns:
        raise ValueError("DataFrame must include a 'daily_return' column")

    # 年化收益率
    daily_return = df['daily_return']
    annualized_return = np.prod(1 + daily_return) ** (252 / len(daily_return)) - 1

    # 年化标准差
    annualized_std = daily_return.std() * np.sqrt(252)

    # 夏普比率
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std

    # 最大回撤
    cumulative_returns = (1 + daily_return).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    return annualized_return, annualized_std, sharpe_ratio, max_drawdown


# 定义时期
periods = {
    'p-1': ('2001-01-01', '2006-12-31'),
    'p-2': ('2007-01-01', '2009-12-31'),
    'p-3': ('2010-01-01', '2019-12-31'),
    'p-4': ('2020-01-01', '2020-04-30'),
    'p-5': ('2020-05-01', '2021-08-31')
}

df = pd.DataFrame(data_list)

# 遍历每个时期
for period_name, (start_date, end_date) in periods.items():

    col_names = [f"{period_name} {metric}" for metric in ["an~return", "an~std", "sharpe", "max~draw"]]
    
    for idx, row in df.iterrows():
        # 提取特定时期数据
        period_data = row['daily_return'].loc[start_date:end_date]
        # 计算金融指标
        if not period_data.empty:
            results = calculate_financial_metrics(period_data)
            # 将结果添加到DataFrame中
            df.loc[idx, col_names[0]] = results[0]  # an~return
            df.loc[idx, col_names[1]] = results[1]  # an~std
            df.loc[idx, col_names[2]] = results[2]  # sharpe
            df.loc[idx, col_names[3]] = results[3]  # max~draw
        else:
            # 填充NaN
            df.loc[idx, col_names] = [np.nan] * 4

df.to_csv('original_method_all_result.csv')
print('全部结果已经保存')
print('开始计算作为基准的SP500的表现')

df_temp = df.drop(['daily_return'],axis=1)
SP500_df = pd.read_excel(r'..\WashedData\SP500indexFromWind.xlsx', index_col=0, header=0)
SP500_return = SP500_df.pct_change().dropna()
df_SP500 = pd.DataFrame(np.zeros(shape=(20,20)))


# 初始化 DataFrame，确保有足够的列
col_names = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown']
df_SP500 = pd.DataFrame(index=periods.keys(), columns=col_names)

# 遍历每个时期
for period_name, (start_date, end_date) in periods.items():
    period_data = pd.DataFrame(SP500_return['SP500'].loc[start_date:end_date])
    period_data['daily_return'] = period_data['SP500']
    
    # 计算指标
    results = calculate_financial_metrics(period_data)
    
    # 将结果添加到相应的行和列
    df_SP500.loc[period_name] = results

print(df_SP500)
print('全部程序运行完成')