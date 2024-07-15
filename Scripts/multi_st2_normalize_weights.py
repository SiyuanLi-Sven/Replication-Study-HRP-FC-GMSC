from multiprocessing import Pool
import os
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import framework_portfolio_weight_performace as portfolio_weight_performace
import warnings
import gc

warnings.filterwarnings('ignore')


def select_and_normalize_weights(k, weight_df_HRP, df_ranked, window_size, FC_GMSC='FC'):
    '''
    对特定的组合, 根据k归一化权重, 保存到excel
    '''
    # 初始化存储选择后权重的DataFrame
    selected_weight = pd.DataFrame(0, index=weight_df_HRP.index, columns=weight_df_HRP.columns)
    
    for date in weight_df_HRP.index:
        # 获取当天边缘程度排名最低的k个证券
        securities = df_ranked.loc[date].iloc[:k].values
        # 在HRP权重中检索这些证券的权重，并赋值到新的DataFrame
        selected_weight.loc[date, securities] = weight_df_HRP.loc[date, securities]
    # 归一化权重
    selected_weight = selected_weight.div(selected_weight.sum(axis=1), axis=0)
    selected_weight.to_excel(f'multi_backtest_result/normalize_weights/{FC_GMSC}_w_{window_size}_k_{k}_weights_normalized.xlsx')


def normalize_weights(file_info):
    '''
    对于单一文件中存储的FC和GMSC两种权重, 应用select_and_normalize_weights函数计算并保存归一化的权重
    '''
    # 读取文件和解析数据
    file_path = file_info[0]
    window_size = file_info[1]

    # 读取数据
    print(f'开始读取excel文件:{file_path}')
    xls = pd.ExcelFile(file_path)
    df_weight_HRP_FC = pd.read_excel(xls, 'df_weight_HRP_FC', index_col=0, header=0)
    df_weight_HRP_GMSC = pd.read_excel(xls, 'df_weight_HRP_GMSC', index_col=0, header=0)
    df_FC_ranked = pd.read_excel(xls, 'df_FC_ranked', index_col=0, header=0)
    df_GMSC_ranked = pd.read_excel(xls, 'df_GMSC_ranked', index_col=0, header=0)
    print(f'excel文件{file_path}读取完成')
    
    # 选择证券和归一化权重
    print(f'{file_info[0]}正在生成重新调整的权重')
    for k in tqdm([5, 10, 20, 30, 50, 100, 200, 300, 350]):
        select_and_normalize_weights(k, df_weight_HRP_FC, df_FC_ranked, window_size=window_size, FC_GMSC='FC')
        select_and_normalize_weights(k, df_weight_HRP_GMSC, df_GMSC_ranked, window_size=window_size, FC_GMSC='GMSC')
    print(f'{file_info[0]}重新调整的权重生成完毕')



def gen_performance():
    '''
    对特定的组合, 生成其表现
    '''
    pass


def one_portfolio(file_info):
    '''
    调用之前定义的函数, 完成一个组合的数据读取, 权重归一化, 权重存储, 权重读取, 表现生成与存储
    '''
    print(file_info)
    normalize_weights(file_info)
    return


if __name__ == '__main__':

    print("请保证当前工作路径在Scripts文件夹下")
    print("当前工作路径:{}".format(os.getcwd()))
    
    # 设定文件夹路径
    folder_path = 'selected_stocks'

    # 获取文件列表
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # 解析文件名并存储文件路径和window_size
    file_info = []
    for filename in files:
        parts = filename.split('_')
        window_size = int(parts[4][:-5])
        file_path = os.path.join(folder_path, filename)
        file_info.append((file_path, window_size))
    
    # 创建一个进程池
    with Pool(processes=6) as pool:
        pool.map(one_portfolio, file_info)