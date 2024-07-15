from multiprocessing import Pool
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import framework_portfolio_weight_performace as portfolio_weight_performace
import warnings
import gc
import os

warnings.filterwarnings('ignore')


# 初始化函数，用于加载数据
def init_process():
    global price_data_global, return_data_global, cumulative_return_SP500
    print('Start read data')
    price_data_global = portfolio_weight_performace.readWindSP500()
    return_data_global = price_data_global.pct_change().dropna()
    SP500_df = pd.read_excel(r'..\WashedData\SP500indexFromWind.xlsx', index_col=0, header=0)
    cumulative_return_SP500 = (1 + SP500_df.pct_change().dropna()).cumprod()
    print('End read data')


# 定义处理文件的函数
def process_file(file_path):

    # 从文件名解析window_size和k
    filename = file_path.split("\\")[-1]
    parts = filename.split('_')
    FC_or_GMSC = parts[0]
    window_size = int(parts[2])  # 窗口大小
    k = int(parts[4].split('.')[0])  # 资产数量, 去掉扩展名

    # 读取Excel文件
    normalized_weight_df = pd.read_excel(file_path, index_col=0)
    print(f'Processing file: {file_path}')
    print(f'Window size: {window_size}, k: {k}')

    performance_obj = portfolio_weight_performace.PortfolioPerformance(
        lis_weights=normalized_weight_df,
        price_df=price_data_global,
        lis_test_date=normalized_weight_df.index.tolist(),
        title=f'{FC_or_GMSC}_w_{window_size}_k_{k}'
    )

    weight_df, daily_return, cumulative_return = performance_obj.plot_charts(if_show_pic=False)
    cumulative_return = cumulative_return - 1
    turnover_rates= performance_obj.turnover_rates
    feefree_cumulative_return = performance_obj.feefree_cumulative_return

    # 将Series转换为DataFrame并设置列名
    daily_return_df = daily_return.to_frame(name='daily_return')
    cumulative_return_df = cumulative_return.to_frame(name='cumulative_return')
    turnover_rates_df = turnover_rates.to_frame(name='turnover_rates')
    feefree_cumulative_return_df = feefree_cumulative_return.to_frame(name='feefree_cumulative_return')

    with pd.ExcelWriter(rf'multi_backtest_result\performance\performance_{FC_or_GMSC}_w_{window_size}_k_{k}.xlsx') as writer:

        cumulative_return_df.to_excel(writer, sheet_name='cumulative_return')
        daily_return_df.to_excel(writer, sheet_name='daily_return')
        turnover_rates_df.to_excel(writer, sheet_name='turnover_rates')

        '''
        feefree_cumulative_return_df.to_excel(writer, sheet_name='feefree_cumulative_return')
        weight_df.to_excel(writer, sheet_name='weight')
        '''
    

def main():

    print("请保证当前工作路径在Scripts文件夹下")
    print("当前工作路径:{}".format(os.getcwd()))

    files = glob.glob(r'multi_backtest_result\normalize_weights\*.xlsx')

    # 创建一个进程池
    # initializer=init_process
    with Pool(initializer=init_process, processes=10) as pool:
        # map函数会自动分配文件到不同的进程中
        results = list(tqdm(pool.imap_unordered(process_file, files), total=len(files)))


if __name__ == '__main__':
    main()
    print("所有文件处理完成。")

