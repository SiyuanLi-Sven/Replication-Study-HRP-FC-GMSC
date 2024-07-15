from multiprocessing import Pool
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import framework_portfolio_weight_performace as portfolio_weight_performace
import warnings
import gc

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
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read().strip()
        stock_codes = content.split(' ')
        stock_codes = [code.strip(',').strip() for code in stock_codes]

        # 在这里添加您的处理逻辑
        temp_kw = file_path.split('\\')[1].split('_')
        k = temp_kw[1]
        w = temp_kw[3]
        type_corr = temp_kw[4].split('.')[0]

        print('process start, k={}, w={}'.format(k, w))

        selected_price_data = price_data_global[stock_codes]
        price_data = selected_price_data
        return_data = price_data.pct_change().dropna()

        """回测"""
        window_size = int(w)    # 窗口大小

        # 初始化一些列表来保存所有的实验结果
        experiment_periods = []

        lis_weights_IV = []
        lis_weights_HRP = []
        lis_weights_HRP_GMSC = []
        lis_weights_EF = []
        lis_weights_ND = []
        lis_test_date = []

        count = 0
        # 迭代DataFrame，创建每个实验期的切片
        for start_day in tqdm(range(len(selected_price_data) - window_size - 1)):

            end_train_day = start_day + window_size

            # 实验期的切片
            experiment_slice = selected_price_data.iloc[start_day:end_train_day]
            
            # 测试期的切片, 从实验期开始后250天开始. 前250天仅用来估计参数了
            test_date = end_train_day + 1
            lis_test_date.append(selected_price_data.index[test_date])
            returns_train = experiment_slice.pct_change().dropna()

            # HRP
            HRP_instance = portfolio_weight_performace.GenPortfolioWeight(returns_train, shrinkage=False)
            weights_HRP = HRP_instance.genHRPweights()
            lis_weights_HRP.append(weights_HRP)

            # HRP-FC 此处省略(与HRP本身相关性太高了, 重合了)
            instance0 = portfolio_weight_performace.CorrProcess(returns_train)
            FC_matrix , GMSC_matrix = instance0.FC_GMSC()
            '''HRP_FC = portfolio_weight_performace.GenPortfolioWeight(returns_train, special_corr=FC_matrix, shrinkage=False)
            weights_FC = HRP_FC.genHRPweights()
            '''

            # HRP GMSC
            HRP_GMSC = portfolio_weight_performace.GenPortfolioWeight(returns_train, special_corr=GMSC_matrix, shrinkage=False)
            weights_GMSC = HRP_GMSC.genHRPweightsSpecialCorr()
            lis_weights_HRP_GMSC.append(weights_GMSC)

            # Inverse Volatility
            experiment_window = portfolio_weight_performace.GenPortfolioWeight(experiment_slice) # 计算IV要收缩估计, 输入为价格数据而非收益率
            weights = experiment_window.genIVweights()
            lis_weights_IV.append(weights.to_dict())
        
        # 生成等权重投资组合
        from collections import OrderedDict

        lis_weights_ND = []
        for series in lis_weights_HRP_GMSC:
            count = len(series)
            equal_weight = 1 / count
            # 创建一个新的Series，索引与原Series相同，每个元素的值都是等权重
            equal_weight_series = pd.Series(equal_weight, index=series.index)
            lis_weights_ND.append(equal_weight_series)
        
        
        """评估"""
        '''评估组合表现'''
        # 生成PortfolioPerformance对象
        portfolio_IV = portfolio_weight_performace.PortfolioPerformance(lis_weights = lis_weights_IV,
                                                                price_df=selected_price_data,
                                                                lis_test_date=lis_test_date,
                                                                title='IV')
        print('portfolio_IV generated')
        portfolio_HRP_GMSC = portfolio_weight_performace.PortfolioPerformance(lis_weights = lis_weights_HRP_GMSC,
                                                                price_df=selected_price_data,
                                                                lis_test_date=lis_test_date,
                                                                title='HRP_GMSC')
        print('portfolio_HRP_GMSC generated')
        portfolio_HRP = portfolio_weight_performace.PortfolioPerformance(lis_weights = lis_weights_HRP,
                                                                price_df=selected_price_data,
                                                                lis_test_date=lis_test_date,
                                                                title='HRP')
        print('portfolio_HRP generated')
        portfolio_ND = portfolio_weight_performace.PortfolioPerformance(lis_weights = lis_weights_ND,
                                                                price_df=selected_price_data,
                                                                lis_test_date=lis_test_date,
                                                                title='ND')
        print('portfolio_ND generated')

        # 调用PortfolioPerformance对象获得数据
        print('调用PortfolioPerformance对象获得数据')
        # weight_df_HRP, daily_return_HRP, cumulative_return_HRP = portfolio_HRP.plot_charts(if_show_pic=False) 
        weight_df_IV, daily_return_IV, cumulative_return_IV = portfolio_IV.plot_charts()
        weight_df_HRP, daily_return_HRP, cumulative_return_HRP = portfolio_HRP.plot_charts() 
        weight_df_HRP_GMSC, daily_return_HRP_GMSC, cumulative_return_HRP_GMSC = portfolio_HRP_GMSC.plot_charts() 
        weight_df_ND, daily_return_ND, cumulative_return_ND = portfolio_ND.plot_charts() 

        performance_df = portfolio_weight_performace.aggregate_performances(portfolio_IV, portfolio_HRP, portfolio_HRP_GMSC, portfolio_ND)


        # 详细数据较大(1.6G?)
        # 保存权重、收益率和累计收益率到Excel文件
        print('保存权重、收益率和累计收益率到一个Excel文件, k={}, w={}'.format(k, w))
        with pd.ExcelWriter(r'multi_backtest_result/weights_returns_k{}_window{}_type{}.xlsx'.format(k, window_size, type_corr)) as writer:
            print('保存权重、收益率和累计收益率到一个Excel文件, Excel文件已经打开, k={}, w={}'.format(k, w))

            pd.DataFrame(selected_price_data.columns).to_excel(writer, sheet_name='selected_assets')

            weight_df_IV.to_excel(writer, sheet_name='Weight_IV')
            daily_return_IV.to_excel(writer, sheet_name='Daily_Return_IV')
            cumulative_return_IV.to_excel(writer, sheet_name='Cumulative_Return_IV')

            weight_df_HRP.to_excel(writer, sheet_name='Weight_HRP')
            daily_return_HRP.to_excel(writer, sheet_name='Daily_Return_HRP')
            cumulative_return_HRP.to_excel(writer, sheet_name='Cumulative_Return_HRP')
                                           
            weight_df_HRP_GMSC.to_excel(writer, sheet_name='Weight_HRP_GMSC')
            daily_return_HRP_GMSC.to_excel(writer, sheet_name='Daily_Return_HRP_GMSC')
            cumulative_return_HRP_GMSC.to_excel(writer, sheet_name='Cumulative_Return_HRP_GMSC')
            
            weight_df_ND.to_excel(writer, sheet_name='Weight_ND')
            daily_return_ND.to_excel(writer, sheet_name='Daily_Return_ND')
            cumulative_return_ND.to_excel(writer, sheet_name='Cumulative_Return_ND')

            
        # 保存性能数据到Excel文件
        print('保存性能数据到另一个Excel文件, k={}, w={}, type={}'.format(k, w, type_corr))
        performance_df.to_excel(r'multi_backtest_result/performance_k{}_window{}_type{}.xlsx'.format(k, window_size,type_corr), sheet_name='Performance')
        print('performance_df.to_excel, k={}, w={}, type={}'.format(k, w, type_corr))

        # 保存累计收益率图表
        plt.figure(figsize=(20, 10))

        # 指定颜色
        colors = {
            'HRP_GMSC': '#70668f',    # 
            'HRP': '#a3799c',    # 
            'IV': '#326658',  # 
            'HRP_FC': '#5da59b',  # 
            'SP500': '#333b43', # 
            'ND': '#c2cfa3',        # 
        }

        # 绘制累计回报率曲线

        plt.plot(cumulative_return_IV, label='IV', color=colors['IV'])
        plt.plot(cumulative_return_HRP, label='HRP', color=colors['HRP'])
        plt.plot(cumulative_return_HRP_GMSC, label='HRP_GMSC', color=colors['HRP_GMSC'])
        plt.plot(cumulative_return_ND, label='ND', color=colors['ND'])
        plt.plot(cumulative_return_SP500, label='SP500', color=colors['SP500'])

        # 设置主标题和副标题
        plt.suptitle('Cumulative Return of Portfolios', fontsize=16)
        plt.title(f'Assets: {k}, Window Size: {window_size} days', fontsize=10)

        # 设置轴标签和图例
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)

        # 保存图表
        print('保存图表, k={}, w={}, type={}'.format(k, w, type_corr))
        plt.savefig(r'multi_backtest_result\cumulative_returns_k{}_window{}_type{}.png'.format(k, window_size, type_corr))
        plt.close()

        # Clearing up variables only after all operations are complete
        # 如果不清理变量, 内存会爆炸, 虚拟内存会侵占c盘空间
        # 在python3.9版本似乎不删除不使用的变量也能运行, 可能3.6以后优化过内存管理
        # 目前看这段代码去掉应该也没事, 但为了避免未知的风险还是留着好了.
        del weight_df_IV, daily_return_IV, cumulative_return_IV
        del weight_df_HRP, daily_return_HRP, cumulative_return_HRP
        del weight_df_HRP_GMSC, daily_return_HRP_GMSC, cumulative_return_HRP_GMSC
        del weight_df_ND, daily_return_ND, cumulative_return_ND
        del performance_df, portfolio_IV, portfolio_HRP, portfolio_HRP_GMSC, portfolio_ND
        gc.collect()  # Explicitly invoke garbage collection



def main():
    # 获取所有txt文件的路径
    files = glob.glob('selected_stocks/**.txt')

    # 创建一个进程池
    with Pool(initializer=init_process, processes=10) as pool:
        # map函数会自动分配文件到不同的进程中
        pool.map(process_file, files)


if __name__ == '__main__':
    main()
    print("所有文件处理完成。")

