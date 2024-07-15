'''
用于实现一些资产配置算法的研究, 包括权重的生成和组合表现的评价

开发&维护: 
    SiyuanLi, SYSU
联系方式: 
    lisiyuansven@foxmail.com
    https://github.com/SiyuanLi-Sven 
'''
'''
类:
    GenPortfolioWeight: 
        可以生成HRP(Hierarchical Risk Parity, 分层风险平价)和IV(Inverse Volatility, 逆波动/逆方差)两个算法的权重, 
        HRP的代码参考自López(2016), IV的算法自拟.

    PortfolioPerformance:
        可以根据输入的权重和收益率数据, 计算组合的收益率数据, 并计算其表现(收益率, 波动率...)

    CorrProcess:
        用于实现一些相关系数矩阵/协方差矩阵的操作. 如去除全局运动的GMSC矩阵(Cho, 2023)


函数:
    readWindSS300:
        读取本地沪深三百股价数据. 

    readWindSP500:
        读取本地标普500股价数据. 


reference
    Cho, Y., & Song, J. W. (2023). Hierarchical risk parity using security selection based on peripheral assets of correlation-based minimum spanning trees. Finance Research Letters, 53, 103608. https://doi.org/10.1016/j.frl.2022.103608
    López De Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. The Journal of Portfolio Management, 42(4), 59–69. https://doi.org/10.3905/jpm.2016.42.4.059

'''


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
from pypfopt_temp import CovarianceShrinkage
from collections import OrderedDict
# from pypfopt.risk_models import CovarianceShrinkage

# 指定字体路径或名称
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 例如，使用微软雅黑作为字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
matplotlib.use('Agg')  # 使用非GUI后端，例如'Agg'用于文件输出

'''
如果没有matplotlib.use('Agg'),在多线程中可能出现以下错误, 导致图片生成失败(?) 参考帖子
https://stackoverflow.com/questions/14694408/runtimeerror-main-thread-is-not-in-main-loop 
另外,可以考虑将绘图放在excel数据保存之后, 这样绘图报错也不影响excel数据. 进程池会自动保持并行数.
'''


class GenPortfolioWeight:
    '''
    此类用于生成投资组合权重, 有HRP和IV两个选择

    :returns_df: 该组合的基础资产的日收益率数据, 为宽格式, 每列为一资产, 每行为一日
                 注意: 计算IV要收缩估计, 输入为价格数据而非收益率
    :lis_test_date: 测试日期列表
    :title: 标题
    '''
    def __init__(self, returns_df, special_corr='', title='', shrinkage=False):
        self.returns_df = returns_df
        self.title = title
        self.special_corr = special_corr
        self.shrinkage = shrinkage

    def getIVP(self, cov, **kargs):
        """Compute the inverse-variance portfolio"""
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def getClusterVar(self, cov, cItems):
        """Compute variance per cluster"""
        cov_ = cov.loc[cItems, cItems]  # matrix slice
        w_ = self.getIVP(cov_).reshape(-1, 1)
        cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return cVar

    def getQuasiDiag(self, link):
        """Sort clustered items by distance"""
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = pd.concat([sortIx, df0]).sort_index()  # combine and re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    def getRecBipart(self, cov, sortIx, if_adv = False):
        """Compute HRP allocation"""
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]   # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = self.getClusterVar(cov, cItems0)
                cVar1 = self.getClusterVar(cov, cItems1)
                if if_adv:
                    cVar0 = self.getClusterVar(cov, cItems0)
                    cVar1 = self.getClusterVar(cov, cItems1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w

    def correlDist(self, corr):
        """A distance matrix based on correlation, where 0<=d[i,j]<=1"""
        dist = ((1 - corr) / 2.)**0.5  # distance matrix
        return dist

    def plotCorrMatrix(self, path, corr, labels=None):
        """Heatmap of the correlation matrix"""
        if labels is None: labels = corr.index.tolist()
        plt.pcolor(corr)
        plt.colorbar()
        plt.yticks(np.arange(.5, corr.shape[0] + .5), labels)
        plt.xticks(np.arange(.5, corr.shape[0] + .5), labels)
        plt.savefig(path)
        plt.clf()
        plt.close()  # reset pylab

    def genHRPweights(self):
        """Main function to run HRP"""

        if self.shrinkage:
            cov = CovarianceShrinkage(self.returns_df).ledoit_wolf()
            # 计算标准差，即协方差矩阵对角线元素的平方根
            std_dev = np.sqrt(np.diag(cov))
            # 创建一个标准差的倒数矩阵，用于归一化协方差矩阵
            inv_std_dev = np.diag(1 / std_dev)
            # 计算相关系数矩阵
            corr = inv_std_dev @ cov @ inv_std_dev
        else:
            cov, corr = self.returns_df.cov(), self.returns_df.corr()
        
        dist = self.correlDist(corr)
        link = sch.linkage(dist, 'single')
        sortIx = self.getQuasiDiag(link)
        sortIx = corr.index[sortIx].tolist()  # recover labels
        df0 = corr.loc[sortIx, sortIx]  # reorder
        hrp_weights = self.getRecBipart(cov, sortIx)
        return hrp_weights
    
    def genIVweights(self):
        """计算逆方差加权的权重，并保留资产标签"""
        cov_matrix = CovarianceShrinkage(self.returns_df).ledoit_wolf() # 计算收缩估计的协方差矩阵
        ivp_weights_array = self.getIVP(cov_matrix)  # 获取逆方差权重
        ivp_weights_series = pd.Series(ivp_weights_array, index=self.returns_df.columns)  # 将权重数组转换为带有资产标签的Series
        return ivp_weights_series
    
    def genHRPweightsSpecialCorr(self):
        """Main function to run HRP"""
        
        if self.shrinkage:
            cov = CovarianceShrinkage(self.returns_df).ledoit_wolf()
            # 计算标准差，即协方差矩阵对角线元素的平方根
            std_dev = np.sqrt(np.diag(cov))
            # 创建一个标准差的倒数矩阵，用于归一化协方差矩阵
            inv_std_dev = np.diag(1 / std_dev)
            # 计算相关系数矩阵
            corr = inv_std_dev @ cov @ inv_std_dev
        else:
            cov, corr = self.returns_df.cov(), self.returns_df.corr()

        dist = self.correlDist(self.special_corr)
        link = sch.linkage(dist, 'single')
        sortIx = self.getQuasiDiag(link)
        sortIx = corr.index[sortIx].tolist()  # recover labels
        df0 = corr.loc[sortIx, sortIx]  # reorder
        hrp_weights = self.getRecBipart(cov, sortIx)
        return hrp_weights



class PortfolioPerformance:
    def __init__(self, lis_weights, price_df, lis_test_date, title='', fee_rate=0.0001, initial_value=1):
        self.title = title
        self.fee_rate = fee_rate
        self.price_df = price_df
        self.returns_df = self.price_df.pct_change().dropna()
        self.lis_weights = lis_weights
        self.lis_test_date = lis_test_date

        self.initial_value = initial_value
        self.weight_df = self.weights_to_dataframe()
        self.weight_df.index = self.ensure_datetime_index(index=self.weight_df.index)
        self.price_df.index = self.ensure_datetime_index(index=self.price_df.index)
        
        self.common_dates = self.weight_df.index.intersection(self.price_df.index)

        self.daily_portfolio_price, self.average_turnover_rate, self.turnover_rates, _ = self.calculate_daily_portfolio_price_and_turnover()
        self.daily_return = self.daily_portfolio_price.pct_change().fillna(0)
        self.feefree_daily_return, self.feefree_cumulative_return = self.calculate_feefree_daily_return()

    def weights_to_dataframe(self):
        '''
        将输入的weight转为df形式
        '''
        # 判断是否self.lis_weights是一个列表
        if isinstance(self.lis_weights, list):
            # 检查列表中的每个元素是否都是OrderedDict
            if all(isinstance(weight, OrderedDict) for weight in self.lis_weights):
                weight_df = pd.DataFrame(self.lis_weights, index=pd.to_datetime(self.lis_test_date))
                return weight_df
            elif all(isinstance(weight, pd.Series) for weight in self.lis_weights):
                weight_df = pd.DataFrame(self.lis_weights, index=pd.to_datetime(self.lis_test_date))
                return weight_df
            elif all(isinstance(weight, dict) for weight in self.lis_weights):
                weight_df = pd.DataFrame(self.lis_weights, index=pd.to_datetime(self.lis_test_date))
                return weight_df
            else:
                raise ValueError("Unsupported type for weights")
        
        # 如果不是列表，检查是否为DataFrame
        elif isinstance(self.lis_weights, pd.DataFrame):
            return self.lis_weights
        
        # 如果都不是，返回一个错误
        else:
            raise ValueError("Unsupported type for weights")
    

    def ensure_datetime_index(self,index):
        '''
        确认索引为日期
        '''
        if not isinstance(index, pd.DatetimeIndex):
            # 将索引转换为DatetimeIndex
            return pd.to_datetime(index)
        return index

    def calculate_daily_portfolio_price_and_turnover(self):
        '''
        计算有手续费的日收益率
        '''
        asset_values = []  # 存储每天的资产面值DataFrame
        portfolio_values = pd.Series(index=self.common_dates, dtype=float)
        portfolio_values.iloc[0] = self.initial_value
        turnover_rates = pd.Series(index=self.common_dates, dtype=float)  # 存储每天的换手率
        
        previous_asset_values = None
        
        for i in range(1, len(self.common_dates)):
            current_day = self.common_dates[i]
            prev_day = self.common_dates[i - 1]
            current_weights = self.weight_df.loc[current_day]
            current_prices = self.price_df.loc[current_day]
            previous_prices = self.price_df.loc[prev_day]
            previous_portfolio_value = portfolio_values.iloc[i - 1]

            current_asset_values = current_weights * (previous_portfolio_value / previous_prices) * current_prices
            asset_values.append(current_asset_values)

            current_portfolio_value = current_asset_values.sum()
            portfolio_values.iloc[i] = current_portfolio_value

            # 计算资产面值变化的绝对值总和
            if previous_asset_values is not None:
                value_changes = (current_asset_values - previous_asset_values).abs().sum()
                # 交易成本
                current_fee_cost = value_changes * self.fee_rate
                # 减去成本后组合净值
                portfolio_values.iloc[i] -= current_fee_cost
                # 换手率
                turnover_rates.iloc[i] = value_changes / previous_portfolio_value
            
            previous_asset_values = current_asset_values

        average_turnover_rate = turnover_rates.mean()  # 计算平均换手率
        return portfolio_values, average_turnover_rate, turnover_rates, asset_values
    
    def calculate_feefree_daily_return(self):
        '''
        无手续费的日收益率和累计收益率
        '''
        feefree_daily_return = (self.returns_df * self.weight_df).sum(axis=1)
        feefree_cumulative_return = (1 + feefree_daily_return).cumprod() - 1
        return feefree_daily_return, feefree_cumulative_return


    def basic_performance(self, risk_free_rate=0.03):
        annual_std = self.daily_return.std() * (252**0.5) # 控制自由度几乎没啥影响
        annual_return = (1 + self.daily_return).prod() ** (252 / len(self.daily_return)) - 1
        sharpe_ratio = (annual_return - risk_free_rate) / annual_std
        cumulative_return = (1 + self.daily_return).cumprod() - 1
        return annual_std, annual_return, sharpe_ratio, cumulative_return

    def improved_performance(self, risk_free_rate=0.03):
        # Value-at-Risk (VaR, 5%)
        var_5 = np.percentile(self.daily_return, 5)

        # Conditional Value-at-Risk (CVaR, 5%)
        cvar_5 = self.daily_return[self.daily_return <= var_5].mean()

        # Drawdown and Maximum drawdown
        cumulative_returns = (1 + self.daily_return).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar ratio
        annual_return = (1 + self.daily_return).prod() ** (252 / len(self.daily_return)) - 1
        calmar_ratio = annual_return / abs(max_drawdown)

        # Sortino ratio
        negative_volatility = self.daily_return[self.daily_return < 0].std() * np.sqrt(252)
        sortino_ratio = (annual_return - risk_free_rate) / negative_volatility

        return var_5, cvar_5, max_drawdown, calmar_ratio, sortino_ratio

    def show_performance(self, quietly=False):
        '''
        quietly为True则不输出组合表现.
        '''
        annual_std, annual_return, sharpe_ratio, cumulative_return = self.basic_performance()
        var_5, cvar_5, max_drawdown, calmar_ratio, sortino_ratio = self.improved_performance()

        if not quietly:
            print("\n ###################  {}  ###################".format(self.title))
            print("Annual Return:", annual_return)
            print("Annual Std:", annual_std)
            print("Sharpe Ratio:", sharpe_ratio)
            print("Turnover Rate:", self.average_turnover_rate)
            print("VaR (5%):", var_5)
            print("CVaR (5%):", cvar_5)
            print("Maximum Drawdown:", max_drawdown)
            print("Calmar Ratio:", calmar_ratio)
            print("Sortino Ratio:", sortino_ratio)

        return [annual_std, annual_return, sharpe_ratio, self.average_turnover_rate, var_5, cvar_5, max_drawdown, calmar_ratio, sortino_ratio]

    def plot_charts(self, if_show_pic=True):
        '''
        传入PortfolioPerfoemance对象, 绘制日收益率\累计收益率
        '''

        if if_show_pic:
            # 设置图表布局
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

            # 第一个子图：Portfolio Weights Over Time
            self.weight_df.plot(kind='area', stacked=True, ax=axes[0], title='Portfolio Weights Over Time '+ self.title)
            axes[0].get_legend().remove()    #不显示图例
            axes[0].set_ylabel('Weight (%)')
            axes[0].set_xlabel('Date')

            # 第二个子图：Daily Return
            self.daily_return.plot(ax=axes[1], title='Daily Return '+ self.title)
            axes[1].set_ylabel('Return')
            axes[1].set_xlabel('Date')

            # 第三个子图：Cumulative Return
            self.daily_portfolio_price.plot(ax=axes[2], title='Cumulative Return '+ self.title)
            axes[2].set_ylabel('Cumulative Return')
            axes[2].set_xlabel('Date')

            plt.tight_layout()
            plt.show()

        return self.weight_df, self.daily_return, self.daily_portfolio_price

class CorrProcess():
    '''
    是否使用收缩估计
    是否准对角化
    是否减去"市场波动"(GMSC)
    计算相关矩阵的信息熵
    Cophenetic相关系数
    '''
    def __init__(self, returns_df, title=''):
        self.returns_df = returns_df
        self.title = title
    
    def getQuasiDiag(self, if_shrinkage=False):
        '''输入收益率数据, 返回一个对角化后的矩阵'''
        if if_shrinkage:
            cov_matrix = CovarianceShrinkage(self.returns_df).ledoit_wolf() # 计算收缩估计的协方差矩阵
            # 计算协方差矩阵的标准差（即方差的平方根，对角线元素）
            std_dev = np.sqrt(np.diag(cov_matrix))
            # 计算相关系数矩阵
            corr = cov_matrix / np.outer(std_dev, std_dev)
        else:
            corr = self.returns_df.corr()
        stock_linkage = linkage(self.returns_df.T, 'ward')
        pwp0 = GenPortfolioWeight(self.returns_df)

        sortIx = corr.index[pwp0.getQuasiDiag(stock_linkage)].tolist()  # recover labels
        quasiDiaged = corr.loc[sortIx, sortIx]

        return quasiDiaged, sortIx
    
    def FC_GMSC(self, if_quasidiaged=False): 
        '''计算FC矩阵和GMSC矩阵'''
        # 1. 归一化(标准化)收益率
        mu = self.returns_df.mean()
        sigma = self.returns_df.std()
        normalized_returns = (self.returns_df - mu) / sigma

        # 2. 计算FC矩阵
        T = len(self.returns_df)
        FC_matrix = normalized_returns.T.dot(normalized_returns) / T

        # 3. 计算GMSC矩阵
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(FC_matrix)
        # 排序特征值（和对应的特征向量）
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 假设根据Marčenko–Pastur分布，只有最大的特征值代表全局运动
        # 去除全局运动的影响
        GMSC_matrix = FC_matrix - np.outer(sorted_eigenvectors[:, 0], sorted_eigenvectors[:, 0]) * sorted_eigenvalues[0]

        if if_quasidiaged:
            # 如果选择进行准对角化，则先获取排序索引
            _, sortIx = self.getQuasiDiag(if_shrinkage=False)
            FC_matrix = FC_matrix.loc[sortIx, sortIx]
            GMSC_matrix = GMSC_matrix.loc[sortIx, sortIx]

        return FC_matrix, GMSC_matrix
    
    def show_corr_profile(self):
        # 计算原始相关系数矩阵
        original_corr = self.returns_df.corr()
        
        # 计算准对角化后的相关系数矩阵
        quasi_diag_corr, _ = self.getQuasiDiag(if_shrinkage=False)
        
        # 计算FC矩阵和GMSC矩阵
        FC_matrix, GMSC_matrix = self.FC_GMSC(if_quasidiaged=False)
        
        # 计算准对角化后的FC矩阵和GMSC矩阵
        _, sortIx = self.getQuasiDiag(if_shrinkage=False)
        sorted_FC_matrix = FC_matrix.loc[sortIx, sortIx]
        sorted_GMSC_matrix = GMSC_matrix.loc[sortIx, sortIx]

        # 绘图
        fig, ax = plt.subplots(3, 2, figsize=(14, 18), sharex='col', sharey='row')
        
        sns.heatmap(original_corr, ax=ax[0, 0], cmap='coolwarm', cbar=True)
        ax[0, 0].set_title('Original Correlation Matrix')
        
        sns.heatmap(quasi_diag_corr, ax=ax[0, 1], cmap='coolwarm', cbar=True)
        ax[0, 1].set_title('Quasi-Diagonalized Correlation Matrix')
        
        sns.heatmap(FC_matrix, ax=ax[1, 0], cmap='coolwarm', cbar=True)
        ax[1, 0].set_title('FC Matrix')
        
        sns.heatmap(sorted_FC_matrix, ax=ax[1, 1], cmap='coolwarm', cbar=True)
        ax[1, 1].set_title('Quasi-Diagonalized FC Matrix')
        
        sns.heatmap(GMSC_matrix, ax=ax[2, 0], cmap='coolwarm', cbar=True)
        ax[2, 0].set_title('GMSC Matrix')
        
        sns.heatmap(sorted_GMSC_matrix, ax=ax[2, 1], cmap='coolwarm', cbar=True)
        ax[2, 1].set_title('Quasi-Diagonalized GMSC Matrix')

        plt.tight_layout()
        plt.show()


def aggregate_performances(*portfolios):
    """
    接收多个 PortfolioPerformance 实例，调用它们的 show_performance 方法，
    并将结果汇总到一个DataFrame中。

    :param portfolios: 任意数量的 PortfolioPerformance 实例。
    :return: 一个Pandas DataFrame，行为性能指标，列为投资组合名称。
    """
    # 指标名称列表
    index_names = ['annual_std', 'annual_return', 'sharpe_ratio', 
                   'average_turnover_rate', 'var_5', 'cvar_5', 
                   'max_drawdown', 'calmar_ratio', 'sortino_ratio']
    
    # 初始化一个空字典来收集每个投资组合的性能数据
    data = {}
    
    # 对于每个投资组合实例，调用 show_performance 并存储结果
    for portfolio in portfolios:
        performance_data = portfolio.show_performance(quietly=True)
        data[portfolio.title] = performance_data  # 假设每个实例都有一个'name'属性
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=index_names)
    
    return df


def readWindSS300(start_date='2007-01-01', end_date='2024-01-01', drop_stable=True):
    '''
    读取Wind的沪深300数据
    '''
    file_path = r'..\WashedData\SS300washedFromWind.xlsx'
    trade_data_df = pd.read_excel(file_path, index_col=0, header=0, engine='openpyxl')
    
    start_time = start_date
    end_time = end_date

    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)

    # 将时间戳转换为时区无关的tz-naive（如果它们是tz-aware的话）
    trade_data_df.index = trade_data_df.index.tz_localize(None)

    # 时间切片
    sliced_df = trade_data_df.loc[start_datetime:end_datetime]
    # 删除含有 0 值的列
    sliced_df = sliced_df.drop(columns=sliced_df.columns[(sliced_df == 0).any()])
    sliced_df = sliced_df.dropna(axis=1)
    
    if drop_stable:
        # 检查并删除存在连续三个一样值的列. 因为我们认为这些股票存在交易的异常
        columns_to_drop = []
        for column in sliced_df.columns:
            if any((sliced_df[column] == sliced_df[column].shift(1)) & 
                (sliced_df[column] == sliced_df[column].shift(2)) & 
                (sliced_df[column] == sliced_df[column].shift(3)) & 
                (sliced_df[column] == sliced_df[column].shift(4))):
                    columns_to_drop.append(column)
        
        sliced_df = sliced_df.drop(columns=columns_to_drop)
    
    return sliced_df


def readWindSP500(start_date='2000-01-01',end_date='2024-01-01'):
    '''
    读取Wind的SP500数据
    '''
    file_path = r'..\WashedData\SP500washedFromWind.xlsx'
    
    trade_data_df = pd.read_excel(file_path, index_col=0, header=0, engine='openpyxl')
    
    start_time = start_date
    end_time = end_date

    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)

    # 将时间戳转换为时区无关的tz-naive（如果它们是tz-aware的话）
    trade_data_df.index = trade_data_df.index.tz_localize(None)

    # 时间切片
    sliced_df = trade_data_df.loc[start_datetime:end_datetime]
    # 删除含有 0 值的列, 因为我们认为这些股票存在交易的异常
    sliced_df = sliced_df.drop(columns=sliced_df.columns[(sliced_df == 0).any()])
    sliced_df = sliced_df.dropna(axis=1)
    
    return sliced_df
