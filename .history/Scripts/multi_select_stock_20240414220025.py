import framework_portfolio_weight_performace as portfolio_weight_performace
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def gen_mst_select(k, window_size, return_data):






def main():

    price_data = portfolio_weight_performace.readWindSP500()
    return_data = price_data.pct_change().dropna()

    instance0 = portfolio_weight_performace.CorrProcess(return_data)
    FC_matrix , GMSC_matrix = instance0.FC_GMSC()

    # ks和window_sizes的定义
    ks = [10, 30, 50, 100, 150, 200, 250, 300, 500]
    window_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 500]
    # ks = [50]
    # window_sizes = [100, 300, 350]

    # 创建参数组合
    params = [(k, window_size, return_data) for k in ks for window_size in window_sizes]

    # 使用Pool并行执行
    with Pool(processes=8) as pool:
        results = tqdm(pool.starmap(run_genetic_algorithm, params))

    # 将结果转换为DataFrame并保存到Excel
    results_df = pd.DataFrame(results, columns=['K', 'Window Size', 'Best Subset'])
    with pd.ExcelWriter('best_subsets.xlsx') as writer:
        results_df.to_excel(writer, index=False)

