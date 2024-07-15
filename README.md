
# HRP-FC/GMSC: A Strategy for Stock Selection Using a Correlation-Based Minimum Spanning Tree and Asset Allocation Using a Hierarchical Risk Parity Model - A Replication Study

**Author**: Li Siyuan

Notice: This file is translated from Chinese Mandarin. If Chinese Mandarin is your mother tone, please go read  "HRP-FC-GMSC-复现报告.pdf".

## Abstract
The Hierarchical Risk Parity (HRP) algorithm, proposed in 2016, integrates the hierarchical structure of a portfolio's correlation matrix to address the distribution of weights within the portfolio, with many scholars finding it performs well in handling risks after backtesting. Cho (2023) proposed a stock trading strategy that employs a minimum spanning tree based on a correlation matrix, minus global movements, for stock selection and uses the HRP algorithm to assign weights to each asset, noting good performance. This study replicates this strategy and includes consideration of transaction costs. Due to uncertainties in specific parameters and experimental methods, we used the simple degree of the minimum spanning tree instead of the mixed ranking strategy in the original article and achieved similar results. We found that at a risk-free rate of 0.02, this strategy does not significantly outperform the benchmark algorithm. Additionally, we believe that subtracting global movements from the correlation matrix may effectively improve the strategy's performance in portfolios with a small number of assets (5-20).

---

## 1. Project Directory and File Description

### 1.1 Project Overview

This paper replicates a trading strategy proposed by Cho (2023). We refer to this strategy as HRP-FC/GMSC. The process begins with generating a correlation matrix based on the time series data of daily returns for several assets, standardizing this matrix, and subtracting global movements. Based on the processed matrix, a minimum spanning tree is drawn, the most peripheral N assets are selected to construct the portfolio, and the HRP algorithm is used to backtest and generate its weights. Finally, the performance of the portfolio is evaluated.

Further, considering that the backtesting window in the literature is fixed at 120 trading days, we attempted to test the performance of this strategy over different backtesting windows. We plan to explore this aspect in future research.

### 1.2 Project File Guide

The root directory of this project is HRP-MST, but the working path for the code is set in HRP-MST/Scripts. In the HRP-MST directory, the Reference path contains the core references for this research; the WashedData path contains the data used for backtesting; the Scripts path contains the core code and the output of the runs.

Specifically, HRP-MST/Scripts/demo_single_portfolio.ipynb is a basic demonstration of stock selection based on the correlation MST as originally described, including visualization of intermediate data. The file original_method_paper.ipynb in the same directory is the core code of this paper, which includes the backtesting process and results for 18 combinations of asset numbers and different correlation matrix processing methods. The performance of each backtest is stored under the HRP-MST/Scripts/original_alpha/performance path, and the overall assessment of all results is in the HRP-MST\Scripts\original_strategy_replication_evaluation.xlsx file.

Further, we attempted to backtest more combinations of backtesting window lengths and asset numbers. Given the long run time of this part of the research, the code uses multiprocessing to accelerate. First, multi_select_stock.py is used for stock selection, with the results saved in the HRP-MST/Scripts/selected_stocks folder. Then, the saved portfolios are backtested using multi_backtest.py, with results in the HRP-MST/Scripts/multi_backtest_result path (this part of the work has not been completed, and we plan to continue it later). The temp.ipynb file tests these results and saves them in the all_result.csv file.



