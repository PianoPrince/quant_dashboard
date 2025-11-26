# core/backtester.py
import pandas as pd
import numpy as np

class VectorBacktester:
    def __init__(self, commission: float = 0.0002, slippage: float = 0.0, initial_principal: float = 1_000_000.0, risk_free_rate: float = 0.018):
        """
        :param commission: 交易佣金费率
        :param slippage: 滑点费率
        :param initial_principal: 初始本金
        :param risk_free_rate: 无风险利率 (用于夏普比率)
        """
        self.commission = commission
        self.slippage = slippage
        self.initial_principal = initial_principal
        self.risk_free_rate = risk_free_rate

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        
        # 1. 计算市场收益率 (Ret_market)
        data['pct_change'] = data['close'].pct_change()
        
        # 2. 计算实际持仓 (Position)
        # T日的信号，T+1日执行
        data['position'] = data['target_weight'].shift(1).fillna(0.0)
        
        # 3. 计算策略收益 (Ret_strategy)
        # 基础收益
        data['strategy_ret'] = data['position'] * data['pct_change']
        
        # 4. 计算交易成本 (佣金 + 滑点)
        # 换手率 Turnover
        turnover = (data['position'] - data['position'].shift(1)).abs()
        data['cost'] = turnover * (self.commission + self.slippage)
        
        # 扣除成本后的净收益
        data['strategy_ret'] = data['strategy_ret'] - data['cost']
        
        # 5. 计算净值曲线 (Equity Curve) - 归一化净值 (起点 1.0)
        data['equity_curve'] = (1 + data['strategy_ret'].fillna(0.0)).cumprod()
        
        # 6. 计算绝对资金曲线 (Absolute Equity) - 真实资金
        data['equity_value'] = data['equity_curve'] * self.initial_principal
        
        # 7. 计算基准化净值 (Rebased Equity)
        first_valid_price = data['close'].dropna().iloc[0]
        data['equity_rebased'] = data['equity_curve'] * first_valid_price
        
        # 8. 计算回撤序列 (用于后续指标计算)
        data['cum_max'] = data['equity_curve'].cummax()
        data['drawdown'] = (data['equity_curve'] - data['cum_max']) / data['cum_max']
        
        return data

    def analyze_range(self, full_results: pd.DataFrame, start_date: str, end_date: str):
        """
        [新增] 特定区间分析功能
        截取指定时间段的数据，重置净值曲线，重新计算各项指标。
        """
        # 1. 切片数据
        # 确保索引是 datetime 类型
        if not isinstance(full_results.index, pd.DatetimeIndex):
            full_results.index = pd.to_datetime(full_results.index)
            
        mask = (full_results.index >= start_date) & (full_results.index <= end_date)
        period_df = full_results.loc[mask].copy()
        
        if period_df.empty:
            print(f"!!! 警告: 区间 {start_date} 至 {end_date} 无数据。")
            return None, None

        # 2. 重置净值曲线 (Rebase Equity)
        # 逻辑：假设在区间开始的第一天投入资金，净值归一化为 1.0
        start_equity = period_df['equity_curve'].iloc[0]
        period_df['equity_curve'] = period_df['equity_curve'] / start_equity
        
        # 3. 重置绝对资金 (可选，便于直观理解该区间的盈亏)
        period_df['equity_value'] = period_df['equity_curve'] * self.initial_principal
        
        # 4. 重置回撤 (Drawdown)
        # 必须基于新区间的净值重新计算，不能沿用历史的最大回撤
        period_df['cum_max'] = period_df['equity_curve'].cummax()
        period_df['drawdown'] = (period_df['equity_curve'] - period_df['cum_max']) / period_df['cum_max']
        
        # 5. 重置可视化用的基准净值 (Rebased Equity)
        # 让策略净值和该区间起点的股价对齐，方便绘图对比
        period_start_price = period_df['close'].iloc[0]
        period_df['equity_rebased'] = period_df['equity_curve'] * period_start_price
        
        # 6. 计算该区间的绩效表
        # 注意：这里会复用 get_summary 的逻辑，它会自动计算该区间内的 Alpha、Sharpe 等
        summary_df = self.get_summary(period_df)
        
        return period_df, summary_df

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算全面的回测绩效指标，并返回对比表格
        """
        # --- 基础数据准备 ---
        days = len(df)
        years = days / 252.0 if days > 0 else 0
        
        # 策略数据
        strategy_returns = df['strategy_ret'].fillna(0)
        strategy_equity = df['equity_curve']
        
        # 基准数据 (Buy & Hold)
        benchmark_returns = df['pct_change'].fillna(0)
        # 重构基准净值曲线
        benchmark_equity = (1 + benchmark_returns).cumprod()
        
        # 1. 计算通用风险收益指标 (策略 vs 基准)
        strategy_metrics = self._calculate_metrics(strategy_returns, strategy_equity, years)
        benchmark_metrics = self._calculate_metrics(benchmark_returns, benchmark_equity, years)
        
        # 2. 计算 Alpha (策略年化 - 基准年化)
        alpha = strategy_metrics['Annualized Return'] - benchmark_metrics['Annualized Return']
        
        # 3. 计算交易统计 (仅策略)
        trade_stats = self._calculate_trade_stats(df, years)
        
        # 4. 组装表格
        # 定义指标展示顺序和名称
        metrics_order = [
            'Total Return', 
            'Annualized Return', 
            'Volatility (Ann.)',
            'Max Drawdown',
            'Recovery Days',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Downside Deviation'
        ]
        
        # 辅助：为 Benchmark 填充交易统计的空值
        trade_stats_keys = list(trade_stats.keys())
        bench_trade_stats = [np.nan] * len(trade_stats_keys)

        data_dict = {
            'Metric': metrics_order + ['Alpha (Excess Return)'] + trade_stats_keys,
            'Strategy': [strategy_metrics[m] for m in metrics_order] + [alpha] + list(trade_stats.values()),
            'Benchmark': [benchmark_metrics[m] for m in metrics_order] + [0.0] + bench_trade_stats
        }
        
        summary_df = pd.DataFrame(data_dict).set_index('Metric')
        return summary_df

    def _calculate_metrics(self, returns, equity, years):
        """内部辅助函数：计算核心风险收益指标"""
        # 收益
        total_ret = equity.iloc[-1] - 1
        ann_ret = (equity.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 下行波动率
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 回撤
        cum_max = equity.cummax()
        drawdown = (equity - cum_max) / cum_max
        max_dd = drawdown.min()
        
        # 回撤修复天数 (Recovery Days)
        # 逻辑：计算创新高的间隔
        is_new_high = equity == cum_max
        high_dates = equity.index[is_new_high].to_series()
        if len(high_dates) > 1:
            recovery_days = high_dates.diff().max().days
        else:
            recovery_days = 0
            
        # 比率
        rf_daily = self.risk_free_rate / 252
        excess_ret = returns - rf_daily
        
        sharpe = (excess_ret.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        sortino = (excess_ret.mean() / downside_returns.std()) * np.sqrt(252) if (len(downside_returns) > 0 and downside_returns.std() != 0) else 0
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'Total Return': total_ret,
            'Annualized Return': ann_ret,
            'Volatility (Ann.)': volatility,
            'Max Drawdown': max_dd,
            'Recovery Days': recovery_days,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Downside Deviation': downside_std
        }

    def _calculate_trade_stats(self, df, years):
        """
        计算交易统计：
        1. Total Executions: 只要仓位变动就算一次交易（买/卖/调仓）
        2. Round-trip Trades: 完整的回合（开->平），用于计算胜率
        """
        # 1. 计算执行次数 (Executions)
        # 只要仓位发生变化 (diff != 0) 且不是第一天 (fillna(0))，就算一次操作
        position_diff = df['position'].diff().fillna(0)
        total_executions = (position_diff != 0).sum()

        # 2. 计算回合交易 (Round-trip) 用于胜率逻辑
        in_market = df['position'] != 0
        trade_starts = (in_market & ~in_market.shift(1).fillna(False))
        trade_ids = trade_starts.cumsum()
        
        trades = df[in_market].copy()
        trades['trade_id'] = trade_ids[in_market]
        
        if len(trades) > 0:
            trade_returns = trades.groupby('trade_id')['strategy_ret'].apply(lambda x: (1 + x).prod() - 1)
            total_round_trips = len(trade_returns)
            winning_trades = (trade_returns > 0).sum()
            win_rate = winning_trades / total_round_trips
            gross_profit = trade_returns[trade_returns > 0].sum()
            gross_loss = abs(trade_returns[trade_returns <= 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        else:
            total_round_trips = 0
            win_rate = 0.0
            profit_factor = np.nan
            
        # 这里的 Trade Freq 改为基于 Executions 计算，或者 Round-trip 均可
        # 为了响应你的需求"仓位变化算一次"，我们展示 Total Executions
        
        return {
            'Total Executions': total_executions,  # 新增：调仓总次数
            'Total Round-trips': total_round_trips, # 原 Total Trades
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
        }

