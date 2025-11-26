# core/strategy_engine.py
import pandas as pd
import numpy as np

class FRAMAStrategy:
    def __init__(self, df: pd.DataFrame, strong_threshold: float = 1.4, weak_threshold: float = 1.7):
        """
        :param df: 包含因子数据的 DataFrame
        :param strong_threshold: 强趋势分形维数阈值 (默认 1.4)
        :param weak_threshold: 弱趋势分形维数阈值 (默认 1.7)
        """
        self.df = df.copy()
        self.strong_th = strong_threshold
        self.weak_th = weak_threshold

    def generate_signals(self) -> pd.DataFrame:
        """
        基于 FRAMA 和分形维数 D 生成目标仓位
        """
        # 1. 提取因子
        if 'FRAMA' not in self.df.columns or 'D' not in self.df.columns:
            raise ValueError("数据缺失 'FRAMA' 或 'D' 列，请先运行 factor_lib 计算因子。")

        close = self.df['close']
        frama = self.df['FRAMA']
        d_value = self.df['D']
        
        # 2. 定义交易逻辑
        # 基础条件：价格在 FRAMA 上方
        price_above = close > frama
        
        # 场景 A: 强趋势区 (Strong Trend)
        # D < 强趋势分形维数阈值
        cond_strong_buy = price_above & (d_value < self.strong_th)
        
        # 场景 B: 弱趋势/震荡爬升区 (Weak Trend)
        # 强趋势分形维数阈值 <= D < 弱趋势分形维数阈值
        cond_weak_buy = price_above & (d_value < self.weak_th)
        
        # 3. 组装条件列表与仓位
        conditions = [
            cond_strong_buy,  # 优先级 1
            cond_weak_buy,    # 优先级 2
        ]
        
        choices = [
            1.0,  # 强买入仓位
            0.5,  # 弱买入仓位
        ]

        # 4. 生成目标仓位
        self.df['target_weight'] = np.select(
            condlist=conditions,
            choicelist=choices,
            default=0.0  # 卖出/空仓/观望
        )

        return self.df

class FRAMA_RSI_Strategy:
    def __init__(self, df: pd.DataFrame, 
                 strong_threshold: float = 1.4, 
                 weak_threshold: float = 1.7,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        """
        :param df: 包含因子数据的 DataFrame
        :param strong_threshold: 强趋势分形维数阈值
        :param weak_threshold: 弱趋势分形维数阈值
        :param rsi_overbought: RSI 超买阈值 (默认 70)
        :param rsi_oversold: RSI 超卖阈值 (默认 30)
        """
        self.df = df.copy()
        self.strong_th = strong_threshold
        self.weak_th = weak_threshold
        self.rsi_high = rsi_overbought
        self.rsi_low = rsi_oversold
    def generate_signals(self) -> pd.DataFrame:
        """
        基于 FRAMA (趋势) 和 RSI (动量) 生成混合信号
        """
        # 1. 校验因子
        required_cols = ['FRAMA', 'D', 'RSI']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"数据缺失，请确保计算了 {required_cols}")

        close = self.df['close']
        frama = self.df['FRAMA']
        d_value = self.df['D']
        rsi = self.df['RSI']

        # ==========================
        # 2. 基础逻辑 (FRAMA Trend)
        # ==========================
        price_above = close > frama

        # A. 强趋势 (D < 1.4)
        trend_strong = price_above & (d_value < self.strong_th)

        # B. 弱趋势 (1.4 <= D < 1.7)
        trend_weak = price_above & (d_value < self.weak_th)

        # ==========================
        # 3. 过滤器逻辑 (RSI Filter)
        # ==========================
        # 超买状态: RSI > 70 (可能回调，不宜满仓追高)
        is_overbought = rsi > self.rsi_high

        # 超卖状态: RSI < 30 (可能反弹，虽趋势向下但已有筑底迹象)
        is_oversold = rsi < self.rsi_low

        # ==========================
        # 4. 信号合成 (多级决策)
        # ==========================

        # 场景 1: 完美风暴 (Strong Buy)
        # 强趋势 + 没有超买 -> 满仓 1.0
        cond_perfect_buy = trend_strong & (~is_overbought)

        # 场景 2: 趋势过热 (Trend but Overheated)
        # 强趋势 + 已经超买 -> 稍微减仓保护，0.8
        cond_hot_trend = trend_strong & is_overbought

        # 场景 3: 弱趋势跟随 (Weak Buy)
        # 弱趋势 + 没有超买 -> 半仓 0.5
        cond_weak_buy = trend_weak & (~is_overbought)

        # 场景 4: 左侧抄底 (Contrarian Buy) [可选]
        # 价格还在均线下方(Trend Down) + RSI超卖(<30) -> 尝试性底仓 0.2
        # 这是一个激进策略，为了稳健可以设为 0.0
        cond_bottom_fish = (~price_above) & is_oversold

        # 组装条件 (优先级从高到低)
        conditions = [
            cond_perfect_buy,  # 1.0
            cond_hot_trend,    # 0.8
            cond_weak_buy,     # 0.5
            cond_bottom_fish,  # 0.2 (左侧)
        ]

        choices = [
            1.0,  # 满仓
            0.8,  # 略减仓
            0.5,  # 半仓
            0.2,  # 底仓
        ]

        # 5. 生成目标仓位
        self.df['target_weight'] = np.select(
            condlist=conditions,
            choicelist=choices,
            default=0.0  # 其他情况空仓 (震荡区 D>1.7 或 跌破均线且非超卖)
        )

        return self.df

class FRAMA_RSI_bb_Strategy:
    def __init__(self, df: pd.DataFrame, 
                 strong_threshold: float = 1.4, 
                 weak_threshold: float = 1.7,
                 rsi_overbought: float = 75,
                 rsi_oversold: float = 25,
                 bb_bw_low_k: float = 0.8,
                 bb_bw_high_k: float = 1.5):
        """
        :param df: 包含因子数据的 DataFrame
        :param bb_bw_low_k: 布林带宽低波动系数 (默认0.8)
        :param bb_bw_high_k: 布林带宽高波动系数 (默认1.5)
        """
        self.df = df.copy()
        self.strong_th = strong_threshold
        self.weak_th = weak_threshold
        self.rsi_high = rsi_overbought
        self.rsi_low = rsi_oversold
        self.bw_low_k = bb_bw_low_k
        self.bw_high_k = bb_bw_high_k

    def generate_signals(self) -> pd.DataFrame:
        """
        混合策略: FRAMA (趋势) + RSI (动量) + Bollinger (均值回归/突破)
        """
        # 1. 校验因子
        required_cols = ['FRAMA', 'D', 'RSI', 'BB_UB', 'BB_LB', 'BB_BW', 'BB_BW_MA']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"数据缺失，请确保计算了 {required_cols}")

        close = self.df['close']
        prev_close = close.shift(1) # P_{t-1}
        
        # FRAMA & RSI
        frama = self.df['FRAMA']
        d_value = self.df['D']
        rsi = self.df['RSI']
        
        # Bollinger
        ub = self.df['BB_UB']
        lb = self.df['BB_LB']
        bw = self.df['BB_BW']
        bw_ma = self.df['BB_BW_MA']
        prev_lb = lb.shift(1)
        prev_ub = ub.shift(1)

        # ==========================
        # 1. 布林带特定逻辑 (图片要求)
        # ==========================
        
        # 逻辑 1: 触及下轨 -> 看多
        # Pt <= LBt
        bb_touch_lb = close <= lb
        
        # 逻辑 2: 低波动后向上突破下轨 -> 强烈看多
        # BW < BW_avg * 0.8  AND  Pt > Pt-1  AND  Pt-1 <= LBt-1
        is_low_vol = bw < (bw_ma * self.bw_low_k)
        is_rebound = (close > prev_close) & (prev_close <= prev_lb)
        bb_strong_buy = is_low_vol & is_rebound
        
        # 逻辑 3: 触及上轨 -> 看空
        # Pt >= UBt
        bb_touch_ub = close >= ub
        
        # 逻辑 4: 高波动后从上轨回落 -> 强烈看空
        # BW > BW_avg * 1.5  AND  Pt < Pt-1  AND  Pt-1 >= UBt-1
        is_high_vol = bw > (bw_ma * self.bw_high_k)
        is_pullback = (close < prev_close) & (prev_close >= prev_ub)
        bb_strong_sell = is_high_vol & is_pullback

        # ==========================
        # 2. FRAMA 趋势逻辑
        # ==========================
        price_above_frama = close > frama
        trend_strong = price_above_frama & (d_value < self.strong_th)
        trend_weak = price_above_frama & (d_value < self.weak_th)
        
        # ==========================
        # 3. 信号综合 (权重分配)
        # ==========================
        
        # 场景 A: 强烈进攻 (Strong Buy)
        # 满足 布林带强买 OR (FRAMA强趋势 且 RSI健康)
        # 权重: 1.0 (满仓)
        cond_strong_signal = bb_strong_buy | (trend_strong & (rsi < self.rsi_high))
        
        # 场景 B: 抄底/低吸 (Buy / Accumulate)
        # 满足 布林带触底 OR (FRAMA弱趋势 且 RSI健康)
        # 权重: 0.6 (6成仓位)
        cond_normal_buy = bb_touch_lb | (trend_weak & (rsi < self.rsi_high))
        
        # 场景 C: 强烈清仓 (Strong Sell / Panic)
        # 满足 布林带强卖 OR 价格有效跌破 FRAMA
        # 权重: 0.0 (清仓)
        cond_strong_exit = bb_strong_sell | (close < frama)
        
        # 场景 D: 减仓/止盈 (Reduce)
        # 满足 布林带触顶 OR RSI超买
        # 权重: 0.4 (保留底仓)
        cond_reduce = bb_touch_ub | (rsi > self.rsi_high)

        # 优先级排序 (np.select 只要匹配到就会停止，所以优先级高的放前面)
        conditions = [
            cond_strong_exit,    # 优先级 1: 风控优先 (强卖) -> 0.0
            cond_strong_signal,  # 优先级 2: 强买 -> 1.0
            cond_normal_buy,     # 优先级 3: 普买 -> 0.6
            cond_reduce,         # 优先级 4: 减仓 -> 0.4
        ]
        
        choices = [
            0.0,
            1.0,
            0.6,
            0.4
        ]

        # 生成目标仓位
        self.df['target_weight'] = np.select(
            condlist=conditions,
            choicelist=choices,
            default=0.0  # 默认空仓
        )

        return self.df