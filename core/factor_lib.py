# core/factor_lib.py
import pandas as pd
import numpy as np

class TechnicalFactors:
    @staticmethod       # 强制子类实现静态方法
    def calculate_frama(df: pd.DataFrame, window: int = 126, n_slow: int = 100) -> pd.DataFrame:
        """
        计算分形自适应移动平均线 (FRAMA)。

        基于分形几何原理，根据价格变化的维度动态调整平滑系数。
        当市场趋势明显时（维度低），FRAMA 紧随价格；
        当市场震荡时（维度高），FRAMA 变平滑以过滤噪音。

        Math
        ----
        D = log2((N1 + N2) / N3) + 1

        Alpha = exp(-4.6 * (D - 1))

        Alpha ∈ [2 / (n_slow + 1), 1.0]

        Parameters
        ----------
        df : pd.DataFrame
            必须包含 'high', 'low', 'close' 列的时间序列数据。
        window : int, optional
            总窗口大小 (n_total)，必须为偶数。默认值为 20。
        n_slow : int, optional
            最慢 EMA 周期限制，用于计算 Alpha 的下限。默认值为 200。

        Returns
        -------
        pd.DataFrame
            返回包含以下新列的 DataFrame：
            - 'FRAMA': 计算出的均线值
            - 'D': 分形维度值
            
        Raises
        ------
        ValueError
            如果 window 不是偶数或列名缺失时抛出。
        """
        # 0. 基础校验
        if window % 2 != 0:
            raise ValueError(f"Window (n_total) 必须是偶数，当前为: {window}")
        
        # 避免修改原始数据
        data = df.copy()
        
        # 准备基础数据
        high = data['high']
        low = data['low']
        close = data['close']
        
        half_window = window // 2
        
        # 1. 向量化计算 N1, N2, N3
        # 辅助函数：计算波幅
        def get_amplitude(h, l, w):
            return h.rolling(w).max() - l.rolling(w).min()

        # N3: 整个窗口的波幅
        n3 = get_amplitude(high, low, window)
        
        # N2: 后半段（当前）波幅
        n2 = get_amplitude(high, low, half_window)
        
        # N1: 前半段（过去）波幅，需要 shift
        n1 = get_amplitude(high, low, half_window).shift(half_window)
        
        # 2. 计算分形维数 D
        # 处理分母为0的情况
        n3 = n3.replace(0, 0.0001)
        
        # 公式：D = (log(N1+N2) - log(N3)) / log(2) + 1
        # 等价于：log2((N1+N2)/N3) + 1
        ratio = (n1 + n2) / n3
        ratio = ratio.replace(0, 1).fillna(1) # 避免 log(0)
        data['D'] = np.log(ratio) / np.log(2) + 1
        data['D'] = data['D'].clip(lower=1.0) # 维数D≥1
        
        # 3. 计算平滑常数 Alpha
        # alpha = exp(-4.6 * (D - 1))
        alpha_series = np.exp(-4.6 * (data['D'] - 1))
        
        # 4. 用EMA中的 Alpha 动态钳制 Alpha 最小值
        min_alpha = 2 / (n_slow + 1)
        alpha_series = alpha_series.clip(lower=min_alpha, upper=1.0)
        
        # 5. 递归计算 FRAMA
        # 将 Series 转换为 numpy array 以加速循环
        price_values = close.values
        alpha_values = alpha_series.values
        frama_values = np.full_like(price_values, np.nan, dtype=float)
        
        # 找到第一个有效 D 值的索引 (window 行数据之后)
        start_idx = window 
        
        # 初始化：第一个有效点的 FRAMA 等于收盘价
        if start_idx < len(price_values):
            frama_values[start_idx] = price_values[start_idx]
            
            # 循环计算 FRAMA[i] = alpha * Price[i] + (1-alpha) * FRAMA[i-1]
            for i in range(start_idx + 1, len(price_values)):
                a = alpha_values[i]
                # 如果 alpha 是 NaN，沿用上一个值
                if np.isnan(a):
                    frama_values[i] = frama_values[i-1]
                else:
                    frama_values[i] = (a * price_values[i]) + ((1 - a) * frama_values[i-1])

        data['FRAMA'] = frama_values
        
        return data

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算相对强弱指数 (RSI)
        
        Math
        ----
        U_t = max(P_t - P_{t-1}, 0)

        D_t = max(P_{t-1} - P_t, 0)

        AvgU = EMA(U, period)

        AvgD = EMA(D, period)

        RS = AvgU / AvgD
        
        RSI = 100 / (1 + 1/RS) = 100 * AvgU / (AvgU + AvgD)

        :param df: 必须包含 'close' 列
        :param period: 计算周期，默认 14
        :return: 包含 'RSI' 列的 DataFrame
        """
        data = df.copy()
        
        # 1. 计算每日价格变化 Delta P
        delta = data['close'].diff()
        
        # 2. 区分上涨幅度 U 和 下跌幅度 D
        # U: 只保留正值，负值变为0
        # D: 只保留负值的绝对值，正值变为0
        u = delta.clip(lower=0)
        d = -1 * delta.clip(upper=0)
        
        # 3. 计算 EMA 平滑平均值 (AvgU, AvgD)
        # 使用 pandas ewm (Exponential Weighted Moving Average)
        # adjust=False 对应标准的递归实现: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
        avg_u = u.ewm(span=period, adjust=False).mean()
        avg_d = d.ewm(span=period, adjust=False).mean()
        
        # 4. 计算 RS 和 RSI
        # RSI = 100 - 100 / (1 + RS) 
        # 等价于 RSI = 100 * AvgU / (AvgU + AvgD) 以避免除以零错误
        sum_avg = avg_u + avg_d
        rsi = 100 * avg_u / sum_avg
        
        # 处理分母为0的极端情况 (填补为50或100，视情况而定，通常填50表示无趋势)
        rsi = rsi.fillna(50)
        
        data['RSI'] = rsi
        return data

    @staticmethod
    def calculate_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        计算布林带 (Bollinger Bands) 及带宽指标
        
        Formulas:
        MB = SMA(Close, 20)

        UB = MB + 2 * std

        LB = MB - 2 * std

        BW = (UB - LB) / MB

        BW_MA = MA(BW, 20)
        """
        data = df.copy()
        close = data['close']
        
        # 1. 计算中轨 (MB) 和 标准差 (sigma)
        mb = close.rolling(window=period).mean()
        sigma = close.rolling(window=period).std()
        
        # 2. 计算上轨 (UB) 和 下轨 (LB)
        ub = mb + std_dev * sigma
        lb = mb - std_dev * sigma
        
        # 3. 计算带宽 (BandWidth) = (UB - LB) / MB
        # 注意处理分母为0的情况 (虽然股价通常不为0)
        bw = (ub - lb) / mb
        
        # 4. 计算带宽的均值 (用于判断低波动/高波动)
        # 这里默认使用与布林带相同的周期 (20) 计算均值
        bw_ma = bw.rolling(window=period).mean()
        
        # 保存结果
        data['BB_UB'] = ub
        data['BB_MB'] = mb
        data['BB_LB'] = lb
        data['BB_BW'] = bw
        data['BB_BW_MA'] = bw_ma
        
        return data

    @staticmethod
    def calculate_atr_stop(df: pd.DataFrame, period=14, multiplier=3.0) -> pd.DataFrame:
        """
        计算 ATR 及 动态止损线
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # 计算吊灯止损线 (Chandelier Exit) 概念：最高价回撤 k * ATR
        long_stop = df['high'].rolling(period).max() - (multiplier * atr)
        
        return pd.DataFrame({'atr': atr, 'stop_price': long_stop})