import os

class Config:
    """
    全局配置类
    集中管理数据路径、回测参数、策略参数
    """
    # ==============================================================================
    # 1. 路径与环境配置 (Path Configuration)
    # ==============================================================================
    # 获取 config.py 当前文件所在的绝对路径 (即项目的根目录 E:\VSCode_Project\quant_research)
    # __file__ 代表当前脚本文件，os.path.abspath 获取绝对路径，os.path.dirname 获取目录名
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # 数据文件夹的绝对路径
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    
    # 报告文件夹的绝对路径
    REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
    
    # 资产配置
    ASSET_SYMBOL = "000300.SH"
    
    # 数据的完整绝对路径 (自动拼接)
    # 这样修改后，无论你在哪里运行命令，Python都能准确找到这个文件
    DATA_FILE_PATH = os.path.join(DATA_DIR, "CSI_300_Index.xlsx")
    SHEET_NAME = "CSI_300_Index"

    # ==============================
    # 2. 回测环境配置 (Backtest Context)
    # ==============================
    # 回测时间窗口 (格式: 'YYYY-MM-DD')
    START_DATE = '2018-01-01'
    END_DATE = '2026-01-01'

    # 账户设置
    INITIAL_PRINCIPAL = 1_000_000.0  # 初始本金
    COMMISSION_RATE = 0.0002         # 交易佣金
    SLIPPAGE = 0.0001                # 滑点
    RISK_FREE_RATE = 0.03            # 无风险利率

    # ==============================
    # 3. 策略参数配置 (Strategy Params)
    # ==============================
    # 3.1 FRAMA 参数
    FRAMA_WINDOW = 126               # 计算周期 n_total
    FRAMA_N_SLOW = 100               # 慢速衰减周期
    
    # FRAMA 交易阈值
    THRESHOLD_STRONG_TREND = 1.4     # 强趋势阈值 D < 1.4
    THRESHOLD_WEAK_TREND = 1.7       # 弱趋势/震荡阈值

    # 3.2 RSI 参数 (超买超卖)
    RSI_PERIOD = 14                  # RSI 计算周期
    RSI_OVERBOUGHT = 70              # 超买阈值 (卖出/减仓警告)
    RSI_OVERSOLD = 30                # 超卖阈值 (反弹/买入关注)

    # 3.3 布林带 (Bollinger Bands) 参数
    BB_WINDOW = 20           # 计算周期 (默认20)
    BB_STD = 2.0             # 标准差倍数 (默认2.0)
    BB_BW_LOW_THRESHOLD = 0.8   # 低波动阈值系数 (BW < Avg * 0.8)
    BB_BW_HIGH_THRESHOLD = 1.5  # 高波动阈值系数 (BW > Avg * 1.5)
    # ==============================
    # 4. 系统设置 (System)
    # =============================
    RANDOM_SEED = 42                 # 随机种子