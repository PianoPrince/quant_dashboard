import streamlit as st
import pandas as pd
import os
import matplotlib.colors as mcolors
from config import Config
from core.data_handlers import ExcelHandler
from core.factor_lib import TechnicalFactors
from core.strategy_engine import FRAMA_RSI_bb_Strategy
from core.backtester import VectorBacktester
from core.visualizer import Visualizer

# ==============================================
# 0. 页面配置与自定义 CSS
# ==============================================
st.set_page_config(
    page_title="量化策略仪表盘 Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #f4f5f8; }
        .header-container {
            background: linear-gradient(90deg, #003366 0%, #004080 100%);
            padding: 1.5rem 2rem;
            border-radius: 0 0 10px 10px;
            margin: -4rem -4rem 2rem -4rem;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header-title { font-size: 24px; font-weight: 600; margin: 0; display: flex; align-items: center; }
        .header-subtitle { font-size: 14px; opacity: 0.8; margin-top: 5px; }
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #004080;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.2s;
        }
        .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        /* metric-value 颜色现在通过行内样式动态控制，这里只保留基础样式 */
        .metric-value { font-family: 'Helvetica Neue', sans-serif; font-size: 28px; font-weight: bold; margin-bottom: 5px; }
        .metric-label { font-size: 14px; color: #666; font-weight: 500; }
        .content-card {
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .formula-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
            margin-bottom: 10px;
        }
        section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
        h3 { color: #003366; font-weight: 600; border-bottom: 2px solid #f0f2f5; padding-bottom: 10px; margin-top: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# 1. 核心计算函数
# ==============================================

@st.cache_data
def load_and_calculate_factors(frama_window, frama_n_slow, rsi_period, bb_window, bb_std):
    if not os.path.exists(Config.DATA_FILE_PATH):
        st.error(f"找不到数据文件: {Config.DATA_FILE_PATH}")
        return None
    try:
        excel_handler = ExcelHandler(file_path=Config.DATA_FILE_PATH, sheet_name=Config.SHEET_NAME)
        raw_data = excel_handler.get_data()
        if raw_data.empty: return None
        raw_data.index = pd.to_datetime(raw_data.index)
        
        data = TechnicalFactors.calculate_frama(raw_data, window=frama_window, n_slow=frama_n_slow)
        data = TechnicalFactors.calculate_rsi(data, period=rsi_period)
        data = TechnicalFactors.calculate_bollinger(data, period=bb_window, std_dev=bb_std)
        return data
    except Exception as e:
        st.error(f"数据处理出错: {e}")
        return None

def run_strategy_and_backtest(data, strong_th, weak_th, rsi_high, rsi_low, bb_low_k, bb_high_k,
                              risk_free_rate, commission_rate, initial_principal, slippage):
    
    strategy = FRAMA_RSI_bb_Strategy(
        data, 
        strong_threshold=strong_th,
        weak_threshold=weak_th,
        rsi_overbought=rsi_high,
        rsi_oversold=rsi_low,
        bb_bw_low_k=bb_low_k,
        bb_bw_high_k=bb_high_k
    )
    data_with_signals = strategy.generate_signals()
    
    backtester = VectorBacktester(
        commission=commission_rate, 
        slippage=slippage, 
        initial_principal=initial_principal, 
        risk_free_rate=risk_free_rate
    )
    
    full_results = backtester.run(data_with_signals)
    return backtester, full_results

def style_dataframe(df):
    rename_dict = {
        'Total Return': 'Total Return (总收益率)',
        'Annualized Return': 'Annualized Return (年化收益率)',
        'Volatility (Ann.)': 'Volatility (年化波动率)',
        'Max Drawdown': 'Max Drawdown (最大回撤)',
        'Downside Deviation': 'Downside Deviation (下行偏差)',
        'Alpha (Excess Return)': 'Alpha (超额收益)',
        'Win Rate': 'Win Rate (胜率)',
        'Recovery Days': 'Recovery Days (修复天数)',
        'Total Executions': 'Total Executions (总执行)',
        'Total Round-trips': 'Total Round-trips (总回合)',
        'Total Trades': 'Total Trades (总交易)',
        'Trade Freq (Yearly)': 'Trade Freq (年均交易)',
        'Sharpe Ratio': 'Sharpe Ratio (夏普)',
        'Sortino Ratio': 'Sortino Ratio (索提诺)',
        'Calmar Ratio': 'Calmar Ratio (卡玛)',
        'Profit Factor': 'Profit Factor (盈亏比)'
    }
    df_renamed = df.rename(index=rename_dict)
    
    format_dict = {
        'Total Return (总收益率)': '{:.2%}',
        'Annualized Return (年化收益率)': '{:.2%}',
        'Volatility (年化波动率)': '{:.2%}',
        'Max Drawdown (最大回撤)': '{:.2%}',
        'Downside Deviation (下行偏差)': '{:.2%}',
        'Alpha (超额收益)': '{:.2%}',
        'Win Rate (胜率)': '{:.2%}',
        'Recovery Days (修复天数)': '{:.0f}',
        'Total Executions (总执行)': '{:.0f}',
        'Total Round-trips (总回合)': '{:.0f}',
        'Trade Freq (年均交易)': '{:.1f}',
        'Sharpe Ratio (夏普)': '{:.3f}',
        'Sortino Ratio (索提诺)': '{:.3f}',
        'Calmar Ratio (卡玛)': '{:.3f}',
        'Profit Factor (盈亏比)': '{:.3f}'
    }
    
    styler = df_renamed.style.format(None, na_rep="-")
    for metric, fmt_str in format_dict.items():
        if metric in df_renamed.index:
            styler.format(fmt_str, subset=pd.IndexSlice[metric, :])

    def color_text(val):
        if not isinstance(val, (int, float)): return ''
        if val > 0: return 'color: #D32F2F; font-weight: bold'
        if val < 0: return 'color: #388E3C; font-weight: bold'
        return 'color: #333333' 
    
    styler.map(color_text)
    styler.set_properties(**{'border-bottom': '1px solid #f0f0f0', 'text-align': 'right', 'padding': '12px', 'font-family': 'Arial, sans-serif'})
    return styler

# ==============================================
# 2. 界面布局 (UI Layout)
# ==============================================

st.markdown("""
    <div class="header-container">
        <div class="header-title">📈 量化回测交互式透视系统 <span style="font-size:14px; margin-left:15px; opacity:0.7;">V5.6 Professional</span></div>
        <div class="header-subtitle">基于 FRAMA + RSI + Bollinger Bands 的多因子复合策略</div>
    </div>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("### ⚙️ 参数控制台")
    
    # 0. 数据加载预处理 (为了获取日期范围)
    data = load_and_calculate_factors(Config.FRAMA_WINDOW, Config.FRAMA_N_SLOW, Config.RSI_PERIOD, Config.BB_WINDOW, Config.BB_STD) # 初始加载使用默认参
    
    if data is not None:
        st.success(f"✅ 数据已就绪 ({Config.ASSET_SYMBOL})")
        min_date, max_date = data.index.min().date(), data.index.max().date()
        cfg_start, cfg_end = pd.to_datetime(Config.START_DATE).date(), pd.to_datetime(Config.END_DATE).date()
        def_start = min(max(min_date, cfg_start), max_date)
        def_end = max(min(max_date, cfg_end), min_date)

        # 1. [修改] 回测区间选择 (移至最上方)
        st.markdown("**📅 回测区间选择**")
        start_date = st.date_input("开始日期", value=def_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("结束日期", value=def_end, min_value=min_date, max_value=max_date)
        if start_date > end_date: st.error("开始日期必须早于结束日期！"); st.stop()
        st.markdown("---")

    # 2. 策略参数配置 (重新加载以应用动态因子参数)
    with st.expander("🧠 策略参数配置", expanded=True):
        st.caption("**因子计算参数 (修改将重新计算)**")
        in_frama_win = st.number_input("FRAMA 周期", value=Config.FRAMA_WINDOW, step=2)
        in_rsi_period = st.number_input("RSI 周期", value=Config.RSI_PERIOD, step=1)
        in_bb_win = st.number_input("布林带 周期", value=Config.BB_WINDOW, step=1)
        in_bb_std = st.number_input("布林带 标准差", value=Config.BB_STD, step=0.1, format="%.1f")
        
        # 重新计算数据 (如果参数变化)
        data = load_and_calculate_factors(in_frama_win, Config.FRAMA_N_SLOW, in_rsi_period, in_bb_win, in_bb_std)

        st.markdown("---")
        st.caption("**交易阈值参数 (修改即时生效)**")
        # [修改] 拓宽 Slider 范围
        in_strong_th = st.slider("FRAMA 强趋势阈值 (D < ?)", 1.0, 1.9, Config.THRESHOLD_STRONG_TREND, 0.05)
        in_rsi_over = st.slider("RSI 超买阈值", 50, 99, Config.RSI_OVERBOUGHT, 1)
        in_rsi_under = st.slider("RSI 超卖阈值", 1, 50, Config.RSI_OVERSOLD, 1)
    
    # 3. 账户与回测参数
    with st.expander("🛠️ 账户与回测设置", expanded=False):
        in_rf = st.number_input("无风险利率", 0.0, 0.2, Config.RISK_FREE_RATE, 0.005, format="%.3f")
        in_comm = st.number_input("交易佣金", 0.0, 0.01, Config.COMMISSION_RATE, 0.0001, format="%.4f")
        in_slip = st.number_input("交易滑点", 0.0, 0.01, Config.SLIPPAGE, 0.0001, format="%.4f")
        in_capital = st.number_input("初始本金", 10000.0, value=Config.INITIAL_PRINCIPAL, step=10000.0)

# --- 主界面 ---
if data is not None:
    backtester, full_results = run_strategy_and_backtest(
        data,
        strong_th=in_strong_th, weak_th=Config.THRESHOLD_WEAK_TREND,
        rsi_high=in_rsi_over, rsi_low=in_rsi_under,
        bb_low_k=Config.BB_BW_LOW_THRESHOLD, bb_high_k=Config.BB_BW_HIGH_THRESHOLD,
        risk_free_rate=in_rf, commission_rate=in_comm, initial_principal=in_capital, slippage=in_slip
    )
    
    period_df, period_summary = backtester.analyze_range(full_results, str(start_date), str(end_date))
    
    if period_df is not None:
        strat_ret = period_summary.loc['Total Return', 'Strategy']
        max_dd = period_summary.loc['Max Drawdown', 'Strategy']
        sharpe = period_summary.loc['Sharpe Ratio', 'Strategy']
        alpha = period_summary.loc['Alpha (Excess Return)', 'Strategy']
        
        col1, col2, col3, col4 = st.columns(4)
        
        # [修改] 指标卡片：根据数值正负显示红绿
        def metric_card(label, value, raw_value):
            # A股风格：正数红，负数绿
            color = "#d32f2f" if raw_value >= 0 else "#388E3C"
            return f"""<div class="metric-card"><div class="metric-value" style="color: {color}">{value}</div><div class="metric-label">{label}</div></div>"""
            
        with col1: st.markdown(metric_card("区间总收益", f"{strat_ret:.2%}", strat_ret), unsafe_allow_html=True)
        with col2: st.markdown(metric_card("最大回撤", f"{max_dd:.2%}", max_dd), unsafe_allow_html=True)
        with col3: st.markdown(metric_card("夏普比率", f"{sharpe:.3f}", sharpe), unsafe_allow_html=True)
        with col4: st.markdown(metric_card("Alpha 超额", f"{alpha:.2%}", alpha), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True) 

        # 策略逻辑展示区域
        with st.expander("📖 策略逻辑详细说明 (Strategy Logic)", expanded=False):
            st.markdown("""
            **核心策略：FRAMA + RSI + Bollinger Bands 多因子混合策略**
            
            ### 1. 趋势识别模块 (FRAMA - Fractal Adaptive Moving Average)
            
            **数学定义**：
            - **价格波幅计算**：$N_{total} = \\max(H_{window}) - \\min(L_{window})$
            - **分形维数**：$D = \\frac{\\log_2(N_1 + N_2)}{\\log_2(N_3)} + 1$，其中 $D \\in [1.0, 2.0]$
            - **动态平滑系数**：$\\alpha = \\exp(-4.6 \\times (D - 1))$，且 $\\alpha \\geq \\frac{2}{N_{slow} + 1}$
            - **FRAMA迭代公式**：$FRAMA_t = \\alpha_t \\times Close_t + (1 - \\alpha_t) \\times FRAMA_{t-1}$
            
            **趋势状态判定**：
            - **强趋势条件**：$Close \\geq FRAMA \\text{ AND } D < """ + str(in_strong_th) + """$
            - **趋势破坏条件**：$Close < FRAMA$
            
            ### 2. 动量过滤模块 (RSI - Relative Strength Index)
            
            **数学定义**（周期=14）：
            - **上涨幅度**：$U_t = \\max(Close_t - Close_{t-1}, 0)$
            - **下跌幅度**：$D_t = \\max(Close_{t-1} - Close_t, 0)$
            - **平均值计算**：$AvgU = EMA(U, 14)$，$AvgD = EMA(D, 14)$
            - **RSI公式**：$RSI = 100 \\times \\frac{AvgU}{AvgU + AvgD}$
            
            **动量状态判定**：
            - **超买阈值**：$RSI > """ + str(in_rsi_over) + """$（限制追高）
            - **超卖阈值**：$RSI < """ + str(in_rsi_under) + """$（提示机会）
            - **健康动量**：$RSI \\leq """ + str(in_rsi_over) + """$
            
            ### 3. 波动性分析模块 (Bollinger Bands)
            
            **数学定义**（窗口=20，标准差倍数=2）：
            - **中轨**：$MB = SMA(Close, 20)$
            - **标准差**：$\\sigma = \\text{StdDev}(Close, 20)$
            - **上轨**：$UB = MB + 2 \\times \\sigma$
            - **下轨**：$LB = MB - 2 \\times \\sigma$
            - **带宽**：$BW = \\frac{UB - LB}{MB}$
            - **平均带宽**：$BW_{MA} = SMA(BW, 20)$
            
            **波动性状态判定**：
            - **低波动条件**：$BW < (BW_{MA} \\times 0.8)$
            - **高波动条件**：$BW > (BW_{MA} \\times 1.5)$
            
            ### 4. 仓位决策逻辑 (优先级顺序)
            
            **决策规则**（按优先级从高到低）：
            
            | 优先级 | 决策条件 | 目标仓位 | 数学表达式 |
            |--------|----------|----------|------------|
            | 1 (最高) | **强清仓** | 0.0 | $(Close < FRAMA) \\text{ OR } [BW > (BW_{MA} \\times 1.5) \\text{ AND } Close_t < Close_{t-1} \\text{ AND } Close_{t-1} \\geq UB_{t-1}]$ |
            | 2 | **强买入** | 1.0 | $[BW < (BW_{MA} \\times 0.8) \\text{ AND } Close_t > Close_{t-1} \\text{ AND } Close_{t-1} \\leq LB_{t-1}] \\text{ OR } [(D < """ + str(in_strong_th) + """) \\text{ AND } (RSI \\leq """ + str(in_rsi_over) + """)]$ |
            | 3 | **普买/触底** | 0.6 | $(Close \\leq LB) \\text{ OR } [(D < 1.7) \\text{ AND } (RSI \\leq """ + str(in_rsi_over) + """)]$ |
            | 4 | **减仓/触顶** | 0.4 | $(Close \\geq UB) \\text{ OR } (RSI > """ + str(in_rsi_over) + """)$ |
            | 默认 | **空仓观望** | 0.0 | 未满足上述任何条件 |
            
            **策略核心原则**：
            1. **风险控制优先**：强清仓条件具有最高优先级，确保及时规避重大风险
            2. **趋势跟随**：在强趋势且动量健康时满仓参与
            3. **波动性自适应**：根据市场波动状态（带宽指标）调整交易策略
            4. **多因子验证**：单一指标不足以触发交易，需多个因子协同确认
            """)
        
        with st.container():
            # st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown("### 📋 详细绩效对比表")
            st_table = style_dataframe(period_summary)
            st.dataframe(st_table, width='stretch', height= 35 * (len(period_summary) + 1) + 3)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 指标说明与公式 ---
        with st.expander("📚 指标计算逻辑与数学公式说明 (Metric Explanations)", expanded=False):
            st.markdown("""
            <div class="formula-box">
            
            #### 1. 收益类指标 (Return Metrics)
            - **总收益率 (Total Return)**: $ R_{total} = \\frac{P_{end} - P_{start}}{P_{start}} $
            - **年化收益率 (Annualized Return)**: $ R_{ann} = (1 + R_{total})^{\\frac{1}{years}} - 1 $
            - **Alpha (超额收益)**: $ \\alpha = R_{ann, strategy} - R_{ann, benchmark} $

            #### 2. 风险类指标 (Risk Metrics)
            - **最大回撤 (Max Drawdown)**: $ MDD = \\min \\left( \\frac{P_t - \\max(P_{0...t})}{\\max(P_{0...t})} \\right) $
            - **年化波动率 (Volatility)**: $ \\sigma_{ann} = \\sigma_{daily} \\times \\sqrt{252} $
            
            #### 3. 风险调整收益 (Risk-Adjusted Return)
            - **夏普比率 (Sharpe Ratio)**: $ Sharpe = \\frac{E[R_p - R_f]}{\\sigma_p} $
            - **索提诺比率 (Sortino Ratio)**: $ Sortino = \\frac{E[R_p - R_f]}{\\sigma_{downside}} $
            - **卡玛比率 (Calmar Ratio)**: $ Calmar = \\frac{R_{ann}}{|MDD|} $
            </div>
            """, unsafe_allow_html=True)
            
        with st.container():
            # st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown("### 📈 策略全景走势图")
            with st.spinner("正在绘制交互式图表..."):
                # [核心修复] 将动态阈值参数传递给 Visualizer，确保图表与侧边栏设置一致
                fig = Visualizer.plot_backtest_result(
                    period_df, 
                    filename=None,
                    strong_th=in_strong_th,
                    weak_th=Config.THRESHOLD_WEAK_TREND,
                    rsi_high=in_rsi_over,
                    rsi_low=in_rsi_under
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.warning("所选区间无数据。")


