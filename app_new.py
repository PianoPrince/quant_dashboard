import streamlit as st
import pandas as pd
import os
from config import Config
from core.data_handlers import ExcelHandler
from core.factor_lib import TechnicalFactors
from core.strategy_engine import FRAMA_RSI_bb_Strategy
from core.backtester import VectorBacktester
from core.visualizer import Visualizer

# ==============================================
# 0. 页面配置与自定义 CSS (美化核心)
# ==============================================
st.set_page_config(
    page_title="量化策略仪表盘 Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS 以复刻金融终端风格
st.markdown("""
    <style>
        /* 全局背景色 - 浅灰 */
        .stApp {
            background-color: #f4f5f8;
        }
        
        /* 顶部标题栏模拟 - 深蓝渐变 */
        .header-container {
            background: linear-gradient(90deg, #003366 0%, #004080 100%);
            padding: 1.5rem 2rem;
            border-radius: 0 0 10px 10px;
            margin: -4rem -4rem 2rem -4rem; /* 抵消 streamlit 默认 padding */
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header-title {
            font-size: 24px;
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
        }
        .header-subtitle {
            font-size: 14px;
            opacity: 0.8;
            margin-top: 5px;
        }

        /* 指标卡片样式 */
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #004080; /* 左侧蓝条装饰 */
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 28px;
            font-weight: bold;
            color: #d32f2f; /* 金融红 */
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            font-weight: 500;
        }
        
        /* 容器卡片化 (表格和图表) */
        .content-card {
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        /* 侧边栏美化 */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }
        
        /* 标题修饰 */
        h3 {
            color: #003366;
            font-weight: 600;
            border-bottom: 2px solid #f0f2f5;
            padding-bottom: 10px;
            margin-top: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# 1. 核心计算函数
# ==============================================

@st.cache_data
def load_and_process_data():
    if not os.path.exists(Config.DATA_FILE_PATH):
        st.error(f"找不到数据文件: {Config.DATA_FILE_PATH}")
        return None
    try:
        excel_handler = ExcelHandler(file_path=Config.DATA_FILE_PATH, sheet_name=Config.SHEET_NAME)
        raw_data = excel_handler.get_data()
        if raw_data.empty: return None
        raw_data.index = pd.to_datetime(raw_data.index)
        data = TechnicalFactors.calculate_frama(raw_data, window=Config.FRAMA_WINDOW, n_slow=Config.FRAMA_N_SLOW)
        data = TechnicalFactors.calculate_rsi(data, period=Config.RSI_PERIOD)
        data = TechnicalFactors.calculate_bollinger(data, period=Config.BB_WINDOW, std_dev=Config.BB_STD)
        return data
    except Exception as e:
        st.error(f"数据处理出错: {e}")
        return None

# 修改 run_strategy_and_backtest 以接收动态参数
def run_strategy_and_backtest(data, 
                              risk_free_rate=Config.RISK_FREE_RATE,
                              commission_rate=Config.COMMISSION_RATE,
                              initial_principal=Config.INITIAL_PRINCIPAL,
                              slippage=Config.SLIPPAGE):
    
    # 这里使用 Config 中的默认阈值，如果需要也可以通过参数传入
    strategy = FRAMA_RSI_bb_Strategy(
        data, 
        strong_threshold=Config.THRESHOLD_STRONG_TREND,
        weak_threshold=Config.THRESHOLD_WEAK_TREND,
        rsi_overbought=Config.RSI_OVERBOUGHT,
        rsi_oversold=Config.RSI_OVERSOLD,
        bb_bw_low_k=Config.BB_BW_LOW_THRESHOLD,
        bb_bw_high_k=Config.BB_BW_HIGH_THRESHOLD
    )
    data_with_signals = strategy.generate_signals()
    
    # 使用传入的动态参数初始化回测引擎
    backtester = VectorBacktester(
        commission=commission_rate, 
        slippage=slippage, 
        initial_principal=initial_principal, 
        risk_free_rate=risk_free_rate
    )
    
    full_results = backtester.run(data_with_signals)
    return backtester, full_results

def style_dataframe(df):
    """应用精确的格式化和智能的红绿配色"""
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
    styler.set_properties(**{
        'border-bottom': '1px solid #f0f0f0',
        'text-align': 'right',
        'padding': '12px',
        'font-family': 'Arial, sans-serif'
    })
    
    return styler

# ==============================================
# 2. 界面布局 (UI Layout) - 深度美化版
# ==============================================

# 自定义 Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">📈 量化回测交互式透视系统 <span style="font-size:14px; margin-left:15px; opacity:0.7;">V5.4 Professional</span></div>
        <div class="header-subtitle">基于 FRAMA + RSI + Bollinger Bands 的多因子复合策略</div>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ 参数控制台")
    
    data = load_and_process_data()
    if data is not None:
        st.success(f"✅ 数据已就绪 ({Config.ASSET_SYMBOL})")
        
        # --- 动态参数调整区域 ---
        with st.expander("🛠️ 账户与回测参数", expanded=False):
            input_risk_free_rate = st.number_input(
                "无风险利率 (Risk Free Rate)", 
                min_value=0.0, max_value=0.2, 
                value=Config.RISK_FREE_RATE, 
                step=0.005, 
                format="%.3f",
                help="用于计算夏普比率和Sortino比率的基准利率"
            )
            
            input_commission = st.number_input(
                "交易佣金 (Commission)", 
                min_value=0.0, max_value=0.01, 
                value=Config.COMMISSION_RATE, 
                step=0.0001, 
                format="%.4f"
            )
            
            input_slippage = st.number_input(
                "交易滑点 (Slippage)", 
                min_value=0.0, max_value=0.01, 
                value=Config.SLIPPAGE, 
                step=0.0001, 
                format="%.4f"
            )
            
            input_principal = st.number_input(
                "初始本金 (Principal)", 
                min_value=10000.0, 
                value=Config.INITIAL_PRINCIPAL, 
                step=10000.0
            )

        min_date = data.index.min().date()
        max_date = data.index.max().date()
        cfg_start = pd.to_datetime(Config.START_DATE).date()
        cfg_end = pd.to_datetime(Config.END_DATE).date()
        default_start = max(min_date, cfg_start) 
        default_start = min(default_start, max_date)
        default_end = min(max_date, cfg_end)
        default_end = max(default_end, min_date)

        st.markdown("---")
        st.markdown("**📅 回测区间选择**")
        start_date = st.date_input("开始日期", value=default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("结束日期", value=default_end, min_value=min_date, max_value=max_date)
        
        if start_date > end_date:
            st.error("开始日期必须早于结束日期！")
            st.stop()
            
        st.markdown("---")
        st.caption(f"📊 数据实际范围: {min_date} ~ {max_date}")
        st.caption("💡 提示: 调整参数后图表将自动刷新")

if data is not None:
    # 传入用户界面设置的参数，而不是 Config 中的静态值
    backtester, full_results = run_strategy_and_backtest(
        data,
        risk_free_rate=input_risk_free_rate,
        commission_rate=input_commission,
        initial_principal=input_principal,
        slippage=input_slippage
    )
    
    period_df, period_summary = backtester.analyze_range(full_results, str(start_date), str(end_date))
    
    if period_df is not None:
        # --- KPI 指标区 (模仿金融卡片) ---
        strat_ret = period_summary.loc['Total Return', 'Strategy']
        max_dd = period_summary.loc['Max Drawdown', 'Strategy']
        sharpe = period_summary.loc['Sharpe Ratio', 'Strategy']
        alpha = period_summary.loc['Alpha (Excess Return)', 'Strategy']
        
        col1, col2, col3, col4 = st.columns(4)
        
        def metric_card(label, value):
            return f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """
            
        with col1: st.markdown(metric_card("区间总收益", f"{strat_ret:.2%}"), unsafe_allow_html=True)
        with col2: st.markdown(metric_card("最大回撤", f"{max_dd:.2%}"), unsafe_allow_html=True)
        with col3: st.markdown(metric_card("夏普比率", f"{sharpe:.3f}"), unsafe_allow_html=True)
        with col4: st.markdown(metric_card("Alpha 超额", f"{alpha:.2%}"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True) 

        # --- 详细报表区 ---
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown("### 📋 详细绩效对比表")
            st_table = style_dataframe(period_summary)
            st.dataframe(st_table, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 图表区 ---
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown("### 📈 策略全景走势图")
            with st.spinner("正在绘制交互式图表..."):
                fig = Visualizer.plot_backtest_result(period_df, filename=None)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True, height=1000)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.warning("所选区间无数据。")
