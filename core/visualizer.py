# core/visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    """
    V5.2 可视化引擎
    集成：价格(含布林带/信号标记)、FRAMA状态、RSI指标、净值、持仓监控
    """

    @staticmethod
    def plot_backtest_result(df: pd.DataFrame, filename: str = 'frama_backtest_report.html'):
        """
        绘制全功能回测报告
        """
        plot_data = df.copy()
        
        # ------------------------------------------------------
        # 1. 数据准备
        # ------------------------------------------------------
        close = plot_data['close']
        high = plot_data['high']
        low = plot_data['low']
        frama = plot_data.get('FRAMA', pd.Series(np.nan, index=plot_data.index))
        d_val = plot_data.get('D', pd.Series(np.nan, index=plot_data.index))
        rsi = plot_data.get('RSI', pd.Series(np.nan, index=plot_data.index))
        equity_rebased = plot_data.get('equity_rebased', close)

        # 布林带数据
        bb_ub = plot_data.get('BB_UB', pd.Series(np.nan, index=plot_data.index))
        bb_lb = plot_data.get('BB_LB', pd.Series(np.nan, index=plot_data.index))

        # ==========================================
        # 2. 信号掩码计算 (用于绘制图标)
        # ==========================================
        
        # A. RSI 信号
        # 筛选出 RSI < 30 的点
        mask_rsi_buy = rsi < 30
        # 筛选出 RSI > 70 的点
        mask_rsi_sell = rsi > 70
        
        # B. 布林带触轨信号
        # 价格触及下轨 (Close <= LB)
        mask_bb_touch_lb = close <= bb_lb
        # 价格触及上轨 (Close >= UB)
        mask_bb_touch_ub = close >= bb_ub
        
        # C. D值强趋势信号 (检测进入 D < 1.4 的瞬间)
        # 逻辑：今天 D < 1.4 且 昨天 D >= 1.4
        mask_d_strong_entry = (d_val < 1.4) & (d_val.shift(1) >= 1.4)

        # ==========================================
        # 3. FRAMA 状态分段 (用于变色线)
        # ==========================================
        mask_breakdown = close < frama
        mask_strong = (~mask_breakdown) & (d_val < 1.4)
        mask_weak = (~mask_breakdown) & (d_val >= 1.4) & (d_val < 1.7)
        mask_noise = (~mask_breakdown) & (d_val >= 1.7)

        def get_segment(mask):
            segment = frama.copy()
            segment[~mask] = np.nan
            return segment

        frama_breakdown = get_segment(mask_breakdown)
        frama_strong = get_segment(mask_strong)
        frama_weak = get_segment(mask_weak)
        frama_noise = get_segment(mask_noise)

        # ------------------------------------------------------
        # 4. 构建画布
        # ------------------------------------------------------
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            # 子图间距，防止标题和坐标轴重叠
            vertical_spacing=0.08,
            # 调整高度比例，留出更多空间给间距
            row_heights=[0.45, 0.15, 0.15, 0.25],
            specs=[[{"secondary_y": False}], 
                   [{"secondary_y": False}], # FRAMA
                   [{"secondary_y": False}], # RSI
                   [{"secondary_y": True}]],
            subplot_titles=('价格行为 & 信号标记', '分形维数 D', 'RSI 相对强弱', '持仓 & 盈亏')
        )

        # === Row 1: 主图 (K线 + 布林带 + FRAMA + 净值) ===
    
        # 1. 布林带通道 (放在最底层)
        # 上轨 (透明线)
        fig.add_trace(go.Scatter(
            x=plot_data.index, y=bb_ub, 
            mode='lines', line=dict(width=0), 
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

        # 下轨 (带填充)
        fig.add_trace(go.Scatter(
            x=plot_data.index, y=bb_lb, 
            mode='lines', line=dict(width=0), 
            fill='tonexty', # 填充到上一个trace (上轨)
            fillcolor='rgba(128, 0, 128, 0.1)', # 浅紫色透明
            name='布林带通道', hoverinfo='skip'
        ), row=1, col=1)

        # 2. K线
        fig.add_trace(go.Candlestick(
            x=plot_data.index,
            open=plot_data['open'], high=plot_data['high'], low=plot_data['low'], close=plot_data['close'],
            name='K线',
            # 上涨/下跌颜色
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
        ), row=1, col=1)

        # 3. FRAMA 状态线
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_strong, mode='lines', line=dict(color='blue', width=2), name='强趋势'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_weak, mode='lines', line=dict(color='orange', width=2), name='弱趋势'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_noise, mode='lines', line=dict(color='gray', width=1, dash='dot'), name='噪音'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_breakdown, mode='lines', line=dict(color='purple', width=2, dash='dash'), name='破坏'), row=1, col=1)

        # 4. 策略净值
        fig.add_trace(go.Scatter(x=plot_data.index, y=equity_rebased, mode='lines', line=dict(color='#FFD700', width=2), name='策略净值'), row=1, col=1)

        # 5 信号标记 (Markers) 
        # A. RSI 超卖 (买入关注) - 绿色向上三角
        # 放在最低价下方 1% 处，避免遮挡
        fig.add_trace(go.Scatter(
            x=plot_data[mask_rsi_buy].index,
            y=low[mask_rsi_buy] * 0.99, 
            mode='markers',
            marker=dict(symbol='triangle-up', size=8, color='#00C853'),
            name='RSI超卖(<30)',
            hovertemplate='RSI超卖: %{x|%Y-%m-%d}'
        ), row=1, col=1)

        # B. RSI 超买 (卖出关注) - 红色向下三角
        # 放在最高价上方 1% 处
        fig.add_trace(go.Scatter(
            x=plot_data[mask_rsi_sell].index,
            y=high[mask_rsi_sell] * 1.01,
            mode='markers',
            marker=dict(symbol='triangle-down', size=8, color='#D50000'),
            name='RSI超买(>70)',
            hovertemplate='RSI超买: %{x|%Y-%m-%d}'
        ), row=1, col=1)

        # C. 布林带触底 (支撑) - 蓝色圆点
        fig.add_trace(go.Scatter(
            x=plot_data[mask_bb_touch_lb].index,
            y=low[mask_bb_touch_lb],
            mode='markers',
            marker=dict(symbol='circle-open', size=6, color='blue', line=dict(width=2)),
            name='触及下轨',
            hovertemplate='布林带支撑'
        ), row=1, col=1)

        # D. D值强趋势启动 (趋势爆发) - 紫色菱形
        fig.add_trace(go.Scatter(
            x=plot_data[mask_d_strong_entry].index,
            y=high[mask_d_strong_entry] * 1.02, # 放得更高一点
            mode='markers',
            marker=dict(symbol='diamond', size=7, color='purple'),
            name='进入强趋势(D<1.4)',
            hovertemplate='强趋势启动'
        ), row=1, col=1)

        # === Row 2: 分形维数 D ===
        fig.add_trace(go.Scatter(x=plot_data.index, y=d_val, line=dict(color='#5c6bc0', width=1), name='D值'), row=2, col=1)
        # 阈值线
        fig.add_hline(y=1.4, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_hline(y=1.7, line_dash="dot", line_color="red", row=2, col=1)
       
        # === Row 3: RSI ===
        fig.add_trace(go.Scatter(x=plot_data.index, y=rsi, line=dict(color='#ab47bc', width=1.5), name='RSI'), row=3, col=1)
        # 阈值区域
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", row=3, col=1) # 超买区
        fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, layer="below", row=3, col=1) # 超卖区
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
       
        # === Row 4: 持仓 & 盈亏 ===        
        #  持仓 (左轴, 面积图)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['position'], fill='tozeroy', line=dict(color='rgba(33, 150, 243, 0.3)'), name='持仓'), row=4, col=1, secondary_y=False)
        colors = np.where(plot_data.get('strategy_ret', 0) >= 0, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data.get('strategy_ret', 0), marker_color=colors, name='盈亏%', opacity=0.6), row=4, col=1, secondary_y=True)
        
        # ------------------------------------------------------
        # 3. 布局配置
        # ------------------------------------------------------
        fig.update_layout(
            height=1200, 
            template='plotly_white',
            # 多子图联动显示
            hovermode='x', 
            hoversubplots='axis', 
            xaxis_rangeslider_visible=False
        )

        # 核心修改: 配置十字光标 (Spikelines) 与 日期格式
        common_xaxis_config = dict(
            showspikes=True,
            spikemode='across',     # 贯穿模式
            spikesnap='cursor',     # 吸附模式
            showline=True,
            spikecolor="grey",
            spikethickness=1,
            spikedash='dash',
            
            # 强制日期格式 (YYYY-MM-DD)
            tickformat='%Y-%m-%d',   # 轴下方的刻度标签格式
            hoverformat='%Y-%m-%d'   # 鼠标悬停时的提示框格式
        )
        
        # 应用配置到所有 X 轴
        fig.update_xaxes(**common_xaxis_config)
        
        # 隐藏周末
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        # Y轴光标
        fig.update_yaxes(
            showspikes=True,
            spikemode='across',
            spikecolor="grey",
            spikethickness=1,
            spikedash='dash'
        )

        # Row 4 右轴 (盈亏%) 强制百分比格式
        fig.update_yaxes(tickformat='.2%', row=4, col=1, secondary_y=True)

        # 保存或返回
        if filename:
            fig.write_html(filename)
        return fig