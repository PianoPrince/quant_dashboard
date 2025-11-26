# core/visualizer.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    """
    V5.6 可视化引擎 (修复参数接收问题)
    """

    @staticmethod
    def plot_backtest_result(df: pd.DataFrame, filename: str = None, strong_th=1.4, weak_th=1.7, rsi_high=70, rsi_low=30):
        """
        绘制全功能回测报告
        
        :param df: 回测结果 DataFrame
        :param filename: (可选) 保存文件名
        :param strong_th: 强趋势阈值 (D值)
        :param weak_th: 弱趋势阈值 (D值)
        :param rsi_high: RSI 超买阈值
        :param rsi_low: RSI 超卖阈值
        """
        plot_data = df.copy()
        
        # 数据提取
        close = plot_data['close']
        high = plot_data['high']
        low = plot_data['low']
        frama = plot_data.get('FRAMA', pd.Series(np.nan, index=plot_data.index))
        d_val = plot_data.get('D', pd.Series(np.nan, index=plot_data.index))
        rsi = plot_data.get('RSI', pd.Series(np.nan, index=plot_data.index))
        equity_rebased = plot_data.get('equity_rebased', close)
        bb_ub = plot_data.get('BB_UB', pd.Series(np.nan, index=plot_data.index))
        bb_lb = plot_data.get('BB_LB', pd.Series(np.nan, index=plot_data.index))

        # 使用传入的动态参数计算信号掩码
        mask_rsi_buy = rsi < rsi_low
        mask_rsi_sell = rsi > rsi_high
        mask_bb_touch_lb = close <= bb_lb
        
        mask_d_strong_entry = (d_val < strong_th) & (d_val.shift(1) >= strong_th)

        mask_breakdown = close < frama
        mask_strong = (~mask_breakdown) & (d_val < strong_th)
        mask_weak = (~mask_breakdown) & (d_val >= strong_th) & (d_val < weak_th)
        mask_noise = (~mask_breakdown) & (d_val >= weak_th)

        def get_segment(mask):
            segment = frama.copy()
            segment[~mask] = np.nan
            return segment

        frama_strong = get_segment(mask_strong)
        frama_weak = get_segment(mask_weak)
        frama_noise = get_segment(mask_noise)
        frama_breakdown = get_segment(mask_breakdown)

        # 画布配置
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08,
            row_heights=[0.45, 0.15, 0.15, 0.25],
            specs=[[{"secondary_y": False}], 
                   [{"secondary_y": False}],
                   [{"secondary_y": False}], 
                   [{"secondary_y": True}]],
            subplot_titles=('价格行为 & 信号标记', '分形维数 D', 'RSI 相对强弱', '持仓 & 盈亏')
        )

        # Row 1: K线 + 布林带 + FRAMA
        fig.add_trace(go.Scatter(x=plot_data.index, y=bb_ub, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=bb_lb, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(128, 0, 128, 0.1)', name='布林带', hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['open'], high=plot_data['high'], low=plot_data['low'], close=plot_data['close'], name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_strong, mode='lines', line=dict(color='blue', width=2), name='强趋势'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_weak, mode='lines', line=dict(color='orange', width=2), name='弱趋势'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_noise, mode='lines', line=dict(color='gray', width=1, dash='dot'), name='噪音'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=frama_breakdown, mode='lines', line=dict(color='purple', width=2, dash='dash'), name='破坏'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=equity_rebased, mode='lines', line=dict(color='#FFD700', width=2), name='策略净值'), row=1, col=1)
        
        # Markers
        fig.add_trace(go.Scatter(x=plot_data[mask_rsi_buy].index, y=low[mask_rsi_buy] * 0.99, mode='markers', marker=dict(symbol='triangle-up', size=8, color='#00C853'), name=f'RSI超卖(<{rsi_low})', hovertemplate=f'RSI超卖: %{{x|%Y-%m-%d}}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data[mask_rsi_sell].index, y=high[mask_rsi_sell] * 1.01, mode='markers', marker=dict(symbol='triangle-down', size=8, color='#D50000'), name=f'RSI超买(>{rsi_high})', hovertemplate=f'RSI超买: %{{x|%Y-%m-%d}}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data[mask_bb_touch_lb].index, y=low[mask_bb_touch_lb], mode='markers', marker=dict(symbol='circle-open', size=6, color='blue', line=dict(width=2)), name='触及下轨', hovertemplate='布林带支撑'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data[mask_d_strong_entry].index, y=high[mask_d_strong_entry] * 1.02, mode='markers', marker=dict(symbol='diamond', size=7, color='purple'), name=f'强趋势启动(D<{strong_th})', hovertemplate='强趋势启动'), row=1, col=1)

        # Row 2: D值
        fig.add_trace(go.Scatter(x=plot_data.index, y=d_val, line=dict(color='#5c6bc0', width=1), name='D值', hovertemplate='%{y:.3f}'), row=2, col=1)
        fig.add_hline(y=strong_th, line_dash="dot", line_color="green", row=2, col=1, annotation_text=f"强趋势 ({strong_th})")
        fig.add_hline(y=weak_th, line_dash="dot", line_color="red", row=2, col=1, annotation_text=f"弱趋势 ({weak_th})")

        # Row 3: RSI
        fig.add_trace(go.Scatter(x=plot_data.index, y=rsi, line=dict(color='#ab47bc', width=1.5), name='RSI', hovertemplate='%{y:.3f}'), row=3, col=1)
        fig.add_hrect(y0=rsi_high, y1=100, fillcolor="red", opacity=0.1, layer="below", row=3, col=1)
        fig.add_hrect(y0=0, y1=rsi_low, fillcolor="green", opacity=0.1, layer="below", row=3, col=1)
        fig.add_hline(y=rsi_high, line_dash="dot", line_color="red", row=3, col=1, annotation_text=f"超买 ({rsi_high})")
        fig.add_hline(y=rsi_low, line_dash="dot", line_color="green", row=3, col=1, annotation_text=f"超卖 ({rsi_low})")

        # Row 4: 持仓 & 盈亏
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['position'], fill='tozeroy', line=dict(color='rgba(33, 150, 243, 0.3)'), name='持仓', hovertemplate='%{y:.1%}'), row=4, col=1, secondary_y=False)
        colors = np.where(plot_data.get('strategy_ret', 0) >= 0, '#ef5350', '#26a69a')
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data.get('strategy_ret', 0), marker_color=colors, name='盈亏%', opacity=0.6, hovertemplate='%{y:.2%}'), row=4, col=1, secondary_y=True)

        # Layout
        fig.update_layout(height=1100, template='plotly_white', hovermode='x', hoversubplots='axis', xaxis_rangeslider_visible=False)
        common_xaxis = dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikecolor="grey", spikethickness=1, spikedash='dash', tickformat='%Y-%m-%d', hoverformat='%Y-%m-%d')
        fig.update_xaxes(**common_xaxis)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        fig.update_yaxes(tickformat='.2%', row=4, col=1, secondary_y=True)
        
        if filename:
            fig.write_html(filename)
        return fig

