import os
import pandas as pd
from config import Config  # 导入配置文件
from core.data_handlers import ExcelHandler
from core.factor_lib import TechnicalFactors
from core.strategy_engine import FRAMAStrategy, FRAMA_RSI_Strategy, FRAMA_RSI_bb_Strategy
from core.backtester import VectorBacktester
from core.visualizer import Visualizer

def main():
    print("==================================================")
    print(f"      V5.3 量化回测系统启动 - {Config.ASSET_SYMBOL}")
    print("==================================================\n")

    # 0. 环境检查
    if not os.path.exists(Config.REPORTS_DIR):
        os.makedirs(Config.REPORTS_DIR)
        print(f">>> 报告目录: {Config.REPORTS_DIR}/")

    # 1. ETL: 加载数据
    print(">>> 1. 加载数据...")
    try:
        excel_handler = ExcelHandler(
            file_path=Config.DATA_FILE_PATH, 
            sheet_name=Config.SHEET_NAME
        )
        raw_data = excel_handler.get_data()
    except Exception as e:
        print(f"!!! 数据加载失败: {e}")
        return

    # 跳过空数据
    if raw_data.empty: return

    # 根据 Config 的时间窗口过滤数据
    # 确保索引是 datetime 类型
    raw_data.index = pd.to_datetime(raw_data.index)
    # 切片数据
    data = raw_data.loc[Config.START_DATE : Config.END_DATE].copy()
    
    print(f"    原始数据: {len(raw_data)} 行")
    print(f"    回测区间: {Config.START_DATE} 至 {Config.END_DATE} ({len(data)} 行)")

    # 2. Feature: 计算因子
    print(f"\n>>> 2.1 计算 FRAMA 因子 (Window={Config.FRAMA_WINDOW})...")
    data = TechnicalFactors.calculate_frama(
        data, 
        window=Config.FRAMA_WINDOW, 
        n_slow=Config.FRAMA_N_SLOW
    )

    print(f"\n>>> 2.2 计算 RSI 因子 (Period={Config.RSI_PERIOD})...")
    data = TechnicalFactors.calculate_rsi(
        data,
        period=Config.RSI_PERIOD
    )

    print(f"    - 布林带 (Window={Config.BB_WINDOW}, Std={Config.BB_STD})")
    data = TechnicalFactors.calculate_bollinger(
        data,
        period=Config.BB_WINDOW,
        std_dev=Config.BB_STD
    )


    # 3. Signal: 生成信号
    print("\n>>> 3. 生成策略信号...")
    # 将 Config 中的阈值传入策略
    
    # # 纯FRAMA策略
    # strategy = FRAMAStrategy(
    #     data_with_factors, 
    #     strong_threshold=Config.THRESHOLD_STRONG_TREND,
    #     weak_threshold=Config.THRESHOLD_WEAK_TREND
    # )

    # # FRAMA+RSI策略
    # strategy = FRAMA_RSI_Strategy(
    #     data, 
    #     strong_threshold=Config.THRESHOLD_STRONG_TREND,
    #     weak_threshold=Config.THRESHOLD_WEAK_TREND,
    #     rsi_overbought=Config.RSI_OVERBOUGHT,
    #     rsi_oversold=Config.RSI_OVERSOLD
    # )

    # # FRAMA+RSI+Bollinger策略
    strategy = FRAMA_RSI_bb_Strategy(
        data, 
        strong_threshold=Config.THRESHOLD_STRONG_TREND,
        weak_threshold=Config.THRESHOLD_WEAK_TREND,
        rsi_overbought=Config.RSI_OVERBOUGHT,
        rsi_oversold=Config.RSI_OVERSOLD,
        bb_bw_low_k=Config.BB_BW_LOW_THRESHOLD,
        bb_bw_high_k=Config.BB_BW_HIGH_THRESHOLD
    )

    # 生成策略信号
    data_with_signals = strategy.generate_signals()

    # 4. Execution: 回测
    print("\n>>> 4. 执行回测...")
    backtester = VectorBacktester(
        commission=Config.COMMISSION_RATE,
        slippage=Config.SLIPPAGE,
        initial_principal=Config.INITIAL_PRINCIPAL,
        risk_free_rate=Config.RISK_FREE_RATE
    )
    
    # 保存完整结果
    full_results = backtester.run(data_with_signals)

    # 打印全区间绩效
    summary = backtester.get_summary(full_results)
    print("\n    -------- 全区间绩效 --------")
    print(summary[['Strategy', 'Benchmark']])
    print("    --------------------------")

    # 全区间报告
    full_report_path = os.path.join(Config.REPORTS_DIR, 'frama_bollinger_full_report.html')
    Visualizer.plot_backtest_result(full_results, filename=full_report_path)
    print(f"\n>>> [全区间报告] 已生成: {os.path.basename(full_report_path)}")

    # ==========================================
    # 5. 交互式区间透视功能
    # ==========================================
    print("\n==================================================")
    print("      >>> 区间透视分析 (Sub-period Analysis) <<<")
    print("==================================================")
    
    while True:
        cmd = input("\n是否进行特定区间分析？(输入 y 开始，其他键退出): ").strip().lower()
        if cmd != 'y':
            break
            
        print("\n请输入分析区间 (格式: YYYY-MM-DD):")
        start_input = input("   开始日期: ").strip()
        end_input = input("   结束日期: ").strip()
        
        try:
            # 调用 backtester 的新功能进行切片分析
            period_df, period_summary = backtester.analyze_range(full_results, start_input, end_input)
            
            if period_df is not None:
                # 1. 打印表格
                print(f"\n    -------- 区间绩效 ({start_input} -> {end_input}) --------")
                print(period_summary[['Strategy', 'Benchmark']])
                print("    --------------------------------------------------------")
                
                # 2. 询问是否生成HTML图表
                generate_html = input("\n是否生成该区间的详细图表？(Y/N): ").strip().lower()
                
                if generate_html == 'y':
                    filename = f"report_{start_input.replace('-','')}_{end_input.replace('-','')}.html"
                    period_report_path = os.path.join(Config.REPORTS_DIR, filename)
                    Visualizer.plot_backtest_result(period_df, filename=period_report_path)
                    print(f"    >>> 图表已生成: {period_report_path}")
                    print("    (请在浏览器中打开该文件查看该区间的详细走势)")
                else:
                    print("    >>> 已跳过图表生成")
                
        except Exception as e:
            print(f"!!! 分析出错: {e}")
            print("请检查日期格式是否正确 (例如 2022-01-01)")

    print("\n>>> 程序结束。")

if __name__ == "__main__":
    main()