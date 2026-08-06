"""
单因子评测脚本：对 roe_ttm 运行完整评测流水线，输出 HTML 网页报告。

用法:
  python eval_roe_ttm.py                              # 最近5年 (2021-2025)
  python eval_roe_ttm.py --start 2010-01-01 --end 2026-12-31  # 全周期 (可能OOM)
  python eval_roe_ttm.py --start 2020-01-01 --end 2023-12-31  # 2020-2023

报告输出:
  factor_data/reports/{tier}/roe_ttm_{timestamp}.html
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from qlworks.config import QLIB_DATA_DIR
from qlworks.evaluation.runner import FactorEvaluator
from qlworks.evaluation.config import EvalConfig, DEFAULT_CONFIG

FACTOR_NAME = "roe_ttm"
FACTOR_EXPR = "roe_ttm"
INSTRUMENTS = "csi500"


def main():
    parser = argparse.ArgumentParser(description="roe_ttm 单因子评测")
    parser.add_argument("--start", default="2021-01-01",
                        help="评测起始日期 (default: 2021-01-01)")
    parser.add_argument("--end", default="2025-12-31",
                        help="评测结束日期 (default: 2025-12-31)")
    parser.add_argument("--output", "-o", default=None,
                        help="HTML 报告输出路径 (default: factor_data/reports/...)")
    args = parser.parse_args()

    start_time = args.start
    end_time = args.end

    print("=" * 60)
    print(f"  单因子评测: {FACTOR_NAME} ({start_time} ~ {end_time})")
    print("=" * 60)

    # 1. 修改配置中的评测时间
    config = EvalConfig(
        start_time=start_time,
        end_time=end_time,
        warehouse_dir=DEFAULT_CONFIG.warehouse_dir,
        factors_dir=DEFAULT_CONFIG.factors_dir,
        cache_dir=DEFAULT_CONFIG.cache_dir,
        report_dir=DEFAULT_CONFIG.report_dir,
        registry_dir=DEFAULT_CONFIG.registry_dir,
        factor_library_dir=DEFAULT_CONFIG.factor_library_dir,
        robustness_sub_periods=[(start_time, end_time)],  # 只设1个子时段
        robustness_sub_pools=DEFAULT_CONFIG.robustness_sub_pools,
        neutralization="none",  # 仓库数据不含行业/市值列，跳过中性化
    )

    # 2. 初始化评测器
    evaluator = FactorEvaluator(config)

    # 3. 加载数据
    print(f"\n[3] 加载因子数据: {FACTOR_NAME}...")
    df = evaluator.load_data(
        FACTOR_EXPR, FACTOR_NAME,
        instruments=INSTRUMENTS,
        start_time=start_time,
        end_time=end_time,
    )
    # 仓库数据列名为 'value'，重命名为因子名
    if "value" in df.columns and FACTOR_NAME not in df.columns:
        df = df.rename(columns={"value": FACTOR_NAME})
    print(f"    数据: {len(df)} 行 × {len(df.columns)} 列")
    print(f"    日期范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"    交易标的: {df['instrument'].nunique()} 只")

    label_col = config.label_name
    if label_col in df.columns:
        print(f"    标签已加载: {label_col}")
    else:
        print(f"    标签未加载（跳过 IC/分层收益等分析）")

    # 4. 执行评测
    print(f"\n[4] 执行完整评测流水线...")
    result = evaluator.evaluate(FACTOR_NAME, df)

    # 5. 输出摘要
    print(f"\n{'=' * 60}")
    print(f"  评测完成!")
    print(f"{'=' * 60}")
    ic = result.get('ic_stats', {})
    ls = result.get('ls_stats', {})
    print(f"  IC 均值:      {ic.get('ic_mean', 'N/A')}")
    print(f"  ICIR:         {ic.get('icir', 'N/A')}")
    print(f"  年化 IC:      {ic.get('annual_ic', 'N/A')}")
    print(f"  胜率:         {ic.get('win_rate', 'N/A')}")
    print(f"  行业IC均值:   {ic.get('industry_ic_mean', 'N/A')}")
    print(f"  行业ICIR:     {ic.get('industry_icir', 'N/A')}")
    print(f"  单调性:       {ic.get('monotonicity', 'N/A')}")
    print(f"  多空年化:     {ls.get('annual_return', 'N/A')}")
    print(f"  多空夏普:     {ls.get('sharpe', 'N/A')}")
    print(f"  最大回撤:     {ls.get('max_drawdown', 'N/A')}")
    print(f"  等级:         {result['qual_result'].get('tier', 'N/A')}")
    print(f"  综合评级:     {result.get('composite_icon', '')} {result.get('composite_grade', 'N/A')} - {result.get('composite_desc', 'N/A')}")
    if result.get('usage_boundaries'):
        ub = result['usage_boundaries']
        print(f"  最佳持有期:   {ub.get('min_holding_days', 'N/A')} 日")
        print(f"  IC 半衰期:    {ub.get('ic_half_life_days', 'N/A')} 日")
        print(f"  最佳市态:     {ub.get('best_regime', 'N/A')}")

    # 6. 报告路径
    report_path = result.get("report_path", "")
    if report_path:
        print(f"\n  报告: {report_path}")
    print("=" * 60)

    print("\n建议评估周期:")
    print("  python eval_roe_ttm.py --start 2021-01-01 --end 2025-12-31   # 最近5年 (推荐)")
    print("  python eval_roe_ttm.py --start 2020-01-01 --end 2022-12-31   # 2020-2022")
    print("  python eval_roe_ttm.py --start 2023-01-01 --end 2025-12-31   # 2023-2025")


if __name__ == "__main__":
    main()
