"""
回测诊断脚本：逐步检查评分→行情→策略数据流，定位零交易问题。
"""
import os
import sys
import pandas as pd
import numpy as np

PROJECT_DIR = r"E:\Quant\Qlibworks"
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

import qlib
from qlib.data import D
from qlworks.config import QLIB_DATA_DIR

SCORE_PATH = os.path.join(PROJECT_DIR, "scripts", "training", "score_tree.csv")
SCORE_THRESHOLD = 0.7
TOP_K = 20

print("=" * 60)
print("回测诊断：追踪评分→行情→策略数据流")
print("=" * 60)

# 1. 检查评分文件
print("\n[1] 检查评分文件...")
pred_df = pd.read_csv(SCORE_PATH, parse_dates=["datetime"])
pred_df.set_index(["datetime", "instrument"], inplace=True)
start_date = pred_df.index.get_level_values("datetime").min()
end_date = pred_df.index.get_level_values("datetime").max()
instruments = pred_df.index.get_level_values("instrument").unique().tolist()
print(f"  日期: {start_date.date()} ~ {end_date.date()}")
print(f"  股票数: {len(instruments)}")
print(f"  总记录: {len(pred_df)}")
print(f"  Score 范围: [{pred_df['score'].min():.4f}, {pred_df['score'].max():.4f}]")
print(f"  Score > {SCORE_THRESHOLD}: {(pred_df['score'] > SCORE_THRESHOLD).sum()} 条 ({(pred_df['score'] > SCORE_THRESHOLD).mean()*100:.1f}%)")

# 2. 检查某一天的情况
first_date = pred_df.index.get_level_values("datetime").unique()[0]
print(f"\n[2] 首个交易日 ({first_date.date()}) 评分情况:")
day_scores = pred_df.xs(first_date, level="datetime")
print(f"  股票数: {len(day_scores)}")
print(f"  Score 统计: min={day_scores['score'].min():.3f}, median={day_scores['score'].median():.3f}, max={day_scores['score'].max():.3f}")
above = day_scores[day_scores['score'] > SCORE_THRESHOLD]
print(f"  Score > {SCORE_THRESHOLD}: {len(above)} 只")
if len(above) > 0:
    print(f"  Top 5:")
    print(above.sort_values('score', ascending=False).head(5))

# 3. 检查行情数据获取
print(f"\n[3] 拉取首个交易日行情 (日期={first_date.date()})...")
qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")

# 只测试前10只股票的行情
test_insts = instruments[:10]
price_data = D.features(
    test_insts, ["$open", "$high", "$low", "$close", "$volume"],
    start_time=first_date, end_time=first_date,
)
price_data.columns = ["open", "high", "low", "close", "volume"]
print(f"  行情 shape: {price_data.shape}")
print(f"  行情索引级别: {price_data.index.names}")
if not price_data.empty:
    print(f"  样例:")
    print(price_data.head(3))

# 4. 模拟 join 过程
print(f"\n[4] 模拟 score join 过程...")
test_inst = test_insts[0]
print(f"  测试股票: {test_inst}")

# 模拟 price_dict 构建
if test_inst in price_data.index.get_level_values("instrument"):
    df = price_data.xs(test_inst, level="instrument").copy()
    print(f"  行情行数: {len(df)}")
    print(f"  行情列: {df.columns.tolist()}")
    print(f"  行情索引名: {df.index.name}")

    # 模拟 bt_runner.py 中的处理
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif df.index.name != 'datetime':
        try:
            df.index = pd.to_datetime(df.index)
            df.index.name = 'datetime'
        except:
            pass

    # 提取预测分数
    inst_pred = pred_df.xs(test_inst, level='instrument')
    if 'score' not in inst_pred.columns and len(inst_pred.columns) == 1:
        inst_pred.columns = ['score']

    print(f"  预测分数行数: {len(inst_pred)}")
    print(f"  预测分数索引: {type(inst_pred.index)}")

    # Join
    df = df.join(inst_pred['score'], how='left')
    print(f"  Join 后列: {df.columns.tolist()}")
    print(f"  Join 后:")
    print(df.head(3))

    # 检查 score 是否为 NaN
    nan_count = df['score'].isna().sum()
    valid_count = df['score'].notna().sum()
    print(f"  Score NaN: {nan_count}, 有效: {valid_count}")
    if valid_count > 0:
        print(f"  Score > {SCORE_THRESHOLD}: {(df['score'] > SCORE_THRESHOLD).sum()}")
else:
    print(f"  {test_inst} 不在行情数据中！")

# 5. 检查全量 join（模拟 bt_runner 中的核心逻辑）
print(f"\n[5] 模拟全量数据流（针对 5 只股票）...")
full_calendar = pd.bdate_range(start_date, end_date)

join_ok = 0
join_fail = 0
score_ok = 0
score_nan = 0

for inst in instruments[:5]:
    if inst not in price_data.index.get_level_values("instrument"):
        join_fail += 1
        continue
    df = price_data.xs(inst, level="instrument").copy()
    df.dropna(subset=["close"], inplace=True)
    if df.empty:
        join_fail += 1
        continue

    valid = df[df["volume"] > 0]
    if not valid.empty:
        df = df[df.index <= valid.index[-1]]

    # 标准化
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif df.index.name != 'datetime':
        try:
            df.index = pd.to_datetime(df.index)
            df.index.name = 'datetime'
        except:
            pass

    # Join score
    inst_pred = pred_df.xs(inst, level='instrument')
    if 'score' not in inst_pred.columns and len(inst_pred.columns) == 1:
        inst_pred.columns = ['score']
    df = df.join(inst_pred['score'], how='left')

    df = df[~df.index.duplicated(keep='first')]
    df = df.reindex(full_calendar)
    df['volume'] = df['volume'].fillna(0)
    df = df.dropna(subset=['close'])
    df = df[df['close'] > 0]

    join_ok += 1
    nan_count = df['score'].isna().sum()
    valid_count = df['score'].notna().sum()
    print(f"  {inst}: 行数={len(df)}, Score NaN={nan_count}, 有效={valid_count}, >0.7={(df['score'] > 0.7).sum()}")
    score_ok += valid_count
    score_nan += nan_count

print(f"\n  汇总: join成功={join_ok}, join失败={join_fail}, score有效={score_ok}, score NaN={score_nan}")

# 6. 关键结论
print(f"\n[6] 诊断结论:")
print(f"  评分文件: 正常 ({len(pred_df)} 条记录)")
print(f"  Score 范围: [{pred_df['score'].min():.3f}, {pred_df['score'].max():.3f}]")
print(f"  每日约有 {(pred_df['score'] > SCORE_THRESHOLD).mean()*100:.0f}% 的股票 score > {SCORE_THRESHOLD}")
print(f"  策略应能在每个调仓日选出 Top {TOP_K} 只股票")

if score_ok > 0 and score_nan > 0:
    print(f"\n  ⚠️  警告: 部分 score 为 NaN，这可能是因为:")
    print(f"      - 行情数据日期与预测数据日期不完全匹配")
    print(f"      - reindex(bdate_range) 引入了预测范围外的交易日")
elif score_ok == 0:
    print(f"\n  ❌ 致命: 所有 score 都为 NaN! Join 失败!")
    print(f"     需检查日期格式是否匹配")
