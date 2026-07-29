"""修正随机基准测试（修复 score dtype 问题）"""
import os, sys, shutil, json, subprocess, pandas as pd
import numpy as np

SCORE_FILE = r"E:\Quant\Qlibworks\scripts\training\score_tree.csv"
BACKUP_FILE = SCORE_FILE.replace('.csv', '_backup2.csv')

# 备份
orig = pd.read_csv(SCORE_FILE)
orig['score'] = orig['score'].astype(float)
shutil.copy(SCORE_FILE, BACKUP_FILE)
print(f"[1] 备份: {len(orig)}行, avg={orig['score'].mean():.4f}")

# 生成随机分数 - 用 np.random.permutation 确保 dtype 稳定
rand = orig.copy()
rand['score'] = rand.groupby('datetime', group_keys=False)['score'].transform(
    lambda x: np.random.RandomState(42).permutation(x.values)
)
rand['score'] = rand['score'].astype(float)
rand.to_csv(SCORE_FILE, index=False)

# 验证
verify = pd.read_csv(SCORE_FILE)
print(f"[2] 随机分数: {len(verify)}行, avg={verify['score'].mean():.4f}, dtype={verify['score'].dtype}")
assert verify['score'].dtype == float or verify['score'].dtype == np.float64, "dtype 不对!"
assert verify['score'].notna().all(), "有 NaN!"
pass_count = (verify['score'] >= 0.7).sum()
print(f"    >=0.7: {pass_count}/{len(verify)} ({100*pass_count/len(verify):.1f}%)")

# 运行回测
print(f"[3] 启动回测（预计30-60分钟）...")
result = subprocess.run(
    [r"E:\Conda_env\Qlib_env\python.exe", "scripts/backtest/tree_label_test.py"],
    cwd=r"E:\Quant\Qlibworks",
    capture_output=True, text=True, timeout=7200
)

# 提取关键输出
for line in result.stdout.split('\n'):
    if any(kw in line for kw in ['期末资金', 'initial_capital', 'final_value', 'max_drawdown',
                                  'win_rate', 'total_return', 'sharpe', '总PnL', 'portfolio',
                                  '实际可用']):
        print(f"  {line.strip()}")
print("\n--- 末20行 ---")
print('\n'.join(result.stdout.split('\n')[-20:]))
if result.stderr:
    err_lines = result.stderr.strip().split('\n')[-10:]
    if any('Http' in l for l in err_lines):
        print(f"\n[注意] ClickHouse HTTP 异常:")
        for l in err_lines:
            print(f"  {l.strip()}")

# 读取 JSON
bt_json = r"E:\Quant\Qlibworks\scripts\backtest\output_label_test\qlib_bt_superplot.json"
if os.path.exists(bt_json):
    d = json.load(open(bt_json, encoding='utf-8'))
    ads = d.get('analysis_data', {})
    pv = d.get('portfolio_data', {}).get('data', [])
    print(f"\n[结果] 随机回测指标:")
    if pv:
        print(f"  PV: {pv[0]['value']:,.0f} -> {pv[-1]['value']:,.0f} = {pv[-1]['value']/pv[0]['value']:.1f}x")
    for k in ['total_return', 'annual_return', 'win_rate', 'total_trades', 'sharpe', 'mdd', 'calmar', 'profit_factor']:
        v = ads.get(k)
        if v is not None and v != {}:
            print(f"  {k}: {v}")
    orders = d.get('orders_data', {}).get('orders', [])
    print(f"  总订单数: {len(orders)}")
    shutil.copy(bt_json, bt_json.replace('.json', '_random2.json'))
    print(f"[保存] 随机结果已保存")

# 恢复
shutil.copy(BACKUP_FILE, SCORE_FILE)
print(f"[恢复] 原分数已恢复")
