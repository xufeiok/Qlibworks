import os, sys, shutil, pandas as pd

SCORE_FILE = r"E:\Quant\Qlibworks\scripts\training\score_tree.csv"
BACKUP_FILE = SCORE_FILE.replace('.csv', '_backup.csv')

# 确认主文件正常
orig = pd.read_csv(SCORE_FILE)
print(f"[备份] 原分数: {len(orig)}行, avg={orig['score'].mean():.4f}")
shutil.copy(SCORE_FILE, BACKUP_FILE)
print(f"[备份] -> {BACKUP_FILE}")

# 生成随机分数（每天内打乱）
rand = orig.copy()
rand['score'] = rand.groupby('datetime', group_keys=False)['score'].apply(
    lambda x: x.sample(frac=1, random_state=42).values
)
rand.to_csv(SCORE_FILE, index=False)
print(f"[写入] 随机分数: avg={rand['score'].mean():.4f}, >=0.7={rand['score'].ge(0.7).sum()/len(rand)*100:.1f}%")

# 运行回测
import subprocess
result = subprocess.run(
    [r"E:\Conda_env\Qlib_env\python.exe", "scripts/backtest/tree_label_test.py"],
    cwd=r"E:\Quant\Qlibworks",
    capture_output=True, text=True, timeout=7200
)

print(f"\n[回测] 退出码={result.returncode}")
# 提取关键行
for line in result.stdout.split('\n'):
    if any(kw in line for kw in ['期末资金','initial_capital','final_value','max_drawdown','win_rate','total_return','sharpe','总PnL','portfolio']):
        print(f"  {line.strip()}")

# 输出最后30行
out_lines = result.stdout.split('\n')
print("\n--- 最后30行输出 ---")
print('\n'.join(out_lines[-30:]))
if result.stderr:
    print(f"\n--- STDERR (末2000字) ---")
    print(result.stderr[-2000:])

# 读取 JSON 结果
import json
bt_json = r"E:\Quant\Qlibworks\scripts\backtest\output_label_test\qlib_bt_superplot.json"
if os.path.exists(bt_json):
    d = json.load(open(bt_json, encoding='utf-8'))
    ads = d.get('analysis_data', {})
    print(f"\n[指标] 随机回测:")
    for k in ['sharpe','calmar','mdd','total_return','annual_return','win_rate','total_trades','profit_factor']:
        print(f"  {k}: {ads.get(k, 'N/A')}")
    pv = d.get('portfolio_data', {}).get('data', [])
    if pv:
        print(f"  PV: {pv[0]['value']:,.0f} -> {pv[-1]['value']:,.0f} = {pv[-1]['value']/pv[0]['value']:.1f}x")
    # 保存结果
    shutil.copy(bt_json, bt_json.replace('.json', '_random.json'))
    print(f"[保存] 随机回测结果已保存")

# 恢复原始分数
shutil.copy(BACKUP_FILE, SCORE_FILE)
print(f"[恢复] 原分数已恢复")
print(f"[完成] {os.path.getsize(BACKUP_FILE)} 字节备份 -> score_tree.csv")
