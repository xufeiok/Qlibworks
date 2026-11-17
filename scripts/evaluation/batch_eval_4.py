#!/usr/bin/env python3
"""
批量单因子评测驱动脚本 — 依次评测仓库中的 4 个因子 (KMID/MA20/ROC20/STR_20d)

用法:
  python batch_eval_4.py
"""
import os
import subprocess
import sys
from pathlib import Path

PY = r"D:\xf_office_draft\QuantVenv\Scripts\python.exe"
# [单脚本化 2026-10-12] run_eval.py 已冻结（scripts/_archive/），唯一评测入口 = factor_screen.py
RUN_EVAL = Path(__file__).resolve().parents[1] / "training" / "factor_screen.py"
SRC = Path(__file__).resolve().parents[2] / "src"
FACTORS = ["KMID", "MA20", "ROC20", "STR_20d"]
START, END, POOL = "2020-01-01", "2025-12-31", "main_board"

env = dict(os.environ)
env["PYTHONPATH"] = str(SRC)

# ── 先补齐 meta.json（避免评测误判仓库无数据而回退 ClickHouse 计算卡死） ──
sys.path.insert(0, str(SRC))
from qlworks.evaluation import FactorStore

store = FactorStore()
for name in FACTORS:
    if store.get_warehouse_meta(name) is None:
        store._update_warehouse_meta(name)
        print(f"[Meta] {name}: meta.json 已补齐", flush=True)
    else:
        print(f"[Meta] {name}: meta.json 已存在", flush=True)

for name in FACTORS:
    print(f"\n{'=' * 70}\n[因子] {name} 评测开始\n{'=' * 70}", flush=True)
    cmd = [
        PY, "-u", str(RUN_EVAL),
        "--deep-eval",
        "--factor", name,
        "--pool", POOL,
        "--start-time", START,
        "--end-time", END,
    ]
    # 子进程输出实时写入独立日志，避免父进程缓冲导致无法监控
    with open(Path(__file__).resolve().parent / f"_eval_{name}.log", "w", encoding="utf-8") as f:
        try:
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                               text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            print(f"\n[因子] {name} 超时(60分钟)被终止", flush=True)
            continue
    print(f"\n[因子] {name} 退出码 {r.returncode}", flush=True)

print("\n全部因子评测完成", flush=True)
