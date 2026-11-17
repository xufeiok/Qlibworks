"""
pipeline_config.py — 全流程单一事实源配置（Single Source of Truth）

[定位]
  集中管理跨阶段共享的管线级参数，避免标签表达式、股票池、统计阈值等
  在 screen_factors / select_factors / train_tree-doubao / 评测 / 准入等
  多个脚本中各自硬编码，导致"幸运一致"而非"机制一致"。

[使用约定]
  - 各脚本顶部配置区优先从本模块导入共享键，再叠加脚本本地特有参数。
  - 仅收录"跨脚本必须一致"的参数；脚本独有参数（rolling_windows、模型超参、
    select_factors 的每类 top_k 等）留在各脚本本地。
"""

from pathlib import Path

# ── 标签管线（四处必须一致：粗筛 / 精选 / 训练 / 评测）──
# T+1 开盘买入、T+5 收盘卖出；DK_L 管线 = CSNeutralize(industry+mv) → CSQuantileNorm
LABEL_EXPR = "Ref($close, -5) / Ref($open, -1) - 1"
LABEL_NAME = "LABEL_5D"

# ── 股票池（全链路统一：筛选 / 评测 / 训练 / 回测）──
INSTRUMENTS = "main_board"  # 沪深主板（沪 600/601/603/605，深 000/001/002/003），支持 PIT 格式


def resolve_instrument_codes(instruments: str | None = None) -> list[str]:
    """
    读取指定 Qlib 股票池并返回标准化代码列表。

    入参:
        instruments: 股票池名称，默认为全局 INSTRUMENTS（当前为 main_board）。

    返回:
        去重且保持升序的股票代码列表。

    异常:
        股票池文件不存在或为空时抛出异常，避免静默回退到 all.txt。
    """
    from qlworks.config import QLIB_DATA_DIR

    pool_name = instruments or INSTRUMENTS
    pool_path = Path(QLIB_DATA_DIR) / "instruments" / f"{pool_name}.txt"
    if not pool_path.exists():
        raise FileNotFoundError(f"股票池文件不存在: {pool_path}")

    codes: list[str] = []
    for line in pool_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            codes.append(parts[0].upper())
    codes = sorted(set(codes))
    if not codes:
        raise ValueError(f"股票池为空: {pool_path}")
    return codes

# ── 主训练时间范围（screen / train 共用主窗口；评测端保留 2010 起的独立长窗口）──
START_TIME = "2020-01-01"
END_TIME = "2025-12-31"

# ── 因子准入数据窗口（P0：杜绝测试期数据泄漏）──
# 候选池将供给 train_tree-doubao.py 的滚动测试窗口（最早 Test_2023 起始于 2023-01-01）。
# 准入评价窗口必须严格早于 ADMISSION_CUTOFF_DATE（截止日不含），
# 否则因子的"准入决策"会用上测试期数据 → 选择偏差泄漏，Test IC 不再是真正的样本外。
ADMISSION_CUTOFF_DATE = "2023-01-01"   # 准入数据截止日（不含），须 ≤ 下游最早测试窗口起点
ADMISSION_WINDOW_YEARS = 3             # 准入评价窗口长度（年），从截止日往前推

# ── 统计阈值（粗筛 / 精选 / 训练统一）──
REDUNDANCY_THRESHOLD = 0.90   # 冗余剔除相关系数阈值
ICIR_WINDOW = 60              # ICIR 滚动窗口（交易日）
ICIR_KEEP_RATIO = 0.8         # ICIR 正向占比保留比例
TOP_K = 60                    # 粗筛阶段一候选数 = 训练端每窗口精选因子数

# ── 准入阈值（run_eval.py --admit 三关，2026-10-12 admit 合并）──
CORR_THRESHOLD = 0.70         # 第一关：与已有池因子相关性上限
ICIR_IMPROVE_MIN = 0.0        # 第二关：增量 ICIR 边际贡献下限（加入后组合 ICIR 须不降）

# ── 可交易性过滤（全链路统一：粗筛 / 评测 / 精选 / 训练 / 实盘 / 回测）──
# 7 个环节必须引用同一组开关，杜绝"4 种设置并存"的口径分裂。
FILTER_ST = True              # 剔除 ST/风险警示股票（依赖 instruments/st_periods.csv）
FILTER_NEW_STOCKS = True      # 剔除上市不足 250 日的次新股
FILTER_SUSPENDED = True       # 剔除当日停牌（volume=0）
FILTER_LIMIT_UPDOWN = True    # 剔除涨跌停/一字板不可交易标签样本（filter_untradeable_labels）
