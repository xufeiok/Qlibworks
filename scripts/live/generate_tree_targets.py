import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

import qlib
from qlworks.backtest.industry import apply_industry_constraint_pit, load_industry_maps_pit
from qlworks.config import QLIB_DATA_DIR
from qlworks.live.targets import build_daily_target_positions
from qlworks.live.tree_strategy import DEFAULT_LIVE_STRATEGY, get_live_strategy_config


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def _resolve_live_context(
    strategy_name: str = DEFAULT_LIVE_STRATEGY,
    runtime_model_name: str | None = None,
) -> dict:
    """
    解析 live 策略配置与输出路径。

    输入:
    - strategy_name: live 策略档案名
    - runtime_model_name: 可选运行目录名；为空时使用策略默认值

    输出:
    - dict: 包含策略参数、score 文件路径和 runtime 输出目录
    """
    config = get_live_strategy_config(strategy_name)
    runtime_name = (runtime_model_name or config["runtime_model_name"]).strip()
    score_path = PROJECT_ROOT / "scripts" / "training" / config["score_file"]
    runtime_root = PROJECT_ROOT / "runtime" / "live" / runtime_name
    return {
        "config": config,
        "score_path": score_path,
        "runtime_root": runtime_root,
        "signal_daily_dir": runtime_root / "signals" / "daily",
        "signal_archive_dir": runtime_root / "signals" / "archive",
        "state_dir": runtime_root / "state",
    }


def generate_targets(
    trade_date: str | None = None,
    strategy_name: str = DEFAULT_LIVE_STRATEGY,
    runtime_model_name: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    context = _resolve_live_context(strategy_name=strategy_name, runtime_model_name=runtime_model_name)
    config = context["config"]
    score_path = context["score_path"]

    if not score_path.exists():
        raise FileNotFoundError(f"找不到预测文件: {score_path}")

    score_df = pd.read_csv(score_path, parse_dates=["datetime"])
    if score_df.empty:
        raise ValueError(f"{config['score_file']} 为空，无法生成目标持仓")

    score_df["datetime"] = pd.to_datetime(score_df["datetime"]).dt.normalize()
    target_date = pd.Timestamp(trade_date).normalize() if trade_date else score_df["datetime"].max()

    if config["industry_neutral"]:
        qlib.init(provider_uri=str(QLIB_DATA_DIR), region="cn")
        instruments = score_df["instrument"].dropna().unique().tolist()
        industry_maps = load_industry_maps_pit(instruments, target_date, target_date)
        constrained = apply_industry_constraint_pit(
            pred_df=score_df.set_index(["datetime", "instrument"]),
            industry_maps=industry_maps,
            top_k=config["top_k"],
            max_per_industry=config["max_per_industry"],
        )
        score_df = (
            score_df.set_index(["datetime", "instrument"])
            .join(constrained.rename(columns={"score": "industry_score"}), how="inner")
            .reset_index()
        )
        score_df["score"] = score_df["industry_score"]
        score_df.drop(columns=["industry_score"], inplace=True)

    # [Filter] 过滤次新股
    from qlworks.factors.filter_utils import filter_codes_post
    _all = score_df.index.get_level_values("instrument").unique().tolist() if isinstance(score_df.index, pd.MultiIndex) else score_df["instrument"].unique().tolist()
    _fi = filter_codes_post(_all, trade_date, filter_new_stocks=True, filter_st=False)
    score_df = score_df[score_df.index.get_level_values("instrument").isin(set(_fi))] if isinstance(score_df.index, pd.MultiIndex) else score_df[score_df["instrument"].isin(set(_fi))]
    target_df = build_daily_target_positions(
        score_df=score_df,
        trade_date=target_date,
        top_k=config["top_k"],
        score_threshold=config["score_threshold"],
        buy_pct=config["buy_pct"],
    )

    signal_daily_dir = context["signal_daily_dir"]
    signal_archive_dir = context["signal_archive_dir"]
    state_dir = context["state_dir"]
    signal_daily_dir.mkdir(parents=True, exist_ok=True)
    signal_archive_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"target_positions_{target_date.strftime('%Y%m%d')}.csv"
    daily_path = signal_daily_dir / file_name
    archive_path = signal_archive_dir / file_name
    target_df.to_csv(daily_path, index=False)
    target_df.to_csv(archive_path, index=False)

    latest_state = {
        "model_name": config["model_name"],
        "runtime_model_name": context["runtime_root"].name,
        "trade_date": target_date.strftime("%Y-%m-%d"),
        "score_file": config["score_file"],
        "signal_file": str(daily_path),
        "signal_count": int(len(target_df)),
        "industry_neutral": bool(config["industry_neutral"]),
        "top_k": int(config["top_k"]),
        "score_threshold": float(config["score_threshold"]),
        "buy_pct": float(config["buy_pct"]),
    }
    with open(state_dir / "latest_target.json", "w", encoding="utf-8") as f:
        json.dump(latest_state, f, ensure_ascii=False, indent=2)

    return target_df, daily_path


def _parse_args():
    parser = argparse.ArgumentParser(description="生成树模型次日目标持仓文件")
    parser.add_argument("--trade-date", type=str, default=None, help="目标交易日，格式 YYYY-MM-DD；默认取 score 文件中的最新日期")
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_LIVE_STRATEGY,
        help="live 策略档案，支持 tree / selected",
    )
    parser.add_argument(
        "--runtime-model-name",
        type=str,
        default=None,
        help="目标持仓输出目录名；默认使用策略档案内置目录",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    target_df, output_path = generate_targets(
        trade_date=args.trade_date,
        strategy_name=args.strategy,
        runtime_model_name=args.runtime_model_name,
    )
    print("=" * 60)
    print("树模型目标持仓生成完成")
    print(f"策略档案: {args.strategy}")
    print(f"交易日: {target_df['trade_date'].iloc[0].date() if not target_df.empty else args.trade_date}")
    print(f"目标数量: {len(target_df)}")
    print(f"输出文件: {output_path}")
    if not target_df.empty:
        print(target_df.head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
