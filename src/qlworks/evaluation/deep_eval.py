# -*- coding: utf-8 -*-
"""
deep_eval.py — 单因子深评编排核心（机构"厚核心薄脚本"，2026-10-12 从 run_eval.py 迁移）

深评流水线（对齐 Citadel Alpha Lab 仓库优先 / AQR 完整评测）：
  仓库优先读取（无数据则 ClickHouse 实时计算）→ 股票池过滤 → 标签合并 →
  数据质量（标签可交易性过滤 / volume+change_pct / mkt_cap+industry）→
  FactorEvaluator 完整评测（IC / 分层回测 / 衰减分析 / 稳健性子池 / walk-forward）→ 返回结果。

下游唯一调用方：factor_screen.py（唯一评测入口脚本）的 --deep-eval / --admit 模式。
run_eval.py 已冻结至 scripts/_archive/（2026-10-12 单脚本化）。

与 factor_stats.py 分工：本模块负责"深评编排"（数据准备 + 评测器调用）；
factor_stats.py 负责"统计核心"（IC/ICIR/相关/三关/候选池）。统计口径统一在 factor_stats。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("deep_eval")


def meta_val(meta: dict, key: str, default=None):
    """兼容新旧 meta 格式读取（data_range / 顶层 / statistics / end_date）。"""
    dr = meta.get("data_range", {}) if meta else {}
    if key in dr and dr[key]:
        return dr[key]
    if meta and key in meta and meta[key]:
        return meta[key]
    stats = meta.get("statistics", {}) if meta else {}
    if key in stats and stats[key]:
        return stats[key]
    if key == "last_date":
        return dr.get("end_date", default)
    return default


def load_pool_stocks(pool_name: str, instruments_dir: Path,
                     log: Optional[logging.Logger] = None) -> set:
    """从 instruments 目录加载指定股票池代码（已过滤过期条目）。"""
    pool_file = instruments_dir / f"{pool_name}.txt"
    if not pool_file.exists():
        if log:
            log.warning(f"股票池文件不存在: {pool_name}.txt")
        return set()
    stocks = set()
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                stocks.add(parts[0])
    if log:
        log.info(f"[股票池] {pool_name}: {len(stocks)} 只股票")
    return stocks


def read_factor_list(path: str, log: Optional[logging.Logger] = None) -> list:
    """读取因子名单。

    支持三种格式：
      txt —— 每行一个因子名（# / // 注释行跳过）
      csv —— 优先 factor_name 列，否则第一列
      json —— A 模块候选清单（family_candidates_*.json：平铺 candidates_by_family 或 final_names）
    """
    p = Path(path)
    if not p.exists():
        if log:
            log.error(f"因子名单文件不存在: {path}")
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    names = []
    if p.suffix.lower() == ".csv":
        import csv
        import io
        rows = list(csv.reader(io.StringIO(text)))
        if rows and rows[0] and "factor_name" in rows[0]:
            idx = rows[0].index("factor_name")
        else:
            idx = 0
        for r in rows[1:]:
            if r and r[idx].strip():
                names.append(r[idx].strip())
    elif p.suffix.lower() == ".json":
        import json as _json
        try:
            data = _json.loads(text)
        except Exception as e:
            if log:
                log.error(f"因子名单 JSON 解析失败: {e}")
            return []
        # A 模块候选清单：candidates_by_family 优先，其次平铺 final_names
        if isinstance(data, dict):
            by_family = data.get("candidates_by_family") or {}
            if isinstance(by_family, dict):
                for fam in sorted(by_family):
                    names.extend(by_family[fam] or [])
            elif isinstance(by_family, list):
                names.extend(by_family)
            flat = data.get("final_names") or []
            if isinstance(flat, list):
                names.extend(flat)
        elif isinstance(data, list):
            names.extend(data)
        # 去重保序
        seen = set()
        names = [n for n in names if not (n in seen or seen.add(n))]
    else:
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "//")):
                names.append(line)
    return names


def enrich_with_extra_fields(df: pd.DataFrame, start_time, end_time,
                             qlib_data_dir: Path,
                             log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """合并 circ_mv（流通市值）+ sw_l1（申万一级行业）到 df。

    路径 A：Qlib（$circ_mv / $sw_l1 + SW 映射解码）；
    路径 B：ClickHouse 后备（QuantDataAPI）。供场景压力测试与控制变量对冲使用。
    """
    if not isinstance(df.index, pd.MultiIndex):
        if log:
            log.warning("[补充字段] DataFrame 非 MultiIndex，跳过")
        return df

    instruments = df.index.get_level_values("instrument").unique().tolist()
    qlib_mc_ok = False

    # ── 路径 A：Qlib ──
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN, mute_warning=True)

        fields = ["$circ_mv", "$sw_l1"]
        col_names = ["circ_mv", "sw_l1"]
        extra = D.features(instruments, fields, str(start_time)[:10], str(end_time)[:10])
        if not extra.empty:
            if isinstance(extra.columns, pd.MultiIndex):
                extra.columns = [c[0] for c in extra.columns]
            keep = [f for f in fields if f in extra.columns]
            if keep:
                extra = extra[keep].copy()
                if "$sw_l1" in extra.columns:
                    mapping_path = qlib_data_dir / "sw_industry_mapping.json"
                    if mapping_path.exists():
                        try:
                            import json
                            with open(mapping_path, encoding="utf-8") as f:
                                id_map = json.load(f)
                            reverse_map = {v: k for k, v in id_map.get("l1", {}).items()}
                            if reverse_map:
                                mapped = extra["$sw_l1"].map(reverse_map)
                                match_rate = mapped.notna().sum() / max(len(mapped), 1)
                                if match_rate > 0.1:
                                    extra["$sw_l1"] = mapped
                                elif log:
                                    log.info(f"[补充字段] 行业编码映射匹配率仅 {match_rate:.1%}，保留原始数值")
                        except Exception:
                            pass
                if df.index.names != extra.index.names:
                    extra = extra.swaplevel()
                extra = extra.reindex(df.index)
                extra.columns = col_names
                df["circ_mv"] = extra["circ_mv"]
                df["sw_l1"] = extra["sw_l1"]

                n_valid_mc = int(df["circ_mv"].notna().sum())
                n_valid_ind = int(df["sw_l1"].notna().sum())
                mc_ok = n_valid_mc > len(df) * 0.1
                if log:
                    log.info(f"[补充字段] Qlib: circ_mv={n_valid_mc:,}行有效({mc_ok}), sw_l1={n_valid_ind:,}行有效 "
                             f"(共{len(instruments)}只股票)")

                from qlworks.evaluation.sw_mapping import decode_sw_series
                ind_ok = n_valid_ind > len(df) * 0.1
                is_numeric_ind = True
                if ind_ok:
                    decoded = decode_sw_series(df["sw_l1"], level=1)
                    n_mapped = decoded.str.startswith("未知(").sum() if len(decoded) > 0 else len(decoded)
                    mapped_ratio = 1 - n_mapped / max(len(decoded), 1)
                    if mapped_ratio > 0.5:
                        df["sw_l1"] = decoded
                        is_numeric_ind = False
                        if log:
                            log.info(f"[补充字段] SW_L1 解码: {1-n_mapped:,}/{len(decoded)} 行成功 ({mapped_ratio:.1%})")
                    elif log:
                        log.info(f"[补充字段] SW_L1 解码覆盖率不足 ({mapped_ratio:.1%})")

                if is_numeric_ind or not ind_ok:
                    try:
                        alt_extra = D.features(instruments, ["$industry"],
                                               str(start_time)[:10], str(end_time)[:10])
                        if not alt_extra.empty:
                            if isinstance(alt_extra.columns, pd.MultiIndex):
                                alt_extra.columns = [c[0] for c in alt_extra.columns]
                            if df.index.names != alt_extra.index.names:
                                alt_extra = alt_extra.swaplevel()
                            alt_extra = alt_extra.reindex(df.index)
                            df["sw_l1"] = alt_extra[alt_fields[0]] if False else alt_extra.iloc[:, 0]
                            n_valid_ind = int(df["sw_l1"].notna().sum())
                            ind_ok = n_valid_ind > len(df) * 0.1
                            is_numeric_ind = False
                            if log:
                                log.info(f"[补充字段] 回退 $industry: {n_valid_ind:,}行有效")
                    except Exception:
                        pass

                if mc_ok:
                    if log:
                        log.info(f"[补充字段] Qlib circ_mv 有效 ({n_valid_mc:,}行)，直接保留")
                else:
                    if log:
                        log.warning(f"[补充字段] Qlib circ_mv 不足 ({n_valid_mc:,}行)，后续 ClickHouse 补充")

                if mc_ok and ind_ok and not is_numeric_ind:
                    if log:
                        log.info(f"[补充字段] Qlib 全达标: circ_mv={n_valid_mc:,}, sw_l1={n_valid_ind:,} → 直接使用")
                    return df
                qlib_mc_ok = mc_ok
                if log:
                    ind_note = f"sw_l1={'数值编码' if is_numeric_ind else '不足'}({n_valid_ind:,}行)"
                    log.info(f"[补充字段] Qlib circ_mv={'有效' if mc_ok else '不足'}, {ind_note} → 仅对 sw_l1 用 ClickHouse 补充")
    except ImportError:
        if log:
            log.info("[补充字段] Qlib 未安装，切换到 ClickHouse 路径...")
    except Exception as e:
        if log:
            log.warning(f"[补充字段] Qlib 路径失败: {e}，切换到 ClickHouse 路径...")

    # ── 路径 B：ClickHouse 后备（仅补充缺失） ──
    try:
        from qlworks.data import QuantDataAPI
        api = QuantDataAPI()
        ch_sql = f"""SELECT p.ts_code AS instrument, p.trade_date AS datetime,
               CAST(i.circ_mv AS DOUBLE) AS circ_mv,
               sw.l1_name AS sw_l1
        FROM daily_prices p
        LEFT JOIN daily_indicators i ON p.ts_code=i.ts_code AND p.trade_date=i.trade_date
        LEFT JOIN sw_industry_members sw ON p.ts_code=sw.ts_code
        WHERE p.trade_date>='{start_time}' AND p.trade_date<='{end_time}'
        ORDER BY p.ts_code, p.trade_date"""
        raw = api.query(ch_sql)
        if raw is not None and not raw.empty:
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            raw = raw.drop_duplicates(subset=["instrument", "datetime"])
            raw = raw.set_index(["instrument", "datetime"])
            raw = raw.reindex(df.index)
            ch_has_sw = "sw_l1" in raw.columns and raw["sw_l1"].notna().any()
            if ch_has_sw:
                df["sw_l1"] = raw["sw_l1"]
                if log:
                    log.info(f"[补充字段] ClickHouse: sw_l1覆盖完成 (非空{raw['sw_l1'].notna().sum():,}行)")
            if not qlib_mc_ok and "circ_mv" in raw.columns:
                df["circ_mv"] = raw["circ_mv"]
                if log:
                    log.info(f"[补充字段] ClickHouse: circ_mv覆盖完成 (非空{raw['circ_mv'].notna().sum():,}行)")
            elif qlib_mc_ok and log:
                log.info(f"[补充字段] 保留 Qlib circ_mv，不做 ClickHouse 覆盖")
            if log:
                log.info(f"[补充字段] ClickHouse: circ_mv={int(df['circ_mv'].notna().sum()):,}行有效, "
                         f"sw_l1={int(df['sw_l1'].notna().sum()):,}行有效 (共{len(instruments)}只股票)")
    except Exception as e:
        if log:
            log.warning(f"[补充字段] ClickHouse 加载失败: {e}")

    return df


def evaluate_factor_list(
    factors: list[dict],
    *,
    start_time: str,
    end_time: str,
    pool: str = "main_board",
    recompute: bool = False,
    walk_forward: bool = False,
    qlib_data_dir: Path,
    instruments_dir: Path,
    log: Optional[logging.Logger] = None,
) -> list[dict]:
    """对因子列表执行完整深评（仓库优先 → 数据质量 → FactorEvaluator 评测）。

    Args:
        factors: [{"name","expr","duckdb_expr","category",...}]
        start_time/end_time: 评测区间
        pool: 股票池（默认 main_board，与训练端对齐）
        recompute: 强制从 ClickHouse 重算（否则仓库优先）
        walk_forward: 启用 Walk-Forward 滚动外推验证
        qlib_data_dir: qlib 数据目录
        instruments_dir: instruments 目录

    Returns:
        [{"name","result","tier","ic_mean","icir_nw","report_dir"}]（按输入顺序，跳过无数据的）
    """
    log = log or logger
    from qlworks.evaluation import FactorStore, DEFAULT_CONFIG
    from dataclasses import replace
    from qlworks.evaluation.runner import FactorEvaluator
    from qlworks.config import QLIB_DATA_DIR
    from qlworks.pipeline_config import FILTER_LIMIT_UPDOWN, FILTER_SUSPENDED

    if qlib_data_dir is None:
        qlib_data_dir = QLIB_DATA_DIR
    if instruments_dir is None:
        instruments_dir = qlib_data_dir / "instruments"

    store = FactorStore()
    config = replace(DEFAULT_CONFIG, instruments=pool, start_time=start_time, end_time=end_time)
    evaluator = FactorEvaluator(config)
    results = []
    pool_stocks = load_pool_stocks(pool, instruments_dir, log)

    for i, f in enumerate(factors, 1):
        name = f["name"]
        expr = f["expr"]
        log.info(f"\n[{i}/{len(factors)}] {name}")

        try:
            # [Citadel Alpha Lab] 仓库优先
            if recompute or not store.get_warehouse_meta(name):
                duckdb_expr = f.get("duckdb_expr", "")
                log.info(f"{name} 仓库无数据或强制重算，从 ClickHouse 计算...")
                store.compute_to_warehouse(name, expr, start_time, end_time,
                                           duckdb_expr=duckdb_expr)
            else:
                warehouse_meta = store.get_warehouse_meta(name)
                wh_start = meta_val(warehouse_meta, "first_date", "")
                wh_end = meta_val(warehouse_meta, "last_date", "")
                total_rows_val = meta_val(warehouse_meta, "total_records", 0) or meta_val(warehouse_meta, "total_rows", 0)
                log.info(f"{name} 从仓库读取 ({wh_start} ~ {wh_end}, {total_rows_val:,} 行)")

            df = store.load_from_warehouse(name, start_time, end_time)
            if df is None or df.empty:
                log.warning(f"{name} 无数据，跳过")
                continue
            df = df.rename(columns={"value": name})

            # 股票池过滤
            if pool_stocks:
                n_before = len(df)
                df = df[df.index.get_level_values("instrument").isin(pool_stocks)]
                log.info(f"[股票池] {pool} 过滤: {n_before:,} → {len(df):,} 行 "
                         f"({df.index.get_level_values('instrument').nunique()} 只)")
                if df.empty:
                    log.warning(f"{name}: 股票池 {pool} 过滤后无数据，跳过")
                    continue

            # 标签合并
            df_label = evaluator._load_labels(start_time, end_time)
            if df_label is not None:
                df = df.join(df_label, how="inner")

            # [数据质量] 标签可交易性过滤（涨跌停/一字板/停牌样本标签置 NaN）
            if FILTER_LIMIT_UPDOWN:
                from qlworks.factors.filter_utils import filter_untradeable_labels
                label_col = evaluator.config.label_name
                if label_col in df.columns:
                    inst_list = df.index.get_level_values("instrument").unique().tolist()
                    label_clean = filter_untradeable_labels(
                        df[[label_col]].copy(), inst_list,
                        str(start_time)[:10], str(end_time)[:10],
                    )
                    if label_clean is not None:
                        df[label_col] = label_clean[label_col]

            # [数据质量] volume / change_pct 补充（使 filter_ashare_constraints 生效）
            if FILTER_SUSPENDED or evaluator.config.limit_up_pct is not None:
                try:
                    import qlib as _q
                    from qlib.config import REG_CN as _REG_CN
                    from qlib.data import D as _D
                    if not evaluator._qinited:
                        _q.init(provider_uri=str(qlib_data_dir), region=_REG_CN)
                        evaluator._qinited = True
                    inst_list = df.index.get_level_values("instrument").unique().tolist()
                    _extra = _D.features(
                        inst_list,
                        ["$volume", "Ref($close, 1) / $close - 1"],
                        str(start_time)[:10], str(end_time)[:10],
                    )
                    if _extra is not None and not _extra.empty:
                        if isinstance(_extra.columns, pd.MultiIndex):
                            _extra.columns = _extra.columns.droplevel(1)
                        _extra = _extra.rename(columns={
                            "$volume": "volume",
                            "Ref($close, 1) / $close - 1": "change_pct",
                        })
                        if df.index.names != _extra.index.names:
                            _extra = _extra.swaplevel()
                        df["volume"] = _extra["volume"]
                        df["change_pct"] = _extra["change_pct"]
                except Exception as _e:
                    log.warning(f"[数据质量] volume/change_pct 加载失败（{_e}），涨跌停/停牌过滤保持不生效")

            # mkt_cap + industry（场景压力测试/控制变量对冲必需）
            df = enrich_with_extra_fields(df, start_time, end_time, qlib_data_dir, log)

            # [AQR] 完整评测流水线
            if walk_forward:
                result = evaluator.walk_forward_evaluate(
                    name, df.reset_index(),
                    config_override={"category": f.get("category", "satellite")},
                )
            else:
                result = evaluator.evaluate(
                    name, df.reset_index(),
                    category=f.get("category", "satellite"),
                )
            q = result["qual_result"]
            icon = {"core": "[OK]", "satellite": "[~]", "archive": "[ ]"}.get(q["tier"], "[ ]")
            cur_meta = store.get_warehouse_meta(name) or {}
            total_str = meta_val(cur_meta, "total_records", 0) or meta_val(cur_meta, "total_rows", 0)
            if total_str:
                total_str = f"{total_str:,}"
            last_dt = meta_val(cur_meta, "last_date", "?")
            last_rows = f"({total_str}行, 至{last_dt})" if not recompute else ""
            print(f"  {icon} 等级={q['tier']:10s} IC={result['ic_stats']['ic_mean']:.4f} "
                  f"ICIR={result['ic_stats']['icir']:.2f} 评分={q['composite_score']:.1f} {last_rows}")

            results.append({
                "name": name,
                "result": result,
                "tier": q["tier"],
                "ic_mean": result["ic_stats"].get("ic_mean"),
                "icir_nw": result["ic_stats"].get("icir_nw") or result["ic_stats"].get("icir"),
                "factor_info": f,
            })
        except Exception as e:
            print(f"  [X] {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n完成! 报告: {config.report_dir}")
    return results
