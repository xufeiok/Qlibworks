"""
单因子评测主引擎：编排从数据加载到报告输出的完整流水线。
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered")

logger = logging.getLogger(__name__)

from .config import EvalConfig, DEFAULT_CONFIG
from .preprocessor import preprocess_factor
from .factor_def import DataQualityReport
from .ic_analysis import calc_daily_ic, calc_ic_stats, calc_decay_analysis, calc_rankic_series
from .group_analysis import (
    quantile_returns, long_short_returns, calc_ls_stats,
    calc_group_avg_returns, calc_monotonicity_score, calc_turnover,
    calc_holding_period_returns,
)
from .robustness import test_sub_periods, test_sub_pools
from .report_generator import FactorReportGenerator
from .factor_selector import (
    evaluate_qualification, update_factor_tier,
    update_factor_registry, handle_candidate_pool_entry,
)
from .lifecycle import LifecycleManager
from .factor_store import FactorStore


class FactorEvaluator:
    """单因子评测器（集成生命周期 + 候选池 + 三级分类导出）。"""

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self._lifecycle = None
        self._qinited = False

    @property
    def lifecycle(self):
        if self._lifecycle is None:
            self._lifecycle = LifecycleManager(self.config.registry_dir)
        return self._lifecycle

    def load_data(self, factor_expr, factor_name, instruments=None,
                  start_time=None, end_time=None, extra_fields=None):
        """加载因子数据和标签。"""
        store = FactorStore(self.config)
        sd = start_time or self.config.start_time
        ed = end_time or self.config.end_time

        try:
            df = store.load(factor_name, factor_expr, sd, ed)
            label_df = self._load_labels(sd, ed)
            if label_df is not None:
                df = df.join(label_df, how="inner")
            return df.reset_index()
        except (FileNotFoundError, RuntimeError):
            pass

        return self._load_via_qlib(factor_expr, factor_name, instruments, sd, ed, extra_fields)

    def _load_labels(self, start_time, end_time):
        """加载标签收益率。

        Qlib 数据已含复权价格，直接从本地 Qlib 计算标签。
        """
        logger.info("[标签] 从本地 Qlib 数据计算标签...")
        try:
            import qlib
            from qlib.config import REG_CN
            from qlib.data import D
            from qlworks.config import QLIB_DATA_DIR

            if not self._qinited:
                qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)
                self._qinited = True
            # Windows 强制单线程，避免 joblib multiprocessing 死锁
            from qlib.config import C as _QC
            _QC.dataloader_workers = 1
            _QC.joblib_backend = "threading"

            # 从 instruments/all.txt 读取股票列表
            ins_file = Path(str(QLIB_DATA_DIR)) / "instruments" / "all.txt"
            all_ins = []
            if ins_file.exists():
                with open(ins_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_ins.append(parts[0])
            if not all_ins:
                return None

            # 分批查询 qlib（每次 2000 只，避免内存爆炸）
            st = pd.Timestamp(start_time) - pd.Timedelta(days=30)
            et = end_time
            all_parts = []
            batch_size = 2000
            logger.info(f"[标签] 开始加载 {len(all_ins)} 只股票数据（{batch_size} 只/批）")
            for i in range(0, len(all_ins), batch_size):
                batch = all_ins[i:i + batch_size]
                try:
                    df_batch = D.features(batch, ["$close", "$open"], str(st)[:10], et)
                    if not df_batch.empty and len(df_batch) > 10:
                        all_parts.append(df_batch)
                except Exception:
                    continue

            if not all_parts:
                return None
            df = pd.concat(all_parts)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            df = df.reset_index()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["instrument"] = df["instrument"].astype(str)
            df = df.rename(columns={"$close": "close", "$open": "open"})
            # 将 qlib 列名映射为 DuckDB 可识别的列名
            df = df.rename(columns={"instrument": "ts_code", "datetime": "trade_date"})

            # 用 DuckDB 本地计算标签（将 Ref 表达式转为 DuckDB SQL）
            conn = duckdb.connect()
            conn.register("_raw", df)
            label_expr = self.config.label_expr
            # 转义 $ 变量为 DuckDB 列名
            label_expr = label_expr.replace("$close", "close").replace("$open", "open")
            # Ref(col, -N) → LEAD(col, N) OVER (...), Ref(col, N) → LAG(col, N) OVER (...)
            import re as _re
            def _translate_ref(m):
                col = m.group(1)
                off = int(m.group(2))
                if off < 0:
                    return f"LEAD({col}, {-off}) OVER (PARTITION BY ts_code ORDER BY trade_date)"
                else:
                    return f"LAG({col}, {off}) OVER (PARTITION BY ts_code ORDER BY trade_date)"
            label_expr = _re.sub(r"Ref\((\w+),\s*(-?\d+)\)", _translate_ref, label_expr)
            r = conn.execute(f"""
                SELECT ts_code AS instrument, trade_date AS datetime,
                       {label_expr} AS {self.config.label_name}
                FROM _raw
                WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL
                ORDER BY ts_code, trade_date
            """).df()
            conn.close()

            if r.empty:
                return None
            r["datetime"] = pd.to_datetime(r["datetime"])
            # 过滤 inf/-inf，并裁剪极端值（避免开源/除零产生的异常收益率污染统计）
            col = self.config.label_name
            r[col] = r[col].replace([np.inf, -np.inf], np.nan)
            r[col] = r[col].clip(-1.0, 10.0)  # -100% ~ +1000%，裁掉明显异常值
            r = r.set_index(["instrument", "datetime"])
            r = r[r.index.get_level_values("datetime") >= start_time]
            r = r[r.index.get_level_values("datetime") <= end_time]
            logger.info(f"[标签] Qlib 标签: {len(r)} 行, {r.index.get_level_values('instrument').nunique()} 只股票")
            return r[[self.config.label_name]]

        except Exception as e:
            logger.warning(f"[标签] Qlib 兜底也失败: {e}")
            return None

    def _load_via_qlib(self, factor_expr, factor_name, instruments, st, et, extra_fields):
        """通过 Qlib D.features() 加载数据。"""
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D
        from qlworks.config import QLIB_DATA_DIR

        if not self._qinited:
            qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN)
            self._qinited = True
        # Windows 强制单线程
        from qlib.config import C as _QC
        _QC.dataloader_workers = 1
        _QC.joblib_backend = "threading"

        ifile = Path(str(QLIB_DATA_DIR)) / "instruments" / "all.txt"
        pool = []
        if ifile.exists():
            with open(ifile, encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split()
                    if p: pool.append(p[0])
        pool = pool[:300]

        all_fields = [factor_expr] + [self.config.label_expr]
        field_names = [factor_name, self.config.label_name]
        if extra_fields:
            all_fields.extend(extra_fields.values())
            field_names.extend(extra_fields.keys())

        df = D.features(pool, all_fields, st, et)
        if df.empty:
            raise ValueError(f"未提取到数据: {factor_expr}")

        if isinstance(df.columns, pd.MultiIndex):
            raw_names = [c[0] for c in df.columns]
        else:
            raw_names = list(df.columns)

        rename = {r: n for r, n in zip(all_fields, field_names) if r in raw_names}
        df = df.rename(columns=rename)
        df.columns = [rename.get(c, c) for c in raw_names]

        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["instrument"] = df["instrument"].astype(str)
        return df.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    def evaluate(self, factor_name, df, extra_config=None,
                 category="satellite", skip_candidate_pool=False):
        """对单因子执行完整评测流水线。"""
        config = self.config
        if extra_config:
            for k, v in extra_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        factor_col, label_col = factor_name, config.label_name

        # 1. 预处理
        preproc_cfg = {
            "winsorize_method": config.winsorize_method,
            "winsorize_threshold": config.winsorize_threshold,
            "standardize_method": config.standardize_method,
            "neutralization": config.neutralization,
        }
        ic_col = "industry" if "industry" in df.columns else None
        mc_col = "mkt_cap" if "mkt_cap" in df.columns else None
        df_proc = preprocess_factor(df, factor_col, ic_col, mc_col, preproc_cfg)
        preproc_info = {
            "去极值": config.winsorize_method,
            "标准化": config.standardize_method,
            "中性化": config.neutralization,
        }

        # 2. IC 分析（含向量化 Spearman + 行业中性 IC）
        ic_series = calc_daily_ic(df_proc, factor_col, label_col, config.ic_method)
        ic_stats = calc_ic_stats(ic_series, config.ic_annual_factor)

        # 2b. 行业中性 Rank IC（如果数据含 industry 列）
        industry_ic_series = None
        if "industry" in df_proc.columns:
            try:
                industry_ic_series = calc_rankic_series(df_proc, factor_col, label_col, group_col="industry")
                ic_ind = calc_ic_stats(industry_ic_series, config.ic_annual_factor)
                ic_stats["industry_ic_mean"] = ic_ind["ic_mean"]
                ic_stats["industry_icir"] = ic_ind["icir"]
            except Exception:
                pass

        # 3. 分层回测
        q_df = quantile_returns(df_proc, factor_col, label_col, config.quantiles)
        group_means = calc_group_avg_returns(q_df) if not q_df.empty else pd.Series()
        mono = calc_monotonicity_score(q_df) if not q_df.empty else 0.0
        ic_stats["monotonicity"] = round(mono, 4)

        ls_df = long_short_returns(q_df, config.quantiles - 1, 0, cost=config.robustness_ls_cost)
        ls_stats = calc_ls_stats(ls_df, config.ic_annual_factor, config.label_horizon)
        ls_stats["monotonicity"] = round(mono, 4)

        turnover_stats = calc_turnover(q_df) if not q_df.empty else {}
        decay_df = calc_decay_analysis(df_proc, factor_col, label_col) if not df_proc.empty else pd.DataFrame()

        # 3b. 多期持有收益分析（不同调仓周期的因子表现）
        hpr_df = calc_holding_period_returns(df_proc, factor_col, label_col, config.quantiles,
                                              horizons=[1, 5, 10, 20]) if not df_proc.empty else pd.DataFrame()
        best_horizon = 5
        if not hpr_df.empty and hpr_df["ls_return"].notna().any():
            best_row_idx = hpr_df["ls_return"].abs().idxmax()
            best_row = hpr_df.loc[best_row_idx] if best_row_idx is not None else None
            best_horizon = int(best_row["horizon"]) if best_row is not None else 5
            logger.info(f"[多期HPR] {factor_name} 最佳调仓周期={best_horizon}日")

        # 4. 稳健性检验 — 子时段 + 子股票池
        robustness_df = test_sub_periods(
            df_proc, factor_col, label_col,
            config.robustness_sub_periods,
            config.ic_annual_factor, config.label_horizon,
        )
        if hasattr(config, 'robustness_sub_pools') and config.robustness_sub_pools:
            try:
                from qlworks.data import QuantDataAPI
                api = QuantDataAPI()
                sub_pool_results = test_sub_pools(
                    lambda pool, fields, st, et: api.query(
                        f"SELECT trade_date, ts_code, {', '.join(fields)} FROM daily_prices WHERE trade_date>='{st}' AND trade_date<='{et}'"
                    ),
                    factor_col, label_col,
                    "csi500",
                    config.start_time, config.end_time,
                    [factor_col, label_col],
                    {factor_col: factor_col, label_col: label_col},
                    config.ic_annual_factor,
                    label_horizon=config.label_horizon,
                )
                n_pools = len(sub_pool_results) if not sub_pool_results.empty else 0
                if n_pools:
                    logger.info(f"[稳健性] {factor_name} 子股票池检验完成: {n_pools} 个池")
            except Exception as e:
                logger.warning(f"[稳健性] {factor_name} 子股票池检验跳过: {e}")

        # 5. 等级判定（增强版：含衰减/换手率/覆盖率评分）
        total_dates = df_proc["datetime"].nunique() if "datetime" in df_proc.columns else 0
        n_dates_all = len(ic_series.dropna())
        coverage_pct = n_dates_all / total_dates if total_dates > 0 else 1.0
        qual_result = evaluate_qualification(
            ic_stats, ls_stats, config,
            decay_df=decay_df,
            turnover_stats=turnover_stats,
            coverage_pct=coverage_pct,
        )
        tier = qual_result["tier"]

        # 5b. 生命周期退化检测 — 对比上次评测记录，持续劣化则自动降级
        consecutive_bad = 0
        if config.enable_lifecycle:
            import json
            reg_path_p = Path(config.registry_dir) / "registry.json"
            if reg_path_p.exists():
                try:
                    with open(reg_path_p, encoding="utf-8") as f:
                        registry = json.load(f)
                    prev = registry.get("factors", {}).get(factor_name, {})
                    prev_ic = prev.get("ic_mean")
                    if prev_ic is not None and prev_ic != 0:
                        is_bad, severity, msg = self.lifecycle.check_degradation(
                            ic_stats["ic_mean"], ic_stats["icir"],
                            prev.get("ic_mean", 0), prev.get("icir", 0),
                        )
                        if severity == "danger":
                            tier = "archive"
                            qual_result["tier"] = "archive"
                            qual_result["reasons"].append(f"生命周期退化: {msg}")
                            logger.warning(f"[生命周期] {factor_name} 严重退化: {msg}")

                    # 监控告警：连续劣化计数（monitor_consecutive_bad）
                    vh = prev.get("version_history", [])
                    if len(vh) >= config.monitor_consecutive_bad:
                        recent = vh[-config.monitor_consecutive_bad:]
                        bad_count = sum(
                            1 for r in recent
                            if abs(r.get("ic_mean", 0)) < config.monitor_ic_warning
                        )
                        if bad_count >= config.monitor_consecutive_bad:
                            consecutive_bad = bad_count
                            logger.warning(
                                f"[监控] {factor_name} 连续{bad_count}次评测IC低于警告阈值"
                            )
                except Exception:
                    pass

        # 6. 报告（按分类，文件名含时间段+时间戳确保每次评测独立）
        report_dir = Path(config.report_dir) / tier
        report_dir.mkdir(parents=True, exist_ok=True)
        gen = FactorReportGenerator(factor_name, str(report_dir),
                                     eval_start=config.start_time,
                                     eval_end=config.end_time)
        html = gen.generate(
            ic_stats=ic_stats, group_means=group_means, ls_stats=ls_stats,
            robustness_df=robustness_df, preprocess_info=preproc_info,
            qual_status=qual_result["qualified"],
            thresholds_info={
                "ic": config.ic_threshold, "icir": config.icir_threshold,
                "win_rate": config.win_rate_threshold,
                "ls_ret": config.ls_annual_return_threshold * 100,
                "ls_sharpe": config.ls_sharpe_threshold,
            },
            eval_period={"start": config.start_time, "end": config.end_time},
            label_expr=config.label_expr, decay_df=decay_df,
            turnover_stats=turnover_stats,
            qual_result=qual_result,
            hpr_df=hpr_df,
        )
        report_path = gen.save(html)
        csv_path = str(report_dir / "_summary.csv")
        gen.export_summary_csv(ic_stats, ls_stats, qual_result["qualified"], csv_path)

        # 7. 注册表 + 生命周期
        reg_path = str(Path(config.registry_dir) / "registry.json")
        update_factor_registry(reg_path, factor_name, qual_result, ic_stats, ls_stats,
                                lifecycle_manager=self.lifecycle if config.enable_lifecycle else None)

        # 8. 候选池
        if not skip_candidate_pool:
            handle_candidate_pool_entry(factor_name, ic_stats, ls_stats, config)

        # 9. 因子等级引用（不复制数据，只写 ref.json 标记）
        #     实际数据在 warehouse 中统一管理
        store = FactorStore(config)
        tier_path = store.link_factor_to_tier(factor_name, tier)
        qualified_path = tier_path

        dq_report = DataQualityReport.from_dataframe(df, factor_col) if not df.empty else None

        return {
            "factor_name": factor_name, "ic_stats": ic_stats,
            "group_means": group_means, "ls_stats": ls_stats,
            "robustness_df": robustness_df, "qual_result": qual_result,
            "report_path": str(report_path), "qualified_path": qualified_path,
            "lifecycle_stage": qual_result.get("lifecycle_stage", ""),
            "data_quality": dq_report,
        }

    def evaluate_batch(self, factors, df_fn):
        """批量评测因子。"""
        results = []
        for f in factors:
            try:
                df = df_fn(f)
                result = self.evaluate(f["name"], df, category=f.get("category", "satellite"))
                results.append(result)
                q = result["qual_result"]
                icon = {"core": "OK", "satellite": "~~", "archive": "  "}.get(q["tier"], "  ")
                print(f"  [{icon}] {f['name']}: tier={q['tier']}, IC={result['ic_stats']['ic_mean']:.4f}, ICIR={result['ic_stats']['icir']:.2f}, 评分={q['composite_score']:.1f}")
            except Exception as e:
                print(f"  [X] {f['name']}: {e}")
                import traceback; traceback.print_exc()
                results.append({"factor_name": f["name"], "error": str(e)})
        return results