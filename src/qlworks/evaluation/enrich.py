"""补充字段工具：加载 circ_mv（流通市值）和 sw_l1（申万一级行业）。

优先走 Qlib 本地数据路径，Qlib 数据不足时回退到 ClickHouse。
"""

import json
import logging
from pathlib import Path

import pandas as pd

from qlworks.config import QLIB_DATA_DIR
from qlworks.evaluation.sw_mapping import decode_sw_series

logger = logging.getLogger("enrich")


def enrich_with_extra_fields(df, start_time, end_time, log=None):
    """加载 circ_mv 和 sw_l1 并合并到 df。

    这样场景压力测试（分市值、分行业板块）和控制变量对冲（双变量分组、残差因子）
    才能获取数据，否则这些评测项目会因缺数据而使用 0.5 中性分。

    Args:
        df: DataFrame with MultiIndex (instrument, datetime), 含因子列和标签列
        start_time: 起始日期
        end_time: 结束日期
        log: 可选的 logger 实例

    Returns:
        合并后的 DataFrame，包含 circ_mv 和 sw_l1 列
    """
    log = log or logger

    if not isinstance(df.index, pd.MultiIndex):
        log.warning("[补充字段] DataFrame 非 MultiIndex，跳过")
        return df

    instruments = df.index.get_level_values("instrument").unique().tolist()

    # ── 路径 A：Qlib 本地数据 ──
    try:
        import qlib
        from qlib.config import REG_CN
        from qlib.data import D

        qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN, mute_warning=True)

        fields = ["$circ_mv", "$sw_l1"]
        col_names = ["circ_mv", "sw_l1"]
        extra = D.features(instruments, fields, str(start_time)[:10], str(end_time)[:10])
        if not extra.empty:
            if isinstance(extra.columns, pd.MultiIndex):
                extra.columns = [c[0] for c in extra.columns]
            keep = [f for f in fields if f in extra.columns]
            if keep:
                extra = extra[keep].copy()

                # 处理 sw_l1 编码
                if "$sw_l1" in extra.columns:
                    mapping_path = Path(str(QLIB_DATA_DIR)) / "sw_industry_mapping.json"
                    if mapping_path.exists():
                        try:
                            with open(mapping_path, encoding="utf-8") as f:
                                id_map = json.load(f)
                            reverse_map = {v: k for k, v in id_map.get("l1", {}).items()}
                            if reverse_map:
                                mapped = extra["$sw_l1"].map(reverse_map)
                                match_rate = mapped.notna().sum() / max(len(mapped), 1)
                                if match_rate > 0.1:
                                    extra["$sw_l1"] = mapped
                                else:
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
                log.info(f"[补充字段] Qlib: circ_mv={n_valid_mc:,}行有效({mc_ok}), "
                         f"sw_l1={n_valid_ind:,}行有效 (共{len(instruments)}只股票)")

                # ── sw_l1 解码：先用本地 SW 映射 ──
                ind_ok = n_valid_ind > len(df) * 0.1
                is_numeric_ind = True

                if ind_ok:
                    decoded = decode_sw_series(df["sw_l1"], level=1)
                    n_mapped = int(decoded.str.startswith("未知(").sum()) if len(decoded) > 0 else len(decoded)
                    mapped_ratio = 1 - n_mapped / max(len(decoded), 1)
                    if mapped_ratio > 0.5:
                        df["sw_l1"] = decoded
                        is_numeric_ind = False
                        log.info(f"[补充字段] SW_L1 解码: {len(decoded)-n_mapped:,}/{len(decoded)} 行成功 ({mapped_ratio:.1%})")
                    else:
                        log.info(f"[补充字段] SW_L1 解码覆盖率不足 ({mapped_ratio:.1%})")

                # 解码失败时回退 $industry
                if is_numeric_ind or not ind_ok:
                    try:
                        alt_fields = ["$industry"]
                        alt_extra = D.features(instruments, alt_fields,
                                               str(start_time)[:10], str(end_time)[:10])
                        if not alt_extra.empty:
                            if isinstance(alt_extra.columns, pd.MultiIndex):
                                alt_extra.columns = [c[0] for c in alt_extra.columns]
                            if df.index.names != alt_extra.index.names:
                                alt_extra = alt_extra.swaplevel()
                            alt_extra = alt_extra.reindex(df.index)
                            df["sw_l1"] = alt_extra[alt_fields[0]]
                            n_valid_ind = int(df["sw_l1"].notna().sum())
                            ind_ok = n_valid_ind > len(df) * 0.1
                            is_numeric_ind = False
                            log.info(f"[补充字段] 回退 $industry: {n_valid_ind:,}行有效")
                    except Exception:
                        pass

                # circ_mv 独立处理
                if mc_ok:
                    log.info(f"[补充字段] Qlib circ_mv 有效 ({n_valid_mc:,}行)，直接保留")
                else:
                    log.warning(f"[补充字段] Qlib circ_mv 不足 ({n_valid_mc:,}行)，后续 ClickHouse 补充")

                # 都满足条件则直接返回
                if mc_ok and ind_ok and not is_numeric_ind:
                    log.info(f"[补充字段] Qlib 全达标: circ_mv={n_valid_mc:,}, sw_l1={n_valid_ind:,} → 直接使用")
                    return df

                ind_note = f"sw_l1={'数值编码' if is_numeric_ind else '不足'}({n_valid_ind:,}行)"
                log.info(f"[补充字段] Qlib circ_mv={'有效' if mc_ok else '不足'}, "
                         f"{ind_note} → 仅对 sw_l1 用 ClickHouse 补充")

    except ImportError:
        log.info("[补充字段] Qlib 未安装，切换到 ClickHouse 路径...")
    except Exception as e:
        log.warning(f"[补充字段] Qlib 路径失败: {e}，切换到 ClickHouse 路径...")

    # ── 路径 B：ClickHouse 后备 ──
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
                log.info(f"[补充字段] ClickHouse: sw_l1覆盖完成 (非空{raw['sw_l1'].notna().sum():,}行)")

            qlib_mc_ok = False
            try:
                qlib_mc_ok = mc_ok
            except NameError:
                qlib_mc_ok = False
            if not qlib_mc_ok and "circ_mv" in raw.columns:
                df["circ_mv"] = raw["circ_mv"]
                log.info(f"[补充字段] ClickHouse: circ_mv覆盖完成 (非空{raw['circ_mv'].notna().sum():,}行)")
            elif qlib_mc_ok:
                log.info(f"[补充字段] 保留 Qlib circ_mv ({n_valid_mc:,}行)，不做 ClickHouse 覆盖")

            n_valid_mc = int(df["circ_mv"].notna().sum())
            n_valid_ind = int(df["sw_l1"].notna().sum())
            log.info(f"[补充字段] ClickHouse: circ_mv={n_valid_mc:,}行有效, "
                     f"sw_l1={n_valid_ind:,}行有效 (共{len(instruments)}只股票)")
    except Exception as e:
        log.warning(f"[补充字段] ClickHouse 路径也失败: {e}")

    return df
