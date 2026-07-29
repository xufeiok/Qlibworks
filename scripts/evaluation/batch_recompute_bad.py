"""
重新计算 warehouse 中数据异常的 14 个因子
等 Qlib 同步完成后执行，确保 Qlib 回退路径返回正确数据
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 确认有异常的 14 个因子
BAD_FACTORS = [
    "Alpha3", "Alpha38", "Alpha40", "Alpha76", "Alpha80",
    "Alpha112", "Alpha118", "Alpha128", "Alpha129", "Alpha167",
    "EXTREME_REV", "LTR_12m", "ROC5", "inv_turn"
]

YAML_PATH = Path(r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml")
START = "2020-01-01"
END = "2026-06-12"

def main():
    # 加载 YAML
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 提取因子的 qlib 表达式
    factors_map = {}
    for fd in data.get("factors", []):
        name = fd.get("name", "")
        if name in BAD_FACTORS:
            expr = fd.get("expression", {})
            if isinstance(expr, dict):
                qlib_expr = str(expr.get("qlib", ""))
                duckdb_expr = str(expr.get("duckdb", ""))
            else:
                qlib_expr = str(expr)
                duckdb_expr = ""
            factors_map[name] = (qlib_expr, duckdb_expr)

    # 非 GTJA 因子手动补充
    extra = {
        "EXTREME_REV": ("-1 * Sum(Return($close, 1), 20)", None),
        "LTR_12m": ("Return($close, 252)", None),
        "ROC5": ("($close - Ref($close, 5)) / Ref($close, 5)", None),
        "inv_turn": ("1 / $turnover_rate", None),
    }
    factors_map.update(extra)

    logger.info(f"待重新计算因子: {list(factors_map.keys())} ({len(factors_map)} 个)")

    from qlworks.evaluation.factor_store import FactorStore
    store = FactorStore()

    for name, (qlib_expr, duckdb_expr) in factors_map.items():
        logger.info(f"{'='*50}")
        logger.info(f"重新计算 {name}")
        try:
            stats = store.compute_to_warehouse(
                name, qlib_expr, START, END,
                duckdb_expr=duckdb_expr,
                overwrite=True
            )
            if stats:
                years_str = ",".join(f"{y}({c})" for y, c in stats.items())
                logger.info(f"  ✅ {name} 完成: {years_str}")
            else:
                logger.info(f"  ⏭️ {name} 已有完整数据")
        except Exception as e:
            logger.error(f"  ❌ {name} 失败: {e}")

    logger.info(f"{'='*50}")
    logger.info("全部完成！")

if __name__ == "__main__":
    main()
