"""
补充计算 GTJA191 中缺失的 14 个因子。
直接读取 archive/gtja191_factor_dictionary.yaml，绕过 FactorLibraryManager 的查找问题。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import yaml
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 缺失的 14 个因子
MISSING = ["Alpha126","Alpha128","Alpha129","Alpha132","Alpha134","Alpha139",
           "Alpha145","Alpha150","Alpha153","Alpha167","Alpha168","Alpha178",
           "Alpha189","Alpha191"]

YAML_PATH = Path(r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml")
START = "2020-01-01"
END = "2026-06-12"

def main():
    # 加载 YAML
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 提取缺失因子的表达式
    factors_map = {}
    for fd in data.get("factors", []):
        name = fd.get("name", "")
        if name in MISSING:
            expr = fd.get("expression", {})
            if isinstance(expr, dict):
                qlib_expr = str(expr.get("qlib", ""))
                duckdb_expr = str(expr.get("duckdb", ""))
            else:
                qlib_expr = str(expr)
                duckdb_expr = ""
            factors_map[name] = (qlib_expr, duckdb_expr)

    logger.info(f"待计算因子: {list(factors_map.keys())} ({len(factors_map)} 个)")

    from qlworks.evaluation.factor_store import FactorStore
    store = FactorStore()

    for name, (qlib_expr, duckdb_expr) in factors_map.items():
        logger.info(f"{'='*50}")
        logger.info(f"开始计算 {name}")
        logger.info(f"  qlib: {qlib_expr[:80]}")
        logger.info(f"  duckdb: {duckdb_expr[:80]}")
        try:
            stats = store.compute_to_warehouse(
                name, qlib_expr, START, END,
                duckdb_expr=duckdb_expr,
                overwrite=True  # 覆盖之前的 placeholder 结果
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
