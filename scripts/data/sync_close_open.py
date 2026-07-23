"""
补全 Qlib close/open 数据 — 从 ClickHouse 写入 .day.bin

功能：
  从 ClickHouse daily_prices 读取 2010~2025 年完整的 close/open 数据，
  写入 Qlib .day.bin 格式，覆盖现有不完整的 close.day.bin 和 open.day.bin 文件。
  不影响 circ_mv、sw_l1 等其他字段。

用法：
  cd E:\Quant\Qlibworks
  python scripts/data/sync_close_open.py

注意事项：
  - 使用前复权价格（与 QlibSynchronizer 默认行为一致）
  - 只处理 features 目录中已有股票目录的股票（不会新增股票）
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _project_root)
sys.path.insert(0, str(Path(_project_root) / "src"))

import numpy as np
from tqdm import tqdm

from qlworks.data.api import QuantDataAPI
from qlworks.data.qlib_sync import QlibSynchronizer


def main():
    QLIB_DATA_DIR = r"E:\Quant\Qlibworks\qlib_data"
    features_dir = Path(QLIB_DATA_DIR) / "features"

    # ── 1. 获取已有股票目录列表，转为大写 ts_code ──
    existing_dirs = sorted([
        d.name for d in features_dir.iterdir()
        if d.is_dir() and (d / "close.day.bin").exists()
    ])
    # 目录名格式: 000001.sz → ClickHouse ts_code: 000001.SZ
    existing_stocks = [c.upper() for c in existing_dirs]
    print(f"现有股票: {len(existing_stocks)} 只")

    # ── 2. 读取日历 ──
    cal_path = Path(QLIB_DATA_DIR) / "calendars" / "day.txt"
    with open(cal_path) as f:
        cal_list = [l.strip() for l in f if l.strip()]
    cal_map = {d: i for i, d in enumerate(cal_list)}
    print(f"日历: {cal_list[0]} ~ {cal_list[-1]} ({len(cal_list)} 天)")

    # ── 3. 连接 ClickHouse 并同步 close/open ──
    with QuantDataAPI() as api:
        syncer = QlibSynchronizer(api)

        # 仅同步 close 和 open，不影响其他字段
        syncer.field_mapping = {"close": "close", "open": "open"}

        print(f"\n开始同步 close/open 数据...")
        print(f"日期范围: 2010-01-04 ~ 2025-12-31\n")

        syncer._sync_features(
            stocks=existing_stocks,
            calendar_list=cal_list,
            calendar_map=cal_map,
            start_date="2010-01-04",
            end_date="2025-12-31",
        )

    # ── 4. 抽样验证 ──
    print("\n" + "=" * 60)
    print("抽样验证")
    print("=" * 60)

    import struct

    def _read_bin_summary(fp: Path) -> dict:
        """读取 .day.bin 文件摘要信息"""
        with open(fp, "rb") as f:
            raw = f.read()
        if len(raw) < 4:
            return {"start_index": None, "count": 0, "valid": 0, "min": None, "max": None}
        start_index = int(struct.unpack("<f", raw[:4])[0])
        n_floats = len(raw) // 4
        data = np.frombuffer(raw, dtype="<f4")
        valid = data[1:][~np.isnan(data[1:])]
        return {
            "start_index": start_index,
            "count": n_floats - 1,
            "valid": len(valid),
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
        }

    sample_dirs = existing_dirs[:5] + existing_dirs[::max(1, len(existing_dirs)//5)][:5]
    sample_dirs = sorted(set(sample_dirs))

    print(f"\n抽样 {len(sample_dirs)} 只股票验证：")
    print(f"{'股票':<16} {'字段':<8} {'起始索引':>10} {'数据点':>8} {'有效值':>8} {'最小值':>12} {'最大值':>12}")
    print("-" * 80)

    for code in sample_dirs:
        for field in ["close", "open"]:
            fp = features_dir / code / f"{field}.day.bin"
            if not fp.exists():
                print(f"{code:<16} {field:<8} {'不存在':>10}")
                continue
            info = _read_bin_summary(fp)
            if info["start_index"] is not None:
                start_date = cal_list[info["start_index"]] if info["start_index"] < len(cal_list) else "?"
            else:
                start_date = "?"
            print(
                f"{code:<16} {field:<8} "
                f"{info['start_index']:>6}({start_date}) "
                f"{info['count']:>8} {info['valid']:>8} "
                f"{str(info['min'])[:12]:>12} {str(info['max'])[:12]:>12}"
            )

    # 检查 000638.sz（之前唯一有完整数据的股票）的覆盖情况
    target = "000638.sz"
    if (features_dir / target).exists():
        print(f"\n之前唯一有完整数据的 {target} 验证：")
        for field in ["close", "open"]:
            fp = features_dir / target / f"{field}.day.bin"
            if fp.exists():
                info = _read_bin_summary(fp)
                if info["start_index"] is not None:
                    start_date = cal_list[info["start_index"]] if info["start_index"] < len(cal_list) else "?"
                    end_idx = info["start_index"] + info["count"] - 1
                    end_date = cal_list[end_idx] if end_idx < len(cal_list) else "?"
                    print(f"  {field}: {info['valid']} 有效值, {start_date} ~ {end_date}")

    print("\n完成！close/open 数据已补全。")


if __name__ == "__main__":
    main()
