"""质量闸门效果验证"""
import csv, json
from collections import defaultdict

score_path = "e:/Quant/Qlibworks/scripts/training/score_tree_selected.csv"
rows = list(csv.DictReader(open(score_path)))
date_rows = defaultdict(list)
for r in rows:
    date_rows[r["datetime"]].append(r)
dates = sorted(date_rows.keys())
anomaly_included = "2024-10-10" in date_rows
year_stats = {}
for d in dates:
    y = d[:4]
    year_stats.setdefault(y, {"dates": set(), "rows": 0})
    year_stats[y]["dates"].add(d)
    year_stats[y]["rows"] += len(date_rows[d])
result = {
    "total_rows": len(rows), "date_count": len(dates),
    "date_range": {"first": dates[0], "last": dates[-1]},
    "anomaly_date_20241010_included": anomaly_included,
    "year_stats": {y: {"dates": sorted(s["dates"]), "count": len(s["dates"]), "rows": s["rows"]} for y, s in sorted(year_stats.items())},
}
open("e:/Quant/Qlibworks/runtime/diagnostics/training_verify.json","w").write(json.dumps(result, ensure_ascii=False, indent=2))
print("OK:", json.dumps(result, ensure_ascii=False))
