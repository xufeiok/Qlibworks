"""从 factors_repo 同步因子评测。"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from qlworks.evaluation import FactorEvaluator, EvalConfig
from qlworks.factors import FactorLibraryManager

CAT_MAP = {"price_volume_factors": "satellite", "quality_factors": "core", "style_factors": "core", "risk_factors": "core", "sentiment_factors": "satellite", "other_factors": "satellite"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor"); parser.add_argument("--category"); parser.add_argument("--all", action="store_true")
    parser.add_argument("--pool", default="csi500"); parser.add_argument("--start", default="2018-01-01"); parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not (args.factor or args.category or args.all): parser.print_help(); return

    m = FactorLibraryManager()
    factors = []
    if args.factor:
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            cfg = m.load_strategy_config(s)
            for fd in cfg.get("factors", []):
                if fd.get("name") == args.factor:
                    expr = fd.get("expression",""); factors.append({"name":args.factor,"expr":str(expr.get("qlib",expr) if isinstance(expr,dict) else expr),"source":s})
    elif args.category:
        cfg = m.load_strategy_config(args.category)
        for fd in cfg.get("factors", []):
            name=fd.get("name"); expr=fd.get("expression",""); factors.append({"name":name,"expr":str(expr.get("qlib",expr) if isinstance(expr,dict) else expr),"source":args.category})
    elif args.all:
        for s in [s for s in m.list_strategies() if "dictionary" not in s]:
            cfg = m.load_strategy_config(s)
            for fd in cfg.get("factors",[]):
                name=fd.get("name"); expr=fd.get("expression",""); factors.append({"name":name,"expr":str(expr.get("qlib",expr) if isinstance(expr,dict) else expr),"source":s})
    if args.limit>0: factors=factors[:args.limit]

    print(f"同步 {len(factors)} 个因子"); config=EvalConfig(instruments=args.pool,start_time=args.start,end_time=args.end)
    ev = FactorEvaluator(config)
    for i,f in enumerate(factors,1):
        print(f"[{i}] {f['name']}")
        try:
            df=ev.load_data(f["expr"],f["name"],start_time=args.start,end_time=args.end)
            layer=CAT_MAP.get(f.get("source",""),"satellite")
            r=ev.evaluate(f["name"],df,category=layer); q=r["qual_result"]
            print(f"  tier={q['tier']} IC={r['ic_stats']['ic_mean']:.4f} ICIR={r['ic_stats']['icir']:.2f} 评分={q['composite_score']:.1f}")
        except Exception as e: print(f"  X {e}")

if __name__=="__main__": main()