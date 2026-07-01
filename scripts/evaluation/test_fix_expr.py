"""快速测试 fix_expr 和计算一个因子"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"e:\Quant\Qlibworks\src")

import yaml
from pathlib import Path

yp = Path(r"e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml")
with open(yp, encoding="utf-8") as f:
    data = yaml.safe_load(f)

wdir = Path(r"e:\Quant\Qlibworks\factor_data\warehouse")
names = [fd["name"] for fd in data["factors"] if fd.get("name")]
completed = [n for n in names if list((wdir/n).glob("*.parquet"))]
missing = [n for n in names if not list((wdir/n).glob("*.parquet"))]
print(f"gtja191 总计: {len(names)}, 已完成: {len(completed)}, 缺失: {len(missing)}")
print()

# 测试 fix_expr
import re

def _extract_func_args(expr, func_name):
    pattern = re.compile(rf'\b{func_name}\(')
    for m in pattern.finditer(expr):
        start = m.start()
        depth = 1
        i = m.end()
        args = []
        cur = i
        while depth > 0 and i < len(expr):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    args.append(expr[cur:i].strip())
            elif expr[i] == ',' and depth == 1:
                args.append(expr[cur:i].strip())
                cur = i + 1
            i += 1
        if depth == 0:
            yield (start, i, args)

def fix_expr(expr):
    e = expr
    for start, end, args in reversed(list(_extract_func_args(e, 'Rank'))):
        if len(args) == 1:
            e = e[:start] + f"Rank({args[0]},1)" + e[end:]
    for fn, repl in [('Ts_Max', 'Max'), ('Ts_Min', 'Min'), ('Ts_Sum', 'Sum')]:
        for start, end, args in reversed(list(_extract_func_args(e, fn))):
            e = e[:start] + f"{repl}({','.join(args)})" + e[end:]
    return e

# 测试 Alpha1
expr1 = "(-1 * Corr(Rank(Delta(Log($volume),1)),Rank((($close-$open)/$open)),6))"
print(f"原始:   {expr1}")
print(f"修复后: {fix_expr(expr1)}")
print()

# 测试 Alpha5
expr5 = "(-1*Ts_Max(Corr(Ts_Rank($volume,5),Ts_Rank($high,5),5),3))"
print(f"原始:   {expr5}")
print(f"修复后: {fix_expr(expr5)}")
print()

# 测试一个有嵌套 Rank 的
expr32 = "(-1*Sum(Rank(Corr(Rank($high),Rank($volume),3)),3))"
print(f"原始:   {expr32}")
print(f"修复后: {fix_expr(expr32)}")
print()

# 显示前 10 个缺失因子
print("=== 前10个缺失因子 ===")
for fd in data["factors"]:
    n = fd.get("name", "")
    if n in missing:
        expr = fd.get("expression", {}).get("qlib", "")
        print(f"  {n}: {fix_expr(str(expr))[:80]}...")
        if len([x for x in missing[:10] if True]) >= 10:
            break
