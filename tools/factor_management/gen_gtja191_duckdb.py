import re
import yaml

def qlib_to_duckdb(expr):
    if not expr or expr.startswith("TODO:"):
        return ""
    
    e = expr
    
    # 替换变量
    e = e.replace("$close", "close").replace("$open", "open").replace("$high", "high").replace("$low", "low")
    e = e.replace("$volume", "volume").replace("$vwap", "vwap").replace("$amount", "amount")
    
    # 替换运算符
    e = e.replace("&&", " AND ").replace("||", " OR ")
    e = e.replace("&", " AND ").replace("|", " OR ")
    
    # 替换基本函数
    e = re.sub(r'(?i)\bLog\(', 'LN(', e)
    e = re.sub(r'(?i)\bAbs\(', 'ABS(', e)
    e = re.sub(r'(?i)\bSign\(', 'SIGN(', e)
    
    # 替换横截面 Rank
    # 匹配 Rank(...)，由于可能存在嵌套，这里用简单的括号匹配（最多处理2层嵌套）
    # 或者用正则表达式：匹配 Rank( 后面的内容直到匹配的 )
    # 为了简化，我们只处理常见的 Rank(...) 形式
    def replace_rank(m):
        inner = m.group(1)
        return f"PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY {inner})"
    
    # 粗略替换 Rank(XXX) (注意处理嵌套)
    # 因为正则很难处理无限嵌套，这里我们用一个简单的括号匹配函数
    def parse_nested(text, func_name, replacement_formatter):
        result = ""
        idx = 0
        while True:
            # 使用正则查找全词匹配的函数名，避免 Rank 匹配到 Ts_Rank
            match = re.search(rf"(?<![a-zA-Z_]){func_name}\(", text[idx:])
            if not match:
                result += text[idx:]
                break
            start = idx + match.start()
            result += text[idx:start]
            
            paren_count = 0
            inner_start = start + len(func_name) + 1
            inner_end = -1
            for i in range(inner_start, len(text)):
                if text[i] == '(':
                    paren_count += 1
                elif text[i] == ')':
                    if paren_count == 0:
                        inner_end = i
                        break
                    paren_count -= 1
            if inner_end == -1: # Unmatched parenthesis fallback
                inner_end = len(text)
                
            inner = text[inner_start:inner_end]
            # 递归处理内部
            inner = parse_nested(inner, func_name, replacement_formatter)
            result += replacement_formatter(inner)
            idx = inner_end + 1
        return result

    # 替换 If(cond, true_val, false_val) -> CASE WHEN cond THEN true_val ELSE false_val END
    def replace_if(inner):
        parts = split_args(inner)
        if len(parts) >= 3:
            return f"(CASE WHEN {parts[0]} THEN {parts[1]} ELSE {parts[2]} END)"
        return f"If({inner})"
    e = parse_nested(e, "If", replace_if)

    # 替换 Rank
    def replace_rank_func(inner):
        return f"PERCENT_RANK() OVER (PARTITION BY trade_date ORDER BY {inner})"
    e = parse_nested(e, "Rank", replace_rank_func)

    # 替换 Ts_Rank(A, d) -> 由于 DuckDB 没有内置时间序列的 PERCENT_RANK，我们近似使用 A 的排序
    # 这里比较复杂，我们暂且把 Ts_Rank 视为一个近似的窗口排序（需要子查询），所以DuckDB中很难完美实现单行 Ts_Rank
    # 但我们可以用一个窗口函数来近似：(A - MIN(A) OVER w) / (MAX(A) OVER w - MIN(A) OVER w)
    def replace_ts_rank(inner):
        parts = split_args(inner)
        if len(parts) == 2:
            A, d = parts
            w = f"OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {int(d)-1} PRECEDING AND CURRENT ROW)"
            return f"(({A} - MIN({A}) {w}) / NULLIF(MAX({A}) {w} - MIN({A}) {w}, 0))"
        return f"Ts_Rank({inner})"
    e = parse_nested(e, "Ts_Rank", replace_ts_rank)

    # 替换 Corr(A, B, d)
    def replace_corr(inner):
        parts = split_args(inner)
        if len(parts) == 3:
            A, B, d = parts
            return f"CORR({A}, {B}) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {int(d)-1} PRECEDING AND CURRENT ROW)"
        return f"Corr({inner})"
    e = parse_nested(e, "Corr", replace_corr)

    # 替换 Cov(A, B, d)
    def replace_cov(inner):
        parts = split_args(inner)
        if len(parts) == 3:
            A, B, d = parts
            return f"COVAR_SAMP({A}, {B}) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {int(d)-1} PRECEDING AND CURRENT ROW)"
        return f"Cov({inner})"
    e = parse_nested(e, "Cov", replace_cov)

    # 替换 Ref(A, d)
    def replace_ref(inner):
        parts = split_args(inner)
        if len(parts) == 2:
            A, d = parts
            return f"LAG({A}, {d}) OVER (PARTITION BY ts_code ORDER BY trade_date)"
        return f"Ref({inner})"
    e = parse_nested(e, "Ref", replace_ref)

    # 替换 Delta(A, d)
    def replace_delta(inner):
        parts = split_args(inner)
        if len(parts) == 2:
            A, d = parts
            return f"({A} - LAG({A}, {d}) OVER (PARTITION BY ts_code ORDER BY trade_date))"
        return f"Delta({inner})"
    e = parse_nested(e, "Delta", replace_delta)

    # 替换 Ts_Max, Ts_Min, Mean, Sum, Std, Sma
    def replace_window(inner, func_name):
        parts = split_args(inner)
        if len(parts) >= 2:
            A, d = parts[0], parts[1]
            return f"{func_name}({A}) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN {int(d)-1} PRECEDING AND CURRENT ROW)"
        return f"{func_name}({inner})"
        
    e = parse_nested(e, "Ts_Max", lambda x: replace_window(x, "MAX"))
    e = parse_nested(e, "Ts_Min", lambda x: replace_window(x, "MIN"))
    e = parse_nested(e, "Mean", lambda x: replace_window(x, "AVG"))
    e = parse_nested(e, "Sum", lambda x: replace_window(x, "SUM"))
    e = parse_nested(e, "Std", lambda x: replace_window(x, "STDDEV_SAMP"))
    e = parse_nested(e, "Sma", lambda x: replace_window(x, "AVG")) # 近似

    # 替换二元操作 Max(A, B) -> GREATEST(A, B), Min(A, B) -> LEAST(A, B)
    def replace_binary(inner, func_name):
        parts = split_args(inner)
        if len(parts) == 2:
            return f"{func_name}({parts[0]}, {parts[1]})"
        return f"{func_name}({inner})"
    
    e = parse_nested(e, "Max", lambda x: replace_binary(x, "GREATEST"))
    e = parse_nested(e, "Min", lambda x: replace_binary(x, "LEAST"))
    e = parse_nested(e, "Greater", lambda x: replace_binary(x, "GREATEST"))
    e = parse_nested(e, "Less", lambda x: replace_binary(x, "LEAST"))

    return e

def split_args(inner):
    # 如果整个 inner 被一对多余的括号包裹，例如 (A, B)，脱掉外层括号
    inner = inner.strip()
    while inner.startswith('(') and inner.endswith(')'):
        # 检查是否这对外层括号包住了整个字符串
        paren_count = 0
        is_enclosing = True
        for i, c in enumerate(inner):
            if c == '(': paren_count += 1
            elif c == ')': paren_count -= 1
            if paren_count == 0 and i < len(inner) - 1:
                is_enclosing = False
                break
        if is_enclosing:
            inner = inner[1:-1].strip()
        else:
            break

    parts = []
    paren_count = 0
    start = 0
    for i, c in enumerate(inner):
        if c == '(':
            paren_count += 1
        elif c == ')':
            paren_count -= 1
        elif c == ',' and paren_count == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    parts.append(inner[start:].strip())
    return parts

yaml_path = r'e:\Quant\Qlibworks\factor_data\factor_library\archive\gtja191_factor_dictionary.yaml'

with open(yaml_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

for factor in data.get('factors', []):
    qlib_expr = factor.get('expression', {}).get('qlib', '')
    if qlib_expr and "TODO:" not in qlib_expr:
        duckdb_expr = qlib_to_duckdb(qlib_expr)
        # Add WITH clause to make it a valid CTE if it's very complex, but for batch_compute, 
        # inline expression is preferred unless it starts with WITH.
        factor['expression']['duckdb'] = duckdb_expr

class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)
        
def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)

with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False, Dumper=MyDumper, default_flow_style=False)

print("DuckDB expressions generated successfully.")
