import json
import yaml
import os
import re

with open(r'e:\Quant\Qlibworks\overrides.json', 'r') as f:
    overrides = json.load(f)

json_path = r'e:\Quant\Qlibworks\extracted_gtja191.json'
target_path = r'e:\Quant\Qlibworks\factors_repo\gtja191_factor_dictionary.yaml'

with open(json_path, 'r', encoding='utf-8') as f:
    formulas = json.load(f)

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)

data = {
    'name': 'gtja191_factor_dictionary',
    'version': '1.0',
    'strategy_type': '国泰君安191 Alpha因子集',
    'author': 'Quant',
    'updated_at': '2026-04-04',
    'description': '国泰君安191个短周期量价特征因子，广泛应用于高频和中高频量化交易中。该因子集基于量价特征挖掘短周期的反转与动量效应。',
    'analysis': {
        'pros': '因子数量庞大且覆盖多种量价逻辑（动量、反转、量价相关性等），非常适合作为机器学习模型的输入池，挖掘非线性关系。',
        'cons': '由于全部为量价因子，同质化较高，且在不同市场环境下的衰减较快；部分公式极度复杂，计算开销较大。',
        'optimization_direction': '可结合 master_factor_dictionary 中的基本面因子进行正交化处理；利用特征选择算法（如Lasso、树模型特征重要性）剔除冗余因子。'
    },
    'factors': []
}

def translate_to_qlib(expr):
    # Variables
    expr = re.sub(r'\bCLOSE\b', '$close', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bOPEN\b', '$open', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bHIGH\b', '$high', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bLOW\b', '$low', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bVOLUME\b', '$volume', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bVWAP\b', '$vwap', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bAMOUNT\b', '$amount', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bRET\b', '($close/Ref($close,1)-1)', expr, flags=re.IGNORECASE)
    
    # Functions
    expr = re.sub(r'\bDELAY\b', 'Ref', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bCORR\b', 'Corr', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bCOVIANCE\b', 'Cov', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bDELTA\b', 'Delta', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bTSRANK\b', 'Ts_Rank', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bYSRANK\b', 'Ts_Rank', expr, flags=re.IGNORECASE) # Fix typo in formula 5
    expr = re.sub(r'\bRANK\b', 'Rank', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bTSMAX\b', 'Ts_Max', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bTSMIN\b', 'Ts_Min', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bSTD\b', 'Std', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bMEAN\b', 'Mean', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bSUM\b', 'Sum', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bSIGN\b', 'Sign', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bABS\b', 'Abs', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bMAX\b', 'Max', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bMIN\b', 'Min', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bCOUNT\b', 'Count', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bSMA\b', 'Sma', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bWMA\b', 'Wma', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bEMA\b', 'Ema', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bDECAYLINEAR\b', 'Decaylinear', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bLOG\b', 'Log', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bEXP\b', 'Exp', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bSQRT\b', 'Sqrt', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bPOW\b', 'Pow', expr, flags=re.IGNORECASE)
    
    return expr

for i in range(1, 192):
    name = f'Alpha{i}'
    raw_expr = formulas.get(str(i), '')
    
    if not raw_expr:
        qlib_expr = f'TODO: 未找到公式'
    else:
        qlib_expr = translate_to_qlib(raw_expr)
        
        # Manual fix for Alpha 3
        if i == 3:
            qlib_expr = 'Sum(If($close == Ref($close,1), 0, If($close > Ref($close,1), $close - Less($low, Ref($close,1)), $close - Greater($high, Ref($close,1)))), 6)'
        elif str(i) in overrides:
            qlib_expr = overrides[str(i)]
        elif '?' in qlib_expr:
            qlib_expr = f'TODO: 存在三元运算符，需手动转换为If()结构 | {qlib_expr}'
        
        # 修复 Alpha1 和 Alpha7 缺失的右括号
        if i in [1, 7] and qlib_expr.count('(') > qlib_expr.count(')'):
            qlib_expr += ')'

    
    meaning = f'【原始公式】\n{raw_expr}\n\n【核心含义】\n国泰君安191因子体系中的 Alpha{i} 因子。具体业务含义参考原始研报公式解析。'
    category = '量价综合 (Price-Volume Synthetic)'
    scenario = '短周期多空博弈、量价背离与动量反转。'
    hint = '高频与中频量价特征。建议作为 LightGBM / CatBoost 等树模型的特征集。'

    factor_obj = {
        'name': name,
        'category': category,
        'expression': {
            'qlib': qlib_expr,
            'duckdb': 'DuckDB暂未实现完整的GTJA191对应逻辑'
        },
        'meaning': meaning,
        'usage_scenario': scenario,
        'strategy_hint': hint
    }
    data['factors'].append(factor_obj)

class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

with open(target_path, 'w', encoding='utf-8') as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False, Dumper=MyDumper, default_flow_style=False)

print(f'Generated {target_path} successfully with {len(data["factors"])} factors.')
