import yaml
import os

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)

target_path = r'e:\Quant\Qlibworks\factors_repo\gtja191_factor_dictionary.yaml'

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

known_alphas = {
    1: {
        'qlib': '(-1 * Corr(Rank(Delta(Log($volume), 1)), Rank((($close - $open) / $open)), 6))',
        'meaning': '计算成交量变化率的排名与日内收益率排名的相关系数，反映量价背离程度。',
        'category': '量价相关性 (Volume-Price Correlation)',
        'usage_scenario': '短线反转，寻找量价背离时的反转点。'
    },
    2: {
        'qlib': '(-1 * Delta((((($close - $low) - ($high - $close)) / ($high - $low + 1e-12)) * $volume), 1))',
        'meaning': '买卖意愿强度的变化率。计算日内多空力量对比乘以成交量后的单日变化。',
        'category': '动量/意愿 (Momentum/Willingness)',
        'usage_scenario': '衡量买卖盘力量的边际变化，适合动量跟踪。'
    },
    3: {
        'qlib': 'Sum(If($close == Ref($close, 1), 0, $close - If($close > Ref($close, 1), Min($low, Ref($close, 1)), Max($high, Ref($close, 1)))), 6)',
        'meaning': '过去6天的真实波动净值求和，衡量短期的买卖压力累积。',
        'category': '压力/支撑 (Pressure/Support)',
        'usage_scenario': '短期动量与超买超卖识别。'
    },
    4: {
        'qlib': 'If((((Sum($close, 8) / 8) + Std($close, 8)) < (Sum($close, 2) / 2)), (-1 * 1), If(((Sum($close, 2) / 2) < ((Sum($close, 8) / 8) - Std($close, 8))), 1, If((1 < ($volume / Mean($volume, 20))), 1, (-1 * 1))))',
        'meaning': '结合均线系统和布林带的突破判断，辅助以成交量放量条件进行综合信号输出。',
        'category': '趋势突破 (Trend Breakout)',
        'usage_scenario': '短线趋势跟踪与放量突破确认。'
    },
    5: {
        'qlib': '(-1 * Ts_Max(Corr(Ts_Rank($volume, 5), Ts_Rank($high, 5), 5), 3))',
        'meaning': '最高价排名与成交量排名的相关系数的滚动最大值的反转，衡量量价齐升后动能衰竭的概率。',
        'category': '量价相关性 (Volume-Price Correlation)',
        'usage_scenario': '高位量价背离的反转信号捕获。'
    }
}

for i in range(1, 192):
    name = f'Alpha{i}'
    
    if i in known_alphas:
        expr = known_alphas[i]['qlib']
        meaning = known_alphas[i]['meaning']
        category = known_alphas[i]['category']
        scenario = known_alphas[i]['usage_scenario']
    else:
        expr = f'TODO: 补充国泰君安191 Alpha{i} 的具体 Qlib 表达式'
        meaning = f'国泰君安191因子体系中的 Alpha{i} 因子。具体业务含义根据原始研报公式解析。'
        category = '量价综合 (Price-Volume Synthetic)'
        scenario = '短周期多空博弈、量价背离与动量反转。'

    factor_obj = {
        'name': name,
        'category': category,
        'expression': {
            'qlib': expr,
            'duckdb': 'DuckDB暂未实现完整的GTJA191对应逻辑'
        },
        'meaning': f'【核心含义】\n{meaning}\n\n【运行原理】\n基于国泰君安191篇研报标准公式，提取量价、均线、相关性等短周期特征进行组合运算。\n反映该特定量价组合下的多空失衡状态。',
        'usage_scenario': scenario,
        'strategy_hint': '高频与中频量价特征。树模型可以直接捕捉其复杂的非线性阈值，建议作为 LightGBM / CatBoost 等模型的特征集。'
    }
    data['factors'].append(factor_obj)

os.makedirs(os.path.dirname(target_path), exist_ok=True)

class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

with open(target_path, 'w', encoding='utf-8') as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False, Dumper=MyDumper, default_flow_style=False)

print(f'Generated {target_path} successfully with {len(data["factors"])} factors.')
