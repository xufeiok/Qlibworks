import os
import yaml
import shutil

repo_path = r'e:\Quant\Qlibworks\factors_repo'
old_files = [
    'alpha158_factor_dictionary.yaml',
    'gtja191_factor_dictionary.yaml',
    'master_factor_dictionary.yaml',
    'weekly_reversal_v1.yaml'
]

# 类别映射规则
def get_main_category(old_cat, factor_name):
    old_cat = str(old_cat).lower()
    
    if any(k in old_cat for k in ['估值', '市值', '红利', '价值', '流动性类', 'style', '风格']):
        return 'style_factors'
    elif any(k in old_cat for k in ['盈利', '成长', '质量', 'roe', '毛利率', '现金流', '营收', '负债', 'quality']):
        return 'quality_factors'
    elif any(k in old_cat for k in ['量价', '价量', 'k线', '动量', '反转', '技术面', '换手', '流动性', 'price_volume']):
        return 'price_volume_factors'
    elif any(k in old_cat for k in ['情绪', '预期', '资金', '融资', 'sentiment']):
        return 'sentiment_factors'
    elif any(k in old_cat for k in ['风险', '波动', '解禁', '质押', 'beta', 'risk']):
        return 'risk_factors'
    
    # 默认回退逻辑，按因子名称猜测
    name_lower = factor_name.lower()
    if 'alpha' in name_lower or 'kmid' in name_lower or 'klen' in name_lower or 'reversal' in name_lower or 'momentum' in name_lower:
        return 'price_volume_factors'
    
    return 'other_factors'

def load_all_old_factors():
    all_factors = {}
    archive_dir = os.path.join(repo_path, 'archive')
    for f in old_files:
        fpath = os.path.join(archive_dir, f)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                if data and 'factors' in data:
                    for factor in data['factors']:
                        # 确保有名称
                        name = factor.get('name')
                        if not name:
                            continue
                        # 归类
                        main_cat = get_main_category(factor.get('category', ''), name)
                        
                        # 特殊处理：gtja191 和 alpha158 都是量价
                        if 'gtja191' in f or 'alpha158' in f:
                            main_cat = 'price_volume_factors'
                            
                        if main_cat not in all_factors:
                            all_factors[main_cat] = {}
                        
                        # 如果已存在同名，选择保留较丰满的一个（此处简单覆盖或保留）
                        if name not in all_factors[main_cat]:
                            all_factors[main_cat][name] = factor
    return all_factors

# 豆包新增因子补充
doubao_factors = [
    # 风格因子
    {"name": "total_mv_log", "category": "风格", "expression": {"qlib": "Log($total_mv)", "duckdb": "ln(total_mv)"}, "meaning": "总市值对数（负向）", "usage_scenario": "小市值策略", "strategy_hint": "控制暴露不直接参与打分"},
    # 质量因子
    {"name": "ocf_to_netprofit", "category": "质量", "expression": {"qlib": "$ocf / $net_profit", "duckdb": "ocf / net_profit"}, "meaning": "盈利现金流含量", "usage_scenario": "排雷、提升夏普", "strategy_hint": "正向"},
    {"name": "q_profit_yoy", "category": "质量", "expression": {"qlib": "$q_profit_yoy", "duckdb": "q_profit_yoy"}, "meaning": "扣非净利润同比", "usage_scenario": "排雷", "strategy_hint": "正向"},
    {"name": "debt_to_assets", "category": "质量", "expression": {"qlib": "$debt_to_assets", "duckdb": "debt_to_assets"}, "meaning": "资产负债率", "usage_scenario": "排雷", "strategy_hint": "负向"},
    {"name": "current_ratio", "category": "质量", "expression": {"qlib": "$current_ratio", "duckdb": "current_ratio"}, "meaning": "流动比率", "usage_scenario": "排雷", "strategy_hint": "正向"},
    # 价量因子
    {"name": "obv", "category": "价量", "expression": {"qlib": "Sum(If($close > Ref($close,1), $volume, If($close < Ref($close,1), -$volume, 0)), 20)", "duckdb": "sum(case when close>lag(close) then vol when close<lag(close) then -vol else 0 end)"}, "meaning": "OBV 能量潮", "usage_scenario": "抓资金", "strategy_hint": "正向"},
    # 情绪因子
    {"name": "eps_forecast_yoy", "category": "情绪", "expression": {"qlib": "($eps_forecast / Ref($eps_last, 240)) - 1", "duckdb": "(eps_forecast / eps_last) - 1"}, "meaning": "预期净利润增速", "usage_scenario": "抓预期差、拐点", "strategy_hint": "正向"},
    {"name": "north_fund_change", "category": "情绪", "expression": {"qlib": "$north_hold - Ref($north_hold, 20)", "duckdb": "north_hold - lag(north_hold, 20)"}, "meaning": "北向资金持股变动", "usage_scenario": "抓外资拐点", "strategy_hint": "正向"},
    {"name": "margin_balance_change", "category": "情绪", "expression": {"qlib": "$rzye - Ref($rzye, 5)", "duckdb": "rzye - lag(rzye, 5)"}, "meaning": "融资余额变化", "usage_scenario": "抓杠杆资金", "strategy_hint": "正向"},
    # 风险因子
    {"name": "beta", "category": "风险", "expression": {"qlib": "$beta", "duckdb": "beta"}, "meaning": "Beta系数", "usage_scenario": "风控、过滤庄股", "strategy_hint": "视策略而定"},
    {"name": "shareholder_change", "category": "风险", "expression": {"qlib": "$stk_holdernumber / Ref($stk_holdernumber, 20) - 1", "duckdb": "stk_holdernumber / lag(stk_holdernumber, 20) - 1"}, "meaning": "股东人数变化", "usage_scenario": "筹码集中度分析", "strategy_hint": "负向（减少代表集中）"},
    {"name": "unlock_pressure", "category": "风险", "expression": {"qlib": "$share_float_30d / $total_mv", "duckdb": "share_float_30d / total_mv"}, "meaning": "解禁压力", "usage_scenario": "防暴雷", "strategy_hint": "负向"},
    {"name": "pledge_ratio", "category": "风险", "expression": {"qlib": "$pledge_ratio", "duckdb": "pledge_ratio"}, "meaning": "质押比例", "usage_scenario": "防暴雷", "strategy_hint": "负向"}
]

def merge_doubao_factors(all_factors):
    for dfactor in doubao_factors:
        cat = get_main_category(dfactor['category'], dfactor['name'])
        if cat not in all_factors:
            all_factors[cat] = {}
        # 避免覆盖已经存在的同名因子
        if dfactor['name'] not in all_factors[cat]:
            all_factors[cat][dfactor['name']] = dfactor

# 头部信息模板
headers = {
    'style_factors': {'name': 'style_factors', 'description': '风格因子库：包含市值、价值、红利等定义策略基础风格的因子。'},
    'quality_factors': {'name': 'quality_factors', 'description': '质量因子库：包含ROE、毛利率、现金流等用于排雷和提升夏普的财务因子。'},
    'price_volume_factors': {'name': 'price_volume_factors', 'description': '价量因子库：包含动量、反转、流动性以及Alpha158/GTJA191等量价特征。'},
    'sentiment_factors': {'name': 'sentiment_factors', 'description': '情绪因子库：包含一致预期、北向资金、融资余额等反应市场情绪的因子。'},
    'risk_factors': {'name': 'risk_factors', 'description': '风险因子库：包含波动率、Beta、解禁压力等用于风控和止损的因子。'},
    'other_factors': {'name': 'other_factors', 'description': '其他因子库：未能明确归类的补充因子。'}
}

class CustomDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter, Dumper=CustomDumper)

def write_new_files(all_factors):
    for cat_key, factors_dict in all_factors.items():
        if not factors_dict:
            continue
            
        factors_list = list(factors_dict.values())
        
        doc = {
            'name': headers[cat_key]['name'],
            'version': '2.0',
            'strategy_type': f"{headers[cat_key]['name']} 因子合集",
            'author': 'Quant',
            'updated_at': '2026-04-22',
            'description': headers[cat_key]['description'],
            'analysis': {
                'pros': '分类清晰，完全符合系统化投研体系。',
                'cons': '部分另类因子需要底层数据支持。',
                'optimization_direction': '持续迭代新的因子表达式。'
            },
            'factors': factors_list
        }
        
        out_path = os.path.join(repo_path, f"{cat_key}.yaml")
        with open(out_path, 'w', encoding='utf-8') as f:
            yaml.dump(doc, f, Dumper=CustomDumper, allow_unicode=True, sort_keys=False, width=1000)

def backup_and_clean_old():
    archive_dir = os.path.join(repo_path, 'archive')
    os.makedirs(archive_dir, exist_ok=True)
    for f in old_files:
        fpath = os.path.join(repo_path, f)
        if os.path.exists(fpath):
            shutil.move(fpath, os.path.join(archive_dir, f))

if __name__ == '__main__':
    print("开始整合因子库...")
    all_factors = load_all_old_factors()
    merge_doubao_factors(all_factors)
    write_new_files(all_factors)
    backup_and_clean_old()
    print("整合完成！")
