import os
import yaml
from tabulate import tabulate

class FactorLibraryManager:
    """
    因子库管理系统
    负责加载、解析和输出因子配置，并将其转换为 Qlib 能够识别的格式
    """
    
    def __init__(self, repo_path: str = None):
        """
        初始化因子库管理器
        :param repo_path: 因子 YAML 配置文件的存放目录
        """
        # 默认路径设为 factors_repo
        if repo_path is None:
            # 假设该脚本在 src/qlworks/factors/manager.py
            # repo 路径在 ../../../factors_repo
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            self.repo_path = os.path.join(base_dir, 'factors_repo')
        else:
            self.repo_path = repo_path
            
        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path, exist_ok=True)
            
    def list_strategies(self):
        """
        列出当前库中所有的策略/因子组合配置
        """
        strategies = []
        for file in os.listdir(self.repo_path):
            if file.endswith('.yaml') or file.endswith('.yml'):
                strategies.append(file.split('.')[0])
        return strategies

    def load_strategy_config(self, strategy_name: str) -> dict:
        """
        加载指定策略的因子配置
        :param strategy_name: 策略名 (yaml 文件名，不含后缀)
        :return: 配置字典
        """
        file_path = os.path.join(self.repo_path, f"{strategy_name}.yaml")
        if not os.path.exists(file_path):
            file_path = os.path.join(self.repo_path, f"{strategy_name}.yml")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到策略配置文件: {strategy_name}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config

    def get_qlib_expressions(self, strategy_name: str) -> tuple:
        """
        提取用于 Qlib DatasetH 初始化的特征表达式和对应的特征名
        :param strategy_name: 策略名
        :return: (fields, names) 列表
        """
        config = self.load_strategy_config(strategy_name)
        fields = []
        names = []
        
        for factor in config.get('factors', []):
            expr = factor.get('expression', '')
            if isinstance(expr, dict):
                qlib_expr = expr.get('qlib', '')
            else:
                qlib_expr = str(expr)
                
            fields.append(qlib_expr)
            names.append(factor.get('name'))
            
        return fields, names

    def get_duckdb_expressions(self, strategy_name: str) -> tuple:
        """
        提取用于 DuckDB 数据库的特征表达式和对应的特征名
        :param strategy_name: 策略名
        :return: (fields, names) 列表
        """
        config = self.load_strategy_config(strategy_name)
        fields = []
        names = []
        
        for factor in config.get('factors', []):
            expr = factor.get('expression', '')
            if isinstance(expr, dict):
                duckdb_expr = expr.get('duckdb', '')
            else:
                duckdb_expr = str(expr)
                
            fields.append(duckdb_expr)
            names.append(factor.get('name'))
            
        return fields, names

    def print_strategy_report(self, strategy_name: str):
        """
        在终端以漂亮的格式打印策略因子的分析报告
        :param strategy_name: 策略名
        """
        config = self.load_strategy_config(strategy_name)
        
        print("="*60)
        print(f"📖 因子组合报告: {config.get('name', strategy_name)} (v{config.get('version', '1.0')})")
        print(f"👤 作者: {config.get('author', 'Unknown')} | 📅 更新时间: {config.get('updated_at', 'Unknown')}")
        print(f"🏷 策略类型: {config.get('strategy_type', 'Unknown')}")
        print("="*60)
        
        print("\n📝 【组合描述】")
        print(config.get('description', '无描述').strip())
        
        print("\n🔍 【组合优劣势分析】")
        analysis = config.get('analysis', {})
        print(f"✅ 优势 (Pros): {analysis.get('pros', '无')}")
        print(f"❌ 劣势 (Cons): {analysis.get('cons', '无')}")
        print(f"🚀 优化方向: {analysis.get('optimization_direction', '无')}")
        
        print("\n📊 【因子库详情】")
        factors = config.get('factors', [])
        
        table_data = []
        for f in factors:
            expr = f.get('expression', '')
            if isinstance(expr, dict):
                qlib_expr = expr.get('qlib', '')
                duckdb_expr = expr.get('duckdb', '')
                expr_display = f"Qlib: {qlib_expr}\nDuckDB: {duckdb_expr}"
            else:
                expr_display = str(expr)
                
            table_data.append([
                f.get('name', ''), 
                f.get('category', ''), 
                expr_display, 
                f.get('meaning', ''),
                f.get('usage_scenario', ''),
                f.get('strategy_hint', '')
            ])
            
        print(tabulate(table_data, headers=['因子名称', '类别', '表达式 (Qlib/DuckDB)', '含义', '使用场景', '策略提示'], tablefmt='grid'))
        print("="*60 + "\n")
