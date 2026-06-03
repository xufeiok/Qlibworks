import os
from pathlib import Path
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
        if repo_path is None:
            base_dir = Path(__file__).resolve().parents[3]  # Qlibworks/
            self.repo_path = str(base_dir / "factor_data" / "factor_library")
        else:
            self.repo_path = repo_path

        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path, exist_ok=True)

    def list_strategies(self):
        strategies = []
        for file in os.listdir(self.repo_path):
            full_path = os.path.join(self.repo_path, file)
            if os.path.isfile(full_path) and (file.endswith('.yaml') or file.endswith('.yml')):
                strategies.append(file.split('.')[0])
        return strategies

    def load_strategy_config(self, strategy_name: str) -> dict:
        file_path = os.path.join(self.repo_path, f"{strategy_name}.yaml")
        if not os.path.exists(file_path):
            file_path = os.path.join(self.repo_path, f"{strategy_name}.yml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到策略配置文件: {strategy_name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def get_expressions(self, strategy_names: str | list[str], lang: str = "qlib") -> tuple:
        """
        提取因子表达式和名称。

        :param strategy_names: 策略名或策略名列表
        :param lang: 表达式语言，'qlib' 或 'duckdb'
        :return: (fields, names) 元组
        """
        if isinstance(strategy_names, str):
            strategy_names = [strategy_names]

        all_fields = []
        all_names = []
        seen_names: set = set()

        for strategy_name in strategy_names:
            config = self.load_strategy_config(strategy_name)
            for factor in config.get('factors', []):
                name = factor.get('name')
                if name in seen_names:
                    print(f"[警告] 发现重名因子 '{name}' (位于 {strategy_name}.yaml 中)，已跳过合并。")
                    continue
                expr = factor.get('expression', '')
                if isinstance(expr, dict):
                    field = expr.get(lang, '')
                else:
                    field = str(expr)
                if not field:
                    continue
                all_fields.append(field)
                all_names.append(name)
                seen_names.add(name)

        return all_fields, all_names

    # 向后兼容
    def get_qlib_expressions(self, strategy_names):
        return self.get_expressions(strategy_names, lang="qlib")

    def get_duckdb_expressions(self, strategy_names):
        return self.get_expressions(strategy_names, lang="duckdb")

    def print_strategy_report(self, strategy_name: str):
        config = self.load_strategy_config(strategy_name)
        print("=" * 60)
        print(f"因子组合报告: {config.get('name', strategy_name)} (v{config.get('version', '1.0')})")
        print(f"作者: {config.get('author', 'Unknown')} | 更新时间: {config.get('updated_at', 'Unknown')}")
        print(f"策略类型: {config.get('strategy_type', 'Unknown')}")
        print("=" * 60)
        print("\n[组合描述]")
        print(config.get('description', '无描述').strip())
        print("\n[组合优劣势分析]")
        analysis = config.get('analysis', {})
        print(f"优势: {analysis.get('pros', '无')}")
        print(f"劣势: {analysis.get('cons', '无')}")
        print(f"优化方向: {analysis.get('optimization_direction', '无')}")
        print("\n[因子库详情]")
        factors = config.get('factors', [])
        table_data = []
        for f in factors:
            expr = f.get('expression', '')
            if isinstance(expr, dict):
                qe = expr.get('qlib', '')
                de = expr.get('duckdb', '')
                expr_display = f"Qlib: {qe}\nDuckDB: {de}"
            else:
                expr_display = str(expr)
            table_data.append([
                f.get('name', ''),
                f.get('category', ''),
                expr_display,
                f.get('meaning', ''),
                f.get('usage_scenario', ''),
                f.get('strategy_hint', ''),
            ])
        print(tabulate(
            table_data,
            headers=['因子名称', '类别', '表达式', '含义', '使用场景', '策略提示'],
            tablefmt='grid',
        ))
        print("=" * 60 + "\n")
