import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd
import sys
import os

# 将 src 目录添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qlworks.processors.neutralize import CSNeutralize
from qlworks.factors.manager import FactorLibraryManager

if __name__ == "__main__":
    # 初始化 Qlib
    qlib.init(provider_uri='e:/Quant/Qlibworks/qlib_data')
    
    print("=" * 60)
    print("🚀 开始行业/市值中性化测试演示")
    print("=" * 60)
    
    # 1. 使用 FactorLibraryManager 加载因子
    manager = FactorLibraryManager(repo_path='e:/Quant/Qlibworks/factors_repo')
    strategy_name = 'alpha158_factor_dictionary'
    
    print(f"📖 正在从 {strategy_name}.yaml 加载因子...")
    fields, names = manager.get_qlib_expressions(strategy_name)
    
    # 选取 Alpha158 中的基于量价的前 3 个因子 (例如 KMAX, KMIN, KMID 等)
    demo_fields = fields[:3]
    demo_names = names[:3]
    print(f"🧪 选取的测试因子: {demo_names}")
    
    # 2. 配置 DataHandlerLP，并将 CSNeutralize 加入 infer_processors
    data_handler_config = {
        "start_time": "2026-01-01",
        "end_time": "2026-02-01",
        "fit_start_time": "2026-01-01",
        "fit_end_time": "2026-02-01",
        "instruments": "csi300", # 使用沪深300作为股票池
        "infer_processors": [
            # 首先可以加入去极值、标准化处理 (可选)
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True, "fit_start_time": "2026-01-01", "fit_end_time": "2026-02-01"}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            # 然后加入我们自定义的行业/市值中性化处理器
            {"class": "CSNeutralize", "module_path": "qlworks.processors.neutralize", "kwargs": {"fields_group": "feature", "industry_field": "industry_code", "market_cap_field": "circ_mv"}}
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
        ],
    }
    
    print("⏳ 正在初始化 DatasetH 并加载数据 (包含处理管道)...")
    try:
        # 使用 DatasetH
        dataset = DatasetH(
            handler={
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": {
                    "instruments": data_handler_config["instruments"],
                    "start_time": data_handler_config["start_time"],
                    "end_time": data_handler_config["end_time"],
                    "infer_processors": data_handler_config["infer_processors"],
                    "learn_processors": data_handler_config["learn_processors"],
                    "data_loader": {
                        "class": "QlibDataLoader",
                        "kwargs": {
                            "config": {
                                "feature": (demo_fields, demo_names),
                                "label": (["Ref($close, -2) / Ref($close, -1) - 1"], ["RETURN"])
                            },
                            "freq": "day",
                        },
                    },
                },
            },
            segments={
                "train": ("2026-01-01", "2026-01-15"),
                "valid": ("2026-01-16", "2026-02-01"),
            },
        )
        
        # 3. 准备数据并查看结果
        df = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_I)
        
        print("\n✅ 中性化处理后的 DataFrame 头部数据:")
        print(df.head())
        print("\n✅ 中性化处理后的 DataFrame 描述性统计:")
        print(df.describe())
        
    except Exception as e:
        print("❌ 演示运行中出现错误:", e)
        import traceback
        traceback.print_exc()

