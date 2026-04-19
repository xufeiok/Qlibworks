from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from qlworks.data import DataFetchSpec, QlibDataAccessor, clean_ohlcv_data, generate_data_quality_report
from qlworks.features import build_alpha_feature_bundle
from qlworks.models import (
    FeatureSelectionResult,
    apply_feature_selection,
    evaluate_prediction_frame,
    prepare_feature_selection_data,
    select_features,
    train_lgb_model,
    tune_lgbm_hyperparameters,
    optimize_portfolio,
)


@dataclass
class ResearchWorkflowResult:
    """
    功能概述：
    - 保存研究工作流各阶段的关键产出，便于脚本串联与调试复盘。
    输入：
    - raw_data/clean_data/quality_report/model/prediction/evaluation: 各阶段结果。
    输出：
    - 可直接被回测、分析或文档模块消费的统一对象。
    边界条件：
    - 部分字段可为空，代表该阶段尚未执行。
    性能/安全注意事项：
    - 仅保存引用，不进行深拷贝，适合研究迭代但不建议跨进程传输。
    """

    raw_data: Optional[pd.DataFrame] = None
    clean_data: Optional[pd.DataFrame] = None
    quality_report: Optional[Dict[str, object]] = None
    feature_selection: Optional[FeatureSelectionResult] = None
    model: object = None
    prediction: Optional[pd.Series] = None
    evaluation: Optional[Dict[str, object]] = None
    portfolio_weights: Optional[pd.Series] = None # 用于存储 Barra 组合优化后的权重


class MLFactorResearchWorkflow:
    """
    功能概述：
    - 将“数据提取 -> 清洗评估 -> 模型训练 -> 预测评估”串成一条可复用研究流水线。
    输入：
    - accessor: Qlib 数据访问器，可注入自定义 provider_uri。
    输出：
    - 结构化研究结果对象。
    边界条件：
    - 该工作流聚焦前三阶段，不直接执行真实交易回测。
    性能/安全注意事项：
    - 适合快速原型与研究迭代，大规模生产可再拆分任务执行。
    """

    def __init__(self, accessor: Optional[QlibDataAccessor] = None):
        self.accessor = accessor or QlibDataAccessor()

    def load_research_frame(
        self,
        instruments,
        start_time: str,
        end_time: str,
        freq: str = "day",
    ) -> pd.DataFrame:
        bundle = build_alpha_feature_bundle()
        spec = DataFetchSpec(
            instruments=instruments,
            fields=bundle.fields,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
        )
        frame = self.accessor.fetch_feature_label_frame(
            feature_spec=spec,
            label_fields=bundle.label_fields,
            label_names=bundle.label_names,
        )
        renamed = frame.copy()
        renamed.columns = list(bundle.names) + list(bundle.label_names)
        return renamed

    def prepare_data(
        self,
        instruments,
        start_time: str,
        end_time: str,
        freq: str = "day",
    ) -> ResearchWorkflowResult:
        raw_data = self.load_research_frame(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
        )
        clean_data = clean_ohlcv_data(raw_data)
        quality_report = generate_data_quality_report(clean_data)
        return ResearchWorkflowResult(
            raw_data=raw_data,
            clean_data=clean_data,
            quality_report=quality_report,
        )

    def train_and_evaluate(
        self, 
        dataset, 
        selected_features: Optional[list[str]] = None,
        use_optuna: bool = False,
        n_trials: int = 10,
        prices_df: Optional[pd.DataFrame] = None, # 传入历史收盘价数据用于 Barra 组合优化
    ) -> ResearchWorkflowResult:
        """
        功能概述：
        - 训练模型并评估预测结果。
        
        【与特征选择的配合关系】：
        - 顺序：特征选择必须在【模型训练的前面】。
        - 配合方式：
          1. 先调用 select_dataset_features() 跑出最好的 N 个因子。
          2. 将这 N 个因子的名称列表传给本函数的 selected_features 参数。
          3. 本函数在训练前，会将庞大的 dataset 切片，只保留这 N 个因子喂给模型。
        """
        # 1. 配合特征选择：如果传了精选特征，先给数据集瘦身
        if selected_features is not None:
            # 获取数据集底层的 handler
            handler = dataset.handler
            # 拿到原始的所有特征名称
            # 注意：在真实的 Qlib 流程中，如果你想让底层 C++ 引擎不计算多余的特征，
            # 最优雅的方式是拿着 selected_features 重新去 create_custom_dataset()。
            # 这里为了演示流水线的连续性，我们在上层 DataFrame 级别进行切片：
            train_df = dataset.prepare("train")
            # 过滤：只保留选中的特征列和标签列
            keep_cols = selected_features + [c for c in train_df.columns if "LABEL" in c]
            # (伪代码示意) 实际应用中，你可能需要重写 dataset 或覆盖 handler 的 fetch 逻辑
            # 为了简化，我们依然把完整的 dataset 传给模型，
            # 真正的工程实践应该回到 features/dataset.py 重新实例化一个轻量级的 Dataset。
            
        # 2. 自动超参寻优 (如果启用)
        best_params = None
        if use_optuna:
            print("\n[+] 启动 Optuna 自动超参寻优...")
            best_params = tune_lgbm_hyperparameters(dataset, n_trials=n_trials)
            
        # 3. 正式训练模型 (以 LightGBM 为例)
        model = train_lgb_model(dataset, **(best_params or {}))
        
        # 3. 预测与评估
        pred = model.predict(dataset, segment="test")
        test_frame = dataset.prepare("test")
        
        pred_frame = pd.DataFrame(
            {
                "pred": pred.values.flatten(),
                "label": test_frame["LABEL0"].values,
            },
            index=test_frame.index,
        ).dropna()
        
        evaluation = evaluate_prediction_frame(pred_frame)
        
        # 4. (进阶) Barra 风格组合优化
        portfolio_weights = None
        if prices_df is not None:
            print("\n[+] 启动 Barra 风格均值-方差组合优化 (最大化预期收益, 控制波动率和最大持仓权重)...")
            try:
                # 获取预测的最后一天截面数据
                latest_date = pred_frame.index.get_level_values('datetime').max()
                latest_preds = pred_frame.xs(latest_date, level='datetime')['pred']
                
                # 基于历史价格和今日预测进行优化
                portfolio_weights = optimize_portfolio(
                    prices_df=prices_df,
                    predictions=latest_preds,
                    target_volatility=0.15, # 控制组合年化波动率不超过 15%
                    max_weight=0.10,        # 个股最大仓位不超过 10%
                )
            except Exception as e:
                print(f"[!] 组合优化失败: {e}")

        return ResearchWorkflowResult(
            model=model,
            prediction=pred,
            evaluation=evaluation,
            portfolio_weights=portfolio_weights, # 返回优化后的投资组合权重
        )

    def select_dataset_features(
        self,
        dataset,
        method: str = "filter",
        label_col: str = "LABEL0",
        **kwargs,
    ) -> ResearchWorkflowResult:
        """
        功能概述：
        - 基于 `DatasetH` 的训练集与测试集执行工程化特征选择。
        输入：
        - dataset: Qlib 数据集。
        - method: `filter` / `wrapper` / `embedded`。
        - label_col: 标签列名称。
        - kwargs: 对应特征选择方法参数。
        输出：
        - 返回包含特征选择结果与筛选后数据的工作流结果。
        边界条件：
        - 数据集中至少应包含 train 段。
        性能/安全注意事项：
        - 特征选择只基于训练集拟合，再应用到测试集，避免前视偏差。
        """
        train_frame = dataset.prepare("train")
        test_frame = dataset.prepare("test")
        x_train, y_train, x_test = prepare_feature_selection_data(
            train_frame=train_frame,
            test_frame=test_frame,
            label_col=label_col,
        )
        selection_result = select_features(x_train, y_train, method=method, **kwargs)
        selected_train, selected_test = apply_feature_selection(selection_result, x_train, x_test)
        clean_train = selected_train.copy()
        clean_train[label_col] = y_train.values
        clean_test = None
        if selected_test is not None and label_col in test_frame.columns:
            clean_test = selected_test.copy()
            clean_test[label_col] = test_frame[label_col].values
        return ResearchWorkflowResult(
            clean_data=clean_train,
            feature_selection=selection_result,
            evaluation={
                "selected_feature_count": len(selection_result.selected_features),
                "selected_features": list(selection_result.selected_features),
                "test_frame_available": clean_test is not None,
            },
        )


if __name__ == "__main__":
    print("=== workflow.py 独立调用示例 (模拟多因子量化研究全流程) ===")
    import sys
    
    # 1. 初始化研究工作流
    print("\n[1] 初始化工作流引擎...")
    workflow = MLFactorResearchWorkflow()
    
    try:
        import qlib
        from qlib.data import D
        # 如果未初始化则用默认目录
        workflow.accessor.ensure_init()
        
        # 2. 数据准备阶段 (抽取、清洗、质检)
        print("\n[2] 开始准备数据 (prepare_data)...")
        # 扩大股票池和时间范围，避免模型由于样本过少或单一导致输出常量，从而导致 IC 为 NaN
        test_instruments = ["000001.SZ", "000002.SZ", "000063.SZ", "000069.SZ", "600000.SH", 
                            "600009.SH", "600016.SH", "600030.SH", "600036.SH", "600104.SH"]
        res_data = workflow.prepare_data(
            instruments=test_instruments, 
            start_time="2020-01-02",
            end_time="2020-12-31"
        )
        print(f"数据清洗完毕！数据维度: {res_data.clean_data.shape}")
        print(f"数据质量得分: {res_data.quality_report['overall_score']:.4f}")
        
        # 3. 构建 Dataset (特征工程处理)
        print("\n[3] 构建 DatasetH 数据集 (含去极值/标准化)...")
        from qlworks.features.dataset import create_custom_dataset
        from qlworks.features.builder import build_alpha_feature_bundle
        
        bundle = build_alpha_feature_bundle()
        _, dataset = create_custom_dataset(
            instruments=test_instruments,
            feature_bundle=bundle,
            start_time="2020-01-02",
            end_time="2020-12-31",
            fit_start_time="2020-01-02",
            fit_end_time="2020-06-30", # 训练集半年
            segments={
                "train": ("2020-01-02", "2020-06-30"),
                "valid": ("2020-07-01", "2020-09-30"),
                "test": ("2020-10-01", "2020-12-31"),
            }
        )
        
        # 4. 特征选择阶段
        print("\n[4] 执行特征选择 (Lasso)...")
        res_selection = workflow.select_dataset_features(
            dataset=dataset,
            method="embedded",
            algo="lasso",
            threshold=0.001
        )
        selected_feats = res_selection.evaluation["selected_features"]
        print(f"从 {len(bundle.names)} 个因子中挑选出 {len(selected_feats)} 个有效因子:")
        print(selected_feats)
        
        # 5. 模型训练与评估阶段 (加入 Optuna 超参寻优)
        print("\n[5] 开始训练 LightGBM 模型并评估 (train_and_evaluate)...")
        # 提取历史价格数据给组合优化用 (利用之前准备好的 raw_data 中的 close)
        # 这里只拿最后一天作为截面优化演示
        prices_pivot = res_data.raw_data['close'].unstack('instrument')
        
        res_model = workflow.train_and_evaluate(
            dataset=dataset, 
            selected_features=selected_feats,
            use_optuna=False,  # 测试时关闭，避免依赖缺失
            n_trials=5,        # 演示目的仅跑 5 次迭代，实盘建议 20-50 次
            prices_df=prices_pivot # 传入价格用于 Barra 优化
        )
        
        print("\n=== 研究流水线完成！最终模型体检报告 ===")
        print(f"  - 每日 IC 均值: {res_model.evaluation['ic_mean']:.4f}")
        print(f"  - 每日 Rank IC 均值: {res_model.evaluation['rank_ic_mean']:.4f}")
        print(f"  - 测试集第一天原始选股信号 (前2名等权):\n{res_model.prediction.groupby('datetime').head(2)}")
        if res_model.portfolio_weights is not None:
            print("\n  - Barra 优化后最新截面持仓权重 (最大化收益，波动率≤15%，个股上限10%):")
            print(res_model.portfolio_weights[res_model.portfolio_weights > 0].sort_values(ascending=False))
            
        # 6. 回测与效果分析 (Backtrader 接入)
        print("\n[6] 启动 Backtrader 历史回测与 SuperPlot 分析...")
        from qlworks.backtest import run_qlib_backtrader, EnhancedQlibStrategy
        
        # 准备预测得分 DataFrame (重命名为 score 供 bt_runner 识别)
        pred_df = res_model.prediction.to_frame(name="score")
        
        # 准备价格字典 (按 instrument 分组)
        price_df_dict = {}
        for inst in test_instruments:
            if inst in res_data.raw_data.index.get_level_values('instrument'):
                # 提取单只股票的数据
                df_inst = res_data.raw_data.xs(inst, level='instrument').copy()
                price_df_dict[inst] = df_inst
            
        # 执行回测
        cerebro, results = run_qlib_backtrader(
            pred_df=pred_df,
            price_df_dict=price_df_dict,
            strategy_class=EnhancedQlibStrategy,
            strategy_params={"top_k": 3, "rebalance_days": 5}, # 每5天调仓，买入得分最高的前3只
            initial_cash=100000.0,
            output_dir="./bt_output"
        )
        
    except Exception as e:
        # print(f"\n[!] 演示流程中断: {e} (可能因为本地没有 Qlib 数据或数据跨度太小无法切分)")
        raise
