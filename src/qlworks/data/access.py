from __future__ import annotations

import os
import sys

# 允许直接运行此文件测试时，能正确识别项目根目录下的 qlworks 包
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd

from qlworks.config import QLIB_DATA_DIR


InstrumentInput = Union[str, Sequence[str]]


@dataclass
class DataFetchSpec:
    """
    功能概述：
    - 统一描述一次 Qlib 数据提取请求，便于数据层、特征层、模型层复用。
    输入：
    - instruments: 股票池名称或股票代码列表。
    - fields: 需要提取的字段/因子表达式列表。
    - start_time/end_time/freq: 时间区间与频率。
    输出：
    - 仅保存查询参数，不直接执行查询。
    边界条件：
    - fields 为空时应由调用方兜底。
    - instruments 为字符串时既可表示股票池，也可表示单只股票。
    性能/安全注意事项：
    - 推荐批量查询，避免碎片化小请求破坏缓存命中率。
    """

    instruments: InstrumentInput
    fields: Sequence[str]
    start_time: str
    end_time: str
    freq: str = "day"


class QlibDataAccessor:
    """
    功能概述：
    - 封装 Qlib 常用数据访问动作，包括初始化、交易日历、股票池与特征/标签提取。
    输入：
    - provider_uri/region: Qlib 数据根目录与市场区域。
    输出：
    - 返回标准 Pandas DataFrame 或列表，供下游清洗、评估、建模模块使用。
    边界条件：
    - 若 Qlib 未安装，调用时会抛出清晰异常。
    - 股票池字符串与显式股票列表均可使用。
    性能/安全注意事项：
    - `ensure_init()` 只在首次调用时初始化，避免重复初始化带来额外开销。
    """

    def __init__(self, provider_uri: Optional[str] = None, region: str = "cn"):
        self.provider_uri = str(provider_uri or QLIB_DATA_DIR)
        self.region = region
        self._initialized = False

    def ensure_init(self) -> None:
        """按需初始化 Qlib。"""
        if self._initialized:
            return
        try:
            import qlib
        except ImportError as exc:  # pragma: no cover
            raise ImportError("未安装 pyqlib，无法使用 Qlib 数据访问能力。") from exc

        qlib.init(provider_uri=self.provider_uri, region=self.region)
        self._initialized = True

    def calendar(self, start_time: str, end_time: str, freq: str = "day"):
        """获取交易日历。"""
        self.ensure_init()
        from qlib.data import D

        return D.calendar(start_time=start_time, end_time=end_time, freq=freq)

    def list_instruments(
        self,
        instruments: InstrumentInput,
        start_time: str,
        end_time: str,
        freq: str = "day",
        as_list: bool = True,
    ) -> List[str]:
        """
        功能概述：
        - 将股票池配置展开为具体股票列表。
        输入：
        - instruments: 股票池名称、股票代码或股票列表。
        输出：
        - 标准股票代码列表。
        边界条件：
        - 若已是列表，则直接返回去重后的结果。
        性能/安全注意事项：
        - 大股票池建议复用结果，避免重复展开。
        """
        self.ensure_init()
        if isinstance(instruments, (list, tuple)):
            return [str(x) for x in instruments]

        from qlib.data import D

        conf = D.instruments(instruments)
        if as_list:
            return D.list_instruments(
                conf,
                start_time=start_time,
                end_time=end_time,
                freq=freq,
                as_list=True,
            )
        return conf

    def fetch_features(self, spec: DataFetchSpec) -> pd.DataFrame:
        """提取 Qlib 特征/因子数据。"""
        self.ensure_init()
        from qlib.data import D

        return D.features(
            instruments=spec.instruments,
            fields=list(spec.fields),
            start_time=spec.start_time,
            end_time=spec.end_time,
            freq=spec.freq,
        )

    def fetch_labels(
        self,
        instruments: InstrumentInput,
        label_fields: Sequence[str],
        start_time: str,
        end_time: str,
        freq: str = "day",
    ) -> pd.DataFrame:
        """提取监督学习标签。"""
        spec = DataFetchSpec(
            instruments=instruments,
            fields=label_fields,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
        )
        return self.fetch_features(spec)

    def fetch_feature_label_frame(
        self,
        feature_spec: DataFetchSpec,
        label_fields: Sequence[str],
        label_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        功能概述：
        - 一次性提取特征与标签并按索引对齐，形成统一研究底表。
        输入：
        - feature_spec: 特征查询规范。
        - label_fields/label_names: 标签表达式与标签名。
        输出：
        - 对齐后的多重索引 DataFrame。
        边界条件：
        - label_names 缺失时自动生成 `LABEL0/LABEL1...`。
        性能/安全注意事项：
        - 只做按索引拼接，不做额外重采样，避免无意引入前视偏差。
        """
        feature_df = self.fetch_features(feature_spec)
        label_df = self.fetch_labels(
            instruments=feature_spec.instruments,
            label_fields=label_fields,
            start_time=feature_spec.start_time,
            end_time=feature_spec.end_time,
            freq=feature_spec.freq,
        )
        if label_names is None:
            label_names = [f"LABEL{i}" for i in range(len(label_df.columns))]
        label_df = label_df.copy()
        label_df.columns = list(label_names)
        return feature_df.join(label_df, how="left")


if __name__ == "__main__":
    print("=== QlibDataAccessor 使用示例 ===")
    
    # 1. 初始化数据访问器 (默认使用 config.py 中的 QLIB_DATA_DIR)
    accessor = QlibDataAccessor()

    # 2. 获取交易日历
    print("\n[1] 获取交易日历 (2020-01-01 至 2020-01-10):")
    cal = accessor.calendar(start_time="2020-01-01", end_time="2020-01-10")
    print(cal)

    # 3. 获取股票池列表 (为加快运行速度，只打印前5只)
    print("\n[2] 获取沪深300成分股 (前5只):")
    try:
        # 注意：2020-01-01 是元旦休息日，本地数据从 2020-01-02 开始
        instruments = accessor.list_instruments(instruments="csi300", start_time="2020-01-02", end_time="2020-01-02")
        print(instruments[:5])
    except Exception as e:
        print(f"获取股票池失败 (可能是本地数据未包含 csi300): {e}")

    # 为了演示快速运行，我们取两只具体的股票 (本地数据格式为 code.EXCHANGE，如 600000.SH)
    demo_instruments = ["600000.SH", "000001.SZ"]

    # 4. 构建特征查询规范
    print("\n[3] 构建特征查询规范 DataFetchSpec...")
    feature_spec = DataFetchSpec(
        instruments=demo_instruments,
        fields=["$close", "$volume", "Ref($close, 1)/$close - 1"],  # 收盘价，成交量，昨日收益率
        start_time="2020-01-02",
        end_time="2020-01-10",
        freq="day"
    )

    # 5. 提取特征数据
    print("\n[4] 提取特征数据 fetch_features (前5行):")
    try:
        features_df = accessor.fetch_features(feature_spec)
        print(features_df.head())
    except Exception as e:
        print(f"提取特征数据失败: {e}")

    # 6. 提取统一研究底表 (特征 + 标签)
    print("\n[5] 提取统一研究底表 fetch_feature_label_frame (特征 + 标签, 前5行):")
    # 假设预测未来2天的收益率作为标签
    label_fields = ["Ref($close, -2)/$close - 1"]
    try:
        dataset_df = accessor.fetch_feature_label_frame(
            feature_spec=feature_spec,
            label_fields=label_fields,
            label_names=["LABEL_2D"]
        )
        print(dataset_df.head(5))
        
        # ==============================================================
        # 演示模块串联：access.py -> cleaning.py -> quality.py
        # ==============================================================
        from qlworks.data.cleaning import clean_ohlcv_data
        from qlworks.data.quality import generate_data_quality_report
        
        print("\n[6] 串联演示：对提取的数据进行基础清洗 (cleaning.py)...")
        cleaned_df = clean_ohlcv_data(dataset_df)
        print("清洗完成。缺失值已填充，极端值已裁剪。")
        
        print("\n[7] 串联演示：对清洗后的数据进行质量评估 (quality.py)...")
        quality_report = generate_data_quality_report(cleaned_df, expected_freq="D")
        
        print("\n=== 数据质量报告 ===")
        print(f"1. 综合质量得分: {quality_report['overall_score']:.4f}")
        print(f"   [{quality_report['metrics_explanation']['overall_score']}]")
        
        print(f"\n2. 完整性得分 (Completeness): {quality_report['completeness']['completeness']:.4f}")
        print(f"   缺失比例统计:\n{quality_report['completeness']['missing_ratio']}")
        print(f"   [{quality_report['metrics_explanation']['completeness']}]")
        
        print(f"\n3. 一致性得分 (Consistency): {quality_report['consistency']['consistency_score']:.4f}")
        print(f"   发现逻辑违规异常数: {quality_report['consistency']['consistency_issues']}")
        print(f"   [{quality_report['metrics_explanation']['consistency']}]")
        
        print(f"\n4. 时效性得分 (Timeliness): {quality_report['timeliness']['timeliness_score']:.4f}")
        print(f"   数据延迟天数: {quality_report['timeliness']['data_lag_days']} 天")
        print(f"   [{quality_report['metrics_explanation']['timeliness']}]")
        
        print(f"\n5. 异常极端值分布 (Outliers):")
        print(f"{quality_report['outliers']}")
        print(f"   [{quality_report['metrics_explanation']['outliers']}]")
        
    except Exception as e:
        print(f"提取研究底表或串联演示失败: {e}")
