"""
[废弃] 此模块已被 src/qlworks/data/api.py 中的 QuantDataAPI 取代。

请使用 QuantDataAPI 替代 QlibDataAccessor：
    from qlworks.data.api import QuantDataAPI
    with QuantDataAPI() as api:
        df = api.get_daily_data(...)

本模块保留仅用于向后兼容，新代码不应再引用。
"""
from __future__ import annotations

import os
import sys
import warnings
warnings.warn(
    "qlworks.data.access 已废弃，请使用 qlworks.data.api.QuantDataAPI 替代",
    DeprecationWarning, stacklevel=2
)

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
    - 统一描述一次数据提取请求，便于数据层、特征层、模型层复用。
    输入：
    - instruments: 股票池名称或股票代码列表。
    - fields: 需要提取的字段/因子表达式列表（在 ClickHouse 模式下为 DuckDB SQL 表达式）。
    - start_time/end_time/freq: 时间区间与频率。
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
    - 维持与 Qlib 原生 `.bin` 文件的接口兼容，确保下游 DatasetH 和模型算子不报错。
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
        """将股票池配置展开为具体股票列表。"""
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
        """一次性提取特征与标签并按索引对齐，形成统一研究底表。"""
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
    
    # 1. 初始化数据访问器
    accessor = QlibDataAccessor()

    # 2. 获取交易日历
    print("\n[1] 获取交易日历 (2020-01-02 至 2020-01-10):")
    try:
        cal = accessor.calendar(start_time="2020-01-02", end_time="2020-01-10")
        print(cal)
    except Exception as e:
        print(f"获取交易日历失败: {e}")

    # 3. 获取股票池列表
    demo_instruments = ["600000.SH", "000001.SZ"]
    print(f"\n[2] 获取标的列表: {demo_instruments}")

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
    label_fields = ["Ref($close, -2)/$close - 1"]
    try:
        dataset_df = accessor.fetch_feature_label_frame(
            feature_spec=feature_spec,
            label_fields=label_fields,
            label_names=["LABEL_2D"]
        )
        print(dataset_df.head(5))
        
        from qlworks.data.cleaning import clean_ohlcv_data
        from qlworks.data.quality import generate_data_quality_report
        
        print("\n[6] 串联演示：对提取的数据进行基础清洗 (cleaning.py)...")
        cleaned_df = clean_ohlcv_data(dataset_df)
        print("清洗完成。缺失值已填充，极端值已裁剪。")
        
        print("\n[7] 串联演示：对清洗后的数据进行质量评估 (quality.py)...")
        quality_report = generate_data_quality_report(cleaned_df, expected_freq="D")
        
        print("\n=== 数据质量报告 ===")
        print(f"1. 综合质量得分: {quality_report['overall_score']:.4f}")
        print(f"\n2. 完整性得分 (Completeness): {quality_report['completeness']['completeness']:.4f}")
        
    except Exception as e:
        print(f"提取研究底表或串联演示失败: {e}")
