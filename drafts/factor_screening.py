import os
import sys
import pandas as pd
import numpy as np
import yaml

# 【基础知识】将项目根目录 src 文件夹加入 sys.path，这样 Python 就能找到并导入我们自己写的 qlworks 包
# 无论是否是主入口运行，都需要确保模块能被找到
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from qlworks.data import DataFetchSpec, QlibDataAccessor, clean_ohlcv_data, generate_data_quality_report
from qlworks.features.builder import build_factor_library_bundle
from qlworks.features.dataset import create_custom_dataset
from qlworks.models import prepare_feature_selection_data, select_features

# ==============================================================================
# [全局配置区] 将本文件需要用到的参数提取到此处，方便您对照学习修改
# ==============================================================================
def load_csi500_instruments():
    """读取中证500股票池"""
    file_path = r"e:\Quant\Qlibworks\qlib_data\instruments\csi500.txt"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t', header=None, names=['instrument', 'start_date', 'end_date'])
        df = df[df['start_date'] <= '2020-01-02'].drop_duplicates(subset=['instrument'])
        return df['instrument'].tolist()
    return ["000001.SZ", "000002.SZ", "600000.SH"]

CONFIG = {
    # 1. 股票池与时间范围
    "instruments": load_csi500_instruments(),
    "start_time": "2020-01-02",
    "end_time": "2020-12-31",
    
    # 2. 数据集切分 (训练/验证/测试)
    "segments": {
        "train": ("2020-01-02", "2020-06-30"),
        "valid": ("2020-07-01", "2020-09-30"),
        "test":  ("2020-10-01", "2020-12-31"),
    },
    
    # 3. 特征选择参数
    "feature_selection": {
        "enable": True,
        "method": "embedded",   # filter / wrapper / embedded
        "algo": "random_forest", # 使用随机森林进行特征过滤
        "threshold": 0.0001,    # 特征重要性阈值
        "label_col": "LABEL0",  # 预测的收益率标签名
    }
}
# ==============================================================================

def load_factor_metadata(factor_files):
    """
    【附加功能】从 YAML 因子库中读取因子的元数据（分类、公式、意义）
    方便最终结果展示时，您能知道每个因子到底是干嘛的。
    """
    metadata = {}
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../factors_repo"))
    
    for file_name in factor_files:
        yaml_path = os.path.join(repo_path, f"{file_name}.yaml")
        if not os.path.exists(yaml_path):
            continue
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if not data or 'factors' not in data:
            continue
            
        for factor in data['factors']:
            name = factor.get('name')
            if not name:
                continue
                
            # 提取 Qlib 表达式
            expr = factor.get('expression', '')
            if isinstance(expr, dict):
                qlib_expr = expr.get('qlib', str(expr))
            else:
                qlib_expr = str(expr)
                
            metadata[name] = {
                '大类 (File)': file_name,
                '子类 (Category)': factor.get('category', '未知'),
                '计算公式 (Expression)': qlib_expr,
                '业务意义 (Meaning)': factor.get('meaning', '').replace('\n', ' ')
            }
    return metadata

def screen_and_rank_factors():
    """
    功能概述：
    - 这是为您量身定制的【大类因子筛选与排序】教学版脚本。
    - 它完整展示了从“读取公式” -> “拉取数据” -> “清洗/去极值/中性化” -> “初步机器学习筛选” -> “深入量化统计(IC/ICIR)” -> “去重选优”的全过程。
    - 我们会在每一步通过 print() 输出结果，方便您对照学习。
    """
    print("="*60)
    print("=== 开始大类因子筛选与排序 (教学演示版) ===")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # [第 0 步] 初始化 Qlib 环境
    # -------------------------------------------------------------------------
    # 【量化逻辑】Qlib 需要知道数据存在哪里（通常是 .bin 文件）。初始化就是告诉它去哪里找数据。
    print("\n[0] 初始化 Qlib 环境...")
    accessor = QlibDataAccessor()
    accessor.ensure_init()
    print(">>> 恭喜！Qlib 底层引擎已成功启动，随时待命提取数据。")
    
    # -------------------------------------------------------------------------
    # [第 1 步] 特征解析与数据拉取
    # -------------------------------------------------------------------------
    # 【量化逻辑】我们需要告诉 Qlib：用哪些股票池？从哪天到哪天？算哪些因子？
    print("\n[1] 解析特征表达式与拉取原始数据...")
    
    # 1.1 我们想要测试的 5 个因子类文件
    factor_files = [
        "style_factors", 
        "quality_factors", 
        "price_volume_factors", 
        "sentiment_factors", 
        "risk_factors"
    ]
    bundle = build_factor_library_bundle(factor_files)
    print(f">>> 成功从 5 个 YAML 文件中解析出 {len(bundle.fields)} 个有效因子的计算公式。")
    
    # 1.2 额外拉取基础行情数据（OHLCV），不仅为了后续的回测，也是为了方便我们人眼检查
    ohlcv_fields = ["$open", "$high", "$low", "$close", "$volume"]
    ohlcv_names = ["open", "high", "low", "close", "volume"]
    
    # 将因子公式和基础行情合并在一起，一次性全部拉取出来
    bundle_fields_with_price = list(bundle.fields) + ohlcv_fields
    bundle_names_with_price = list(bundle.names) + ohlcv_names
    
    spec = DataFetchSpec(
        instruments=CONFIG["instruments"],     # 股票池 (例如：中证500)
        fields=bundle_fields_with_price,       # 需要提取的所有字段表达式
        start_time=CONFIG["start_time"],       # 起始时间
        end_time=CONFIG["end_time"],           # 结束时间
    )
    
    # 真正去本地 .bin 文件里把庞大的表格拉出来
    raw_data = accessor.fetch_feature_label_frame(
        feature_spec=spec,
        label_fields=bundle.label_fields,      # 收益率标签 (如：明天收盘价/今天收盘价 - 1)
        label_names=bundle.label_names,        # 标签列名叫 "LABEL0"
    )
    
    # 给拉出来的表格列名换成好懂的名字
    raw_data.columns = bundle_names_with_price + list(bundle.label_names)
    print(f">>> 数据拉取完毕！我们得到了一个超级大表格：")
    print(f"    - 总行数 (不同日期 * 不同股票): {raw_data.shape[0]} 行")
    print(f"    - 总列数 (因子数量 + 行情 + LABEL0): {raw_data.shape[1]} 列")
    print(f"    - 【数据抽样展示】表格的右上角一瞥:")
    # iloc[:3, -5:] 意思是取前3行，以及最后5列展示，不让屏幕刷屏
    print(raw_data.iloc[:3, -5:]) 
    
    # -------------------------------------------------------------------------
    # [第 2 步] 数据清洗
    # -------------------------------------------------------------------------
    # 【量化逻辑】真实市场里会有停牌、一字涨跌停等导致数据异常，甚至存在无限大 (Inf)。必须清洗。
    print("\n[2] 数据清洗...")
    clean_data = clean_ohlcv_data(raw_data)
    print(f">>> 清洗后数据剩余 {clean_data.shape[0]} 行。")
    
    # -------------------------------------------------------------------------
    # [第 3 步] 构建 DatasetH (特征工程/中性化/去极值)
    # -------------------------------------------------------------------------
    # 【量化逻辑】机器学习模型最怕“离群极大值”和“量纲不统一”。
    # DatasetH 内部会执行 RobustZScoreNorm（去极值+标准化）和 Fillna（缺失值填充）。
    print("\n[3] 构建 DatasetH (特征工程/标准化/去极值)...")
    _, dataset = create_custom_dataset(
        instruments=CONFIG["instruments"],
        feature_bundle=bundle,
        start_time=CONFIG["start_time"],
        end_time=CONFIG["end_time"],
        fit_start_time=CONFIG["segments"]["train"][0],
        fit_end_time=CONFIG["segments"]["train"][1],
        segments=CONFIG["segments"]
    )
    print(">>> 数据已经被加工成了适合机器学习的形态。")
    
    # -------------------------------------------------------------------------
    # [第 4 步] 初步特征选择 (降维)
    # -------------------------------------------------------------------------
    # 【量化逻辑】因子太多了（200多个），里面很多是噪音。
    # 这里我们用“随机森林（Random Forest）”快速扫一遍，把对收益率毫无帮助的废因子直接砍掉。
    print("\n[4] 执行初步特征选择 (机器学习过滤)...")
    fs_conf = CONFIG["feature_selection"]
    train_frame = dataset.prepare("train") # 提取训练集数据
    test_frame = dataset.prepare("test")   # 提取测试集数据
    
    x_train, y_train, _ = prepare_feature_selection_data(
        train_frame=train_frame,
        test_frame=test_frame,
        label_col=fs_conf["label_col"]
    )
    
    fs_result = select_features(
        x_train, y_train, 
        method=fs_conf["method"], 
        algo=fs_conf["algo"], 
        threshold=fs_conf["threshold"]
    )
    
    selected_features = list(fs_result.selected_features)
    print(f">>> 随机森林火眼金睛，从 {len(bundle.names)} 个原始因子中，")
    print(f"    剔除了毫无作用的因子，保留下 {len(selected_features)} 个有效因子。")
    print(f"    - 展示的其中前 10 个因子: {selected_features[:10]}")
    
    # -------------------------------------------------------------------------
    # [第 5 步] 因子深度分析：信息系数 (IC) 与 信息比率 (ICIR)
    # -------------------------------------------------------------------------
    # 【量化逻辑】
    # 1. IC (Information Coefficient): 因子值与第二天真实收益率的相关性。IC 绝对值越大，说明预测越准。
    # 2. ICIR (Information Ratio): IC 均值 / IC 标准差。衡量因子赚钱是否“稳”。如果一个因子偶尔很准但平时瞎指，ICIR就很低。
    print("\n[5] 因子深度分析：计算相关性 (IC) 和 稳定性 (ICIR)...")
    
    ic_dict = {}
    icir_dict = {}
    
    import warnings
    warnings.filterwarnings('ignore', message='An input array is constant; the correlation coefficient is not defined.')
    
    for feature in selected_features:
        # 每天单独算一次该因子和第二天收益率的相关性
        daily_ic = train_frame.groupby('datetime').apply(
            lambda x: x[feature].corr(x[fs_conf["label_col"]], method='spearman')
        )
        daily_ic = daily_ic.dropna() # 剔除由于某些天数据缺失导致的 NaN
        
        ic_mean = daily_ic.mean()  # 这段时间每天IC的平均值
        ic_std = daily_ic.std()    # 这段时间每天IC的波动程度（标准差）
        
        ic_dict[feature] = ic_mean
        
        # 如果标准差极小，说明因子退化成常数了，ICIR 设为0
        if pd.isna(ic_std) or ic_std < 1e-6:
            icir_dict[feature] = 0
        else:
            # 乘以 sqrt(252) 是为了把它年化 (一年约252个交易日)
            icir_dict[feature] = ic_mean / ic_std * np.sqrt(252) 
        
    ic_df = pd.DataFrame({
        'Factor': list(ic_dict.keys()),
        'IC': list(ic_dict.values()),
        'ICIR': list(icir_dict.values())
    }).set_index('Factor')
    
    ic_df = ic_df.dropna(subset=['IC'])
    print(f">>> IC 与 ICIR 计算完成！")
    print(f"    - 【数据抽样展示】前 10 个因子的量化指标:")
    print(ic_df.head(10))
    
    # -------------------------------------------------------------------------
    # [第 6 步] 剔除高相关冗余因子，排序生成最终因子集
    # -------------------------------------------------------------------------
    # 【量化逻辑】如果两个因子长得差不多（比如 MA5 和 MA10），它们提供的信息是重复的。
    # 同时放进模型会导致“多重共线性”，让模型变笨。
    # 我们的做法：算出所有因子的综合得分，得分高的留下。碰到两个高度相似的，把得分低的那个直接踢掉！
    print("\n[6] 剔除高相关冗余因子，排序生成最终核心因子库...")
    
    print("    - 正在计算因子间的自相关矩阵 (找出谁和谁长得像)...")
    corr_matrix = train_frame[ic_df.index].corr(method='spearman').abs()
    
    # 综合得分 = |IC| * |ICIR| (既要赚得多，又要稳)
    ic_df['Abs_IC'] = ic_df['IC'].abs()
    ic_df['Abs_ICIR'] = ic_df['ICIR'].abs()
    ic_df['Score'] = ic_df['Abs_IC'] * ic_df['Abs_ICIR']
    
    # 按照综合得分从高到低排好队
    sorted_factors = ic_df.sort_values(by='Score', ascending=False).index.tolist()
    
    final_selected_factors = []
    correlation_threshold = 0.7  # 相似度超过 70% 就认为是冗余的
    
    for factor in sorted_factors:
        is_redundant = False
        # 和已经选入“名人堂”的因子比一比
        for selected in final_selected_factors:
            if corr_matrix.loc[factor, selected] > correlation_threshold:
                is_redundant = True
                break
        
        # 只要和名人堂里的都不像，就可以入选！
        if not is_redundant:
            final_selected_factors.append(factor)
            
    print(f">>> 经过严苛的选拔和去重，最终从 {len(sorted_factors)} 个因子中，")
    print(f"    优中选优，提炼出 {len(final_selected_factors)} 个绝对核心因子！")
    
    # -------------------------------------------------------------------------
    # [最后] 输出与保存结果
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("=== 🏆 最终有效大类因子集合 (按分类展示 & 综合得分排行) 🏆 ===")
    print("="*80)
    
    # 从本地 YAML 文件中加载每个因子的详细业务说明
    metadata = load_factor_metadata(factor_files)
    
    # 构建包含详细说明的结果表
    result_records = []
    for factor in final_selected_factors:
        record = {
            '因子名称 (Factor)': factor,
            '综合得分 (Score)': round(ic_df.loc[factor, 'Score'], 6),
            'IC均值 (IC)': round(ic_df.loc[factor, 'IC'], 4),
            '信息比率 (ICIR)': round(ic_df.loc[factor, 'ICIR'], 2),
        }
        
        # 加上元数据（如果在YAML里没找到就填空）
        meta = metadata.get(factor, {
            '大类 (File)': '未归类',
            '子类 (Category)': '未归类',
            '计算公式 (Expression)': '',
            '业务意义 (Meaning)': ''
        })
        record.update(meta)
        result_records.append(record)
        
    final_result_df = pd.DataFrame(result_records)
    
    # 按照大类分组，看看各路豪杰分别属于哪个门派
    grouped = final_result_df.groupby('大类 (File)')
    print("\n【各分类有效因子入选分布情况】")
    for group_name, group_df in grouped:
        print(f"[{group_name}]: 入选 {len(group_df)} 个因子")
        
    # 给新手补充一句话解释打印的这三个列
    print("\n (释义：IC 绝对值越大收益预测越准，ICIR 绝对值越大越稳，Score 是综合分)")
    
    # 为了不在屏幕上刷屏太乱，终端只按得分从高到低打印 Top 10，并截取部分说明
    print("\n【全市场综合得分 Top 10】:")
    top10_df = final_result_df.sort_values(by='综合得分 (Score)', ascending=False).head(10)
    
    # 设置 pandas 显示选项，让终端打印更好看
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 50)
    
    print(top10_df[['因子名称 (Factor)', '大类 (File)', '综合得分 (Score)', 'IC均值 (IC)', '信息比率 (ICIR)', '计算公式 (Expression)']])
    print("...\n")
    
    # 保存完整榜单（包含所有信息）到 drafts 目录，按照 大类 和 得分 进行双重排序
    final_result_df = final_result_df.sort_values(by=['大类 (File)', '综合得分 (Score)'], ascending=[True, False])
    output_path = os.path.join(os.path.dirname(__file__), "factor_screening_results_with_meaning.csv")
    final_result_df.to_csv(output_path, index=False, encoding='utf-8-sig') # utf-8-sig 防止 Excel 乱码
    print(f"- 带有【公式与中文含义】的完整筛选榜单已保存至: {output_path}")
    print("  强烈建议用 Excel 打开它，按大类查看各因子的具体意义！")
    print("="*80)
    
    return final_selected_factors

if __name__ == "__main__":
    screen_and_rank_factors()