# -*- coding: utf-8 -*-
"""批量补全新因子 meta.json：补 usage_scenario 和 strategy_hint 字段。"""
import json
from pathlib import Path

WAREHOUSE = Path(r"E:\Quant\Qlibworks\factor_data\warehouse")

# 每个因子的补充说明
UPDATES = {
    "sent_elg_net_ratio": {
        "usage_scenario": "机构/超大单资金动向跟踪，大盘蓝筹股主力建仓识别。",
        "strategy_hint": "正向因子（超大单净流入=机构买入），适合中线持仓策略。",
    },
    "sent_lg_net_ratio": {
        "usage_scenario": "大户资金动向跟踪，中盘股主力行为分析。",
        "strategy_hint": "正向因子（大单净流入=大户买入），适合波段策略。",
    },
    "sent_md_net_ratio": {
        "usage_scenario": "散户/中户资金动向，情绪反向指标参考。",
        "strategy_hint": "常作反向因子（中单净流入=散户追涨，需结合大盘环境）。",
    },
    "sent_mf_strength_20d": {
        "usage_scenario": "综合主力资金强度，全市场资金流向排序。",
        "strategy_hint": "正向因子（主力持续净流入=中期上涨动能），适合周频调仓。",
    },
    "volume_ratio_5d": {
        "usage_scenario": "成交量异动检测，放量突破/缩量回调识别。",
        "strategy_hint": "正向因子（量比突增=资金关注），需结合价格方向判断。",
    },
    "gross_margin_chg_4q": {
        "usage_scenario": "基本面质量改善选股，毛利率拐点捕捉。",
        "strategy_hint": "正向因子（毛利率提升=产品竞争力增强），季度频调仓。",
    },
    "ep_ind_neutral": {
        "usage_scenario": "行业内估值排序，避免价值因子集中在低估值行业。",
        "strategy_hint": "正向因子（EP高于市场均值=相对低估），适合价值选股。",
    },
    "holder_num_chg_20d": {
        "usage_scenario": "筹码集中度跟踪，机构建仓/出货识别。",
        "strategy_hint": "反向因子（股东户数减少=筹码集中=利好），季度频。",
    },
    "holder_num_trend_4q": {
        "usage_scenario": "筹码趋势性变化，持续集中/分散识别。",
        "strategy_hint": "反向因子（户数斜率为负=持续集中=机构吸筹）。",
    },
    "lhb_count_20d": {
        "usage_scenario": "游资关注度跟踪，短线热门股识别。历史短(2025-09起)。",
        "strategy_hint": "正向因子（上榜次数多=游资追捧），短线策略参考。",
    },
    "lhb_org_net_rate": {
        "usage_scenario": "龙虎榜机构席位买卖力度，机构通过龙虎榜的态度。历史短(2025-09起)。",
        "strategy_hint": "正向因子（机构净买入率高=机构看好）。",
    },
    "limit_up_height_20d": {
        "usage_scenario": "打板情绪周期位置判断，连板股高度跟踪。",
        "strategy_hint": "正向因子（连板高度高=情绪强），但高位注意退潮风险。",
    },
    "limit_seal_strength": {
        "usage_scenario": "涨停封板质量评估，封单力度=做多决心。",
        "strategy_hint": "正向因子（封单强=封板坚决），次日溢价概率高。",
    },
    "hot_rank_chg_20d": {
        "usage_scenario": "散户搜索关注度变化，情绪过热识别。历史短(2025-09起)。",
        "strategy_hint": "反向因子（排名急升=散户过热，短期见顶风险）。",
    },
    "break_count_20d": {
        "usage_scenario": "涨停股封板质量评估，炸板多=上方抛压重。历史短(2025-09起)。",
        "strategy_hint": "负向因子（炸板次数多=封板不坚决=弱势）。",
    },
    "limit_down_count_20d": {
        "usage_scenario": "弱势股排雷，跌停多=恐慌抛售。历史短(2025-09起)。",
        "strategy_hint": "负向因子（跌停次数多=坚决回避）。",
    },
    "hot_money_net_rate_20d": {
        "usage_scenario": "游资净买入强度，短线资金追捧跟踪。历史短(2025-09起)。",
        "strategy_hint": "正向因子（游资净流入=短线做多），适合短线策略。",
    },
}

updated = 0
for factor_name, extra in UPDATES.items():
    meta_path = WAREHOUSE / factor_name / "meta.json"
    if not meta_path.exists():
        print(f"  [SKIP] {factor_name}: meta.json 不存在")
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    changed = False
    for k, v in extra.items():
        if k not in meta or not meta[k]:
            meta[k] = v
            changed = True
    if changed:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] {factor_name}: 补全 usage_scenario / strategy_hint")
        updated += 1
    else:
        print(f"  [--] {factor_name}: 字段已存在")

print(f"\n完成: {updated}/{len(UPDATES)} 个因子 meta.json 已更新")
