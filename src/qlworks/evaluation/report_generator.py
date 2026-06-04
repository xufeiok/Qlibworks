"""
单因子评测报告生成器：输出交互式 HTML 报告。
支持最新 8 维评分、行业中性 IC、多期持有收益分析等全部优化功能。
"""

import json
import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def _load_plotly_js() -> str:
    """从本地 Plotly 包中读取 plotly.min.js，用于内嵌到 HTML 实现离线查看。"""
    import plotly as _pl
    _js_path = Path(_pl.__file__).parent / "package_data" / "plotly.min.js"
    if _js_path.exists():
        return _js_path.read_text(encoding="utf-8")
    # fallback: 尝试从 CDN 加载（已安装 plotly 时不应走到这里）
    return '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'


class FactorReportGenerator:
    """生成单因子测评 HTML 报告。"""

    def __init__(self, factor_name: str, output_dir: str,
                 eval_start: Optional[str] = None, eval_end: Optional[str] = None):
        self.factor_name = factor_name
        self.output_dir = Path(output_dir)
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fmt(val, fmt: str = ".4f") -> str:
        """安全格式化数值，NaN/Inf 替换为 'N/A'。"""
        import math
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return 'N/A'
        try:
            return f'{val:{fmt}}'
        except (ValueError, TypeError):
            return str(val)

    @staticmethod
    def _fmt_pct(val) -> str:
        """安全格式化百分数。"""
        import math
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return 'N/A'
        try:
            return f'{val:.2%}'
        except (ValueError, TypeError):
            return str(val)

    def _ic_plot(self, ic_series: pd.Series) -> str:
        """IC 时间序列 + 累计 IC 图。"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("每日 IC 值", "累计 IC"),
        )

        dates = ic_series.index
        if hasattr(dates, "str"):
            dates = [str(d)[:10] for d in dates]

        fig.add_trace(
            go.Bar(x=dates, y=ic_series.values, name="IC",
                   marker_color=["red" if v > 0 else "green" for v in ic_series.values]),
            row=1, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        cumic = ic_series.cumsum()
        fig.add_trace(
            go.Scatter(x=dates, y=cumic.values, mode="lines",
                       name="累计 IC", line=dict(color="navy", width=2)),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(height=500, margin=dict(l=40, r=40, t=40, b=40),
                          showlegend=False)
        fig.update_xaxes(tickangle=45, nticks=20)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _group_plot(self, group_means: pd.Series) -> str:
        """分组收益率柱状图。"""
        fig = go.Figure()
        colors = px.colors.sequential.Blues[2:2 + max(len(group_means), 1)]
        vals_safe = []
        texts = []
        any_valid = False
        for v in group_means.values:
            import math as _m
            if hasattr(v, 'item'):
                v = v.item()
            if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
                vals_safe.append(0.0)
                texts.append('N/A')
            else:
                any_valid = True
                vals_safe.append(float(v) * 100)
                texts.append(f'{float(v) * 100:.4f}%')
        # 全部 NaN 也画图，但显示提示信息
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="数据不足（因子分层后各分组平均收益均为 NaN）" if not any_valid else "",
            showarrow=False, font=dict(size=14, color="#94a3b8"),
        )
        fig.add_trace(go.Bar(
            x=[f"Q{i+1}" for i in group_means.index],
            y=vals_safe,
            marker_color=colors,
            text=texts,
            textposition="outside",
        ))
        fig.update_layout(
            title="分层收益率",
            xaxis_title="分组",
            yaxis_title="平均收益率 (%)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _long_short_plot(self, cum_series: pd.Series, ls_stats: dict) -> str:
        """多空组合净值曲线。"""
        fig = go.Figure()
        dates = cum_series.index
        if hasattr(dates, "str"):
            dates = [str(d)[:10] for d in dates]

        fig.add_trace(go.Scatter(
            x=dates, y=cum_series.values, mode="lines",
            name="多空净值", line=dict(color="darkorange", width=2),
            fill="tozeroy", fillcolor="rgba(255,165,0,0.1)",
        ))

        annotations = [
            f"年化收益: {self._fmt(ls_stats.get('annual_return', 'N/A'), '.2f')}%",
            f"夏普比率: {self._fmt(ls_stats.get('sharpe', 'N/A'), '.4f')}",
            f"最大回撤: {self._fmt(ls_stats.get('max_drawdown', 'N/A'), '.2f')}%",
        ]
        for i, ann in enumerate(annotations):
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.95 - i * 0.06,
                text=ann, showarrow=False,
                font=dict(size=12, color="darkorange"),
                bgcolor="rgba(255,255,255,0.8)",
            )

        fig.update_layout(
            title="多空组合净值",
            xaxis_title="日期", yaxis_title="净值",
            height=400, margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _ic_dist_plot(self, ic_series: pd.Series) -> str:
        """IC 分布直方图。"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ic_series.values, nbinsx=40,
            marker_color="steelblue", opacity=0.75,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="IC 分布",
            xaxis_title="IC 值", yaxis_title="频次",
            height=300, margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _sub_period_plot(self, sp_df: pd.DataFrame) -> str:
        """子时段对比图。"""
        if sp_df.empty:
            return ""
        import math as _m
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("IC 均值", "ICIR", "多空年化收益 (%)"),
        )
        periods = sp_df["period"].tolist()
        for col_idx, metric in enumerate(["ic_mean", "icir", "ls_ann_ret"], 1):
            vals = []
            for v in sp_df[metric].values:
                vv = v.item() if hasattr(v, 'item') else v
                if isinstance(vv, float) and (_m.isnan(vv) or _m.isinf(vv)):
                    vals.append(0.0)
                else:
                    vals.append(float(vv))
            colors = ["green" if x > 0 else "red" for x in vals]
            fig.add_trace(
                go.Bar(x=periods, y=vals, marker_color=colors,
                       text=[f"{v:.4f}" for v in vals], textposition="outside"),
                row=1, col=col_idx,
            )
        fig.update_layout(
            height=350, margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _turnover_plot(self, turnover_stats: dict) -> str:
        """换手率分析柱状图。"""
        q_data = turnover_stats.get('monthly_turnover_by_q', {})
        if not q_data:
            return ''
        labels = sorted(q_data.keys())
        values = [q_data[k] for k in labels]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color='#f59e0b',
            text=[f'{v:.1%}' for v in values],
            textposition='outside',
        ))
        fig.update_layout(
            title='各分组的月均换手率',
            xaxis_title='分组', yaxis_title='月均换手率',
            height=350, margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _decay_plot(self, decay_df: pd.DataFrame) -> str:
        """IC 衰减折线图。"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=decay_df['horizon'].values, y=decay_df['ic_mean'].values,
            mode='lines+markers', name='IC 均值',
            line=dict(color='#8b5cf6', width=2), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=decay_df['horizon'].values, y=decay_df['icir'].values,
            mode='lines+markers', name='ICIR',
            line=dict(color='#f97316', width=2, dash='dash'), marker=dict(size=8),
            yaxis='y2',
        ))
        fig.update_layout(
            title='IC 衰减分析',
            xaxis_title='预测期（天）',
            yaxis=dict(title='IC 均值'),
            yaxis2=dict(title='ICIR', overlaying='y', side='right'),
            height=350, margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(x=0.02, y=0.98),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _hpr_plot(self, hpr_df: pd.DataFrame) -> str:
        """多期持有收益分析柱状图。"""
        if hpr_df.empty:
            return ''
        import math as _m
        safe_vals = []
        safe_texts = []
        any_valid = False
        for v in hpr_df['ls_return']:
            vv = v.item() if hasattr(v, 'item') else v
            if isinstance(vv, float) and (_m.isnan(vv) or _m.isinf(vv)):
                safe_vals.append(0.0)
                safe_texts.append('N/A')
            else:
                any_valid = True
                safe_vals.append(float(vv) * 100)
                safe_texts.append(f'{float(vv) * 100:.2f}%')
        fig = go.Figure()
        if not any_valid:
            fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5,
                text="数据不足（各持有期收益均为 NaN）", showarrow=False,
                font=dict(size=14, color="#94a3b8"))
        fig.add_trace(go.Bar(
            x=[f'{h}日' for h in hpr_df['horizon']],
            y=safe_vals,
            marker_color='#10b981',
            text=safe_texts,
            textposition='outside',
        ))
        fig.update_layout(
            title='不同调仓周期的多空收益',
            xaxis_title='持有周期', yaxis_title='多空收益 (%)',
            height=300, margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def generate(
        self,
        ic_stats: dict,
        group_means: pd.Series,
        ls_stats: dict,
        robustness_df: pd.DataFrame,
        preprocess_info: Optional[dict] = None,
        qual_status: bool = False,
        thresholds_info: Optional[dict] = None,
        eval_period: Optional[dict] = None,
        label_expr: Optional[str] = None,
        decay_df: Optional[pd.DataFrame] = None,
        turnover_stats: Optional[dict] = None,
        qual_result: Optional[dict] = None,
        hpr_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """生成完整 HTML 报告。"""
        ic_series = ic_stats.get("ic_series", pd.Series())
        ls_cum = ls_stats.get("cumulative", pd.Series())

        tier = (qual_result or {}).get("tier", "N/A")
        composite_score = (qual_result or {}).get("composite_score", 0)
        reasons = (qual_result or {}).get("reasons", [])
        dim_scores = (qual_result or {}).get("scores", {})

        # 构建核心指标
        summary_rows = [
            ("IC 均值", f"{ic_stats.get('ic_mean', 0):.6f}"),
            ("IC 标准差", f"{ic_stats.get('ic_std', 0):.6f}"),
            ("ICIR（年化）", f"{ic_stats.get('icir', 0):.4f}"),
            ("IC 胜率", f"{ic_stats.get('win_rate', 0):.2%}"),
            ("IC > 0 占比", f"{ic_stats.get('ic_positive_ratio', 0):.2%}"),
            ("t 统计量", f"{ic_stats.get('t_stat', 0):.4f}"),
            ("单调性得分", f"{ic_stats.get('monotonicity', 0):.4f}"),
        ]

        # 行业中性 IC
        ind_ic = ic_stats.get("industry_ic_mean")
        if ind_ic is not None:
            summary_rows.append(("行业中性 IC", f"{ind_ic:.6f}"))
            summary_rows.append(("行业中性 ICIR", f"{ic_stats.get('industry_icir', 0):.4f}"))

        summary_rows += [
            ("多空年化收益", f"{ls_stats.get('annual_return', 0):.2f}%"),
            ("多空年化波动", f"{ls_stats.get('annual_vol', 0):.2f}%"),
            ("多空夏普比率", f"{ls_stats.get('sharpe', 0):.4f}"),
            ("多空最大回撤", f"{ls_stats.get('max_drawdown', 0):.2f}%"),
        ]

        if preprocess_info:
            for k, v in preprocess_info.items():
                summary_rows.append((k, str(v)))

        summary_rows.append(("等级", tier))
        summary_rows.append(("综合评分", f"{composite_score:.1f}"))
        summary_rows.append(("评测结论", "√ 通过" if qual_status else "X 未通过"))

        # 开始组装 HTML
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>单因子评测报告 — {self.factor_name}</title>
<script>{_load_plotly_js()}</script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f5f7fa; color: #333; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #1a365d 0%, #2563eb 100%); color: white; padding: 30px 40px; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .subtitle {{ opacity: 0.85; font-size: 14px; }}
.section {{ background: white; border-radius: 10px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 18px; color: #1a365d; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
th {{ background: #f8fafc; font-weight: 600; color: #475569; font-size: 13px; }}
td {{ font-size: 14px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
.summary-item {{ background: #f8fafc; border-radius: 8px; padding: 14px; text-align: center; }}
.summary-item .label {{ font-size: 12px; color: #94a3b8; margin-bottom: 4px; }}
.summary-item .value {{ font-size: 20px; font-weight: 700; color: #1e293b; }}
.summary-item .value.pass {{ color: #059669; }}
.summary-item .value.fail {{ color: #dc2626; }}
.chart-container {{ margin-top: 16px; }}
.status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }}
.status-badge.core {{ background: #d1fae5; color: #065f46; }}
.status-badge.satellite {{ background: #fef3c7; color: #92400e; }}
.status-badge.archive {{ background: #e2e8f0; color: #475569; }}
.footer {{ text-align: center; color: #94a3b8; font-size: 12px; padding: 20px; }}
.score-bar {{ height: 8px; border-radius: 4px; margin-top: 4px; }}
</style>
</head>
<body>

<div class="header">
  <h1>单因子评测报告</h1>
  <div class="subtitle">
    因子名称: <strong>{self.factor_name}</strong> &nbsp;|&nbsp;
    生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
    等级: <span class="status-badge {tier}">{tier}</span>
  </div>
  <div class="header-meta" style="margin-top:12px;font-size:13px;opacity:0.9;display:flex;flex-wrap:wrap;gap:8px 20px;">
    <span>评测区间: {eval_period.get("start", "N/A") if eval_period else "N/A"} ~ {eval_period.get("end", "N/A") if eval_period else "N/A"}</span>
    <span>标签计算: {label_expr if label_expr else "N/A"}</span>
    <span>准入标准: IC>={thresholds_info.get("ic", 0.05) if thresholds_info else 0.05} | IR>={thresholds_info.get("icir", 1.0) if thresholds_info else 1.0} | 胜率>={self._fmt_pct(thresholds_info.get("win_rate", 0.65)) if thresholds_info else "65%"} | 多空>={thresholds_info.get("ls_ret", 15) if thresholds_info else 15}% | 夏普>={thresholds_info.get("ls_sharpe", 1.25) if thresholds_info else 1.25}</span>
  </div>
</div>
"""

        # 核心指标卡片
        html += """<div class="section">
  <h2>核心指标概览</h2>
  <div class="summary-grid">"""
        for label, val in summary_rows:
            css = "pass" if label == "等级" and val == "core" else ("fail" if label == "评测结论" and "X" in val else "pass" if label == "评测结论" and "√" in val else "")
            html += f"""    <div class="summary-item"><div class="label">{label}</div><div class="value {css}">{val}</div></div>"""
        html += """  </div>
</div>
"""

        # 8 维评分详情
        if dim_scores:
            html += """<div class="section">
  <h2>8 维评分详情（满分 100）</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;">"""
            dim_names = {
                "ic": ("IC 均值", 20), "icir": ("ICIR 稳定性", 15),
                "win_rate": ("胜率", 15), "ls_return": ("多空收益", 15),
                "ls_sharpe": ("多空夏普", 10), "decay": ("IC 衰减", 10),
                "turnover": ("换手率", 5), "coverage": ("覆盖率", 10),
            }
            bar_colors = {"ic": "#2563eb", "icir": "#7c3aed", "win_rate": "#059669",
                          "ls_return": "#d97706", "ls_sharpe": "#dc2626",
                          "decay": "#8b5cf6", "turnover": "#f59e0b", "coverage": "#0ea5e9"}
            for k, (label, weight) in dim_names.items():
                sc = dim_scores.get(k, 0)
                score_pct = sc * 100
                color = bar_colors.get(k, "#94a3b8")
                html += f"""<div style="background:#f8fafc;border-radius:8px;padding:12px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;">
        <span>{label}</span>
        <span><strong>{sc:.2f}</strong> / {weight} 分 <span style="color:#94a3b8;font-size:11px;">({weight}%)</span></span>
      </div>
      <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-top:6px;">
        <div style="background:{color};width:{score_pct:.0f}%;height:6px;border-radius:4px;"></div>
      </div>
    </div>"""
            html += f"""</div>
    <div style="margin-top:12px;font-size:14px;text-align:center;padding:10px;background:#f0fdf4;border-radius:8px;">
      <strong>综合评分: {composite_score:.1f} / 100</strong> &nbsp;|&nbsp; 等级: {tier}
    </div>
  </div>"""

        # IC 分析
        if not ic_series.empty:
            html += f"""<div class="section">
  <h2>IC 分析</h2>
  <div class="chart-container">{self._ic_plot(ic_series)}</div>
  <div class="chart-container" style="display:inline-block;width:48%">{self._ic_dist_plot(ic_series)}</div>
"""

        # 分层收益率
        if not group_means.empty:
            safe_vals = [0 if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v for v in group_means.values]
            group_str = f"""
  <div class="chart-container" style="display:inline-block;width:55%">{self._group_plot(group_means)}</div>
  <div style="display:inline-block;width:42%;vertical-align:top;padding-left:20px">
    <h3 style="font-size:14px;color:#475569;margin-bottom:8px">多空组合统计</h3>
    <table>
      <tr><th>指标</th><th>数值</th></tr>
      <tr><td>年化收益率</td><td>{self._fmt(ls_stats.get('annual_return', 'N/A'), '.2f')}%</td></tr>
      <tr><td>年化波动率</td><td>{self._fmt(ls_stats.get('annual_vol', 'N/A'), '.2f')}%</td></tr>
      <tr><td>夏普比率</td><td>{self._fmt(ls_stats.get('sharpe', 'N/A'), '.4f')}</td></tr>
      <tr><td>最大回撤</td><td>{self._fmt(ls_stats.get('max_drawdown', 'N/A'), '.2f')}%</td></tr>
      <tr><td>单调性得分</td><td>{self._fmt(ls_stats.get('monotonicity', 'N/A'), '.4f')}</td></tr>
    </table>
  </div>
</div>"""
            html += group_str
        else:
            html += "</div>"

        # 多空净值
        if not ls_cum.empty:
            html += f"""<div class="section">
  <h2>多空组合净值曲线</h2>
  <div class="chart-container">{self._long_short_plot(ls_cum, ls_stats)}</div>
</div>"""

        # 换手率
        if turnover_stats and turnover_stats.get('monthly_turnover_by_q'):
            html += f"""<div class="section">
  <h2>换手率分析</h2>
  <div class="chart-container">{self._turnover_plot(turnover_stats)}</div>
  <div style="margin-top:8px;font-size:13px;color:#64748b;">
    平均换手率: {turnover_stats.get('avg_turnover', 0):.2%} &nbsp;|&nbsp;
    最高分组换手率: {turnover_stats.get('max_turnover', 0):.2%}
  </div>
</div>"""

        # IC 衰减
        if decay_df is not None and not decay_df.empty:
            html += f"""<div class="section">
  <h2>IC 衰减分析</h2>
  <div class="chart-container">{self._decay_plot(decay_df)}</div>"""
            if len(decay_df) > 1:
                decay_ratio = abs(decay_df["ic_mean"].iloc[-1]) / max(abs(decay_df["ic_mean"].iloc[0]), 1e-8)
                html += f"""  <div style="margin-top:8px;font-size:13px;color:#64748b;">
    60日 IC 保留率: {decay_ratio:.1%} (1日 IC={abs(decay_df['ic_mean'].iloc[0]):.6f}, {decay_df['horizon'].iloc[-1]}日 IC={abs(decay_df['ic_mean'].iloc[-1]):.6f})
  </div>"""
            html += "</div>"

        # 多期持有收益
        if hpr_df is not None and not hpr_df.empty:
            if hpr_df["ls_return"].notna().any() and (hpr_df["ls_return"].abs() > 0).any():
                best_horizon = hpr_df.loc[hpr_df["ls_return"].abs().idxmax(), "horizon"]
            else:
                best_horizon = 5
            html += f"""<div class="section">
  <h2>多期持有收益分析</h2>
  <div class="chart-container">{self._hpr_plot(hpr_df)}</div>
  <div style="margin-top:8px;font-size:13px;color:#64748b;">
    最佳调仓周期: <strong>{int(best_horizon)} 日</strong> &nbsp;|&nbsp;
  </div>
  <div style="margin-top:12px">
    <table><tr><th>持有期</th><th>Q1收益</th><th>Q5收益</th><th>多空收益</th><th>单调性</th></tr>"""
            def _safe_pct(v):
                import math as _m
                if v is None:
                    return 'N/A'
                if hasattr(v, 'item'):
                    v = v.item()
                if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
                    return 'N/A'
                return f'{float(v) * 100:.4f}%'
            for _, row in hpr_df.iterrows():
                ls = _safe_pct(row.get("ls_return", 0))
                q0 = _safe_pct(row.get("q0_mean", 0))
                q5 = _safe_pct(row.get("q4_mean", row.get("q3_mean", 0)))
                html += f"<tr><td>{int(row['horizon'])}日</td><td>{q0}</td><td>{q5}</td><td>{ls}</td><td>{row.get('monotonicity',0):.4f}</td></tr>"
            html += """</table>
  </div>
</div>"""

        # 稳健性检验
        if isinstance(robustness_df, pd.DataFrame) and not robustness_df.empty:
            html += f"""<div class="section">
  <h2>稳健性检验（子时段）</h2>
  <div class="chart-container">{self._sub_period_plot(robustness_df)}</div>
  <div style="margin-top:16px">
    <table><tr>"""
            for col in robustness_df.columns:
                html += f"<th>{col}</th>"
            html += "</tr>"
            for _, row in robustness_df.iterrows():
                html += "<tr>"
                for val in row:
                    html += f"<td>{val if not isinstance(val, float) else f'{val:.4f}'}</td>"
                html += "</tr>"
            html += """</table>
  </div>
</div>"""

        # 等级判定详情
        if reasons:
            html += f"""<div class="section">
  <h2>等级判定详情</h2>
  <div style="margin-top:8px;">
    <p><strong>等级: {tier}</strong> （综合评分: {composite_score:.1f}）</p>
    <p style="margin-top:8px;color:#dc2626;">未达标项:</p>
    <ul style="margin-left:20px;color:#dc2626;">"""
            for r in reasons:
                html += f"<li style='margin-top:4px;'>{r}</li>"
            html += """</ul>
  </div>
</div>"""

        html += f"""<div class="footer">
  单因子评测系统 · Qlibworks · 报告自动生成
</div>
</body>
</html>"""
        return html

    def save(self, html_content: str) -> str:
        """保存 HTML 报告到文件（文件名含因子名+评测时间段+时间戳）。"""
        safe_name = self.factor_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        period = ""
        if self.eval_start or self.eval_end:
            s = (self.eval_start or "")[:10]
            e = (self.eval_end or "")[:10]
            period = f"_{s}_{e}"
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{safe_name}{period}_{ts}.html"
        path.write_text(html_content, encoding="utf-8")
        return str(path)

    def export_summary_csv(self, ic_stats: dict, ls_stats: dict, qual_status: bool, path: str):
        """导出评测摘要 CSV。"""
        rows = []
        base = {
            "factor": self.factor_name,
            "ic_mean": ic_stats.get("ic_mean", 0),
            "icir": ic_stats.get("icir", 0),
            "win_rate": ic_stats.get("win_rate", 0),
            "ls_annual_return": ls_stats.get("annual_return", 0),
            "ls_sharpe": ls_stats.get("sharpe", 0),
            "ls_max_drawdown": ls_stats.get("max_drawdown", 0),
            "qualified": qual_status,
        }
        rows.append(base)
        df = pd.DataFrame(rows)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["factor"], keep="last")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path
