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

    def _decile_nav_plot(self, decile_nav: pd.DataFrame) -> str:
        """分层净值曲线：显示全部 10 条分位组的累计净值。

        这是判断因子是否「长期分化」、「阶段性失效」的核心图表。
        G1（因子最小）→ 红色渐变，G10（因子最大）→ 绿色渐变。
        """
        if decile_nav.empty:
            return ''
        fig = go.Figure()

        # 使用从红→橙→黄→绿的易区分渐变, G1~G10 顺序
        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c',
                  '#1976d2', '#1565c0', '#6a1b9a', '#c2185b', '#e91e63']
        for i, col in enumerate(decile_nav.columns):
            is_top = (i >= len(decile_nav.columns) - 2)
            is_bottom = (i <= 1)
            fig.add_trace(go.Scatter(
                x=decile_nav.index,
                y=decile_nav[col].values,
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=3 if is_top or is_bottom else 1.2),
                opacity=0.9 if is_top or is_bottom else 0.6,
            ))

        # 标注各组的最终值（右末端标注）
        last_idx = decile_nav.index[-1]
        for i, col in enumerate(decile_nav.columns):
            is_top = (i >= len(decile_nav.columns) - 2)
            is_bottom = (i <= 1)
            if is_top or is_bottom:
                fig.add_annotation(
                    x=last_idx, y=decile_nav[col].iloc[-1],
                    text=f"{col} ({decile_nav[col].iloc[-1]:.3f})",
                    showarrow=False, xanchor='left',
                    font=dict(size=10, color=colors[i], weight='bold'),
                )

        fig.update_layout(
            title='分层净值曲线（十分位组累计净值，G1=最小  G10=最大）',
            xaxis_title='日期', yaxis_title='累计净值',
            height=480, margin=dict(l=40, r=140, t=40, b=60),
            legend=dict(orientation='v', x=1.02, y=1, xanchor='left', yanchor='top',
                       font=dict(size=9), itemsizing='constant'),
            hovermode='x unified',
            hoverlabel=dict(font_size=10),
        )
        fig.update_xaxes(tickangle=45, nticks=15)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    # ── 场景压力测试图表 ──

    def _market_cap_ic_plot(self, df: pd.DataFrame) -> str:
        """分市值 IC 柱状图。"""
        if df.empty:
            return ''
        fig = go.Figure()
        colors = ['#3b82f6', '#f59e0b', '#ef4444']
        for i, (_, row) in enumerate(df.iterrows()):
            fig.add_trace(go.Bar(
                name=row['bucket'],
                x=['IC 均值', 'ICIR', '多空年化%', '夏普'],
                y=[row.get('ic_mean', 0) * 100, row.get('icir', 0),
                   row.get('ls_ann_ret', 0), row.get('ls_sharpe', 0)],
                marker_color=colors[i % len(colors)],
                text=[f'{v:.3f}' for v in [row.get('ic_mean', 0) * 100, row.get('icir', 0),
                      row.get('ls_ann_ret', 0), row.get('ls_sharpe', 0)]],
                textposition='outside',
            ))
        fig.update_layout(
            title='分市值检验（大/中/小盘）',
            barmode='group',
            height=350, margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _regime_ic_plot(self, df: pd.DataFrame) -> str:
        """牛熊分段 IC 柱状图。"""
        if df.empty:
            return ''
        regime_order = ['牛市', '震荡', '熊市']
        df_plot = df.copy()
        df_plot['regime_order'] = df_plot['regime'].apply(lambda x: regime_order.index(x) if x in regime_order else 99)
        df_plot = df_plot.sort_values('regime_order')
        fig = make_subplots(rows=1, cols=3, subplot_titles=('IC 均值', 'ICIR', '多空年化收益 (%)'))
        for col_idx, metric in enumerate(['ic_mean', 'icir', 'ls_ann_ret'], 1):
            colors = ['#22c55e' if r == '牛市' else '#f97316' if r == '震荡' else '#ef4444' for r in df_plot['regime']]
            fig.add_trace(go.Bar(
                x=df_plot['regime'], y=df_plot[metric].values,
                marker_color=colors,
                text=[f'{v:.4f}' for v in df_plot[metric].values],
                textposition='outside',
            ), row=1, col=col_idx)
        fig.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40), showlegend=False)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _sector_ic_plot(self, df: pd.DataFrame) -> str:
        """分行业板块 IC 图。"""
        if df.empty:
            return ''
        fig = make_subplots(rows=1, cols=3, subplot_titles=('IC 均值', 'ICIR', '多空年化收益 (%)'))
        colors = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899']
        for col_idx, metric in enumerate(['ic_mean', 'icir', 'ls_ann_ret'], 1):
            marker_c = colors[:len(df)]
            fig.add_trace(go.Bar(
                x=df['sector'].values, y=df[metric].values,
                marker_color=marker_c,
                text=[f'{v:.4f}' for v in df[metric].values],
                textposition='outside',
            ), row=1, col=col_idx)
        fig.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40), showlegend=False)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _bivariate_plot(self, df: pd.DataFrame) -> str:
        """双变量分组柱状图。"""
        if df.empty:
            return ''
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'Q{g+1}' for g in df['primary_group']],
            y=(df['ls_return'].values * 100),
            marker_color=px.colors.sequential.Viridis[:len(df)],
            text=[f'{v:.4f}%' for v in df['ls_return'].values * 100],
            textposition='outside',
        ))
        fig.update_layout(
            title='双变量分组（先按市值分5组，再按因子二次分层）',
            xaxis_title='市值分组（从小到大）',
            yaxis_title='组内多空收益 (%)',
            height=350, margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _residual_summary_section(self, result: dict) -> str:
        """残差因子评测摘要。"""
        ic_s = result.get('residual_ic_stats', {})
        ls_s = result.get('residual_ls_stats', {})
        if not ic_s or 'ic_mean' not in ic_s:
            return ''
        control_cols = result.get('control_cols', [])
        html = f'''<div class="section">
  <h2>控制变量对冲二：残差因子评测</h2>
  <div style="padding:6px 0;">
    <p style="color:#64748b;font-size:13px;">
      控制变量: {" + ".join(control_cols)} — 每日截面回归取残差，剔除已知风险因子干扰，确认因子本源收益
    </p>
  </div>
  <table>
    <tr><th>指标</th><th>残差因子值</th></tr>
    <tr><td>IC 均值</td><td>{ic_s.get("ic_mean", 0):.6f}</td></tr>
    <tr><td>ICIR</td><td>{ic_s.get("icir", 0):.4f}</td></tr>
    <tr><td>胜率</td><td>{ic_s.get("win_rate", 0):.2%}</td></tr>
    <tr><td>多空年化收益</td><td>{ls_s.get("annual_return", 0):.2f}%</td></tr>
    <tr><td>多空夏普</td><td>{ls_s.get("sharpe", 0):.4f}</td></tr>
    <tr><td>单调性</td><td>{ls_s.get("monotonicity", 0):.4f}</td></tr>
  </table>
</div>'''
        return html

    def _size_neutral_section(self, result: dict) -> str:
        """规模分组检验（市值分布验证）。"""
        factor_stats = result.get('cap_group_factor_stats')
        cap_ic = result.get('cap_group_ic')
        if factor_stats is None or factor_stats.empty:
            return ''

        html = '<div class="section">\n  <h2>控制变量对冲三：规模分组检验（市值中性化验证）</h2>'
        html += '<div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">按市值分5组，查看因子均值分布——理想情况：各组因子均值接近0，验证中性化是否到位，避免分层收益只是市值带来的。</p></div>'
        html += '<table><tr><th>市值组</th><th>因子均值</th><th>因子标准差</th><th>标签均值</th><th>样本数</th></tr>'
        for _, row in factor_stats.iterrows():
            html += f'<tr><td>{row["cap_group"]}</td><td>{row["factor_mean"]:.6f}</td><td>{row["factor_std"]:.6f}</td><td>{row["label_mean"]:.6f}</td><td>{int(row["count"])}</td></tr>'
        html += '</table>'

        if cap_ic is not None and not cap_ic.empty:
            html += '<div style="margin-top:12px;"><table><tr><th>市值组</th><th>IC 均值</th><th>ICIR</th><th>有效天数</th></tr>'
            for _, row in cap_ic.iterrows():
                html += f'<tr><td>{row["cap_group"]}</td><td>{row["ic_mean"]:.6f}</td><td>{row["icir"]:.4f}</td><td>{int(row["n_days"])}</td></tr>'
            html += '</table></div>'

        html += '</div>'
        return html

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
        decile_nav: Optional[pd.DataFrame] = None,
        scenario_results: Optional[dict] = None,
        control_results: Optional[dict] = None,
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
    <span>评分体系: 10 维综合评分（新增: 场景稳健性 + 残差独立性）</span>
  </div>
</div>"""

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
  <h2>10 维评分详情（满分 100）</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;">"""
            dim_names = {
                "ic": ("IC 均值", 16), "icir": ("ICIR 稳定性", 12),
                "win_rate": ("胜率", 12), "ls_return": ("多空收益", 12),
                "ls_sharpe": ("多空夏普", 8), "decay": ("IC 衰减", 8),
                "turnover": ("换手率", 4), "coverage": ("覆盖率", 8),
                "scenario_robustness": ("场景稳健性*", 12),
                "residual_independence": ("残差独立性*", 8),
            }
            bar_colors = {"ic": "#2563eb", "icir": "#7c3aed", "win_rate": "#059669",
                          "ls_return": "#d97706", "ls_sharpe": "#dc2626",
                          "decay": "#8b5cf6", "turnover": "#f59e0b", "coverage": "#0ea5e9",
                          "scenario_robustness": "#06b6d4", "residual_independence": "#14b8a6"}
            for k, (label, weight) in dim_names.items():
                sc = dim_scores.get(k, 0)
                score_pct = sc * 100
                color = bar_colors.get(k, "#94a3b8")
                star = "*" if k in ("scenario_robustness", "residual_independence") else ""
                # 检测是否因数据缺失导致的默认中性分
                is_default = (k in ("scenario_robustness", "residual_independence") and abs(sc - 0.5) < 0.01)
                badge = '<span style="font-size:11px;color:#f59e0b;margin-left:4px;">⚠️ 无数据</span>' if is_default else ""
                bar_color = "#d1d5db" if is_default else color
                html += f"""<div style="background:#f8fafc;border-radius:8px;padding:12px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;">
        <span>{label}{star}{badge}</span>
        <span><strong>{sc:.2f}</strong> / {weight} 分 <span style="color:#94a3b8;font-size:11px;">({weight}%)</span></span>
      </div>
      <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-top:6px;">
        <div style="background:{bar_color};width:{score_pct:.0f}%;height:6px;border-radius:4px;"></div>
      </div>
    </div>"""
            html += f"""</div>
    <div style="margin-top:12px;font-size:13px;color:#64748b;padding:8px 12px;background:#f0f9ff;border-radius:8px;">
      <strong>* 新增维度</strong>：场景稳健性 = 跨市值/牛熊/行业的表现一致性打分；残差独立性 = 剔除市值/行业干扰后纯因子预测力。<br>
      ⚠️ <strong>无数据</strong> = 因输入数据不完整（缺少市值/行业/板块等信息），该维度使用 0.5 中性分替代实际评分。<br>
      <strong>判定标准</strong>：core = 0 项不达标；satellite = 综合分 ≥ {thresholds_info.get("satellite_min", 40) if thresholds_info else 40}；archive = 其余。
    </div>
    <div style="margin-top:8px;font-size:14px;text-align:center;padding:10px;background:#f0fdf4;border-radius:8px;">
      <strong>综合评分: {composite_score:.1f} / 100</strong> &nbsp;|&nbsp; 等级: <span class="status-badge {tier}">{tier}</span>
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

        # 分层净值曲线（10条分位组累计净值）
        if decile_nav is not None and not decile_nav.empty:
            html += f"""<div class="section">
  <h2>分层净值曲线</h2>
  <div style="padding:6px 0;">
    <p style="color:#64748b;font-size:13px;">
      十分位分组累计净值，G1=因子值最小组合，G10=因子值最大组合。
      理想状态：G10 >> G1 长期分化，且曲线走势平稳；若某段所有组交织在一起，说明因子阶段性失效。
    </p>
  </div>
  <div class="chart-container">{self._decile_nav_plot(decile_nav)}</div>
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

        # ── 场景压力测试：分市值IC ──
        if scenario_results:
            mc_df = scenario_results.get('market_cap_ic', pd.DataFrame())
            if isinstance(mc_df, pd.DataFrame) and not mc_df.empty:
                html += f'''<div class="section">
  <h2>场景压力测试一：分市值分组检验</h2>
  <div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">检验因子在大盘/中盘/小盘股中是否都有效，避免因子仅在小盘生效、大盘完全失效。</p></div>
  <div class="chart-container">{self._market_cap_ic_plot(mc_df)}</div>
  <div style="margin-top:12px">
    <table><tr><th>市值组</th><th>IC均值</th><th>ICIR</th><th>胜率</th><th>多空年化%</th><th>夏普</th><th>最大回撤%</th><th>单调性</th><th>天数</th></tr>'''
                for _, row in mc_df.iterrows():
                    html += f'<tr><td>{row["bucket"]}</td><td>{row["ic_mean"]:.6f}</td><td>{row["icir"]:.4f}</td><td>{row["win_rate"]:.2%}</td><td>{row["ls_ann_ret"]:.2f}</td><td>{row["ls_sharpe"]:.4f}</td><td>{row["max_drawdown"]:.2f}</td><td>{row["monotonicity"]:.4f}</td><td>{int(row["n_days"])}</td></tr>'''
                html += '</table></div></div>'

            regime_df = scenario_results.get('market_regime', pd.DataFrame())
            if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
                html += f'''<div class="section">
  <h2>场景压力测试二：牛熊/震荡市分段检验</h2>
  <div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">很多因子只在牛市有效，熊市持续回撤。按沪深300实际走势分段独立测算IC、分层收益。</p></div>
  <div class="chart-container">{self._regime_ic_plot(regime_df)}</div>
  <div style="margin-top:12px">
    <table><tr><th>行情</th><th>时段</th><th>IC均值</th><th>ICIR</th><th>胜率</th><th>多空年化%</th><th>夏普</th><th>最大回撤%</th><th>单调性</th><th>天数</th></tr>'''
                for _, row in regime_df.iterrows():
                    html += f'<tr><td>{row["regime"]}</td><td>{row.get("period","")}</td><td>{row["ic_mean"]:.6f}</td><td>{row["icir"]:.4f}</td><td>{row["win_rate"]:.2%}</td><td>{row["ls_ann_ret"]:.2f}</td><td>{row["ls_sharpe"]:.4f}</td><td>{row["max_drawdown"]:.2f}</td><td>{row["monotonicity"]:.4f}</td><td>{int(row["n_days"])}</td></tr>'''
                html += '</table></div></div>'

            sector_df = scenario_results.get('industry_sector', pd.DataFrame())
            if isinstance(sector_df, pd.DataFrame) and not sector_df.empty:
                html += f'''<div class="section">
  <h2>场景压力测试三：分行业板块检验</h2>
  <div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">周期/消费/制造/金融/科技五大板块独立评测，验证因子是否全行业通用。</p></div>
  <div class="chart-container">{self._sector_ic_plot(sector_df)}</div>
  <div style="margin-top:12px">
    <table><tr><th>板块</th><th>IC均值</th><th>ICIR</th><th>胜率</th><th>多空年化%</th><th>夏普</th><th>最大回撤%</th><th>单调性</th><th>股票数</th></tr>'''
                for _, row in sector_df.iterrows():
                    html += f'<tr><td>{row["sector"]}</td><td>{row["ic_mean"]:.6f}</td><td>{row["icir"]:.4f}</td><td>{row["win_rate"]:.2%}</td><td>{row["ls_ann_ret"]:.2f}</td><td>{row["ls_sharpe"]:.4f}</td><td>{row["max_drawdown"]:.2f}</td><td>{row["monotonicity"]:.4f}</td><td>{int(row.get("n_stocks",0))}</td></tr>'''
                html += '</table></div></div>'

        # ── 控制变量对冲 ──
        if control_results:
            bv_df = control_results.get('bivariate', pd.DataFrame())
            if isinstance(bv_df, pd.DataFrame) and not bv_df.empty:
                html += f'''<div class="section">
  <h2>控制变量对冲一：双变量分组（剔除市值干扰）</h2>
  <div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">先按市值分5组，再在每个市值组内按因子分5组，检验因子在控制市值后是否仍有独立预测力。</p></div>
  <div class="chart-container">{self._bivariate_plot(bv_df)}</div>
  <div style="margin-top:12px">
    <table><tr><th>市值组</th><th>Q1收益</th><th>Q5收益</th><th>组内多空收益</th><th>天数</th></tr>'''
                for _, row in bv_df.iterrows():
                    html += f'<tr><td>Q{int(row["primary_group"])+1}</td><td>{float(row["q0_mean"]):.6f}</td><td>{float(row.get("q4_mean", row["qN_mean"])):.6f}</td><td>{float(row["ls_return"]):.6f}</td><td>{int(row["n_days"])}</td></tr>'
                html += '</table></div></div>'

            residual_res = control_results.get('residual', {})
            if residual_res:
                html += self._residual_summary_section(residual_res)

            size_res = control_results.get('size_neutral', {})
            if size_res:
                html += self._size_neutral_section(size_res)

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
