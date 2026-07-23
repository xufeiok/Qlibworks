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
    """使用 CDN 加载 Plotly.js，避免 3MB 内联脚本导致 IDE 预览卡死。

    使用 cdn.bootcdn.net（国内可用）替代 cdn.plot.ly（国内被墙）。
    """
    return '<script src="https://cdn.bootcdn.net/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>'


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
        """IC 时间序列（柱状图）+ 累计 IC（折线图）。

        始终使用 Bar 展示每日 IC，确保围绕 0 轴分布可见。
        所有 numpy 数组显式转为 Python list，避免 Plotly 6.5.0 的二进制序列化。
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("每日 IC 值", "累计 IC"),
        )

        dates = ic_series.index
        if hasattr(dates, "str"):
            dates = [str(d)[:10] for d in dates]
        vals = ic_series.values.tolist()  # numpy → list

        # 始终使用 Bar，密集时自动压缩柱宽
        colors = ["red" if v > 0 else "green" for v in vals]
        fig.add_trace(
            go.Bar(x=dates, y=vals, name="IC",
                   marker_color=colors, marker_line_width=0),
            row=1, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        cumic = ic_series.cumsum().values.tolist()  # numpy → list
        fig.add_trace(
            go.Scatter(x=dates, y=cumic, mode="lines",
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
            x=dates, y=cum_series.values.tolist(), mode="lines",  # numpy→list
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
        import math as _m
        vals = [v for v in ic_series.values if not (isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)))]
        fig = go.Figure()
        if not vals:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="IC 数据全部为 NaN/Inf，无法绘制分布图",
                showarrow=False, font=dict(size=14, color="#94a3b8"),
            )
        else:
            fig.add_trace(go.Histogram(
                x=vals, nbinsx=40,
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
        """IC 衰减折线图。

        如果所有 IC 值都接近 0，会自动调整 y 轴范围使其可见。
        如果数据完全无效，显示提示文本。
        """
        import math as _m
        x = decay_df['horizon'].values
        ic_mean = [0.0 if (isinstance(v, float) and (_m.isnan(v) or _m.isinf(v))) else float(v) for v in decay_df['ic_mean'].values]
        icir = [0.0 if (isinstance(v, float) and (_m.isnan(v) or _m.isinf(v))) else float(v) for v in decay_df['icir'].values]

        fig = go.Figure()

        # 检查是否所有 IC 均为 0
        all_zero = all(abs(v) < 1e-10 for v in ic_mean) and all(abs(v) < 1e-10 for v in icir)
        if all_zero:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.5, y=0.5,
                text="IC 衰减数据全部为 0（可能因持有期标签导致各 horizon IC 均失效）",
                showarrow=False, font=dict(size=14, color="#94a3b8"),
            )

        fig.add_trace(go.Scatter(
            x=x.tolist(), y=ic_mean,  # numpy→list
            mode='lines+markers', name='IC 均值',
            line=dict(color='#8b5cf6', width=2), marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=icir,  # numpy→list
            mode='lines+markers', name='ICIR',
            line=dict(color='#f97316', width=2, dash='dash'), marker=dict(size=8),
            yaxis='y2',
        ))

        # 强制 y 轴包含数据范围
        y1_min, y1_max = min(ic_mean), max(ic_mean)
        y2_min, y2_max = min(icir), max(icir)
        if not all_zero:
            y1_pad = max(abs(y1_max - y1_min) * 0.2, 0.001) if y1_max != y1_min else 0.001
            y2_pad = max(abs(y2_max - y2_min) * 0.2, 0.001) if y2_max != y2_min else 0.001
            y1_range = [y1_min - y1_pad, y1_max + y1_pad]
            y2_range = [y2_min - y2_pad, y2_max + y2_pad]
        else:
            y1_range = [-0.01, 0.01]
            y2_range = [-0.01, 0.01]

        fig.update_layout(
            title='IC 衰减分析',
            xaxis_title='预测期（天）',
            xaxis=dict(tickmode='array', tickvals=x.tolist()),
            yaxis=dict(title='IC 均值', range=y1_range),
            yaxis2=dict(title='ICIR', overlaying='y', side='right', range=y2_range),
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
            width=0.5,
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
        G1（因子最小）→ 橙色/红，G10（因子最大）→ 蓝/紫。

        如果分组数 < 2 或各组净值完全相同，显示提示信息。
        """
        if decile_nav.empty:
            return ''

        n_groups = len(decile_nav.columns)
        if n_groups < 2:
            return f'<div style="padding:20px;text-align:center;color:#94a3b8;">分层组数不足 ({n_groups} 组)，无法绘制净值曲线</div>'

        fig = go.Figure()

        # 日期索引转字符串（numpy → list）
        dates = decile_nav.index
        if hasattr(dates, "str"):
            dates = [str(d)[:10] for d in dates]
        else:
            dates = list(dates)

        # 10 种高区分度颜色
        colors = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#388e3c',
                  '#1976d2', '#1565c0', '#6a1b9a', '#c2185b', '#e91e63']

        # 检查是否所有组终值几乎相同
        final_vals = decile_nav.iloc[-1].values
        val_range = float(final_vals.max() - final_vals.min())

        # 收集各组 y 数据为 Python list，避免 Plotly 二进制序列化
        y_data = {}
        for col in decile_nav.columns:
            y_data[col] = decile_nav[col].values.tolist()

        for i, col in enumerate(decile_nav.columns):
            is_top = (i >= n_groups - 2)
            is_bottom = (i <= 1)
            fig.add_trace(go.Scatter(
                x=dates,
                y=y_data[col],  # Python list
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=3 if is_top or is_bottom else 1.5),
                opacity=0.95 if is_top or is_bottom else 0.7,
            ))

        # 标注首尾两组的终值
        last_date = dates[-1]
        for i, col in enumerate(decile_nav.columns):
            if i == 0 or i == n_groups - 1:
                val = float(decile_nav[col].iloc[-1])
                fig.add_annotation(
                    x=last_date, y=val,
                    text=f"{col}={val:.3f}",
                    showarrow=False, xanchor='left',
                    font=dict(size=10, color=colors[i], weight='bold'),
                )

        # 如果终值差异极小，添加说明
        note = ''
        if val_range < 0.01:
            note = ' [⚠ 各组终值差异极小，因子分层效果微弱]'

        fig.update_layout(
            title=f'分层净值曲线（十分位组累计净值，G1=最小  G10=最大）{note}',
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
        """分市值 IC 柱状图：按市值分组（大/中/小盘），每个指标一张子图。"""
        if df.empty:
            return ''
        metrics = [
            ('ic_mean', 'IC 均值'),
            ('icir', 'ICIR'),
            ('ls_ann_ret', '多空年化%'),
            ('ls_sharpe', '夏普'),
        ]
        fig = make_subplots(rows=1, cols=4, subplot_titles=[m[1] for m in metrics])
        colors = ['#3b82f6', '#f59e0b', '#ef4444']
        for col_idx, (metric, _) in enumerate(metrics, 1):
            import math as _m
            vals = []
            for v in df[metric].values:
                if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
                    vals.append(0.0)
                else:
                    vals.append(float(v))
            fig.add_trace(go.Bar(
                x=df['bucket'].values.tolist(),
                y=vals,
                marker_color=colors[:len(df)],
                text=[f'{v:.4f}' for v in vals],
                textposition='outside',
            ), row=1, col=col_idx)
        fig.update_layout(
            title='分市值检验（大/中/小盘）',
            height=350, margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _regime_ic_plot(self, df: pd.DataFrame) -> str:
        """牛熊分段 IC 柱状图。"""
        if df.empty:
            return ''
        import math as _m
        regime_order = ['牛市', '震荡', '熊市']
        df_plot = df.copy()
        df_plot['regime_order'] = df_plot['regime'].apply(lambda x: regime_order.index(x) if x in regime_order else 99)
        df_plot = df_plot.sort_values('regime_order')
        fig = make_subplots(rows=1, cols=3, subplot_titles=('IC 均值', 'ICIR', '多空年化收益 (%)'))
        for col_idx, metric in enumerate(['ic_mean', 'icir', 'ls_ann_ret'], 1):
            colors = ['#22c55e' if r == '牛市' else '#f97316' if r == '震荡' else '#ef4444' for r in df_plot['regime']]
            vals = [0.0 if (isinstance(v, float) and (_m.isnan(v) or _m.isinf(v))) else float(v) for v in df_plot[metric].values]
            fig.add_trace(go.Bar(
                x=list(df_plot['regime']), y=vals,  # numpy→list
                marker_color=colors,
                text=[f'{v:.4f}' for v in vals],
                textposition='outside',
            ), row=1, col=col_idx)
        fig.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40), showlegend=False)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _sector_ic_plot(self, df: pd.DataFrame) -> str:
        """分行业板块 IC 图。"""
        if df.empty:
            return ''
        import math as _m
        fig = make_subplots(rows=1, cols=3, subplot_titles=('IC 均值', 'ICIR', '多空年化收益 (%)'))
        colors = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899']
        for col_idx, metric in enumerate(['ic_mean', 'icir', 'ls_ann_ret'], 1):
            marker_c = colors[:len(df)]
            vals = [0.0 if (isinstance(v, float) and (_m.isnan(v) or _m.isinf(v))) else float(v) for v in df[metric].values]
            fig.add_trace(go.Bar(
                x=df['sector'].values.tolist(), y=vals,
                marker_color=marker_c,
                text=[f'{v:.4f}' for v in vals],
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
            y=(df['ls_return'].values * 100).tolist(),
            marker_color=px.colors.sequential.Viridis[:len(df)],
            text=[f'{v:.4f}%' for v in (df['ls_return'].values * 100).tolist()],
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
        # 新增：时序统计检验、风险分析、边际检验
        statistical_tests: Optional[dict] = None,
        ic_half_life: Optional[dict] = None,
        rolling_ic_stability: Optional[dict] = None,
        risk_metrics: Optional[dict] = None,
        q1_q10_significance: Optional[dict] = None,
    ) -> str:
        """生成完整 HTML 报告。"""
        ic_series = ic_stats.get("ic_series", pd.Series())
        ls_cum = ls_stats.get("cumulative", pd.Series())

        # ── 诊断信息（HTML 注释，不展示） ──
        _diag_lines = []
        _diag_lines.append(f"decile_nav: shape={decile_nav.shape if decile_nav is not None else 'None'}, cols={list(decile_nav.columns) if decile_nav is not None and not decile_nav.empty else '[]'}")
        _diag_lines.append(f"decay_df: shape={decay_df.shape if decay_df is not None else 'None'}, ic_mean={decay_df['ic_mean'].tolist() if decay_df is not None and not decay_df.empty else '[]'}")
        _diag_lines.append(f"ic_series: len={len(ic_series)}, mean={ic_series.mean():.6f}, first_5={ic_series.head(5).tolist() if not ic_series.empty else '[]'}")
        _diag_lines.append(f"circ_mv_in_data: {(decile_nav is not None and 'circ_mv' in str(decile_nav.columns)) if False else 'N/A'}")
        html_debug = "<!--\n" + "\n".join(_diag_lines) + "\n-->"

        tier = (qual_result or {}).get("tier", "N/A")
        composite_score = (qual_result or {}).get("composite_score", 0)
        reasons = (qual_result or {}).get("reasons", [])
        core_failures = (qual_result or {}).get("core_failures", [])
        characteristic_notes = (qual_result or {}).get("characteristic_notes", [])
        recommendation = (qual_result or {}).get("recommendation", "N/A")
        recommendation_detail = (qual_result or {}).get("recommendation_detail", "")
        dim_scores = (qual_result or {}).get("scores", {})

        # 构建核心指标
        summary_rows = [
            ("IC 均值", f"{ic_stats.get('ic_mean', 0):.6f}"),
            ("IC 标准差", f"{ic_stats.get('ic_std', 0):.6f}"),
            ("ICIR（年化）", f"{ic_stats.get('icir', 0):.4f}"),
            ("ICIR_NW（自相关修正）", f"{ic_stats.get('icir_nw', 0):.4f}"),
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

        summary_rows.append(("综合评分", f"{composite_score:.1f}"))
        summary_rows.append(("等级", tier))
        summary_rows.append(("推荐结论", recommendation))

        # 开始组装 HTML
        html = f"""<!DOCTYPE html>
<!-- DIAGNOSTICS BELOW -->
{html_debug}
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>单因子评测报告 — {self.factor_name}</title>
{_load_plotly_js()}
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
.status-badge.watch {{ background: #e2e8f0; color: #64748b; }}
.status-badge.reject {{ background: #fee2e2; color: #991b1b; }}
.recommend-tag {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.recommend-tag.strong {{ background: #d1fae5; color: #065f46; }}
.recommend-tag.ok {{ background: #fef3c7; color: #92400e; }}
.recommend-tag.watch {{ background: #f1f5f9; color: #64748b; }}
.recommend-tag.reject {{ background: #fee2e2; color: #991b1b; }}
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
    &nbsp;|&nbsp; 推荐: <span class="recommend-tag {'strong' if tier=='core' else 'ok' if tier=='satellite' else 'watch' if tier=='watch' else 'reject'}">{recommendation}</span>
  </div>
  <div class="header-meta" style="margin-top:12px;font-size:13px;opacity:0.9;display:flex;flex-wrap:wrap;gap:8px 20px;">
    <span>评测区间: {eval_period.get("start", "N/A") if eval_period else "N/A"} ~ {eval_period.get("end", "N/A") if eval_period else "N/A"}</span>
    <span>标签计算: {label_expr if label_expr else "N/A"}</span>
    <span>评分体系: 10 维综合评分 + 时序统计检验 + IC 质量 + 尾部风险 + 显著性检验</span>
    <span style="color:#fef3c7;">{recommendation_detail}</span>
  </div>
</div>"""

        # 核心指标卡片
        html += """<div class="section">
  <h2>核心指标概览</h2>
  <div class="summary-grid">"""
        for label, val in summary_rows:
            css = ""
            if label == "等级":
                css = "pass" if val in ("core", "satellite") else "fail" if val == "reject" else ""
            elif label == "推荐结论":
                css = "pass" if "推荐" in val and "不" not in val else "fail" if "不推荐" in val else ""
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
                weighted = sc * weight
                html += f"""<div style="background:#f8fafc;border-radius:8px;padding:12px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;">
        <span>{label}{star}{badge}</span>
        <span><strong>{weighted:.2f}</strong> / {weight} 分 <span style="color:#94a3b8;font-size:11px;">(归一分 {sc:.2f} × {weight}%)</span></span>
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

        # ── 时序统计检验板块（ADF + KPSS + Ljung-Box） ──
        if statistical_tests:
            adf = statistical_tests.get("adf", {})
            kpss_res = statistical_tests.get("kpss", {})
            lb = statistical_tests.get("ljungbox", {})
            verdict = statistical_tests.get("stationarity_verdict", "N/A")
            html += f"""<div class="section">
  <h2>时序统计检验</h2>
  <div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">
    因子截面均值的时序分析。ADF + KPSS 联合判定序列平稳性；Ljung-Box 检验是否存在自相关结构（白噪声检验）。
    对反转因子：稳定平稳 + 非白噪声 → 因子存在可预测的自相关结构。
  </p></div>
  <table>
    <tr><th>检验</th><th>统计量</th><th>P 值</th><th>结论</th></tr>
    <tr>
      <td>ADF 单位根检验</td>
      <td>{adf.get("adf_stat", "N/A")}</td>
      <td>{adf.get("p_value", "N/A")}</td>
      <td><span style="color:{'#059669' if adf.get('is_stationary') else '#dc2626'};">{'平稳' if adf.get('is_stationary') else '非平稳'}</span></td>
    </tr>
    <tr>
      <td>KPSS 平稳性检验</td>
      <td>{kpss_res.get("kpss_stat", "N/A")}</td>
      <td>{kpss_res.get("p_value", "N/A")}</td>
      <td><span style="color:{'#059669' if kpss_res.get('is_stationary') else '#dc2626'};">{'平稳' if kpss_res.get('is_stationary') else '非平稳'}</span></td>
    </tr>
    <tr>
      <td>Ljung-Box 白噪声检验</td>
      <td>{lb.get("lb_stat", "N/A")}</td>
      <td>{lb.get("lb_pvalue", "N/A")}</td>
      <td><span style="color:{'#dc2626' if not lb.get('is_white_noise') else '#059669'};">{'非白噪声（有自相关结构）' if not lb.get('is_white_noise') else '白噪声'}</span></td>
    </tr>
    <tr>
      <td colspan="4" style="text-align:center;font-weight:600;">
        综合平稳性判定: <span style="color:{'#059669' if verdict in ('平稳',) else '#d97706'};">{verdict}</span>
        &nbsp;|&nbsp; 样本量: {adf.get("n_obs", 0)} 个交易日
      </td>
    </tr>
  </table>
</div>"""

        # ── IC 半衰期 + 滚动 IC 稳定性 ──
        if ic_half_life or rolling_ic_stability:
            hl = ic_half_life or {}
            rs = rolling_ic_stability or {}
            hl_days = hl.get("half_life_days")
            hl_display = f"{hl_days} 天" if hl_days is not None else "N/A"
            hl_color = "#059669" if (hl_days and hl_days <= 60) else "#d97706" if hl_days else "#94a3b8"
            html += f"""<div class="section">
  <h2>IC 质量分析</h2>
  <table>
    <tr><th>指标</th><th>数值</th><th>说明</th></tr>
    <tr>
      <td>IC 半衰期</td>
      <td style="color:{hl_color};font-weight:600;">{hl_display}</td>
      <td>IC 自相关衰减到 50% 所需天数。越短 → 调仓频率需越高；越长 → 信号越持久</td>
    </tr>
    <tr>
      <td>IC 衰减率 (lag-1 ACF)</td>
      <td>{hl.get("decay_rate", "N/A")}</td>
      <td>IC 序列的一阶自相关系数</td>
    </tr>
    <tr>
      <td>滚动 ICIR 均值</td>
      <td>{rs.get("rolling_icir_mean", "N/A")}</td>
      <td>252 日滚动 ICIR 的均值（越高越好）</td>
    </tr>
    <tr>
      <td>滚动 ICIR 标准差</td>
      <td style="color:{(rs.get('rolling_icir_std') or 0) < 2 and '#059669' or (rs.get('rolling_icir_std') or 0) < 4 and '#d97706' or '#dc2626'};">{rs.get("rolling_icir_std", "N/A")}</td>
      <td>滚动 ICIR 的标准差（越低越稳定）</td>
    </tr>
    <tr>
      <td>滚动 IC 波动率</td>
      <td>{rs.get("ic_rolling_vol", "N/A")}</td>
      <td>滚动窗口内 IC 标准差的均值</td>
    </tr>
  </table>
</div>"""

        # ── VaR/CVaR 尾部风险 ──
        if risk_metrics:
            var_95 = risk_metrics.get("var_95", 0)
            cvar_95 = risk_metrics.get("cvar_95", 0)
            var_99 = risk_metrics.get("var_99", 0)
            cvar_99 = risk_metrics.get("cvar_99", 0)
            mdd = risk_metrics.get("max_drawdown", 0)
            mdd_dur = risk_metrics.get("max_drawdown_duration", 0)
            html += f"""<div class="section">
  <h2>尾部风险分析</h2>
  <div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">
    多空组合日收益的风险度量。VaR 衡量正常市场下的最大损失；CVaR 衡量极端情况下的平均损失。
  </p></div>
  <table>
    <tr><th>指标</th><th>数值</th></tr>
    <tr><td>VaR (95%)</td><td>{var_95:.6f}</td></tr>
    <tr><td>CVaR (95%)</td><td>{cvar_95:.6f}</td></tr>
    <tr><td>VaR (99%)</td><td>{var_99:.6f}</td></tr>
    <tr><td>CVaR (99%)</td><td>{cvar_99:.6f}</td></tr>
    <tr><td>最大回撤</td><td style="color:#dc2626;">{mdd:.2%}</td></tr>
    <tr><td>最长回撤持续期</td><td>{mdd_dur} 个交易日</td></tr>
    <tr><td colspan="2" style="text-align:center;font-size:12px;color:#94a3b8;">参考: VaR/CVaR 为负值代表损失；正值代表收益（多空组合整体为正时）</td></tr>
  </table>
</div>"""

        # ── Q1 vs Q10 差异显著性 ──
        if q1_q10_significance:
            t_sig = q1_q10_significance.get("t_significant", False)
            mw_sig = q1_q10_significance.get("mw_significant", False)
            t_p = q1_q10_significance.get("t_pvalue", 1)
            mw_p = q1_q10_significance.get("mw_pvalue", 1)
            diff = q1_q10_significance.get("diff_mean", 0)
            html += f"""<div class="section">
  <h2>Q1 vs Q10 差异显著性检验</h2>
  <div style="padding:6px 0;"><p style="color:#64748b;font-size:13px;">
    Q1（最低因子分组）与 Q10（最高因子分组）的收益差异是否统计上显著。
    若两种检验都显著 → 因子的分层能力不只是随机噪声。
  </p></div>
  <table>
    <tr><th>检验方法</th><th>统计量</th><th>P 值</th><th>是否显著</th></tr>
    <tr>
      <td>Welch's t-test（均值差异）</td>
      <td>{q1_q10_significance.get("t_stat", "N/A")}</td>
      <td>{t_p}</td>
      <td><span style="color:{'#059669' if t_sig else '#94a3b8'};">{'显著 (p<0.05)' if t_sig else '不显著'}</span></td>
    </tr>
    <tr>
      <td>Mann-Whitney U 检验（非参数）</td>
      <td>{q1_q10_significance.get("mw_stat", "N/A")}</td>
      <td>{mw_p}</td>
      <td><span style="color:{'#059669' if mw_sig else '#94a3b8'};">{'显著 (p<0.05)' if mw_sig else '不显著'}</span></td>
    </tr>
    <tr>
      <td colspan="4" style="text-align:center;">
        Q1 均值: {q1_q10_significance.get("q1_mean", 0):.6f} &nbsp;|&nbsp;
        Q10 均值: {q1_q10_significance.get("q10_mean", 0):.6f} &nbsp;|&nbsp;
        Q1 - Q10 差异: <span style="color:{'#059669' if diff < 0 else '#dc2626'};font-weight:600;">{diff:.6f}</span>
        &nbsp;|&nbsp; Q1 样本: {q1_q10_significance.get("q1_n", 0)} &nbsp; Q10 样本: {q1_q10_significance.get("q10_n", 0)}
      </td>
    </tr>
  </table>
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
    <p>
      <strong>等级: {tier}</strong> （综合评分: {composite_score:.1f}）&nbsp;|&nbsp;
      推荐: <span class="recommend-tag {'strong' if tier=='core' else 'ok' if tier=='satellite' else 'watch' if tier=='watch' else 'reject'}">{recommendation}</span>
      &nbsp;—&nbsp; {recommendation_detail}
    </p>"""
            if core_failures:
                html += f"""    <p style="margin-top:10px;color:#dc2626;font-weight:600;">🔴 核心缺陷（否决项）:</p>
    <ul style="margin-left:20px;color:#dc2626;">"""
                for r in core_failures:
                    html += f"<li style='margin-top:4px;'>{r}</li>"
                html += """</ul>"""
            if characteristic_notes:
                html += f"""    <p style="margin-top:10px;color:#94a3b8;font-weight:600;">⚪ 特征提示（不影响入库判断）:</p>
    <ul style="margin-left:20px;color:#94a3b8;">"""
                for r in characteristic_notes:
                    html += f"<li style='margin-top:4px;'>{r}</li>"
                html += """</ul>"""
            if not core_failures and not characteristic_notes:
                html += """    <p style="margin-top:8px;color:#059669;font-weight:600;">✓ 所有维度均达标，无否决项</p>"""
            html += """
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
