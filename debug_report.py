# -*- coding: utf-8 -*-
"""解析报告中的plotly图表，检查各图表数据是否正确。"""
import json, re, sys
from collections import Counter

with open('factor_data/reports/watch/STR_20d_2018-01-01_2025-12-31_20260704_234537.html', 'r', encoding='utf-8') as f:
    content = f.read()

sys.stdout.reconfigure(encoding='utf-8')

# 找到所有 plotly-graph-div 的 data-plotly-json
# 格式为: <div class="plotly-graph-div" ... data-plotly-json='{"...":...}'>
matches = re.findall(r'class="plotly-graph-div".*?data-plotly-json=\'(.+?)\'', content)
print(f'发现 {len(matches)} 个 plotly 图表\n')

for i, raw in enumerate(matches):
    try:
        d = json.loads(raw)
        # Plotly JSON format: {config:..., layout:..., traces:[...]}
        traces = d.get('traces', d.get('data', []))
        layout = d.get('layout', {})
        title = ''
        if isinstance(layout, dict):
            title = layout.get('title', {})
            if isinstance(title, dict):
                title = title.get('text', '')
        
        fig_type = 'make_subplots' if layout.get('grid') else '单图'
        
        total_pts = 0
        for t in traces:
            if isinstance(t, dict):
                y = t.get('y', [])
                if isinstance(y, list):
                    total_pts += len(y)
        
        n_traces = len(traces)
        print(f'【图表{i+1}】title="{title[:30]}" | {fig_type} | {n_traces}条轨迹 | {total_pts}个数据点')
        
        for j, t in enumerate(traces):
            if isinstance(t, dict):
                ttype = t.get('type', '?')
                name = t.get('name', f'trace{j}')
                y = t.get('y', [])
                x = t.get('x', [])
                ylen = len(y) if isinstance(y, list) else '?'
                xlen = len(x) if isinstance(x, list) else '?'
                has_nan = sum(1 for v in y if isinstance(v, float) and (v != v)) if isinstance(y, list) else 0  # NaN check
                print(f'    #{j} "{name}" type={ttype} x={xlen}点 y={ylen}点 nan={has_nan}')
                
                # 检查 y 值是否全为 0
                if isinstance(y, list) and len(y) > 0:
                    y_vals = [v for v in y if isinstance(v, (int, float))]
                    if len(y_vals) > 0:
                        y_min, y_max = min(y_vals), max(y_vals)
                        print(f'      y范围: [{y_min:.6f}, {y_max:.6f}]')
    except Exception as e:
        print(f'  图表{i+1}: 解析失败 - {e}')

# 单独检查 decile_nav 曲线条数
g_names = set(re.findall(r'"name":\s*"G(\d+)"', content))
print(f'\n分层净值曲线中的分组: {len(g_names)}条 (G{", G".join(sorted(g_names))})')
