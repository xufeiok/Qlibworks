import glob, re
files = sorted(glob.glob(r'e:\Quant\Qlibworks\factor_data\reports\watch\STR_20d*.html'))
with open(files[-1], 'r', encoding='utf-8') as f:
    c = f.read()

print(f'最新报告: {files[-1].split(chr(92))[-1]}')
print()

# 找 decile_nav trace 数据
y_arrays = []
for m in re.finditer(r'"y":\[', c):
    start = m.end()
    end_idx = c.find(']', start)
    if end_idx > 0:
        y_content = c[start:end_idx]
        vals = y_content.split(',')
        if len(vals) > 1000:
            # 找name
            before = c[max(0,start-200):start]
            name_m = re.search(r'"name":"([^"]+)"', before)
            name = name_m.group(1) if name_m else 'unknown'
            last_vals = [float(v) for v in vals[-200:]]
            zeros = sum(1 for v in last_vals if abs(v) < 0.0001)
            y_arrays.append((name, len(vals), zeros))

print(f'长时间序列(>1000): {len(y_arrays)} 条')
for name, n, zeros in y_arrays:
    print(f'{name:>5s}: len={n}, 最后200中0={zeros}/200')
print()

# 找 ICIR_NW
idx = c.find('ICIR_NW')
print(f'ICIR_NW: {c[idx:idx+60]}' if idx>=0 else 'ICIR_NW: 未找到')
