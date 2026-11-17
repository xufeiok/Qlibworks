import os

features_dir = r'e:\Quant\Qlibworks\qlib_data\features'
instruments_dir = r'e:\Quant\Qlibworks\qlib_data\instruments'

# Get all instrument names from features directory
# They are like 000001.sz, we need SH000001 or 000001.SZ or just 000001
# Wait, let's see exactly what's in features_dir
insts = []
for f in os.listdir(features_dir):
    if f.endswith('.sz') or f.endswith('.sh') or f.endswith('.bj'):
        insts.append(f)

# Let's save them exactly as they are in features_dir, e.g., 000001.sz
# Wait, earlier test with `000963.SZ` worked! But `000963` didn't. So Qlib expects the exact filename or uppercase?
# Windows is case-insensitive, so 000001.SZ will match 000001.sz.
insts_upper = [f.upper() for f in insts]

with open(os.path.join(instruments_dir, 'all.txt'), 'w') as f:
    for inst in insts_upper:
        f.write(f"{inst}\t2010-01-01\t2099-12-31\n")

# Copy to csi500.txt for testing
with open(os.path.join(instruments_dir, 'csi500.txt'), 'w') as f:
    for inst in insts_upper:
        f.write(f"{inst}\t2010-01-01\t2099-12-31\n")

print(f"Restored {len(insts_upper)} instruments to all.txt and csi500.txt")
