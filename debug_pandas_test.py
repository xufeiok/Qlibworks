"""Quick pandas column type test"""
import pandas as pd
import numpy as np

# Simulate warehouse_df
idx = pd.MultiIndex.from_tuples([("2020-01-01", "000001.sz"), ("2020-01-02", "000001.sz")], names=["datetime", "instrument"])
df = pd.DataFrame({"STR_20d": [0.1, 0.2], "STR_5d": [0.3, 0.4], "MOM_6m": [0.5, 0.6]}, index=idx)

print(f"Before injection:")
print(f"  type: {type(df.columns)}")
print(f"  columns: {df.columns.tolist()}")
print(f"  is MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")

# Simulate label injection
df[("label", "LABEL_5D")] = [0.7, 0.8]

print(f"\nAfter injection:")
print(f"  type: {type(df.columns)}")
print(f"  columns: {df.columns.tolist()}")
print(f"  is MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")

# Test get_loc
print(f"\nget_loc('STR_20d'): ", end="")
try:
    print(df.columns.get_loc("STR_20d"))
except Exception as e:
    print(f"Error: {e}")

print(f"get_loc('label'): ", end="")
try:
    result = df.columns.get_loc("label")
    print(f"Found: {result}")
except Exception as e:
    print(f"KeyError: {e}")

print(f"\nget_group_columns(df, 'label') with DropnaLabel logic")
try:
    from qlib.data.dataset.processor import get_group_columns
    cols = get_group_columns(df, "label")
    print(f"  Found columns: {cols}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test: can we access LABEL_5D column?
print(f"\nAccess df[('label', 'LABEL_5D')]: ", end="")
try:
    print(f"OK: shape={df[('label', 'LABEL_5D')].shape}")
except Exception as e:
    print(f"Error: {e}")

print(f"Access df['LABEL_5D']: ", end="")
try:
    print(f"OK: shape={df['LABEL_5D'].shape}")
except Exception as e:
    print(f"KeyError: {e}")
