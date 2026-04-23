import os
import yaml
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import qlib
from qlib.data import D

# Initialize Qlib
qlib.init(provider_uri="E:/Quant/Qlibworks/qlib_data", region="cn")

repo_path = r'e:\Quant\Qlibworks\factors_repo'
files_to_check = [
    'style_factors.yaml',
    'quality_factors.yaml',
    'price_volume_factors.yaml',
    'sentiment_factors.yaml',
    'risk_factors.yaml',
    'other_factors.yaml'
]

test_instrument = 'SH600000'
start_time = '2020-01-02'
end_time = '2020-01-10'

def check_expression(expr):
    try:
        # Check if expr contains anything Qlib can evaluate
        df = D.features([test_instrument], [expr], start_time=start_time, end_time=end_time)
        return True
    except Exception as e:
        return False

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)

def reformat_and_save(file_path, original_data, valid_factors):
    # Group factors by category
    categories = {}
    for factor in valid_factors:
        cat = factor.get('category', 'Uncategorized')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(factor)

    # 1. Dump the header part (everything except 'factors')
    header_data = {k: v for k, v in original_data.items() if k != 'factors'}
    header_str = yaml.dump(header_data, allow_unicode=True, sort_keys=False, width=1000)
    
    # 2. Build factors string
    factors_str = "factors:\n"
    
    for c_idx, (cat, f_list) in enumerate(categories.items()):
        for f_idx, f in enumerate(f_list):
            f_str = yaml.dump(f, allow_unicode=True, sort_keys=False, width=1000)
            # Replace first line with "  - " and indent rest with "    "
            lines = f_str.strip().split('\n')
            formatted_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    formatted_lines.append("  - " + line)
                else:
                    formatted_lines.append("    " + line)
            
            factors_str += '\n'.join(formatted_lines) + "\n\n" # 1 empty line between factors
            
        if c_idx < len(categories) - 1:
            # 3 empty lines between categories (since we already added \n\n, add two more \n)
            factors_str += "\n\n"
            
    with open(file_path, 'w', encoding='utf-8') as out_f:
        out_f.write(header_str + factors_str.rstrip() + "\n")

def main():
    for fname in files_to_check:
        fpath = os.path.join(repo_path, fname)
        if not os.path.exists(fpath):
            continue
            
        print(f"Processing {fname}...")
        with open(fpath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if not data or not data.get('factors'):
            continue
            
        valid_factors = []
        for factor in data['factors']:
            expr = factor.get('expression', {}).get('qlib', '')
            name = factor.get('name', 'Unknown')
            if not expr:
                print(f"  [-] Skip {name}: No qlib expression")
                continue
                
            is_valid = check_expression(expr)
            if is_valid:
                valid_factors.append(factor)
            else:
                print(f"  [X] Removed {name} due to calculation failure")
                
        print(f"  => Valid factors: {len(valid_factors)} / {len(data['factors'])}")
        
        reformat_and_save(fpath, data, valid_factors)

if __name__ == '__main__':
    main()
