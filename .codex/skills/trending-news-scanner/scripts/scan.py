import argparse
import json
import sys
import os

# Add current directory to path to find get_trending.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from get_trending import fetch_weibo_hot, fetch_baidu_hot
except ImportError:
    print("Error: Could not import get_trending.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    # Try Weibo first, then Baidu
    news_items = fetch_weibo_hot()
    if not news_items:
        news_items = fetch_baidu_hot()
    
    if not news_items:
        print("No news found", file=sys.stderr)
        # Write empty list to avoid crashing next step
        news_items = []

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    abs_output_path = os.path.abspath(args.output)
    
    with open(abs_output_path, 'w', encoding='utf-8') as f:
        json.dump(news_items, f, ensure_ascii=False, indent=2)
    
    # Print absolute path for workflow orchestration
    print(abs_output_path)

if __name__ == "__main__":
    main()
