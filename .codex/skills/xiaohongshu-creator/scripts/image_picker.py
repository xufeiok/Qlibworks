#!/usr/bin/env python3
"""
图片提取器：从 FnOS 本地目录或夸克网盘挂载目录按条件选图
"""
import argparse
import glob
import os
import random
from pathlib import Path

# 常见图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# 默认搜索路径（FnOS / 夸克网盘挂载点）
DEFAULT_PATHS = [
    "/vol1/1000/KnowledgeBase/lib-地产/raw/images",
    "/vol1/1000/KnowledgeBase/lib-量化交易/raw/images",
    "/vol1/1000/Pictures",
    "/mnt/quarkdisk",           # 夸克网盘挂载路径（如安装）
    "/media/quark",             # 备用夸克挂载路径
]


def scan_images(source_dir: str, keyword: str = None,
                 count: int = 3, recursive: bool = True,
                 random_pick: bool = True) -> list[str]:
    """扫描目录，按条件筛选图片"""
    source = Path(source_dir)
    if not source.exists():
        print(f"⚠ 目录不存在: {source_dir}")
        return []

    pattern = "**/*" if recursive else "*"
    candidates = []
    for f in source.glob(pattern):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            if keyword:
                if keyword.lower() not in f.stem.lower():
                    continue
            candidates.append(str(f.resolve()))

    if not candidates:
        print(f"⚠ 未找到符合条件的图片 (dir={source_dir}, keyword={keyword})")
        return []

    if random_pick:
        selected = random.sample(candidates, min(count, len(candidates)))
    else:
        selected = candidates[:count]

    return selected


def list_sources():
    """列出所有可用的图片源目录"""
    print("可用图片源目录:")
    print("-" * 50)
    for p in DEFAULT_PATHS:
        path = Path(p)
        status = "✓ 存在" if path.exists() else "✗ 不存在"
        if path.exists():
            img_count = sum(1 for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
            print(f"  [{status}] {p} ({img_count} 张图片)")
        else:
            print(f"  [{status}] {p}")
    print()
    print("可自定义 --source 指定其他路径")


def main():
    parser = argparse.ArgumentParser(description="小红书图片提取器")
    parser.add_argument("--source", type=str, default=None,
                        help="图片目录路径（默认自动扫描可用路径）")
    parser.add_argument("--keyword", type=str, default=None,
                        help="按关键词筛选文件名")
    parser.add_argument("--count", type=int, default=3, help="需要几张图 (1-18)")
    parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    parser.add_argument("--no-random", action="store_true", help="按修改时间排序而非随机")
    parser.add_argument("--list-sources", action="store_true", help="列出可用图片源")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    if args.source:
        sources = [args.source]
    else:
        sources = [p for p in DEFAULT_PATHS if Path(p).exists()]

    if not sources:
        print("没有可用的图片目录，请用 --source 指定或先挂载夸克网盘")
        return

    all_selected = []
    for src in sources:
        selected = scan_images(
            src, keyword=args.keyword,
            count=args.count, recursive=not args.no_recursive,
            random_pick=not args.no_random
        )
        all_selected.extend(selected)
        if len(all_selected) >= args.count:
            break

    all_selected = all_selected[:args.count]

    if not all_selected:
        print("未找到任何图片")
        return

    print(f"已选择 {len(all_selected)} 张图片：")
    for i, img in enumerate(all_selected, 1):
        print(f"  [{i}] {img}")

    # 输出 JSON 格式便于其他脚本解析
    import json
    result = {"images": all_selected, "count": len(all_selected)}
    print(f"\nJSON_OUTPUT:{json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
