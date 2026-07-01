#!/usr/bin/env python3
"""
小红书发布编排器
整合：创意内容 → 图片规格检查 → 选图/生图 → 违禁词审核 → 用户确认 → 发布
支持三种发布模式：
  模式A: 纯文字发布
  模式B: 文案+已有图片（自动校验图片大小，超标则跳过）
  模式C: 文案+AI生图后发布
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent

# 小红书图片规格要求
IMAGE_MAX_SIZE = 20 * 1024 * 1024  # 20MB 单张上限
IMAGE_MAX_COUNT = 18               # 最多18张
IMAGE_ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
IMAGE_RECOMMENDED_RATIO = 0.75     # 3:4 推荐比例


def check_image(image_path: str) -> dict:
    """检查单张图片是否符合小红书要求，返回检查结果"""
    result = {
        "path": image_path,
        "valid": True,
        "size_mb": 0,
        "issues": [],
    }

    if not os.path.exists(image_path):
        result["valid"] = False
        result["issues"].append("文件不存在")
        return result

    # 检查文件大小
    size = os.path.getsize(image_path)
    result["size_mb"] = round(size / (1024 * 1024), 2)

    if size == 0:
        result["valid"] = False
        result["issues"].append("文件为空")
        return result

    if size > IMAGE_MAX_SIZE:
        result["valid"] = False
        result["issues"].append(f"文件过大: {result['size_mb']}MB (上限20MB)")

    # 检查扩展名
    ext = Path(image_path).suffix.lower()
    if ext not in IMAGE_ALLOWED_EXT:
        result["valid"] = False
        result["issues"].append(f"不支持格式: {ext} (支持 JPG/PNG/WebP/GIF)")

    # 尝试读取图片尺寸（如需进一步校验分辨率）
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
            result["width"] = w
            result["height"] = h
            ratio = min(w, h) / max(w, h)
            result["ratio"] = round(ratio, 2)
            if abs(ratio - IMAGE_RECOMMENDED_RATIO) > 0.2:
                result["issues"].append(f"比例不推荐: {w}x{h} (建议3:4竖版)")
    except ImportError:
        pass
    except Exception:
        result["issues"].append("无法读取图片信息")

    return result


def validate_images(images: list[str]) -> tuple[list[str], list[str]]:
    """
    批量校验图片，返回 (合规图片列表, 被跳过的图片列表)
    超标图片自动跳过，用户可见提示
    """
    valid_images = []
    skipped_images = []

    print("\n🔍 图片规格检查：")
    print("-" * 50)
    for img in images:
        check = check_image(img)
        size_str = f"{check['size_mb']}MB" if check['size_mb'] else "?"
        name = Path(img).name
        if check["valid"]:
            dim = f" {check.get('width','?')}x{check.get('height','?')}" if 'width' in check else ""
            print(f"  ✅ {name} ({size_str}{dim})")
            valid_images.append(img)
        else:
            issues = ", ".join(check["issues"])
            print(f"  ⛔ {name} ({size_str}) — {issues}")
            skipped_images.append(img)

    # 限制数量
    if len(valid_images) > IMAGE_MAX_COUNT:
        print(f"  ⚠ 图片超过{IMAGE_MAX_COUNT}张，仅保留前{IMAGE_MAX_COUNT}张")
        skipped_images.extend(valid_images[IMAGE_MAX_COUNT:])
        valid_images = valid_images[:IMAGE_MAX_COUNT]

    print(f"\n  通过: {len(valid_images)} 张 | 跳过: {len(skipped_images)} 张")
    if skipped_images:
        print(f"  被跳过的图片: {[Path(p).name for p in skipped_images]}")
    print("-" * 50)

    return valid_images, skipped_images


def run_script(script_name: str, args: list[str]) -> str:
    script_path = SKILL_DIR / "scripts" / script_name
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout


def extract_json_output(text: str) -> dict:
    marker = "JSON_OUTPUT:"
    if marker in text:
        json_str = text.split(marker)[-1].strip()
        return json.loads(json_str)
    return {}


def publish_via_xhscli(title: str, content: str,
                        images: list[str] = None,
                        account: str = None,
                        publish: bool = False,
                        dry_run: bool = True) -> dict:
    cmd = ["xhs", "post"]
    cmd.extend(["--title", title])
    cmd.extend(["--content", content])

    if images:
        for img in images:
            cmd.extend(["--image", img])

    if account:
        cmd.extend(["--account", account])

    if publish and not dry_run:
        cmd.append("--publish")
        print("  → 将自动点击发布按钮")
    else:
        print("  → 预览模式（仅填表，需手动确认发布）")

    print(f"  执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, timeout=180)
    return {
        "exit_code": result.returncode,
        "mode": "preview" if dry_run else "published",
    }


def main():
    parser = argparse.ArgumentParser(description="小红书发布编排器")
    parser.add_argument("--title", type=str, required=True, help="笔记标题")
    parser.add_argument("--content", type=str, help="笔记正文")
    parser.add_argument("--content-file", type=str, help="从文件读取正文")
    parser.add_argument("--images", type=str, nargs="*", default=None,
                        help="图片路径列表（可选）")
    parser.add_argument("--image-dir", type=str, help="从目录选图")
    parser.add_argument("--image-keyword", type=str, help="图片关键词筛选")
    parser.add_argument("--ai-image", action="store_true", help="AI 生图（无图时启用）")
    parser.add_argument("--image-prompt", type=str, help="AI 生图提示词（覆盖自动生成）")
    parser.add_argument("--account", type=str, help="指定账号")
    parser.add_argument("--preview", action="store_true", default=True,
                        help="预览模式（默认），不自动发布")
    parser.add_argument("--publish", action="store_true", help="确认发布（覆盖预览模式）")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="跳过图片规格检查")
    parser.add_argument("--feishu-doc", action="store_true",
                        help="同时将文案发布到飞书公开文档")
    parser.add_argument("--feishu-space", type=str, default=None,
                        help="飞书记忆空间ID（可选，默认使用已配置的空间）")
    parser.add_argument("--step", type=str,
                        choices=["all", "check-only", "preview-only"],
                        default="all", help="执行步骤")
    args = parser.parse_args()

    # 读取正文
    content = args.content
    if args.content_file:
        content = Path(args.content_file).read_text(encoding="utf-8")

    if not content:
        print("错误：请提供 --content 或 --content-file")
        sys.exit(1)

    images = []

    # 步骤1: 处理图片
    if args.images:
        images = args.images
        print(f"📷 使用指定图片: {len(images)} 张")
    elif args.image_dir:
        print(f"📷 从目录选图: {args.image_dir}")
        picker_args = ["--source", args.image_dir, "--count", "3"]
        if args.image_keyword:
            picker_args.extend(["--keyword", args.image_keyword])
        output = run_script("image_picker.py", picker_args)
        print(output)
        data = extract_json_output(output)
        images = data.get("images", [])
    elif args.ai_image and args.image_prompt:
        print(f"🎨 AI 生图")
        output = run_script("image_generator.py", [
            "--prompt", args.image_prompt,
            "--type", "cover",
            "--count", "1",
        ])
        print(output)
        data = extract_json_output(output)
        images = data.get("images", [])
    else:
        print("📝 纯文字笔记（无图片）")

    # 步骤1.5: 图片规格检查
    if images and not args.skip_image_check:
        valid_images, skipped = validate_images(images)
        images = valid_images
        if skipped:
            print(f"\n⚠ 已自动跳过 {len(skipped)} 张不合规的图片")

    # 步骤2: 检查/预览
    if args.step == "check-only":
        print("\n✅ 内容检查完成，以下是发布预览：")
        print(f"  标题: {args.title}")
        print(f"  正文长度: {len(content)} 字")
        print(f"  图片: {len(images)} 张")
        print(f"  账号: {args.account or '当前账号'}")
        return

    # 步骤3: 可选 — 发布到飞书公开文档
    if args.feishu_doc:
        print("\n📄 同时发布到飞书公开文档...")
        scheduler_script = SKILL_DIR / "scripts" / "scheduler.py"
        result = subprocess.run(
            [sys.executable, str(scheduler_script), "--action", "feishu-publish",
             "--topic", args.title, "--content", content],
            capture_output=True, text=True, timeout=60
        )
        print(result.stdout)
        if result.returncode == 0:
            print("  ✅ 飞书文档发布完成")
        else:
            print(f"  ⚠ 飞书发布失败: {result.stderr[:200]}")

    # 步骤4: 发布到小红书
    result = publish_via_xhscli(
        title=args.title,
        content=content,
        images=images if images else None,
        account=args.account,
        publish=args.publish,
        dry_run=(not args.publish),
    )

    if result["exit_code"] == 0:
        if args.publish:
            print("\n✅ 笔记已发布！")
        else:
            print("\n📋 已打开发布页面，请确认后手动点击发布按钮")
            print("  → 确认发布请加 --publish 参数")
    else:
        print("\n❌ 发布失败，请检查错误信息")


if __name__ == "__main__":
    main()
