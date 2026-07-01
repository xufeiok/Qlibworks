#!/usr/bin/env python3
"""
AI 生图编排器
调用 ComfyUI / Hermes image_generate 工具 / MCP 生图
根据创意文案自动生成小红书封面图和内容配图
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = SKILL_DIR / "templates"


def load_cover_template(style: str = "default") -> dict:
    """加载封面模板配置"""
    tpl_file = TEMPLATES_DIR / "cover_templates.yaml"
    if tpl_file.exists():
        import yaml
        templates = yaml.safe_load(tpl_file.read_text())
        return templates.get(style, templates.get("default", {}))
    return {
        "aspect_ratio": "portrait",  # 3:4 竖版
        "style": "clean minimal",
        "colors": ["warm", "natural"],
    }


def generate_via_comfyui(prompt: str, style: str = "cover") -> list[str]:
    """
    调用 ComfyUI API 生图。
    ComfyUI 默认地址: http://192.168.10.102:8188 （根据实际配置）
    返回生成图片的本地路径列表。
    """
    # ComfyUI API 端点
    comfyui_url = os.environ.get("COMFYUI_URL", "http://192.168.10.102:8188")

    payload = {
        "prompt": prompt,
        "negative_prompt": "文字,水印,模糊,低质量",
        "width": 768 if style == "cover" else 1024,
        "height": 1024 if style == "cover" else 768,
        "batch_size": 1,
    }

    try:
        req = urllib.request.Request(
            f"{comfyui_url}/prompt",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read())
        # ComfyUI 返回 images 路径列表
        images = result.get("images", [])
        print(f"  ComfyUI 生图完成: {len(images)} 张")
        return images
    except Exception as e:
        print(f"  ComfyUI 调用失败: {e}")
        return []


def generate_via_hermes(prompt: str, aspect: str = "portrait") -> list[str]:
    """
    调用 Hermes image_generate 工具生图。
    此函数返回占位信息，实际由 Hermes Agent 调用 image_generate 工具。
    """
    print(f"  使用 Hermes image_generate 生图")
    print(f"  提示词: {prompt}")
    print(f"  比例: {aspect}")
    print(f"  [注意] 在 Hermes 环境中，此函数应由主 Agent 调用 image_generate 工具完成")
    return []


def enhance_prompt(base_prompt: str, template: dict) -> str:
    """根据封面模板增强提示词"""
    style = template.get("style", "minimal")
    colors = template.get("colors", ["warm"])
    ratio = template.get("aspect_ratio", "portrait")

    style_map = {
        "portrait": "3:4 vertical orientation, phone wallpaper aspect",
        "landscape": "16:9 widescreen",
        "square": "1:1 square",
    }

    enhancements = [
        f"style: {style}",
        f"color palette: {', '.join(colors)}",
        f"aspect: {style_map.get(ratio, '3:4 portrait')}",
        "小红书 style cover, clean typography space at top or bottom",
        "high quality, 4K, detailed, professional",
    ]

    enhanced = base_prompt + "\n" + "\n".join(enhancements)
    return enhanced


def main():
    parser = argparse.ArgumentParser(description="AI 生图编排器")
    parser.add_argument("--prompt", type=str, required=True, help="生图提示词")
    parser.add_argument("--provider", type=str, default="auto",
                        choices=["auto", "comfyui", "hermes", "none"],
                        help="生图引擎")
    parser.add_argument("--type", type=str, default="cover",
                        choices=["cover", "content"],
                        help="生图类型：封面图 or 内容配图")
    parser.add_argument("--style", type=str, default="default",
                        help="封面风格模板（cover_templates.yaml 中定义）")
    parser.add_argument("--count", type=int, default=1, help="生成数量")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认自动创建）")
    args = parser.parse_args()

    # 加载封面模板
    template = load_cover_template(args.style)
    enhanced_prompt = enhance_prompt(args.prompt, template)

    print(f"🎨 开始生图")
    print(f"  类型: {args.type}")
    print(f"  引擎: {args.provider}")
    print(f"  增强提示词: {enhanced_prompt[:200]}...")

    images = []

    if args.provider == "comfyui" or args.provider == "auto":
        images = generate_via_comfyui(enhanced_prompt, args.type)
        if images:
            print(f"  ✓ ComfyUI 生成成功: {len(images)} 张")

    if not images and (args.provider == "hermes" or args.provider == "auto"):
        images = generate_via_hermes(
            enhanced_prompt,
            "portrait" if args.type == "cover" else "landscape"
        )

    if not images:
        print("  ⚠ 所有生图引擎均未返回结果")
        if args.provider == "auto":
            print("  请检查 ComfyUI 是否运行，或手动运行 Hermes 的 image_generate 工具")

    result = {
        "provider_used": args.provider,
        "enhanced_prompt": enhanced_prompt,
        "images": images,
        "count": len(images),
    }
    print(f"\nJSON_OUTPUT:{json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
