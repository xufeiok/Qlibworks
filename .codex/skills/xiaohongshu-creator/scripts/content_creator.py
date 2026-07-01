#!/usr/bin/env python3
"""
小红书内容创意生成器 + 违禁词审核
功能：
- 根据主题+模板生成标题/正文/标签/生图提示词
- 双通道违禁词审核（规则库+LLM）
- 爆款分析参考（可选的搜索参考）
"""
import argparse
import json
import os
import random
import sys
import yaml
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = SKILL_DIR / "templates"
FORBIDDEN_WORDS_FILE = SKILL_DIR / "references" / "forbidden-words.txt"

# 默认违禁词库（内置兜底）
DEFAULT_FORBIDDEN_WORDS = [
    "第一", "最", "国家级", "首个", "遥遥领先", "绝对", "百分百",
    "全网", "唯一", "独创", "首创", "永久", "包治", "速效",
    "致富", "躺赚", "日赚", "月入过万", "点击领钱",
    "微信", "QQ", "加我", "私聊", "私信我", "联系我",
]


def load_forbidden_words() -> list:
    words = DEFAULT_FORBIDDEN_WORDS.copy()
    if FORBIDDEN_WORDS_FILE.exists():
        extra = FORBIDDEN_WORDS_FILE.read_text().strip().split("\n")
        words.extend([w.strip() for w in extra if w.strip() and not w.startswith("#")])
    return list(set(words))


def load_template(name: str = "default") -> dict:
    filepath = TEMPLATES_DIR / f"{name}.yaml"
    if not filepath.exists():
        filepath = TEMPLATES_DIR / "default.yaml"
        if not filepath.exists():
            return {
                "name": "default",
                "style": "通用",
                "title_length": 20,
                "body_length": 1000,
                "tone": "亲切自然",
                "tag_count": 5,
                "cover_style": "简洁清晰",
            }
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_content(topic: str, template_name: str = "default",
                     style: str = "专业", count: int = 1) -> list[dict]:
    """
    生成创意内容。
    实际使用时，此函数应调用 LLM 生成。
    这里提供默认的结构化占位逻辑。
    """
    template = load_template(template_name)

    # 构造给 LLM 的提示
    system_prompt = f"""你是一个小红书内容创作专家。
根据用户提供的主题，生成{count}篇小红书笔记。

每篇笔记包含：
- title: 标题（不超过{template.get('title_length', 20)}字）
- body: 正文（不超过{template.get('body_length', 1000)}字，使用{template.get('tone', '亲切自然')}的语气）
- tags: 3-6个标签（#开头）
- image_prompt: 封面图生成提示词（描述画面风格和内容，适合用 AI 绘图生成，3:4竖版比例）
- image_prompts: 可选的内容配图提示词列表（每个元素对应一张配图）

风格：{style}
模板风格：{template.get('style', '通用')}

注意：
- 标题要有吸引力，使用数字、反问、悬念等技巧
- 正文分段清晰，每段不要太长
- 标签要覆盖热门话题
- 图片提示词要详细，包含场景、色调、构图等
- 不要使用违禁词
"""

    user_prompt = f"主题：{topic}\n\n请生成{count}篇小红书笔记。"

    # 注意：此函数应在 Hermes 环境中由主 Agent 调用 LLM 完成
    # 这里返回占位结构说明
    results = []
    for i in range(count):
        results.append({
            "id": i + 1,
            "title": f"[AI生成建议] {topic} - 第{i+1}篇",
            "body": f"这是关于「{topic}」的内容建议。实际使用时由 LLM 根据模板生成完整正文。",
            "tags": [f"#{topic}", "#推荐", "#干货分享"],
            "image_prompt": f"小红书封面图，{topic}主题，3:4竖版，简约大气风格，温暖色调",
            "image_prompts": [
                f"{topic}相关内容配图1",
                f"{topic}相关内容配图2",
            ],
            "template": template_name,
        })

    return results


def check_forbidden_words(text: str) -> dict:
    """
    规则库违禁词审核。
    返回：{has_issue, matched_words, suggestions}
    """
    forbidden = load_forbidden_words()
    matched = []
    for word in forbidden:
        if word.lower() in text.lower():
            matched.append(word)

    return {
        "has_issue": len(matched) > 0,
        "matched_words": matched,
        "suggestions": [f"建议替换: '{w}' → 用更温和的表达" for w in matched],
        "risk_level": "high" if len(matched) > 3 else ("medium" if matched else "safe"),
    }


def llm_audit(content: dict) -> dict:
    """
    LLM 审核入口（占位）。
    实际由 Hermes 主 Agent 调用 LLM 完成深度审核。
    检查：夸大宣传、虚假信息、敏感话题、合规风险等。
    """
    return {
        "llm_risk_level": "pending",
        "llm_suggestions": ["请使用主 Agent 的 LLM 能力进行深度审核"],
    }


def format_preview(content: dict, audit_result: dict) -> str:
    """格式化预览内容"""
    lines = []
    lines.append("=" * 50)
    lines.append(f"📝 笔记 #{content['id']}")
    lines.append("=" * 50)
    lines.append(f"标题：{content['title']}")
    lines.append(f"正文：{content['body'][:200]}{'...' if len(content['body']) > 200 else ''}")
    lines.append(f"标签：{' '.join(content['tags'])}")
    lines.append("")
    lines.append("🎨 封面提示词：")
    lines.append(f"  {content['image_prompt']}")
    if content.get('image_prompts'):
        lines.append("📷 配图提示词：")
        for p in content['image_prompts']:
            lines.append(f"  - {p}")
    lines.append("")
    lines.append("🔍 违禁词审核：")
    if audit_result["risk_level"] == "safe":
        lines.append("  ✅ 未发现违禁词（规则库）")
    else:
        for w in audit_result["matched_words"]:
            lines.append(f"  ⚠️ 发现违禁词: '{w}'")
        for s in audit_result["suggestions"]:
            lines.append(f"  💡 {s}")
    lines.append(f"  风险等级: {audit_result['risk_level']}")
    lines.append("=" * 50)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="小红书内容创意生成器")
    parser.add_argument("--topic", type=str, required=True, help="创作主题")
    parser.add_argument("--template", type=str, default="default", help="模板名称")
    parser.add_argument("--style", type=str, default="专业",
                        choices=["专业", "通俗", "种草", "故事", "干货"])
    parser.add_argument("--count", type=int, default=1, help="生成篇数")
    parser.add_argument("--audit-only", type=str, help="仅审核指定文本")
    parser.add_argument("--output", type=str, help="输出到文件")
    args = parser.parse_args()

    if args.audit_only:
        result = check_forbidden_words(args.audit_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 生成创意内容
    contents = generate_content(args.topic, args.template, args.style, args.count)

    # 审核
    all_output = []
    for c in contents:
        audit = check_forbidden_words(c["title"] + " " + c["body"] + " " + " ".join(c["tags"]))
        preview = format_preview(c, audit)
        all_output.append({"content": c, "audit": audit, "preview": preview})
        print(preview)
        print()

    # 如指定输入出文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_output, f, ensure_ascii=False, indent=2)
        print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
