#!/usr/bin/env python3
"""
小红书定时发布编排器 + 飞书公开文档发布
配合 Hermes cron 实现定时发布，可选同步到飞书公开文档：
1. 预先生成内容并保存
2. 注册 Hermes cron job
3. 到时间自动发布
4. 可选：同时发布到飞书公开文档（需先配置飞书凭证）
"""
import argparse
import json
import os
import subprocess
import sys
import requests
from datetime import datetime
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent

# 飞书配置
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
FEISHU_CONFIG_FILE = Path.home() / ".lark-cli" / "hermes" / "config.json"

SCHEDULE_DIR = SKILL_DIR / ".scheduled"
SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)


def create_schedule(topic: str, schedule: str, template: str = "default",
                    account: str = None, count: int = 1,
                    ai_image: bool = False, feishu: bool = False) -> dict:
    task_id = f"xhs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task = {
        "id": task_id,
        "topic": topic,
        "template": template,
        "schedule": schedule,
        "account": account,
        "count": count,
        "ai_image": ai_image,
        "feishu": feishu,
        "created_at": datetime.now().isoformat(),
        "status": "pending",
    }
    task_file = SCHEDULE_DIR / f"{task_id}.json"
    task_file.write_text(json.dumps(task, ensure_ascii=False, indent=2))
    return task


def list_schedules() -> list[dict]:
    tasks = []
    for f in sorted(SCHEDULE_DIR.glob("*.json")):
        tasks.append(json.loads(f.read_text()))
    return tasks


def remove_schedule(task_id: str) -> bool:
    task_file = SCHEDULE_DIR / f"{task_id}.json"
    if task_file.exists():
        task_file.unlink()
        return True
    return False


def publish_to_feishu(topic: str, content: str) -> dict:
    """
    将文案发布到飞书公开文档（全网可看）。
    支持飞书 CLI 和 Python requests 两种方式。
    """
    # 方案A: 用飞书 CLI
    lark_bin = None
    for p in ["/vol1/1000/HermesAgent/venv/bin/lark", "/usr/local/bin/lark"]:
        if os.path.exists(p):
            lark_bin = p
            break

    if lark_bin and FEISHU_CONFIG_FILE.exists():
        try:
            env = os.environ.copy()
            env["GODEBUG"] = "netdns=go"

            # 创建文档
            doc_title = f"📕 小红书文案 - {topic[:30]}"
            r = subprocess.run(
                [lark_bin, "docx", "create", "--title", doc_title],
                capture_output=True, text=True, timeout=15, env=env
            )
            if r.returncode != 0:
                raise Exception(r.stderr[:200])

            doc_info = json.loads(r.stdout)
            doc_token = doc_info.get("document_id") or \
                doc_info.get("data", {}).get("document", {}).get("document_id", "")
            if not doc_token:
                raise Exception("无法获取文档ID")

            # 写入内容（逐行）
            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                bt = 4 if line.startswith("# ") else (5 if line.startswith("## ") else 3)
                subprocess.run(
                    [lark_bin, "api", "POST",
                     f"/open-apis/docx/v1/documents/{doc_token}/blocks/{doc_token}/children",
                     "--data", json.dumps({
                         "children": [{"block_type": bt, "text": {
                             "elements": [{"text_run": {"content": line}}]
                         }}]
                     })],
                    capture_output=True, text=True, timeout=15, env=env
                )

            # 设置公开权限
            subprocess.run(
                [lark_bin, "api", "PATCH",
                 f"/open-apis/drive/v1/permissions/{doc_token}/public",
                 "--data", json.dumps({
                     "external_access_entity": "open",
                     "security_entity": "anyone",
                     "comment_entity": "anyone",
                     "share_entity": "anyone",
                     "link_share_entity": "anyone_readable"
                 })],
                capture_output=True, text=True, timeout=15, env=env
            )

            return {"success": True, "url": f"https://x0k1x5b6h3.feishu.cn/docx/{doc_token}", "error": ""}
        except Exception as e:
            return _publish_via_api(topic, content)

    # 方案B: Python requests 直接调用 API
    return _publish_via_api(topic, content)


def _publish_via_api(topic: str, content: str) -> dict:
    if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
        return {"success": False, "url": "",
                "error": "未配置飞书，请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET 环境变量，或配置飞书 CLI"}

    try:
        # 1. 获取 token
        r = requests.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET}, timeout=15
        )
        token = r.json().get("tenant_access_token", "")
        if not token:
            return {"success": False, "url": "", "error": "获取飞书Token失败"}

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # 2. 创建文档
        cr = requests.post("https://open.feishu.cn/open-apis/docx/v1/documents",
            json={"title": f"📕 小红书文案 - {topic[:30]}"}, headers=headers, timeout=15)
        doc_token = cr.json().get("data", {}).get("document", {}).get("document_id", "")
        if not doc_token:
            return {"success": False, "url": "", "error": f"创建文档失败"}

        # 3. 写入内容（逐段，避免批量 1770001 错误）
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            bt = 4 if line.startswith("# ") else (5 if line.startswith("## ") else 3)
            requests.post(
                f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks/{doc_token}/children",
                json={"children": [{"block_type": bt, "text": {
                    "elements": [{"text_run": {"content": line}}]
                }}]},
                headers=headers, timeout=15
            )

        # 4. 设置公开权限
        requests.patch(
            f"https://open.feishu.cn/open-apis/drive/v1/permissions/{doc_token}/public",
            json={"external_access_entity": "open", "security_entity": "anyone",
                  "comment_entity": "anyone", "share_entity": "anyone",
                  "link_share_entity": "anyone_readable"},
            headers=headers, timeout=15
        )

        return {"success": True, "url": f"https://x0k1x5b6h3.feishu.cn/docx/{doc_token}", "error": ""}
    except Exception as e:
        return {"success": False, "url": "", "error": str(e)}


def show_cron_instruction(schedule: str, task_id: str, feishu: bool = False) -> str:
    extra = "\n4. 同时发布到飞书公开文档" if feishu else ""
    prompt = f"""使用小红书技能定时发布笔记。

已预生成内容，任务ID: {task_id}。
请加载 xiaohongshu-creator 技能，然后执行以下流程：

1. 运行 content_creator.py 生成内容
2. 对生成内容进行违禁词审核
3. 运行 publisher.py --preview 预览
4. 用户确认后，运行 publisher.py --publish 发布{extra}

注意：每次运行前先清理上次内容，避免重复发布。"""

    return f"""
要注册定时任务，运行：

cronjob action=create \\
  schedule="{schedule}" \\
  name="小红书-{task_id[:8]}" \\
  skills=["xiaohongshu-creator"] \\
  prompt="{prompt.strip()}"

或手动添加到 Hermes cron 配置。"""


def main():
    parser = argparse.ArgumentParser(description="小红书定时发布编排 + 飞书公开文档")
    parser.add_argument("--action", type=str, required=True,
                        choices=["create", "list", "remove", "cron-help", "feishu-publish"],
                        help="操作类型: create/list/remove/cron-help/feishu-publish")
    parser.add_argument("--topic", type=str, help="创作主题/文档标题")
    parser.add_argument("--content", type=str, help="正文内容（feishu-publish 用）")
    parser.add_argument("--schedule", type=str, help="cron 表达式（create用）")
    parser.add_argument("--template", type=str, default="default", help="模板")
    parser.add_argument("--account", type=str, help="指定账号")
    parser.add_argument("--ai-image", action="store_true", help="AI 生图")
    parser.add_argument("--feishu", action="store_true", help="同时发布到飞书公开文档")
    parser.add_argument("--task-id", type=str, help="任务ID（remove用）")

    args = parser.parse_args()

    if args.action == "create":
        if not args.topic or not args.schedule:
            print("错误: create 需要 --topic 和 --schedule")
            sys.exit(1)
        task = create_schedule(
            args.topic, args.schedule, args.template,
            args.account, ai_image=args.ai_image, feishu=args.feishu
        )
        print(f"✓ 定时任务已创建: {task['id']}")
        print(f"  主题: {task['topic']}")
        print(f"  时间: {task['schedule']}")
        if args.feishu:
            print(f"  同步飞书: 是")
        print()
        print(show_cron_instruction(args.schedule, task['id'], args.feishu))

    elif args.action == "list":
        tasks = list_schedules()
        if not tasks:
            print("没有待处理的定时任务")
            return
        print(f"共 {len(tasks)} 个定时任务：")
        print(f"{'ID':30s} {'主题':20s} {'时间':15s} {'飞书':6s} {'状态':10s}")
        print("-" * 80)
        for t in tasks:
            feishu_flag = "✓" if t.get("feishu") else ""
            print(f"{t['id']:30s} {t['topic']:20s} {t['schedule']:15s} {feishu_flag:6s} {t['status']:10s}")

    elif args.action == "remove":
        if not args.task_id:
            print("错误: remove 需要 --task-id")
            sys.exit(1)
        if remove_schedule(args.task_id):
            print(f"✓ 任务 {args.task_id} 已删除")
        else:
            print(f"✗ 任务 {args.task_id} 不存在")

    elif args.action == "cron-help":
        print("Hermes Cron 使用指南：")
        print()
        print("1. 创建定时内容：")
        print("   python scheduler.py --action create --topic '主题' --schedule '0 22 * * 1'")
        print()
        print("2. 同时同步飞书：")
        print("   python scheduler.py --action create --topic '主题' --schedule '0 22 * * 1' --feishu")
        print()
        print("3. 注册到 cron：")
        print('   cronjob action=create \\')
        print('     schedule="0 22 * * 1" \\')
        print('     name="小红书定时发布" \\')
        print('     skills=["xiaohongshu-creator"] \\')
        print('     prompt="加载 xiaohongshu-creator 技能，执行定时发布任务"')
        print()
        print("4. 查看已有 cron job：")
        print("   cronjob action=list")
        print()
        print("5. 单独发布文案到飞书：")
        print('   python scheduler.py --action feishu-publish --topic "标题" --content "正文"')

    elif args.action == "feishu-publish":
        if not args.topic or not args.content:
            print("错误: feishu-publish 需要 --topic 和 --content")
            sys.exit(1)
        print(f"📄 发布到飞书公开文档...")
        result = publish_to_feishu(args.topic, args.content)
        if result["success"]:
            print(f"  ✅ 飞书文档已创建，全网可看！")
            print(f"  🔗 {result['url']}")
        else:
            print(f"  ❌ 发布失败: {result['error']}")
            print(f"  💡 如需配置飞书，请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET 环境变量")


if __name__ == "__main__":
    main()
