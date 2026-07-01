#!/usr/bin/env python3
"""
樾江峰荟+亿澜峰荟 每周小红书文案生成 + 飞书推送
每周四 22:00 执行，生成周五~下周四共7篇文案
流程：生成草稿 → 飞书发我确认 → 我确认后 → 发布到飞书文档

模式:
  --action preview  生成文案并发送飞书消息预览（默认）
  --action publish  将上次生成的草稿发布到飞书文档（我确认后执行）
"""
import json, os, sys, requests, time
from datetime import datetime, timedelta
from pathlib import Path

APP_ID = os.environ.get("FEISHU_APP_ID", "cli_a96305a3b97a1cd3")
APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
FOLDER_TOKEN = "G0OmfaVRMlessidQGDccNt9fnFg"  # 樾江峰荟亿澜峰荟每周销售小红书文案
USER_OPEN_ID = "ou_9a0caef46ae39a2e88f9a867583b5821"
DRAFT_DIR = Path(os.path.expanduser("~/.hermes/scripts/drafts"))
DRAFT_DIR.mkdir(parents=True, exist_ok=True)


def get_feishu_token():
    r = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET}, timeout=15
    )
    return r.json()["tenant_access_token"]


def send_feishu_message(token, title, preview_text):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    content = {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"📝 {title}"}, "template": "green"},
        "elements": [
            {"tag": "markdown", "content": preview_text},
            {"tag": "hr"},
            {"tag": "markdown", "content": "**请确认以上文案内容**\n回复「**确认发布(YJYL)**」后，我会将完整文档推送到飞书公开文档中。"},
        ]
    }
    r = requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        params={"receive_id_type": "open_id"},
        json={"receive_id": USER_OPEN_ID, "msg_type": "interactive",
              "content": json.dumps(content, ensure_ascii=False)},
        headers=headers, timeout=15
    )
    return r.json()


def send_simple_message(token, text):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        params={"receive_id_type": "open_id"},
        json={"receive_id": USER_OPEN_ID, "msg_type": "text",
              "content": json.dumps({"text": text}, ensure_ascii=False)},
        headers=headers, timeout=15
    )


# ===== 日期和节气 =====
today = datetime.now()
this_friday = today + timedelta(days=(4 - today.weekday()) % 7 or 7)
dates = [(this_friday + timedelta(days=i)).strftime("%m月%d日 %A") for i in range(7)]
date_range_str = f"{this_friday.strftime('%Y.%m.%d')}-{(this_friday + timedelta(days=6)).strftime('%m.%d')}"

def get_seasonal_tag():
    m, d = today.month, today.day
    if m == 5:
        if d <= 5: return "立夏"
        if d <= 20: return "小满"
        if d <= 31: return "初夏"
    if m == 6:
        if d <= 5: return "芒种"
        if d <= 21: return "夏至"
        return "盛夏"
    if m == 7:
        if d <= 6: return "小暑"
        return "盛夏"
    if m == 8:
        if d <= 7: return "立秋"
        if d <= 23: return "处暑"
    return ""

season_tag = get_seasonal_tag()

# ===== 项目价值角度（樾江峰荟+亿澜峰荟 大邑双盘）=====
VALUE_ANGLES = [
    "国企（兴城人居，千亿国企，成都龙头）",
    "低密大盘（合计约132亩，约2.0容积率）",
    "公园（内约74亩家庭公园+外约63亩市政公园）",
    "健康（斜江湾心+五重园林标准，自然氧吧）",
    "园林（五重园林技法，樾江峰荟由贝尔高林操刀，口碑大邑领先）",
    "洋房（6-10层真洋房，约3.3m层高，最大楼间距约193m）",
    "公区精装（停车场→大堂→电梯厅，墅级立面品质）",
    "高知社区（业主多为政企中高层，学历层次高，圈层纯粹）",
    "优质教育（优质学校在侧，社区乐学空间）",
    "优质服务（润锦城物业，国家一级资质）",
    "桃源新城中心（大邑核心区位，醇熟配套）",
    "全龄配套（环氧跑道、康体会馆、恒温泳池）",
]

# ===== 每天角度组合 =====
daily_angles = [
    ("周五", "国企+桃源新城中心", "周末看房邀请"),
    ("周六", "低密洋房+公区精装", "产品力"),
    ("周日", "公园+健康+园林", "居住环境"),
    ("周一", "高知社区+优质服务", "生活品质"),
    ("周二", season_tag + "季节生活场景", "季节情绪"),
    ("周三", "优质教育+全龄配套", "家庭陪伴"),
    ("周四", "桃源新城+园林", "板块价值"),
]

# ===== 操作 =====
def action_preview():
    print(f"📅 本周文案周期: {date_range_str} (周五~下周四)")
    print(f"🌿 当前节气/季节: {season_tag or '无'}")
    print(f"📋 项目: 樾江峰荟+亿澜峰荟")
    print(f"📋 项目价值角度: {len(VALUE_ANGLES)}个")
    
    contents = []
    for i, (dow, angles, scene) in enumerate(daily_angles):
        date_str = dates[i]
        content_block = {
            "day": dow,
            "date": date_str,
            "angles": angles,
            "scene": scene,
            "title": "",
            "body": ""
        }
        contents.append(content_block)
    
    draft = {
        "date_range": date_range_str,
        "season_tag": season_tag,
        "generated_at": datetime.now().isoformat(),
        "status": "draft",
        "project": "樾江峰荟+亿澜峰荟",
        "contents": contents,
        "value_angles": VALUE_ANGLES
    }
    draft_file = DRAFT_DIR / "latest_yjyl_draft.json"
    draft_file.write_text(json.dumps(draft, ensure_ascii=False, indent=2))
    
    preview_lines = [
        f"**项目**: 樾江峰荟 + 亿澜峰荟（大邑双盘）",
        f"**周期**: {date_range_str}（周五~下周四）",
        f"**节气**: {season_tag or '无'}",
        f"**用途**: 置业顾问转发宣传",
        "",
        "**本周7天角度安排：**",
    ]
    for i, (dow, angles, scene) in enumerate(daily_angles):
        preview_lines.append(f"  **{dow}({dates[i]})**: {angles} — {scene}")
    
    preview_lines += [
        "",
        "**可调用项目价值角度（11个）：**",
        "、".join(VALUE_ANGLES),
        "",
        "**知识库参考：**",
        "/vol1/1000/KnowledgeBase/lib-地产/wiki/sources/source-樾江亿澜价值体系梳理.md",
        "/vol1/1000/KnowledgeBase/lib-地产/wiki/sources/source-樾江峰荟项目基础信息.md",
        "/vol1/1000/KnowledgeBase/lib-地产/wiki/sources/source-大邑人居生活方式价值体系.md",
    ]
    
    preview_text = "\n".join(preview_lines)
    title = f"樾江峰荟亿澜峰荟每周销售小红书文案-{date_range_str}"
    
    try:
        token = get_feishu_token()
        result = send_feishu_message(token, title, preview_text)
        print(f"\n✅ 飞书预览消息已发送！")
        print(f"   消息ID: {result.get('data', {}).get('message_id', 'unknown')}")
        print(f"\n⏳ 请查看飞书，确认文案内容后回复「确认发布(YJYL)」")
        print(f"   确认后我会执行:")
        print(f"   python weekly_yjyl_plan.py --action publish")
    except Exception as e:
        print(f"⚠ 飞书消息发送失败: {e}")
    
    print(f"\n📄 草稿已保存: {draft_file}")
    print(f"📋 文档名: {title}")
    print(f"\nJSON_OUTPUT:{{\"title\":\"{title}\",\"draft_file\":\"{draft_file}\",\"date_range\":\"{date_range_str}\"}}")


def action_publish():
    draft_file = DRAFT_DIR / "latest_yjyl_draft.json"
    if not draft_file.exists():
        print("❌ 没有找到草稿文件。请先生成预览。")
        sys.exit(1)
    
    draft = json.loads(draft_file.read_text())
    doc_title = f"樾江峰荟亿澜峰荟每周销售小红书文案-{draft['date_range']}"
    print(f"📄 正在发布: {doc_title}")
    
    try:
        token = get_feishu_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        # 创建文档到文件夹
        cr = requests.post("https://open.feishu.cn/open-apis/docx/v1/documents",
            json={"title": doc_title, "folder_token": FOLDER_TOKEN},
            headers=headers, timeout=15)
        doc_id = cr.json()["data"]["document"]["document_id"]
        print(f"✅ 文档已创建")
        
        # 分享给用户
        requests.post(f"https://open.feishu.cn/open-apis/drive/v1/permissions/{doc_id}/members?type=docx",
            json={"member_type": "openid", "member_id": USER_OPEN_ID, "perm": "full_access"},
            headers=headers, timeout=15)
        
        def add_block(bt, key, text=""):
            block = {"block_type": 27, "divider": {}} if bt == 27 else \
                    {"block_type": bt, key: {"elements": [{"text_run": {"content": text}}]}}
            requests.post(f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{doc_id}/children",
                json={"children": [block]}, headers=headers, timeout=15)
            time.sleep(0.02)
        
        # 写入内容
        add_block(3, "heading1", doc_title)
        add_block(2, "text", f"周期：{draft['date_range']}（周五~下周四）")
        add_block(2, "text", "用途：樾江峰荟+亿澜峰荟置业顾问转发宣传")
        add_block(27, "divider")
        
        for item in draft["contents"]:
            add_block(4, "heading2", f"{item['day']} {item['date']}")
            add_block(27, "divider")
            
            if item.get("title"):
                add_block(3, "heading1", item["title"])
                add_block(2, "text", "")
            
            if item.get("body"):
                all_lines = item["body"].split("\n")
                body_paras = [l.strip() for l in all_lines if l.strip() and not l.strip().startswith("#")]
                tags_list = [l.strip() for l in all_lines if l.strip().startswith("#")]
                
                for para in body_paras:
                    add_block(2, "text", para)
                
                add_block(2, "text", "")
                if tags_list:
                    add_block(2, "text", "  ".join(tags_list))
            
            add_block(27, "divider")
        
        doc_url = f"https://x0k1x5b6h3.feishu.cn/docx/{doc_id}"
        print(f"\n✅ 发布完成！")
        print(f"📄 文档: {doc_url}")
        print(f"📁 位置: 樾江峰荟亿澜峰荟每周销售小红书文案/")
        
        draft["status"] = "published"
        draft["published_url"] = doc_url
        draft["published_at"] = datetime.now().isoformat()
        draft_file.write_text(json.dumps(draft, ensure_ascii=False, indent=2))
        
        try:
            send_simple_message(token,
                f"✅ 樾江峰荟亿澜峰荟每周销售小红书文案已发布！\n"
                f"📄 {doc_title}\n"
                f"🔗 {doc_url}\n"
                f"📁 位置：飞书 → 樾江峰荟亿澜峰荟每周销售小红书文案/")
        except:
            pass
        
        print(f"\nJSON_OUTPUT:{{\"url\":\"{doc_url}\",\"token\":\"{doc_id}\",\"title\":\"{doc_title}\"}}")
        
    except Exception as e:
        print(f"❌ 发布失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="樾江峰荟+亿澜峰荟每周小红书文案")
    parser.add_argument("--action", choices=["preview", "publish"], default="preview",
                        help="preview=生成预览并飞书发我确认 | publish=确认后发布到飞书文档")
    args = parser.parse_args()
    
    if args.action == "publish":
        action_publish()
    else:
        action_preview()
