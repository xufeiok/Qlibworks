#!/bin/bash
# 小红书技能全自动安装脚本
# 用法: bash setup.sh
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== 小红书创作者技能 安装开始 ==="
echo "技能目录: $SKILL_DIR"

# 1. 安装 xhs-cli
echo ""
echo ">>> [1/3] 安装 xhs-cli (npm)..."
if command -v xhs &>/dev/null; then
    echo "  xhs-cli 已安装，跳过"
else
    npm install -g xhs-cli 2>&1 || {
        echo "  [警告] npm install 失败，请检查 Node.js 版本 (需 ≥20)"
    }
fi

# 2. 安装 Python 依赖
echo ""
echo ">>> [2/3] 安装 Python 依赖..."
cd "$SKILL_DIR"
pip install requests pyyaml 2>&1 || echo "  [警告] pip 安装失败"

# 3. 部署 xiaohongshu-mcp
echo ""
echo ">>> [3/3] 部署 xiaohongshu-mcp..."
bash "$SKILL_DIR/scripts/setup_mcp.sh" || echo "  [警告] MCP 部署未完成，可稍后手动运行 setup_mcp.sh"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "下一步操作："
echo "  1. 添加小红书账号: xhs account add my-account"
echo "  2. 登录账号:       xhs login"
echo "  3. 检查登录状态:   python scripts/login_manager.py --check"
echo "  4. 写第一篇笔记:   python scripts/content_creator.py --topic '试试小红书创作'"
echo ""
echo "详细用法请阅读 SKILL.md"
