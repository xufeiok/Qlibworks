#!/bin/bash
# 单独部署 xiaohongshu-mcp MCP Server
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MCP_DIR="$SKILL_DIR/.mcp/xiaohongshu-mcp"

echo "=== 部署 xiaohongshu-mcp ==="

# 检查 Docker
if command -v docker &>/dev/null; then
    echo "  [选项A] 使用 Docker 部署 (推荐)"
    echo "  运行:"
    echo "    docker run --rm -i -v /tmp/xiaohongshu-mcp:/app/configs ghcr.io/xpzouying/xiaohongshu-mcp:latest"
    echo ""
fi

# 检查 Go
if command -v go &>/dev/null; then
    echo "  [选项B] 使用 Go 源码编译"
    if [ ! -d "$MCP_DIR" ]; then
        mkdir -p "$(dirname "$MCP_DIR")"
        git clone https://github.com/xpzouying/xiaohongshu-mcp.git "$MCP_DIR" 2>/dev/null || {
            echo "  克隆失败，请手动执行:"
            echo "    git clone https://github.com/xpzouying/xiaohongshu-mcp.git $MCP_DIR"
            exit 1
        }
    fi
    cd "$MCP_DIR"
    go build -o xiaohongshu-mcp . 2>&1 || echo "  Go 编译失败，请检查 Go 版本"
    sudo cp xiaohongshu-mcp /usr/local/bin/ 2>/dev/null || cp xiaohongshu-mcp "$SKILL_DIR/.mcp/"
    echo "  MCP 二进制已编译到 /usr/local/bin/xiaohongshu-mcp"
fi

echo ""
echo "部署后，在 Hermes config.yaml 添加:"
echo ""
echo '  mcp_servers:'
echo '    xiaohongshu:'
echo '      command: /usr/local/bin/xiaohongshu-mcp'
echo '      args: []'
echo ""
echo "然后重启 Hermes 或运行: hermes mcp reload"
