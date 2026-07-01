#!/bin/bash
# 单独安装 xhs-cli
set -euo pipefail

echo "=== 安装 xhs-cli ==="
if command -v xhs &>/dev/null; then
    echo "  已安装: $(xhs --version 2>/dev/null || echo '未知版本')"
else
    npm install -g xhs-cli
    echo "  安装完成: $(xhs --version 2>/dev/null || echo 'ok')"
fi

echo ""
echo "快速开始:"
echo "  xhs account add my-account    # 添加账号"
echo "  xhs account use my-account    # 设为当前账号"
echo "  xhs login                     # 扫码登录"
echo "  xhs help                      # 查看全部命令"
