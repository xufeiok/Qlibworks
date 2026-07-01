#!/usr/bin/env python3
"""
小红书账号登录管理器
管理多账号的登录、状态检测、切换
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

XHS_CLI_DIR = Path.home() / ".xhs-cli" / ".cache"
REGISTRY_FILE = XHS_CLI_DIR / "accounts" / "registry.json"


def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {"accounts": {}, "currentAccount": None}


def list_accounts():
    registry = load_registry()
    current = registry.get("currentAccount")
    accounts = registry.get("accounts", {})
    if not accounts:
        print("没有已配置的账号。请运行: xhs account add <name>")
        return
    print(f"{'当前':4s} {'账号名':20s} {'别名':20s} {'已登录':6s}")
    print("-" * 55)
    for name, info in accounts.items():
        prefix = "*" if name == current else " "
        alias = info.get("alias", "")
        logged = "✓" if info.get("loggedIn") else "✗"
        print(f"{prefix:4s} {name:20s} {alias:20s} {logged:6s}")


def check_login():
    """通过 xhs-cli 检查当前登录状态"""
    result = subprocess.run(
        ["xhs", "metrics"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print("✓ 登录状态正常")
        print(result.stdout[:500])
    else:
        print("✗ 未登录或登录已过期")
        print(result.stderr[:300])
        print("请运行: xhs login")


def add_account(name: str, alias: str = ""):
    result = subprocess.run(
        ["xhs", "account", "add", name],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        print(f"✓ 账号 '{name}' 已添加")
        if alias:
            # 写别名到 registry
            registry = load_registry()
            if name in registry.get("accounts", {}):
                registry["accounts"][name]["alias"] = alias
                REGISTRY_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
            print(f"   别名设为: {alias}")
        print("接下来: xhs account use {name} && xhs login")
    else:
        print(f"✗ 添加失败: {result.stderr}")


def set_current(name: str):
    result = subprocess.run(
        ["xhs", "account", "use", name],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        print(f"✓ 当前账号已切换为: {name}")
    else:
        print(f"✗ 切换失败: {result.stderr}")


def login(name: str = None):
    cmd = ["xhs", "login"]
    if name:
        cmd.extend(["--account", name])
    print("正在打开浏览器进行扫码登录...")
    subprocess.run(cmd)
    # 登录后检查状态
    check_login()


def main():
    parser = argparse.ArgumentParser(description="小红书账号管理")
    parser.add_argument("--list", action="store_true", help="列出所有账号")
    parser.add_argument("--check", action="store_true", help="检查当前登录状态")
    parser.add_argument("--add", type=str, help="添加账号 (参数: 账号名)")
    parser.add_argument("--alias", type=str, default="", help="账号别名")
    parser.add_argument("--use", type=str, help="切换当前账号")
    parser.add_argument("--login", type=str, nargs="?", const="", help="登录指定账号（默认当前）")

    args = parser.parse_args()

    if args.list:
        list_accounts()
    elif args.check:
        check_login()
    elif args.add:
        add_account(args.add, args.alias)
    elif args.use:
        set_current(args.use)
    elif args.login is not None:
        login(args.login if args.login else None)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
