# xiaohongshu-mcp MCP 工具参考
# 来源: https://github.com/xpzouying/xiaohongshu-mcp

## MCP Tools 列表

| 工具 | 功能 | 必填参数 |
|------|------|----------|
| login | 登录小红书（扫码） | — |
| check_login | 检查登录状态 | — |
| post_image | 发布图文笔记 | title, content, images[] |
| post_video | 发布视频笔记 | title, content, video |
| search_content | 搜索内容 | keyword |
| get_recommend_list | 获取推荐列表 | — |
| get_note_detail | 获取笔记详情 | note_id, xsec_token |
| post_comment | 发表评论 | feed_id, xsec_token, content |
| reply_comment | 回复评论 | feed_id, xsec_token, comment_id, content |
| like_note | 点赞/取消点赞 | feed_id, xsec_token |
| favorite_note | 收藏/取消收藏 | feed_id, xsec_token |
| get_user_profile | 获取用户主页 | user_id, xsec_token |

## 配置方式

### Docker 运行
```yaml
# 添加到 Hermes config.yaml 的 mcp_servers:
xiaohongshu:
  command: docker
  args:
    - run
    - --rm
    - -i
    - -v
    - /tmp/xiaohongshu-mcp:/app/configs
    - ghcr.io/xpzouying/xiaohongshu-mcp:latest
```

### Go 二进制
```bash
git clone https://github.com/xpzouying/xiaohongshu-mcp.git
cd xiaohongshu-mcp
go build -o xiaohongshu-mcp .
sudo mv xiaohongshu-mcp /usr/local/bin/
```

### 浏览器插件版 (不需要 Docker)
https://github.com/xpzouying/x-mcp

## 重要提示
1. 首次使用必须先调用 login
2. 同一个账号不能在多个网页端同时登录
3. 标题 ≤ 20字，正文 ≤ 1000字
4. 支持本地图片路径和 HTTP(S) 图片链接
5. 每天发帖量上限约 50 篇
