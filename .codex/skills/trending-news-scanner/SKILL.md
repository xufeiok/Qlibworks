---
name: trending-news-scanner
description: Scans the whole web (Baidu, Weibo) for the top 10 hottest and top 10 fastest rising news. Outputs a structured table with Category, Rank, Title, Brief, and Link.
---

# Trending News Scanner

## Overview

This skill automatically scans major news aggregators (Baidu Hot Search, Weibo Hot Search) to identify:
1.  **Top 10 Hottest News**: Based on Baidu Realtime Hot Board.
2.  **Top 10 Fastest Rising News**: Based on Weibo Realtime Search (representing fast-moving social trends).

It organizes the output into a clear, classified table format.

## Workflow

### Step 1: Fetch Trending Data

Execute the bundled Python script to fetch the latest data.

```bash
python scripts/get_trending.py
```

### Step 2: Process and Format

The script returns a JSON object with `hot_news` and `rising_news` lists.
Each item contains: `rank`, `title`, `brief`, `link`, `category`.

**Output Requirement:**
Present the data in two separate Markdown tables.
Note: While this scanner fetches "all" trending news, the downstream consumer (News Topic Generator) will filter these for Personal Growth relevance.

Columns must be: **Category | Index | Title | Brief Content | Link**
(对应中文：**分类 | 序号 | 新闻标题 | 新闻简要内容 | 新闻链接**)

**Handling Missing Data:**
- If the script fails or returns empty lists, fallback to using `WebSearch` with the query: "当前全网热点新闻排行榜" (Current whole web hot news ranking).
- Manually extract 10 items for Hot and 10 for Rising if possible.

### Step 3: Classification

- **Category (分类)**: Use the category provided by the script (e.g., "热点", "微博"). If the script returns "Unknown", infer the category from the title (e.g., "Entertainment", "Politics", "Tech", "Society").
- **Index (序号)**: Use the `rank` field.
- **Title (新闻标题)**: Use the `title` field.
- **Brief Content (新闻简要内容)**: Use the `brief` field. If brief is "点击链接查看详情", keep it or summarize the title if obvious.
- **Link (新闻链接)**: Use the `link` field. Ensure it is clickable.

## Example Output

### 🔥 全网热点新闻 Top 10

| 分类 | 序号 | 新闻标题 | 新闻简要内容 | 新闻链接 |
| :--- | :--- | :--- | :--- | :--- |
| 热点 | 1 | [Example Title] | Example brief description... | [Link](http://...) |
| ... | ... | ... | ... | ... |

### 🚀 热度上升最快新闻 Top 10

| 分类 | 序号 | 新闻标题 | 新闻简要内容 | 新闻链接 |
| :--- | :--- | :--- | :--- | :--- |
| 微博 | 1 | [Rising Title] | Social media discussion... | [Link](http://...) |
| ... | ... | ... | ... | ... |
