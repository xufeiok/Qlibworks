---
name: publish-to-wechat
description: Automates publishing local Markdown files to WeChat Official Account Drafts with AI optimization, compliance review, and formatting.
---

# Publish to WeChat Skill

This skill allows you to automatically publish local Markdown files to your WeChat Official Account (公众号) draft box. It includes an AI-powered pipeline to review content (preserve meaning), check for compliance, and format the article with a beautiful theme.

## Features
- **Batch Processing**: Scan folders for `.md` files.
- **Content Review**: Fixes errors and checks compliance without changing meaning.
- **Compliance Review**: Checks for grammar errors and prohibited terms.
- **Auto-Formatting**: Converts Markdown to WeChat-compatible HTML with Juejin-like styles.
- **Image Handling**: Automatically removes all images.
- **Notifications**: Sends a completion message to your personal WeChat (via `wxauto` or Webhook).

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For `wxauto` notification, you need the WeChat PC client logged in.*

2. **Configuration**:
   Edit `.trae/skills/publish-to-wechat/config.yaml` with your credentials:
   - `wechat.appid` & `wechat.appsecret`: From WeChat Official Account Admin.
   - `llm.api_key`: Your LLM API key (OpenAI/DeepSeek).
   - `notification`: Choose `wxauto` (local) or `webhook`.

## Usage

**Publish a single file:**
```bash
python .trae/skills/publish-to-wechat/main.py "path/to/article.md"
```

**Publish all files in a folder:**
```bash
python .trae/skills/publish-to-wechat/main.py "path/to/articles_folder/"
```

**Skip AI processing (faster):**
```bash
python .trae/skills/publish-to-wechat/main.py "path/to/article.md" --no-ai
```

## Requirements
- Python 3.8+
- WeChat Official Account (Service or Subscription)
- Verified Account for advanced APIs (optional, but recommended for stability)