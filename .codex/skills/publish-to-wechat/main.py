import re
import argparse
import os
import sys
from src.scanner import FileScanner
from src.ai_processor import AIProcessor
from src.converter import MarkdownConverter
from src.wechat_api import WeChatAPI
from src.notifier import Notifier

def main():
    parser = argparse.ArgumentParser(description="Publish Markdown files to WeChat Official Account Drafts")
    parser.add_argument("path", help="Path to markdown file or directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI optimization and review")
    parser.add_argument("--template", help="CSS template to use (tech, prose, colorful, or auto)")
    args = parser.parse_args()

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Initialize components
    print("Initializing components...")
    scanner = FileScanner()
    ai_processor = AIProcessor(config_path)
    converter = MarkdownConverter(config_path)
    wechat = WeChatAPI(config_path)
    notifier = Notifier(config_path)

    # Scan files
    files = scanner.scan(args.path)
    if not files:
        print("No markdown files found.")
        sys.exit(0)

    print(f"Found {len(files)} files to process.")

    for file_path in files:
        print(f"\nProcessing: {file_path}")
        try:
            # 1. Read Content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove images (User Requirement)
            print("  - Removing images...")
            content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
            content = re.sub(r'<img.*?>', '', content, flags=re.IGNORECASE)
            
            # 2. AI Optimization & Review
            template_name = args.template
            
            if not args.no_ai:
                # 2.1 Process Metadata (Title & Abstract) - NEW
                print("  - Processing metadata (Title/Abstract)...")
                content = ai_processor.process_metadata(content)
                
                print("  - Optimizing content with AI...")
                # content = ai_processor.optimize_content(content) # Disabled per user request
                print("  - Reviewing content with AI...")
                content = ai_processor.review_content(content)
                
                # Auto-select template if not specified or set to auto
                if not template_name or template_name == "auto":
                    print("  - Auto-selecting template with AI...")
                    template_name = ai_processor.recommend_template(content)
                    print(f"    Selected template: {template_name}")

            # Extract title (filename or first h1) - Moved here to capture generated title
            filename = os.path.basename(file_path)
            # Try to find title in content first
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                title = os.path.splitext(filename)[0]
            
            # 3. Convert to HTML
            print(f"  - Converting to HTML (Template: {template_name or 'default'})...")
            html = converter.convert(content, template_name)
            
            # 4. Upload Images & Process Content
            print("  - Uploading images and processing content...")
            # Use the directory of the markdown file as base for relative images
            file_dir = os.path.dirname(file_path)
            final_html, thumb_media_id = wechat.process_content_and_upload_images(html, file_dir)
            
            if not thumb_media_id:
                print("  - Warning: No cover image found. Draft creation might fail if account requires it.")
                # Could set a default media_id from config if available
            
            # 5. Create Draft
            print("  - Creating draft...")
            media_id = wechat.add_draft(title, final_html, thumb_media_id)
            
            # 6. Notify
            print("  - Sending notification...")
            notifier.send(f"Published Draft: {title}", f"Media ID: {media_id}\nFile: {file_path}")
            
            print(f"Success! Draft created for {filename}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            notifier.send(f"Publish Failed: {filename}", str(e))

if __name__ == "__main__":
    main()
