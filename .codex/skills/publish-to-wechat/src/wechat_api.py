import requests
import json
import os
import yaml
from bs4 import BeautifulSoup

class WeChatAPI:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.appid = self.config.get('wechat', {}).get('appid')
        self.secret = self.config.get('wechat', {}).get('appsecret')
        self.access_token = None
        
        if self.appid == "YOUR_APP_ID":
            print("Warning: WeChat AppID not configured.")

    def get_token(self):
        if self.access_token:
            return self.access_token
        
        url = f"https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={self.appid}&secret={self.secret}"
        resp = requests.get(url)
        data = resp.json()
        
        if 'access_token' in data:
            self.access_token = data['access_token']
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {data}")

    def upload_article_image(self, file_path: str) -> str:
        """
        Upload image for use INSIDE the article content.
        Returns the URL.
        """
        token = self.get_token()
        url = f"https://api.weixin.qq.com/cgi-bin/media/uploadimg?access_token={token}"
        
        with open(file_path, 'rb') as f:
            files = {'media': f}
            resp = requests.post(url, files=files)
            
        data = resp.json()
        if 'url' in data:
            return data['url']
        else:
            raise Exception(f"Failed to upload article image: {data}")

    def upload_cover_image(self, file_path: str) -> str:
        """
        Upload image for use as COVER (Thumb).
        Returns the media_id.
        """
        token = self.get_token()
        # Use add_material for permanent storage, or media/upload for temporary
        # Drafts usually need permanent material for the cover? 
        # Actually media/upload (temp) works for 3 days, add_material is better.
        url = f"https://api.weixin.qq.com/cgi-bin/material/add_material?access_token={token}&type=image"
        
        with open(file_path, 'rb') as f:
            files = {'media': f}
            resp = requests.post(url, files=files)
            
        data = resp.json()
        if 'media_id' in data:
            return data['media_id']
        else:
            raise Exception(f"Failed to upload cover image: {data}")

    def process_content_and_upload_images(self, html_content: str, base_path: str):
        """
        Parse HTML, find local images, upload them, replace src with WeChat URL.
        Also returns the media_id of the first image to use as cover.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        images = soup.find_all('img')
        thumb_media_id = None
        
        for i, img in enumerate(images):
            src = img.get('src')
            if not src.startswith('http'):
                # Local file
                # Handle relative paths
                if not os.path.isabs(src):
                    # Assuming base_path is the directory of the markdown file
                    img_path = os.path.join(base_path, src)
                else:
                    img_path = src
                
                if os.path.exists(img_path):
                    print(f"Uploading image: {img_path}")
                    try:
                        # Upload for content
                        wechat_url = self.upload_article_image(img_path)
                        img['src'] = wechat_url
                        
                        # Use first image as cover
                        if i == 0:
                            print(f"Uploading cover image: {img_path}")
                            thumb_media_id = self.upload_cover_image(img_path)
                    except Exception as e:
                        print(f"Error uploading image {img_path}: {e}")
                else:
                    print(f"Warning: Image not found: {img_path}")
        
        # Fallback to default cover if no image found in content
        if not thumb_media_id:
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                default_cover_path = os.path.join(base_dir, 'assets', 'default_cover.jpg')
                if os.path.exists(default_cover_path):
                    print(f"No images in article. Uploading default cover: {default_cover_path}")
                    thumb_media_id = self.upload_cover_image(default_cover_path)
                else:
                    print(f"Warning: Default cover not found at {default_cover_path}")
            except Exception as e:
                print(f"Error uploading default cover: {e}")

        return str(soup), thumb_media_id

    def add_draft(self, title: str, content: str, thumb_media_id: str, author: str = ""):
        token = self.get_token()
        url = f"https://api.weixin.qq.com/cgi-bin/draft/add?access_token={token}"
        
        article = {
            "title": title,
            "author": author,
            "digest": "",
            "content": content,
            "content_source_url": "",
            "thumb_media_id": thumb_media_id,
            "need_open_comment": 0,
            "only_fans_can_comment": 0
        }
        
        payload = {
            "articles": [article]
        }
        
        # Ensure UTF-8 encoding for json dump
        resp = requests.post(url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        data = resp.json()
        
        if 'media_id' in data:
            print(f"Draft created successfully! Media ID: {data['media_id']}")
            return data['media_id']
        else:
            raise Exception(f"Failed to create draft: {data}")
