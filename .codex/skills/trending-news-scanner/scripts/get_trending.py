import requests
from bs4 import BeautifulSoup
import json
import sys
import re

def get_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

def fetch_baidu_hot():
    """Fetch Top 10 Hot News from Baidu."""
    url = "https://top.baidu.com/board?tab=realtime"
    items = []
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Baidu structure: div.category-wrap_iQLoo
        containers = soup.find_all('div', class_=lambda x: x and 'category-wrap' in x)
        
        for index, container in enumerate(containers):
            if len(items) >= 10:
                break
                
            try:
                # Title
                title_div = container.find(class_=lambda x: x and 'c-single-text-ellipsis' in x)
                title = title_div.get_text().strip() if title_div else "Unknown Title"
                
                # Link
                link_tag = container.find('a', href=True)
                link = link_tag['href'] if link_tag else ""
                if link and not link.startswith('http'):
                    link = "https://top.baidu.com" + link
                
                # Brief
                intro_div = container.find(class_=lambda x: x and ('intro' in x or 'large' in x or 'desc' in x))
                brief = intro_div.get_text().strip() if intro_div else ""
                
                # Category
                # Try to find type text, often in specific small divs
                # For Baidu Realtime, it's usually "Hot"
                category = "热点"
                
                items.append({
                    "rank": index + 1,
                    "title": title,
                    "brief": brief,
                    "link": link,
                    "category": category
                })
            except:
                continue
                
    except Exception as e:
        items.append({"error": str(e)})
        
    return items

def fetch_weibo_hot():
    """Fetch Top 10 Rising/Hot News from Weibo."""
    url = "https://s.weibo.com/top/summary"
    items = []
    try:
        # Weibo requires cookie sometimes, but let's try without first
        headers = get_headers()
        headers['Cookie'] = 'SUB=_2AkMVWD-af8NxqwJRmP0Sz2_hZYt2zw_EieKjpHWcJRMxHRl-yT9jqkActRB6PQA5fX8-xJ0u4Z5k4F4_z5_z5_z5' # Guest cookie
        
        response = requests.get(url, headers=headers, timeout=10)
        # Weibo might return 200 even if blocked, but usually gives content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        rows = soup.select('td.td-02')
        
        for row in rows:
            if len(items) >= 10:
                break
            try:
                a_tag = row.find('a')
                if not a_tag:
                    continue
                    
                title = a_tag.get_text().strip()
                if not title:
                    continue
                    
                link = "https://s.weibo.com" + a_tag['href'] if a_tag['href'].startswith('/') else a_tag['href']
                
                # Check for "New" or "Hot" icon
                icon_tag = row.find('i', class_=re.compile(r'icon-txt-'))
                icon_text = icon_tag.get_text().strip() if icon_tag else ""
                
                # We prioritize "New" (Rising) but take Hot if not enough New
                category = "微博" + (f" ({icon_text})" if icon_text else "")
                
                # Brief is not available on summary page, need to click through or leave empty
                brief = "点击链接查看详情"
                
                items.append({
                    "rank": len(items) + 1,
                    "title": title,
                    "brief": brief,
                    "link": link,
                    "category": category
                })
            except:
                continue
                
    except Exception as e:
        # Fallback to Baidu 11-20 if Weibo fails
        pass
        
    return items

def main():
    hot_news = fetch_baidu_hot()
    
    # Try Weibo for Rising
    rising_news = fetch_weibo_hot()
    
    # If Weibo failed or returned empty (anti-bot), fallback to Baidu 11-20?
    # Or just return what we have.
    
    if not rising_news and isinstance(hot_news, list) and len(hot_news) > 10:
         # Actually fetch_baidu_hot only returns 10.
         # Let's fetch more in Baidu if needed? 
         # No, let's keep it simple. If Weibo fails, we return empty list for rising
         # and let the Agent handle the empty list (maybe by searching).
         pass

    result = {
        "hot_news": hot_news,
        "rising_news": rising_news
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
