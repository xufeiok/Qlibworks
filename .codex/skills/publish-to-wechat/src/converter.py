import markdown
import yaml
import os
from bs4 import BeautifulSoup
from premailer import transform

class MarkdownConverter:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Base directory for templates
        # config_path is usually in the skill root directory
        self.skill_dir = os.path.dirname(os.path.abspath(config_path))
        self.templates_dir = os.path.join(self.skill_dir, 'templates')

    def load_template(self, template_name: str) -> str:
        """Load CSS content from a template file."""
        # Clean template name to prevent directory traversal
        template_name = os.path.basename(template_name)
        if not template_name.endswith('.css'):
            template_name += '.css'
            
        template_path = os.path.join(self.templates_dir, template_name)
        
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error loading template {template_name}: {e}")
        
        # Fallback to default or empty if not found
        print(f"Template {template_name} not found, checking 'tech.css' or using default.")
        default_path = os.path.join(self.templates_dir, 'tech.css')
        if os.path.exists(default_path):
             with open(default_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        return self.config.get('content', {}).get('css_theme', '')

    def convert(self, markdown_text: str, template_name: str = None) -> str:
        """
        Convert Markdown to HTML and inject CSS styles.
        """
        # Determine template
        if not template_name:
            template_name = self.config.get('content', {}).get('default_template', 'tech')
        
        css_content = self.load_template(template_name)

        # Convert Markdown to HTML
        html_content = markdown.markdown(
            markdown_text,
            extensions=['fenced_code', 'tables', 'attr_list', 'nl2br']
        )

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Add inline styles for images to ensure they fit
        for img in soup.find_all('img'):
            img['style'] = "max-width: 100%; height: auto; display: block; margin: 20px auto; border-radius: 8px;"

        # Wrap in a container with the custom style and use premailer to inline CSS
        # WeChat requires inline styles as <style> tags are often stripped or ignored
        html_with_style = f"""
        <html>
        <head>
            <style>
            {css_content}
            </style>
        </head>
        <body>
            <div class="markdown-body" id="wechat-content">
                {soup.prettify()}
            </div>
        </body>
        </html>
        """
        
        # Transform CSS to inline styles
        # remove_classes=False: Keep classes for structure/JS (though WeChat ignores JS)
        # strip_important=False: Keep !important if present
        try:
            inlined_html = transform(html_with_style, remove_classes=False, strip_important=False)
            
            # Extract the content div back
            inlined_soup = BeautifulSoup(inlined_html, 'html.parser')
            content_div = inlined_soup.find('div', id='wechat-content')
            
            if content_div:
                return str(content_div)
            return inlined_html
        except Exception as e:
            print(f"Error inlining CSS: {e}. Falling back to style tag method.")
            # Fallback to original method
            return f"""
            <section id="wechat-article-container">
                <style>
                {css_content}
                </style>
                <section class="markdown-body">
                    {soup.prettify()}
                </section>
            </section>
            """
