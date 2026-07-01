import yaml
from openai import OpenAI

class AIProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.llm_config = self.config.get('llm', {})
        self.api_key = self.llm_config.get('api_key')
        self.base_url = self.llm_config.get('base_url')
        self.model = self.llm_config.get('model', 'deepseek-chat')
        self.prompts = self.config.get('prompts', {})
        
        if not self.api_key or self.api_key == "YOUR_LLM_API_KEY":
            print("Warning: LLM API Key not configured. AI features will be skipped.")
            self.client = None
        else:
            # Zhipu AI / OpenAI Compatible Client
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def optimize_content(self, text: str, style: str = "professional") -> str:
        """
        Rewrite content to match a specific style.
        """
        if not self.client:
            return text

        # Get prompt from config or fallback to default
        prompt_template = self.prompts.get('optimize', "")
        if not prompt_template:
            # Fallback prompt
            prompt_template = f"""
            You are an expert content editor for a WeChat Official Account (Self-media).
            Please rewrite the following article content to make it more engaging, readable, and suitable for the "{style}" style.
            Keep the original meaning and Markdown formatting (headers, code blocks, links, images) intact.
            Do not add introductory or concluding remarks, just return the optimized Markdown content.
            
            Content:
            """
        
        # Format the prompt with style if needed (assuming user might keep {style} placeholder)
        try:
            prompt = prompt_template.format(style=style)
        except KeyError:
            # In case the user removed {style} or added other keys
            prompt = prompt_template
        
        # Append content
        full_prompt = f"{prompt}\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error optimizing content: {e}")
            return text

    def recommend_template(self, text: str) -> str:
        """
        Analyze content and recommend a CSS template.
        """
        if not self.client:
            return "premium" # Default

        prompt = self.prompts.get('recommend_template', "")
        if not prompt:
            return "premium"

        # Limit preview text to save tokens
        preview_text = text[:1000]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt + "\n" + preview_text}]
            )
            template = response.choices[0].message.content.strip().lower()
            # Basic validation
            if template in ['premium', 'tech', 'prose', 'colorful']:
                return template
            return "premium"
        except Exception as e:
            print(f"Error recommending template: {e}")
            return "premium"

    def review_content(self, text: str) -> str:
        """
        Check for grammar errors and prohibited terms.
        Returns the corrected text.
        """
        if not self.client:
            return text

        prompt = self.prompts.get('review', "")
        if not prompt:
            prompt = """
            You are a content compliance officer for WeChat Official Accounts.
            Please review the following article content:
            1. Correct any grammar or spelling errors.
            2. Replace any sensitive words, prohibited terms (e.g., political sensitivity, absolute superlatives like 'best', 'No.1' if potentially violating ad laws), or inappropriate language with compliant alternatives.
            3. Ensure the tone is positive and compliant with platform rules.
            4. Keep the Markdown formatting intact.
            5. Return ONLY the corrected Markdown content.
            
            Content:
            """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt + "\n" + text}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error reviewing content: {e}")
            return text

    def process_metadata(self, content: str) -> str:
        """
        Ensure content has a title and abstract.
        If missing, generate them using AI.
        Returns the updated content with Title and Abstract.
        """
        if not self.client:
            return content

        lines = content.split('\n')
        
        # 1. Check/Generate Title
        # Heuristic: First line starting with # is title
        has_title = False
        title = ""
        # Check first 5 lines for a title
        for line in lines[:5]: 
            if line.strip().startswith('# '):
                has_title = True
                title = line.strip()[2:]
                break
        
        if not has_title:
            print("  - Generating title with AI...")
            prompt = self.prompts.get('generate_title', "Generate a title for this article.")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt + "\n\n" + content[:2000]}]
                )
                title = response.choices[0].message.content.strip().replace('"', '')
                # Add title at the beginning
                content = f"# {title}\n\n{content}"
            except Exception as e:
                print(f"Error generating title: {e}")

        # 2. Check/Generate Abstract
        # Heuristic: Look for blockquote `> ` in the first few lines
        has_abstract = False
        # Re-split content in case title was added
        lines = content.split('\n')
        
        for i, line in enumerate(lines[:10]):
            if line.strip().startswith('> '):
                has_abstract = True
                break
            if "摘要" in line and len(line) < 20: 
                has_abstract = True
                break
        
        if not has_abstract:
            print("  - Generating abstract with AI...")
            prompt = self.prompts.get('generate_abstract', "Generate an abstract.")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt + "\n\n" + content}]
                )
                abstract = response.choices[0].message.content.strip()
                
                # Insert abstract
                if content.strip().startswith('# '):
                    # Find end of title line
                    first_newline = content.find('\n')
                    if first_newline != -1:
                        # Insert after title
                        content = content[:first_newline] + f"\n\n> {abstract}\n" + content[first_newline:]
                    else:
                        content = content + f"\n\n> {abstract}"
                else:
                    content = f"> {abstract}\n\n{content}"
                    
            except Exception as e:
                print(f"Error generating abstract: {e}")

        return content
