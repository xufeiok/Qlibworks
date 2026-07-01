import argparse
import sys
import os
import re

# Add path to AIProcessor
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat'))

try:
    from src.ai_processor import AIProcessor
except ImportError:
    print("Error: Could not import AIProcessor")
    sys.exit(1)

class VisualStoryDesigner:
    def __init__(self, ai_client):
        self.ai = ai_client

    def generate(self, article_content):
        prompt = f"""
You are an expert **Editorial Designer** and **Information Architect**. Your goal is to distill articles into **High-Impact Typographic Slides** (SVG).

**Your output is NOT about drawing complex illustrations. It is about typesetting text beautifully.**
You turn boring articles into "Insta-worthy" or "Xiaohongshu-style" text cards.

## Core Philosophy
- **Content is King**: The text *is* the visual. Use font size, weight, and color to create hierarchy.
- **Rich Context**: Don't just give slogans. Provide **actionable details** and **nuanced explanations**. We want depth, not just clickbait.
- **Manual Typesetting**: SVG does not auto-wrap text. **You must manually break lines** using `<tspan>`.
- **No Overlap**: You must carefully calculate line heights (`dy`) to ensure text never overlaps.

## Workflow

### 1. Content Distillation
- Read the input text.
- **Deep Dive**: Extract 3-5 core insights, but keep the *context* and *nuance*.
- **Rewrite**: Convert insights into a structured format:
  - **Headline**: The core concept (Punchy, 5-10 chars).
  - **Explanation**: The "Why" and "How" (Rich detail, 30-50 chars).
  - **Key Data/Quote**: A supporting fact or golden sentence.

### 2. Storyboard Structure (4-7 Slides)
| Slide | Role | Content Focus |
| :--- | :--- | :--- |
| **1. Cover** | The Hook | Big Headline + Subtitle + Decorative Shape/Line. |
| **2-N. Body** | The Value | **Structure**: Number -> Headline -> Rich Explanation. |
| **Last. Outro** | The Ask | "Summary" + Call to Action (Save/Share). |

### 3. SVG Visual System
The SVG output must use the following advanced design system. Do not use plain white backgrounds.

#### Design Language: "Modern Editorial"
- **Dimensions**: `viewBox="0 0 1080 1440"` (Vertical 3:4/9:16)
- **Typography**: Google Fonts (Noto Sans SC) via `@import`.
  - **IMPORTANT**: You MUST escape the ampersand in the URL: `...wght@400;900&amp;display=swap`
- **Color Palette**:
  - **Backgrounds**: Soft gradients or dark mode glassmorphism.
  - **Accents**: High contrast colors (Red #E02020, Gold #FFD700, Electric Blue #007AFF).

#### Standard `<defs>` Template (Copy this exact block)
```xml
<defs>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;700;900&amp;display=swap');
    .title {{ font-family: 'Noto Sans SC', sans-serif; font-weight: 900; }}
    .subtitle {{ font-family: 'Noto Sans SC', sans-serif; font-weight: 700; }}
    .text {{ font-family: 'Noto Sans SC', sans-serif; font-weight: 400; }}
    .detail {{ font-family: 'Noto Sans SC', sans-serif; font-weight: 300; }}
  </style>
  <!-- Gradients -->
  <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" style="stop-color:#F5F7FA;stop-opacity:1" />
    <stop offset="100%" style="stop-color:#C3CFE2;stop-opacity:1" />
  </linearGradient>
  <linearGradient id="grad_dark" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" style="stop-color:#232526;stop-opacity:1" />
    <stop offset="100%" style="stop-color:#414345;stop-opacity:1" />
  </linearGradient>
  <linearGradient id="grad_gold" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" style="stop-color:#F2994A;stop-opacity:1" />
    <stop offset="100%" style="stop-color:#F2C94C;stop-opacity:1" />
  </linearGradient>
  <!-- Filters -->
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="10"/>
    <feOffset dx="5" dy="10" result="offsetblur"/>
    <feComponentTransfer>
      <feFuncA type="linear" slope="0.3"/>
    </feComponentTransfer>
    <feMerge> 
      <feMergeNode/>
      <feMergeNode in="SourceGraphic"/> 
    </feMerge>
  </filter>
  <filter id="glass" x="0" y="0" width="100%" height="100%">
    <feGaussianBlur in="SourceGraphic" stdDeviation="10" />
  </filter>
</defs>
```

#### Layout Components
1. **Glass Card**: `<rect fill="rgba(255,255,255,0.1)" stroke="rgba(255,255,255,0.2)" rx="30" filter="url(#shadow)" ... />`
2. **Highlight Marker**: `<path d="..." fill="url(#grad_gold)" opacity="0.8" />`
3. **Typography Hierarchy** (Strict Sizing):
   - **Title**: 80px (dy=100), Heavy weight (900). Max 10 chars/line.
   - **Subtitle**: 50px (dy=75), Bold weight (700). Max 15 chars/line.
   - **Body/Detail**: 35px (dy=55), Light/Regular (300/400). Max 22 chars/line. **CRITICAL: Generous line height prevents overlap.**

### Content-to-SVG Rules
- **Text Wrapping**: SVG does not wrap text. You MUST manually split text into `<tspan>` lines.
- **Line Spacing (dy)**:
  - For Title (80px), use `dy="100"`.
  - For Subtitle (50px), use `dy="75"`.
  - For Body (35px), use `dy="55"`.
- **Safe Area**: Keep text within x=80 to x=1000. Avoid writing below y=1300.

## Output Generation
Generate a Markdown report containing the SVGs.

## Output Format
```markdown
# Visual Summary: [Title]

## Slide 1: Cover
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1440">
  ...
</svg>
```
...
```

Article Content:
{article_content}
"""
        resp = self.ai.client.chat.completions.create(
            model=self.ai.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input Article Markdown file")
    args = parser.parse_args()

    # Config
    config_path = os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    ai = AIProcessor(config_path)
    if not ai.client:
        print("Error: AI Client not initialized")
        sys.exit(1)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        article_content = f.read()

    designer = VisualStoryDesigner(ai)
    storyboard_md = designer.generate(article_content)

    # Output filename: [original]_Storyboard.md
    base_name = os.path.splitext(args.input_file)[0]
    output_file = f"{base_name}_Storyboard.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(storyboard_md)
        
    print(output_file)

if __name__ == "__main__":
    main()
