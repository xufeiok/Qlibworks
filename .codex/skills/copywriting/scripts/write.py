import argparse
import json
import sys
import os
import re
import time

# Add path to AIProcessor in publish-to-wechat skill
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat'))

try:
    from src.ai_processor import AIProcessor
except ImportError:
    print("Error: Could not import AIProcessor from publish-to-wechat skill")
    sys.exit(1)

class ArticleWriter:
    """
    Implements the 'copywriting' skill (Professional Article Writer mode).
    Ref: .trae/skills/copywriting/SKILL.md
    """
    def __init__(self, ai_client):
        self.ai = ai_client

    def write(self, topic, news_title):
        # Phase 1: Planning
        plan = self._phase_planning(topic, news_title)
        
        # Phase 2: Writing
        content = self._phase_writing(plan, topic)
        
        return content

    def _phase_planning(self, topic, news_title):
        prompt = f"""
        You are a Professional Article Writer.
        Task: Plan an article about "{news_title}" with the specific angle: "{topic['title']}".
        
        SAFETY CONSTRAINT:
        - Avoid controversial political opinions.
        - Focus on personal growth, human nature, and positive values.
        
        Follow this workflow:
        1. Define Subject & Audience.
        2. Core Thesis (One sentence).
        3. Pyramid Structure Outline:
           - Introduction: Must include a 'News Hook' that briefly describes the event ({topic.get('news_context', news_title)}) to leverage the trend.
           - Arguments: 3 key points.
           - Actionable Advice: Concrete steps for the reader.
           - Conclusion.
        
        Output the plan.
        """
        resp = self.ai.client.chat.completions.create(
            model=self.ai.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    def _phase_writing(self, plan, topic):
        prompt = f"""
        You are a Professional Article Writer.
        Write a high-quality article based on this plan:
        {plan}
        
        SAFETY CONSTRAINT:
        - Ensure content is safe, positive, and compliant with WeChat regulations.
        - NO sensitive keywords (politics, violence).
        
        Follow these specific rules:
        1. Headline: Number + Emotion + Benefit (or use "{topic['title']}" if better).
        2. Opening: MUST explicitly mention the news event '{topic.get('news_context', 'current event')}' to grab attention (The "Hot Topic Hook"). Then transition to the pain point.
        3. Flow & Transitions (CRITICAL):
           - Use transitional phrases between paragraphs (e.g., "However", "This leads us to...", "More importantly").
           - Ensure the text reads like a coherent narrative, not a disjointed list.
           - Avoid abrupt jumps between arguments.
        4. Body: Topic sentences, diverse evidence (stories, data), active voice.
        5. Actionable Advice: When giving suggestions, provide SPECIFIC, EXECUTABLE STEPS (e.g., "Step 1:...", "Use tool X..."), not just vague theory.
        6. Ending: Summary + Emotional Resonance + Call to Action.
        7. Language: Simplified Chinese. Vivid, viral style.
        8. Format: Markdown.
        
        Return ONLY the Markdown content.
        """
        resp = self.ai.client.chat.completions.create(
            model=self.ai.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Topic JSON file")
    parser.add_argument("--output", required=True, help="Output Markdown file")
    args = parser.parse_args()

    # Use config from publish-to-wechat
    config_path = os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
        
    ai = AIProcessor(config_path)
    if not ai.client:
        print("Error: AI Client not initialized")
        sys.exit(1)
    
    with open(args.input, 'r', encoding='utf-8') as f:
        topic = json.load(f)
        
    writer = ArticleWriter(ai)
    
    # We need a fallback for news_title if not in topic
    # The topic object usually has 'news_context' which is a summary.
    # We can use that or just the topic title.
    news_context = topic.get('news_context', topic['title'])
    
    article_content = writer.write(topic, news_context)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    abs_output_path = os.path.abspath(args.output)
    
    with open(abs_output_path, 'w', encoding='utf-8') as f:
        f.write(article_content)
        
    # Print absolute path for workflow orchestration
    print(abs_output_path)

if __name__ == "__main__":
    main()
