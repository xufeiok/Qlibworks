import os
import sys
import json
import time
import subprocess
import re

# Add skill paths
# Current file: e:\Quant\.trae\skills\workflow-orchestrator\workflows\auto_news_publisher.py
# Project root (e:\Quant) is 4 levels up
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_root, '.trae', 'skills', 'trending-news-scanner', 'scripts'))
sys.path.append(os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat'))

try:
    from get_trending import fetch_weibo_hot, fetch_baidu_hot
    from src.ai_processor import AIProcessor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class NewsTopicGenerator:
    """
    Implements the 'news-topic-generator' skill workflow.
    Ref: .trae/skills/news-topic-generator/SKILL.md
    """
    def __init__(self, ai_client):
        self.ai = ai_client

    def run(self, news_items):
        print("--- [Skill: News Topic Generator] ---")
        
        # Step 1: Selection
        print("Step 1: Selecting best news from candidates...")
        selected_news = self._step1_selection(news_items)
        print(f"Selected: {selected_news['title']}")
        
        # Step 2: Dismantling
        print("Step 2: Dismantling news...")
        analysis = self._step2_dismantling(selected_news)
        
        # Step 3: Ideation
        print("Step 3: Generating topics...")
        topics = self._step3_ideation(selected_news, analysis)
        
        # Step 4: Evaluation
        print("Step 4: Evaluating topics...")
        best_topic = self._step4_evaluation(topics)
        print(f"Best Topic: {best_topic['title']} (Score: {best_topic['score']})")
        
        return best_topic

    def _step1_selection(self, news_items):
        # Limit to top 5 to save tokens
        candidates = news_items[:5]
        candidates_str = json.dumps([{k: v for k, v in item.items() if k in ['title', 'brief', 'category']} for item in candidates], ensure_ascii=False)
        
        prompt = f"""
        You are a News Editor focused on Personal Growth.
        
        CRITICAL FILTERING (SAFETY FIRST):
        You MUST REJECT any news that involves:
        - National Leaders (e.g., Presidents, PMs, senior officials)
        - International Politics / Diplomacy / Military
        - Government Policy Announcements
        - Crime / Violence / Disasters
        
        Task: Select the best "Safe" news item for a Personal Growth article.
        
        Evaluate SAFE items on:
        Layer 1: General Quality (Timeliness, Social Relevance, Debate, Emotion)
        Layer 2: Strategic Fit (Can it generate topics on: Personal Growth, Career, Family, Reading, Skills?)
        
        Decision Logic:
        1. Filter out ALL unsafe items.        
        2. From the safe ones, pick the one with the highest Personal Growth Potential.
        
        News Items:
        {candidates_str}
        
        Return ONLY the index (0-based).
        """
        try:
            resp = self.ai.client.chat.completions.create(
                model=self.ai.model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content.strip()
            # Extract first number
            match = re.search(r'\d+', content)
            idx = int(match.group()) if match else 0
            if 0 <= idx < len(candidates):
                return candidates[idx]
            return candidates[0]
        except:
            return candidates[0]

    def _step2_dismantling(self, news):
        prompt = f"""
        Deeply analyze this news: "{news['title']}" - "{news.get('brief', '')}"
        
        Perform "Deep Dismantling":
        1. 5W1H Re-focus (Who, What, When, Where, Why, How) - Look for hidden angles.
        2. 4-Dimension Extension (Depth, Breadth, Temperature, Attitude).
        
        Output a concise analysis paragraph.
        """
        resp = self.ai.client.chat.completions.create(
            model=self.ai.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    def _step3_ideation(self, news, analysis):
        prompt = f"""
        Based on this news "{news['title']}" and analysis:
        {analysis}
        
        Task: Generate 4 distinct writing topics.
        
        CONSTRAINT 1 (DIRECTION): Every topic MUST strictly belong to one of these 5 directions:
        1. Personal Growth (Self-improvement, mindset)
        2. Career & Learning (Work skills, efficiency, career path)
        3. Family Life (Relationships, parenting, balance)
        4. Reading & Knowledge (Book reviews, cognitive models)
        5. Personal Skills (Practical abilities, tools)
        
        CONSTRAINT 2 (SAFETY):
        - NO political commentary.
        - If the news is political, PIVOT to a soft skill or abstract lesson (e.g., "Leadership under pressure" rather than "Policy analysis").
        
        Apply these Creative Methods to the directions above:
        1. Niche Perspective (Focus on the ignored)
        2. Reverse Thinking (Rational counter-point)
        3. Small-to-Big (Microcosm)
        4. History/Culture Contrast
        
        Return JSON array: [{{"title": "...", "news_context": "Brief 1-sentence summary of the news event", "direction": "...", "method": "...", "angle": "..."}}, ...]
        """
        try:
            resp = self.ai.client.chat.completions.create(
                model=self.ai.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            # Handle potential non-JSON wrapping
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            # Robust extraction
            if isinstance(data, list):
                topics = data
            elif isinstance(data, dict):
                topics = data.get('topics', [])
                if not topics:
                    # Try to find any list in the dict values
                    for v in data.values():
                        if isinstance(v, list):
                            topics = v
                            break
            else:
                topics = []

            if not topics:
                 raise ValueError("No topics found in JSON")
                 
            return topics
        except Exception as e:
            print(f"Step 3 Error: {e}")
            # Fallback
            return [
                {"title": f"Deep Analysis of {news['title']}", "news_context": f"News about {news['title']}", "direction": "Personal Growth", "method": "Standard", "angle": "General"},
                {"title": f"What {news['title']} teaches us about Life", "news_context": f"News about {news['title']}", "direction": "Personal Skills", "method": "Small-to-Big", "angle": "Lesson"}
            ]

    def _step4_evaluation(self, topics):
        if not topics:
             return {"title": "Default Topic", "score": 5}
             
        topics_str = json.dumps(topics, ensure_ascii=False)
        prompt = f"""
        Evaluate these topics (1-10 score) on Novelty, Depth, and Writeability.
        Return the index of the BEST topic. Return ONLY the number.
        
        Topics:
        {topics_str}
        """
        try:
            resp = self.ai.client.chat.completions.create(
                model=self.ai.model,
                messages=[{"role": "user", "content": prompt}]
            )
            match = re.search(r'\d+', resp.choices[0].message.content)
            idx = int(match.group()) if match else 0
            if 0 <= idx < len(topics):
                t = topics[idx]
                t['score'] = 9.0 # Placeholder score
                return t
            return topics[0]
        except:
            return topics[0]

class ArticleWriter:
    """
    Implements the 'copywriting' skill (Professional Article Writer mode).
    Ref: .trae/skills/copywriting/SKILL.md
    """
    def __init__(self, ai_client):
        self.ai = ai_client

    def write(self, topic, news_title):
        print("--- [Skill: Copywriting (Article Writer)] ---")
        
        # Step 1-3: Plan
        print("Phase 1: Planning Structure...")
        plan = self._phase_planning(topic, news_title)
        
        # Step 4-8: Write
        print("Phase 2: Writing Content...")
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
    # Init AI
    config_path = os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    ai = AIProcessor(config_path)
    if not ai.client:
        print("AI Client not initialized.")
        return

    # 1. Fetch
    print(">>> [Scanner] Fetching News...")
    news_items = fetch_weibo_hot()
    if not news_items:
        news_items = fetch_baidu_hot()
    if not news_items:
        print("No news found.")
        return

    # 2. Generate Topic
    topic_gen = NewsTopicGenerator(ai)
    best_topic = topic_gen.run(news_items)

    # 3. Write Article
    writer = ArticleWriter(ai)
    article_content = writer.write(best_topic, news_items[0]['title']) # Pass original title for context

    # 4. Save
    output_dir = os.path.join(project_root, 'articles')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract title from content or use topic title
    title_match = re.search(r'^#\s+(.+)$', article_content, re.MULTILINE)
    file_title = title_match.group(1) if title_match else best_topic['title']
    safe_title = "".join([c for c in file_title if c.isalnum() or c in (' ', '-', '_')]).strip()[:30]
    
    filename = f"auto_{safe_title}_{int(time.time())}.md"
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(article_content)
    print(f"\nArticle saved to: {file_path}")

    # 5. Publish
    print("\n>>> [Publisher] Publishing to WeChat...")
    publish_script = os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat', 'main.py')
    cmd = [sys.executable, publish_script, file_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
