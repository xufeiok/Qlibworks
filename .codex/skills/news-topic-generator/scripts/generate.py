import argparse
import json
import sys
import os
import re

# Add path to AIProcessor in publish-to-wechat skill
# Assuming we are in .trae/skills/news-topic-generator/scripts/
# We need to go up 3 levels to .trae/skills/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(os.path.join(project_root, '.trae', 'skills', 'publish-to-wechat'))

try:
    from src.ai_processor import AIProcessor
except ImportError:
    print("Error: Could not import AIProcessor from publish-to-wechat skill")
    sys.exit(1)

class NewsTopicGenerator:
    """
    Implements the 'news-topic-generator' skill workflow.
    Ref: .trae/skills/news-topic-generator/SKILL.md
    """
    def __init__(self, ai_client):
        self.ai = ai_client

    def run(self, news_items):
        # Step 1: Selection
        selected_news = self._step1_selection(news_items)
        
        # Step 2: Dismantling
        analysis = self._step2_dismantling(selected_news)
        
        # Step 3: Ideation
        topics = self._step3_ideation(selected_news, analysis)
        
        # Step 4: Evaluation
        best_topic = self._step4_evaluation(topics)
        
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
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            if isinstance(data, list):
                topics = data
            elif isinstance(data, dict):
                topics = data.get('topics', [])
                if not topics:
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
                t['score'] = 9.0
                return t
            return topics[0]
        except:
            return topics[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input News JSON file")
    parser.add_argument("--output", required=True, help="Output Topic JSON file")
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
        news_items = json.load(f)
        
    if not news_items:
        print("Error: No news items in input file")
        sys.exit(1)
        
    generator = NewsTopicGenerator(ai)
    best_topic = generator.run(news_items)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    abs_output_path = os.path.abspath(args.output)
    
    with open(abs_output_path, 'w', encoding='utf-8') as f:
        json.dump(best_topic, f, ensure_ascii=False, indent=2)

    # Print absolute path for workflow orchestration
    print(abs_output_path)

if __name__ == "__main__":
    main()
