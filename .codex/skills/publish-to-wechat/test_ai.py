from src.ai_processor import AIProcessor
import os

config_path = os.path.abspath(".trae/skills/publish-to-wechat/config.yaml")
processor = AIProcessor(config_path)

print("Testing AI Connection...")
text = "这是一个测试文章。我们需要测试AI是否能正常工作。这是关于财务自由的。"
try:
    print("1. Testing Optimization...")
    optimized = processor.optimize_content(text)
    print("Optimization Result Prefix:", optimized[:100])
    
    print("\n2. Testing Template Recommendation...")
    template = processor.recommend_template(text)
    print("Recommended Template:", template)
    
    print("\nTest Complete.")
except Exception as e:
    print(f"Error: {e}")
