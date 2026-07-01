from PIL import Image, ImageDraw, ImageFont
import os

def generate_default_cover():
    # Settings
    width, height = 900, 383  # WeChat cover aspect ratio approx 2.35:1
    bg_color = (66, 185, 131) # WeChat Green
    text_color = (255, 255, 255)
    text = "WECHAT\nPUBLISH"
    
    # Create Image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add simple pattern (circles)
    draw.ellipse((800, -50, 950, 100), fill=(255, 255, 255, 50))
    draw.ellipse((-50, 300, 100, 450), fill=(255, 255, 255, 50))
    
    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, '..', 'assets', 'default_cover.jpg')
    img.save(output_path, quality=85)
    print(f"Default cover generated at: {output_path}")

if __name__ == "__main__":
    generate_default_cover()
