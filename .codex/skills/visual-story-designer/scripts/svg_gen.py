import re
import os
import sys
import argparse
import xml.etree.ElementTree as ET

try:
    import cairosvg
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    print("Warning: cairosvg not found. PNG conversion will be skipped.")

def validate_svg(svg_content):
    """
    Validates SVG content using xml.etree.ElementTree.
    Returns (is_valid, error_message)
    """
    try:
        ET.fromstring(svg_content)
        return True, ""
    except ET.ParseError as e:
        return False, str(e)

def parse_svgs(md_content):
    """
    Parses the markdown content to find SVG code blocks associated with slides.
    Returns a list of dictionaries: [{'slide_num': 1, 'svg_code': '...'}]
    """
    slides = []
    
    # Split by "## Slide" or "### Slide" to isolate sections
    # Matches "## Slide" or "### Slide"
    sections = re.split(r'#{2,3} Slide', md_content)
    
    for section in sections:
        if not section.strip():
            continue
            
        # Try to find the slide number
        # Matches " 1: Title" or " 1"
        header_match = re.match(r'\s*(\d+)', section)
        if not header_match:
            continue
            
        slide_num = int(header_match.group(1))
        
        # Find SVG code block
        # Look for ```svg ... ``` or ```xml ... ```
        svg_match = re.search(r'```(?:svg|xml)\s*(.*?)```', section, re.DOTALL | re.IGNORECASE)
        
        if svg_match:
            svg_code = svg_match.group(1).strip()
            
            # XML Validation
            if svg_code.startswith('<svg'):
                is_valid, error_msg = validate_svg(svg_code)
                if not is_valid:
                    print(f"Warning: Invalid XML in Slide {slide_num}: {error_msg}")
                    # Try to auto-fix common errors
                    if "undefined entity" in error_msg and "&" in svg_code:
                         print("Attempting to fix unescaped ampersands...")
                         svg_code = svg_code.replace("&display", "&amp;display")
                         # Re-validate
                         is_valid, error_msg = validate_svg(svg_code)
                         if is_valid:
                             print("Fix successful!")
                         else:
                             print(f"Fix failed. Still invalid: {error_msg}")

            slides.append({
                'slide_num': slide_num,
                'svg_code': svg_code
            })
        else:
            print(f"[Warning] No SVG block found for Slide {slide_num}")
            
    return slides

def main():
    parser = argparse.ArgumentParser(description="Extract SVGs from Storyboard Markdown")
    parser.add_argument("file_path", help="Path to the markdown file")
    args = parser.parse_args()

    file_path = args.file_path
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Reading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    slides = parse_svgs(content)
    
    if not slides:
        print("No SVG slides found.")
        return

    # Create output directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), "images", base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output Directory: {output_dir}")

    for slide in slides:
        slide_num = slide['slide_num']
        svg_code = slide['svg_code']
        
        # Validate SVG start/end
        if not svg_code.strip().startswith('<svg'):
             # Sometimes there might be a comment before <svg
             # Try to find the first <svg
             match = re.search(r'(<svg.*?</svg>)', svg_code, re.DOTALL)
             if match:
                 svg_code = match.group(1)
             else:
                 print(f"[Warning] Slide {slide_num} does not look like valid SVG. Skipping.")
                 continue

        filename = f"Slide_{slide_num}.svg"
        save_path = os.path.join(output_dir, filename)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(svg_code)
            print(f"Saved: {filename}")
            
            # Convert to PNG if available
            if CAIRO_AVAILABLE:
                png_filename = filename.replace('.svg', '.png')
                png_save_path = os.path.join(output_dir, png_filename)
                try:
                    cairosvg.svg2png(url=save_path, write_to=png_save_path, output_width=1080, output_height=1440)
                    print(f"Converted: {png_filename}")
                except Exception as e:
                    print(f"PNG Conversion failed for {filename}: {e}")
                    
        except Exception as e:
            print(f"Failed to save {filename}: {e}")

    print("\nDone! You can view the SVGs in your browser.")

if __name__ == "__main__":
    main()
