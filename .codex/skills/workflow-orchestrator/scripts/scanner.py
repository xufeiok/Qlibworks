import os
import yaml
import json
import glob

def scan_skills(skills_root):
    skills = []
    # skills_root is e:\Quant\.trae\skills
    # We are in scripts/, so up two levels
    if not os.path.isabs(skills_root):
        skills_root = os.path.abspath(skills_root)

    print(f"Scanning skills in: {skills_root}")
    
    for skill_dir in os.listdir(skills_root):
        skill_path = os.path.join(skills_root, skill_dir)
        if not os.path.isdir(skill_path):
            continue
            
        skill_info = {
            "name": skill_dir,
            "path": skill_path,
            "description": "",
            "executable": False,
            "scripts": []
        }
        
        # Parse SKILL.md for description
        skill_md_path = os.path.join(skill_path, "SKILL.md")
        if os.path.exists(skill_md_path):
            try:
                with open(skill_md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # extract frontmatter
                    if content.startswith('---'):
                        parts = content.split('---', 2)
                        if len(parts) >= 3:
                            frontmatter = yaml.safe_load(parts[1])
                            if frontmatter and 'description' in frontmatter:
                                skill_info['description'] = frontmatter['description']
            except Exception as e:
                print(f"Error reading {skill_md_path}: {e}")

        # Check for scripts
        scripts_dir = os.path.join(skill_path, "scripts")
        if os.path.isdir(scripts_dir):
            skill_info['executable'] = True
            for f in os.listdir(scripts_dir):
                if f.endswith(('.py', '.sh', '.js')):
                    skill_info['scripts'].append(f)
        
        skills.append(skill_info)
    
    return skills

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming scripts/ is inside the skill dir, and skill dir is inside .trae/skills
    # e:\Quant\.trae\skills\workflow-orchestrator\scripts -> e:\Quant\.trae\skills
    skills_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    skills = scan_skills(skills_root)
    print(json.dumps(skills, indent=2, ensure_ascii=False))
