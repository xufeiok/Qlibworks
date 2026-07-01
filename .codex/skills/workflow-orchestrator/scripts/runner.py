import os
import yaml
import argparse
import subprocess
import sys
import re

def load_workflow(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve_skill_path(skill_name):
    # Assuming standard structure e:\Quant\.trae\skills\<skill_name>
    # We are in e:\Quant\.trae\skills\workflow-orchestrator\scripts\runner.py
    # So skill root is ../../<skill_name>
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    skill_dir = os.path.join(base_dir, skill_name)
    if not os.path.exists(skill_dir):
        raise FileNotFoundError(f"Skill '{skill_name}' not found in {base_dir}")
    return skill_dir

def run_step(step, context):
    print(f"--- Running Step: {step.get('name', step['id'])} ---")
    
    skill_name = step['skill']
    skill_path = resolve_skill_path(skill_name)
    
    command_template = step['command']
    
    # Variable substitution
    # Replace {step_id.output} with actual output
    for key, value in context.items():
        placeholder = f"{{{key}.output}}"
        if placeholder in command_template:
            command_template = command_template.replace(placeholder, value.strip())
            
    # Also support {project_root}
    project_root = os.path.abspath(os.path.join(skill_path, "../../../"))
    command_template = command_template.replace("{project_root}", project_root)

    # Prepend skill path to script if it's a relative path in the command
    # Heuristic: if command starts with "python scripts/", make it absolute
    # Actually, better to set cwd to the skill directory
    cwd = skill_path
    
    print(f"Command: {command_template}")
    print(f"CWD: {cwd}")
    
    try:
        # We use shell=True to allow complex commands, but we need to be careful
        result = subprocess.run(
            command_template, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise Exception(f"Step {step['id']} failed")
            
        print(f"Output: {result.stdout[:200]}...") # truncate log
        return result.stdout
        
    except Exception as e:
        print(f"Execution failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Workflow Runner")
    parser.add_argument("workflow_file", help="Path to workflow YAML file")
    args = parser.parse_args()
    
    try:
        workflow = load_workflow(args.workflow_file)
        print(f"Starting Workflow: {workflow.get('name', 'Untitled')}")
        
        context = {}
        
        for step in workflow['steps']:
            output = run_step(step, context)
            context[step['id']] = output
            
        print("Workflow completed successfully.")
        
    except Exception as e:
        print(f"Workflow failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
