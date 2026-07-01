import schedule
import time
import argparse
import subprocess
import yaml
import os
import sys

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_workflow(workflow_path, script_dir):
    runner_script = os.path.join(script_dir, "runner.py")
    print(f"Executing scheduled workflow: {workflow_path}")
    subprocess.run(["python", runner_script, workflow_path])

def main():
    parser = argparse.ArgumentParser(description="Workflow Scheduler")
    parser.add_argument("workflow_file", help="Path to workflow YAML file")
    args = parser.parse_args()
    
    workflow_path = os.path.abspath(args.workflow_file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        config = load_config(workflow_path)
        if 'schedule' not in config:
            print("No 'schedule' field found in workflow file.")
            sys.exit(1)
            
        time_str = config['schedule']
        print(f"Scheduling workflow '{config.get('name')}' at {time_str} daily...")
        
        # Simple daily scheduler
        schedule.every().day.at(time_str).do(run_workflow, workflow_path, script_dir)
        
        while True:
            schedule.run_pending()
            time.sleep(10) # check every 10s
            
    except KeyboardInterrupt:
        print("Scheduler stopped.")
    except Exception as e:
        print(f"Scheduler error: {e}")

if __name__ == "__main__":
    main()
