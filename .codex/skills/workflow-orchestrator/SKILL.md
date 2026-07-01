---
name: workflow-orchestrator
description: A skill for discovering available skills, designing custom workflows, and orchestrating their execution (manual or scheduled).
---

# Workflow Orchestrator

This skill allows you to scan the local skill library, design automated workflows chaining multiple skills, and execute them on demand or on a schedule.

## Capabilities

1.  **Skill Discovery**: Scan local directories to find available skills and their executable scripts.
2.  **Workflow Planning**: Design YAML-based workflows to solve complex tasks using multiple skills.
3.  **Execution**: Run the designed workflows.
4.  **Scheduling**: Schedule workflows to run at specific times.

## Tools & Scripts

All scripts are located in `scripts/`.

-   **`scanner.py`**: Scans `.trae/skills` and lists available skills with their descriptions and scripts.
-   **`runner.py`**: Executes a workflow defined in a YAML file.
    -   Usage: `python scripts/runner.py <path_to_workflow.yaml>`
-   **`scheduler.py`**: Runs a workflow periodically based on the `schedule` field in the YAML.
    -   Usage: `python scripts/scheduler.py <path_to_workflow.yaml>`

## Workflow Template

The workflow YAML format is:

```yaml
name: "Task Name"
schedule: "09:00" # Optional: Daily run time (HH:MM)
steps:
  - id: step_unique_id
    skill: skill-directory-name
    command: command to run (e.g., python scripts/main.py --arg value)
```

## How to Use This Skill

### 1. Discovery Phase
When the user asks to "check available skills" or "what can I do?", run the scanner:
```bash
python scripts/scanner.py
```
Read the JSON output to understand what skills are available for orchestration.

### 2. Planning Phase
When the user wants to automate a task:
1.  Understand the goal.
2.  Match the goal to available skills (from the scan).
3.  Propose a workflow YAML file.
4.  Write the YAML file to the user's workspace (e.g., `workflows/my_task.yaml`).

### 3. Execution Phase
To run a workflow immediately:
```bash
python scripts/runner.py <path_to_workflow.yaml>
```

### 4. Scheduling Phase
To schedule a workflow:
```bash
python scripts/scheduler.py <path_to_workflow.yaml>
```
*Note: The scheduler script must be kept running (e.g., in a dedicated terminal) to trigger the task.*
