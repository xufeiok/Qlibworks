---
name: skill-from-github
description: Create skills by learning from high-quality GitHub projects. Use when users want to accomplish a task and you want to find existing tools/projects to learn from (e.g., "I want to convert markdown to PDF", "Analyze sentiment in customer reviews").
model: sonnet
---

# Skill from GitHub

When users want to accomplish something, search GitHub for quality projects that solve the problem, understand them deeply, then create a skill based on that knowledge.

## When to Use

When users describe a task and you want to find existing tools to learn from:

- "I want to convert markdown to PDF"
- "Help me analyze sentiment in customer reviews"
- "I need to generate API documentation from code"

## Workflow

### Step 1: Understand User Intent

Clarify:
- What is the input?
- What is the expected output?
- Any constraints (language, framework)?

### Step 2: Search GitHub

```
{task keywords} language:{preferred} stars:>100 sort:stars
```

**Quality filters (must meet ALL):**
- Stars > 100
- Updated within last 12 months
- Has README with clear documentation
- Has actual code

### Step 3: Present Options

Show top 3-5 candidates:

```markdown
## Found X projects

### Option 1: [project-name](github-url)
- Stars: xxx | Last updated: xxx
- What it does: one-line description
- Why it's good: specific strength

Which one should I dive into?
```

**Wait for user confirmation.**

### Step 4: Deep Dive into Selected Project

1. Read README - Understand purpose, features, usage
2. Read core source files - Understand how it works
3. Check examples - See real usage patterns
4. Note dependencies - What it relies on
5. Identify key concepts - The mental model

Extract:
- Core algorithm/approach
- Input/output formats
- Error handling patterns
- Best practices encoded

### Step 5: Summarize Understanding

```markdown
## Understanding [project-name]

### Core Approach
How it solves the problem...

### Key Techniques
1. ...
2. ...

### Limitations
- ...

### What I'll encode in the skill
- ...

Does this look right?
```

**Wait for user confirmation.**

### Step 6: Create Skill via skill-creator

Hand off to skill-creator with:

1. Clear purpose
2. Methodology - The approach learned from the project
3. Step-by-step process
4. Quality criteria
5. Common pitfalls

**Important:** The skill should encode the *knowledge* from the project, not just wrap the tool.

## Example

**User**: "I want to write better commit messages"

**Step 2 - Search:**
```
commit message convention stars:>100 sort:stars
```

**Step 3 - Present:**
```
## Found 4 projects

### Option 1: conventional-commits (4.2k stars)
- Specification for readable commit messages
- Why: Industry standard

### Option 2: commitizen/cz-cli (15k stars)
- CLI tool for creating conventional commits

Which one?
```

**User**: "conventional commits"

**Step 4 - Deep dive:**
- Read the specification
- Understand types (feat, fix, docs)
- Learn about scopes, breaking changes

**Step 5 - Summarize:**
```
## Understanding Conventional Commits

### Core Approach
<type>(<scope>): <description>

### Key Types
- feat: new feature
- fix: bug fix
- docs: documentation

Ready to create?
```

**Step 6 - Create skill with these principles encoded**

## Important Notes

1. Always get user confirmation at step 3 and 5
2. Prefer learning over wrapping
3. Check license - mention if restrictive
4. Credit the source
5. Quality over speed
