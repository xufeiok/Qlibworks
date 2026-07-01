---
name: skill-from-masters
description: Help users create high-quality skills by discovering and incorporating proven methodologies from domain experts. Use this skill BEFORE skill-creator when users want to create a new skill - it enhances skill-creator by first identifying expert frameworks and best practices to incorporate. Triggers on requests like "help me create a skill for X" or "I want to make a skill that does Y".
---

# Skill From Masters

Create skills that embody the wisdom of domain masters. This skill helps users discover and incorporate proven methodologies from recognized experts before generating a skill.

## Core Philosophy

Most professional domains have outstanding practitioners who have codified their methods through books, talks, interviews, and frameworks. A skill built on these proven methodologies is far more valuable than one created from scratch.

The goal is not just "good enough" — it's reaching the highest level of human expertise in that domain.

## Critical Requirements for Non-Technical Skills

**Technical skills have standard answers.** Writing code, debugging, or configuring systems — these have relatively objective quality bars.

**Non-technical skills vary dramatically in quality.** Skills involving decision-making, communication, persuasion, or judgment can range from mediocre to world-class. The difference comes from incorporating deep expertise.

For non-technical skills (writing, sales, hiring, product decisions, etc.), follow these requirements:

### 1. Narrow, Specific Task Definition
- The task must be **extremely specific and well-defined**
- BAD: "Write a sales email" (too broad)
- GOOD: "Write a B2B cold outreach email to enterprise CTOs"
- Different contexts require completely different skills

### 2. Model Selection: Opus Required
- Non-technical skills MUST use **Claude Opus** (claude-opus-4-5)
- DO NOT use Sonnet, Haiku, or any other model
- The quality difference is substantial for these domains

### 3. Methodology Research: Clear & Reliable Conclusions
- Continue searching until you reach **very clear, reliable conclusions**
- Don't stop at surface-level research
- Sources to exhaust: model knowledge, web search, golden examples, counter-examples

## Workflow

### Step 1: Understand and Narrow the Skill Intent

Use the **5-Layer Narrowing Framework**:

**Layer 1: Domain Identification**
```
"[Broad task] can mean different things. Which is closest?
- [Domain A]
- [Domain B]
- [Domain C]"
```

**Layer 2: Context Constraints (5W1H)**
- WHO: Role, seniority, expertise level
- WHAT: Specific output or decision
- WHERE: Organizational context
- WHEN: Stage/timing
- WHY: Primary goal/outcome
- HOW: Constraints

**Layer 3: Comparative Narrowing**
Present 2-3 similar but distinct scenarios for the user to choose.

**Layer 4: Boundary Validation (Via Negativa)**
Confirm what the skill does NOT include.

**Layer 5: Concrete Case Anchoring**
Ask for a real, specific example from their experience.

**Stop Condition: Is It Narrow Enough?**
1. ✅ Unique methodology
2. ✅ Clear quality bar
3. ✅ Specific constraints
4. ✅ Concrete example
5. ✅ Excludes alternatives

### Step 2: Identify Skill Type

| Type | Core Operation | Key Question |
|------|----------------|--------------|
| Summary | Compress | Need comprehensive coverage? |
| Insight | Extract | Need to find what really matters? |
| Generation | Create | Need new content created? |
| Decision | Choose | Need to make a choice? |
| Evaluation | Judge | Need quality judgment? |
| Diagnosis | Trace | Need to find root cause? |
| Persuasion | Bridge | Need to change minds? |
| Planning | Decompose | Need a roadmap? |
| Research | Discover | Need knowledge gathered? |
| Facilitation | Elicit | Need to extract info? |
| Transformation | Map | Need format conversion? |

### Step 3: Surface Expert Methodologies (Until Crystal Clear)

**Layer 1: Local Database**
Consult references/methodology-database.md

**Layer 2: Web Search for Experts**
Search: "[domain] best practices expert"

**Layer 3: Deep Dive on Selected Experts**
Search: "[expert name] methodology interview"

**Keep iterating until you reach very clear, reliable conclusions.**

### Step 4: Find Golden Examples

Search for exemplary outputs to define the quality bar.

### Step 5: Collaborative Selection

Present methodologies to the user and discuss which frameworks resonate.

### Step 6: Extract Actionable Principles

For each methodology, distill:
- The Why (Core Principles)
- The How (Concrete Process)
- The What (Quality Criteria)
- The Pitfalls (Common Mistakes)

### Step 7: Cross-Validate

Compare insights across multiple sources.

### Step 8: Design Test Scenarios

Design comprehensive test scenarios:
- Typical scenarios
- Edge cases
- Failure modes
- Context variations

### Step 9: Generate the Skill

**Invoke skill-creator:**

```
Use the Skill tool with: skill: "skill-creator:skill-creator"
```

**For non-technical skills, CRITICAL:**
- Add `model: opus` in YAML frontmatter

### Step 10: Test, Review, and Iterate

1. Run Test Scenarios
2. Evaluate Results
3. Identify Gaps
4. Refine Methodology
5. Regenerate
6. Repeat

## Quality Checklist

- [ ] Task definition is narrow and specific
- [ ] Searched beyond local database
- [ ] Found primary sources (not just summaries)
- [ ] Reached clear, reliable conclusions
- [ ] Found golden examples
- [ ] Identified common mistakes
- [ ] Cross-validated across experts
- [ ] For non-technical skills: Used Opus model
- [ ] Encoded specific, actionable steps
- [ ] Designed comprehensive test scenarios
- [ ] Tested and iterated based on results
