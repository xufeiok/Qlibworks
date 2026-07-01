---
name: search-skill
description: Search and recommend Claude Code skills from trusted marketplaces. Triggers when users describe a need and want to find an existing Skill to solve it (e.g., "Is there a skill that can auto-generate changelogs?", "Find me a skill for frontend design").
model: sonnet
---

# Search Skill

Search and recommend Claude Code Skills from trusted marketplaces based on user requirements.

## When to Use

When users describe a need and want to find an existing Skill:

- "Is there a skill that can auto-generate changelogs?"
- "Find me a skill for frontend design"
- "I need a skill that can automate browser actions"

## Data Sources (by trust level)

### Tier 1 - Official / High Trust (show first)
- anthropics/skills: github.com/anthropics/skills (Official examples)
- ComposioHQ/awesome-claude-skills: github.com/ComposioHQ/awesome-claude-skills (12k+ stars)

### Tier 2 - Community Curated (secondary)
- travisvn/awesome-claude-skills: github.com/travisvn/awesome-claude-skills (21k+ stars)
- skills.sh: skills.sh (Vercel's official directory)

### Tier 3 - Aggregators (use with caution)
- skillsmp.com: skillsmp.com (Auto-scraped, requires extra filtering)

## Search Process

### Step 1: Parse User Intent

Extract from user description:
- Core functionality keywords (e.g., changelog, browser, frontend)
- Use case (development, testing, design)
- Special requirements (language, framework)

### Step 2: Multi-Source Search

**IMPORTANT: Only search these 5 sources.**

```
1. Search Tier 1 first
2. If fewer than 5 results, continue to Tier 2
3. If still insufficient, search Tier 3 with strict filtering
4. If nothing found, tell user honestly
```

Search queries:
```
site:github.com/anthropics/skills {keywords}
site:github.com/ComposioHQ/awesome-claude-skills {keywords}
site:github.com/travisvn/awesome-claude-skills {keywords}
site:skills.sh {keywords}
site:skillsmp.com {keywords}
```

### Step 3: Quality Filtering

**Must filter out:**

| Filter Condition | Reason |
|-----------------|--------|
| GitHub stars < 10 | Not verified |
| Last update > 6 months ago | Possibly abandoned |
| No SKILL.md file | Non-standard |
| README too sparse | Quality concerns |
| Suspicious code patterns | Security risk |

### Step 4: Rank Results

```
Score = Source Weight × 0.4 + Stars Weight × 0.3 + Recency × 0.2 + Relevance × 0.1

Source weights:
- Tier 1: 1.0
- Tier 2: 0.7
- Tier 3: 0.4
```

### Step 5: Format Output

```markdown
## Found X relevant Skills

### Recommended
1. **[skill-name](github-url)** - Source: anthropics/skills
   - Function: xxx
   - Install: `/plugin marketplace add xxx`

### Worth considering
2. **[skill-name](github-url)** - Source: ComposioHQ
   - ...

### Not recommended
- [skill-name](url) - Reason: low stars / not maintained
```

## Example

**User**: "Is there a skill that helps write commit messages?"

**Search process**:
1. Keywords: commit, message, git
2. Find: git-commit-assistant in anthropics/skills
3. Filter: Exclude results with stars < 10
4. Rank: Official sources first

**Output**:
```
## Found 3 relevant Skills

### Recommended
1. **git-commit-assistant** - Source: anthropics/skills (official)
   - Function: Generate semantic commit messages

2. **semantic-commit** - Source: ComposioHQ
   - Function: Follow conventional commits spec
   - Stars: 890
```

## Important Notes

1. Never recommend unverified Skills
2. Stay cautious with Tier 3 sources
3. If nothing suitable found, suggest using skill-from-masters
4. Security concerns: clearly inform users
