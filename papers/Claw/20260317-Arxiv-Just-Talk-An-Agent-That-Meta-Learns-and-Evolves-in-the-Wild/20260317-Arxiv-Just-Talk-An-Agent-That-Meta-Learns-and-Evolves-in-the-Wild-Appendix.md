<a id="section-7"></a>

## 7 Prompts and Templates

This appendix documents the core prompt templates used in MetaClaw-Bench evaluations and the MetaClaw framework components.

- [7.1 Agent System Prompt (MetaClaw-Bench Part I)](#71-agent-system-prompt-metaclaw-bench-part-i)
- [7.2 Agent Identity Context (MetaClaw-Bench Part II)](#72-agent-identity-context-metaclaw-bench-part-ii)
- [7.3 Task Question Templates](#73-task-question-templates)
- [7.4 Skill Evolver Prompt Template](#74-skill-evolver-prompt-template)
- [7.5 Skill Injection Format](#75-skill-injection-format)
- [7.6 System Prompt Compression Prompt](#76-system-prompt-compression-prompt)
- [7.7 Part II Implicit Preference Rules](#77-part-ii-implicit-preference-rules)

<a id="section-7-1"></a>

### 7.1 Agent System Prompt (MetaClaw-Bench Part I)

Part I evaluates agents on OpenClaw CLI tasks via a programmatic rollout loop. The agent receives the following fixed system prompt, which may be replaced by a compressed variant (see [Section 7.6](#section-7-6)) after the first session:

The single tool exposed to the agent is run_command:

<a id="section-7-2"></a>

### 7.2 Agent Identity Context (MetaClaw-Bench Part II)

Part II (CALMB-14) injects the following workspace context files into the agent’s session at initialization, defining the agent’s role, user profile, and behavioral principles.

<a id="section-7-3"></a>

### 7.3 Task Question Templates

Each day in MetaClaw-Bench presents two question types. Below are representative examples.

Multi-choice question (Part II, Day 01 / r1):

File-check question (Part II, Day 01 / r21):

Part I task instruction format (real OpenClaw session, train.jsonl):

<a id="section-7-4"></a>

### 7.4 Skill Evolver Prompt Template

After each session in which failed trajectories are collected, the MetaClaw skill evolver submits the following prompt to synthesize new behavioral skills:

<a id="section-7-5"></a>

### 7.5 Skill Injection Format

Retrieved skills are appended to the agent’s system message by SkillManager.format_for_conversation:

<a id="section-7-6"></a>

### 7.6 System Prompt Compression Prompt

To prevent context overflow during long sessions, MetaClaw compresses OpenClaw’s native system prompt using the following instruction:

<a id="section-7-7"></a>

### 7.7 Part II Implicit Preference Rules

MetaClaw-Bench Part II introduces five implicit preference rules progressively across 14 days. These rules are _not_ stated in the agent’s system prompt; they must be inferred from task feedback and internalized through skill evolution or RL training.

<a id="table-4"></a>

> Table 4: The five implicit preference rules in MetaClaw-Bench Part II, introduced progressively across 6 learning arcs. Each rule is verified by a dedicated automated checker script.

| Rule | Category       | Requirement                                               | Active from |
| ---- | -------------- | --------------------------------------------------------- | ----------- | --------------------- | ------ |
| P1   | Timestamp      | All date/time fields: YYYY-MM-DDTHH:MM:SS+08:00           | Day 01      |
| P2   | File naming    | Output files: YYYYMMDD_description.ext (snake_case)       | Day 04      |
| P3   | Metadata       | Every output file must include created_at, author, status | Day 06      |
| P4   | Backup         | Create <file>.bak before modifying any existing file      | Day 08      |
| P5   | Completion log | Append [DONE] <timestamp>                                 | <task_id>   | <summary> to done.log | Day 10 |
