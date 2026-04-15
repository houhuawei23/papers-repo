<a id="section-7"></a>

## 7 提示词与模板（Prompts and Templates）

本附录记录了在 **MetaClaw-Bench** 评估和 **MetaClaw框架（MetaClaw framework）** 组件中使用的核心提示词模板。

- [7.1 智能体系统提示词（MetaClaw-Bench 第一部分）](#71-agent-system-prompt-metaclaw-bench-part-i)
- [7.2 智能体身份上下文（MetaClaw-Bench 第二部分）](#72-agent-identity-context-metaclaw-bench-part-ii)
- [7.3 任务问题模板](#73-task-question-templates)
- [7.4 技能进化器提示词模板](#74-skill-evolver-prompt-template)
- [7.5 技能注入格式](#75-skill-injection-format)
- [7.6 系统提示词压缩提示词](#76-system-prompt-compression-prompt)
- [7.7 第二部分隐式偏好规则](#77-part-ii-implicit-preference-rules)

<a id="section-7-1"></a>

### 7.1 智能体系统提示词（MetaClaw-Bench 第一部分）

第一部分通过程序化 **回滚循环（rollout loop）** 在 **OpenClaw CLI** 任务上评估智能体。智能体接收以下固定的系统提示词，该提示词可能在第一次会话后被压缩版本替换（参见[第7.6节](#section-7-6)）：

暴露给智能体的唯一工具是 `run_command`：

<a id="section-7-2"></a>

### 7.2 智能体身份上下文（MetaClaw-Bench 第二部分）

第二部分（ **CALMB-14** ）在初始化时将以下工作空间上下文文件注入智能体的会话中，定义智能体的角色、用户档案和行为原则。

<a id="section-7-3"></a>

### 7.3 任务问题模板

**MetaClaw-Bench** 中的每一天呈现两种问题类型。以下是代表性示例。

**多项选择题（第二部分，第01天 / r1）** ：

**文件检查题（第二部分，第01天 / r21）** ：

**第一部分任务指令格式（真实的OpenClaw会话，train.jsonl）** ：

<a id="section-7-4"></a>

### 7.4 技能进化器提示词模板

在每次收集到失败轨迹的会话后， **MetaClaw技能进化器（MetaClaw skill evolver）** 提交以下提示词以合成新的行为技能：

<a id="section-7-5"></a>

### 7.5 技能注入格式

检索到的技能通过 `SkillManager.format_for_conversation` 附加到智能体的系统消息中：

<a id="section-7-6"></a>

### 7.6 系统提示词压缩提示词

为防止长会话期间的 **上下文溢出（context overflow）** ， **MetaClaw** 使用以下指令压缩 **OpenClaw** 的原生系统提示词：

<a id="section-7-7"></a>

### 7.7 第二部分隐式偏好规则

**MetaClaw-Bench** 第二部分在14天内逐步引入了五条隐式偏好规则。这些规则 **未** 在智能体的系统提示词中说明；必须从任务反馈中推断，并通过技能进化或 **强化学习（Reinforcement Learning, RL）** 训练内化。

<a id="table-4"></a>

> **表4：MetaClaw-Bench第二部分中的五条隐式偏好规则，在6个学习弧中逐步引入。每条规则由专用的自动化检查脚本验证。**

| 规则 | 类别     | 要求                                                    | 生效起始日 |
| ---- | -------- | ------------------------------------------------------- | ---------- |
| P1   | 时间戳   | 所有日期/时间字段：YYYY-MM-DDTHH:MM:SS+08:00            | 第01天     |
| P2   | 文件命名 | 输出文件：YYYYMMDD\_描述.ext（snake_case）              | 第04天     |
| P3   | 元数据   | 每个输出文件必须包含 created_at、author、status         | 第06天     |
| P4   | 备份     | 在修改任何现有文件前创建 <file>.bak                     | 第08天     |
| P5   | 完成日志 | 追加 [DONE] <timestamp> <task_id> <summary> 到 done.log | 第10天     |
