## 说明

- **来源**：原论文 Markdown 中标题为「2 Problem Setup」的小节（非标准章节名，使用通用解析模板）。
- **解析模板**：`prompts/paper/section-generic.md`
- **提示摘要**：你是 **计算机与人工智能领域**的论文解读助手。下面小节在原论文中的标题为「{section_heading}」，可能无法简单归入 Abstract / Introduction / Methods 等固定结构（例如：威胁模型、伦理讨论、扩展实验、某节模型细节、Benchmark 说明等）。请对该小节正文进行**中文**结构化解析。

---

## role_positioning
该小节在全文中的功能是定义论文研究的核心问题框架和系统模型，为后续方法（如技能库适应、策略优化）的展开提供形式化基础。

## key_technical_points
- **问题设定**：考虑一个已部署的 CLI 代理，服务于来自非平稳分布 $p_{t}(\tau)$ 的任务流 $\tau_{1},\tau_{2},\ldots$。
- **任务定义**：每个任务 $\tau_{i}$ 包含用户指令和环境上下文（文件系统状态、shell 历史等），代理需生成动作序列 $a_{1:T}$ 完成任务。
- **元模型定义**：代理行为由元模型 $\mathcal{M}=(\theta,\mathcal{S})$ 完全决定，其中 $\theta$ 是基础 LLM 策略的参数，$\mathcal{S}=\{s_{1},s_{2},\ldots,s_{K}\}$ 是技能指令库（简洁、可复用的行为指令，在推理时注入系统提示）。
- **动作生成**：给定任务 $\tau$，代理根据 $a\sim\pi_{\theta}\!\left(\cdot\mid\tau,\;\textsc{Retrieve}(\mathcal{S},\tau)\right)$ 生成动作，其中 $\textsc{Retrieve}(\mathcal{S},\tau)\subseteq\mathcal{S}$ 通过基于嵌入的检索选择最相关技能。
- **轨迹数据类型**：基于在元模型演化中的作用，区分两种轨迹数据：支持数据 $\mathcal{D}^{\text{sup}}$（失败轨迹驱动技能库 $\mathcal{S}$ 适应，反映适应前行为）和查询数据 $\mathcal{D}^{\text{qry}}$（适应后收集的轨迹，反映适应后行为，用于优化策略参数 $\theta$）。
- **数据分离原则**：严格分离支持和查询数据至关重要，混合会导致 $\theta$ 针对过时奖励信号优化，不再反映代理当前能力。
- **系统目标**：MetaClaw 的目标是在任务流中持续改进 $\mathcal{M}$，不仅孤立解决每个任务，而是随着新任务到达，逐步变得更好于适应新任务。
- **系统定位**：这使 MetaClaw 定位为一个持续元学习系统：代理从非平稳任务流中学习，同时改进自身适应能力。
## connection_with_context
原文未明确衔接（小节末尾有锚点 `<a id="section-3"></a>`，但未说明与前后文的逻辑关系）。

## formulas_figures_tables
- **公式 (1)**：$\mathcal{M}=(\theta,\mathcal{S})$，用于形式化定义代理的元模型结构，区分参数和技能库组件。
- **公式 (2)**：$a\sim\pi_{\theta}\!\left(\cdot\mid\tau,\;\textsc{Retrieve}(\mathcal{S},\tau)\right)$，用于描述代理的动作生成过程，结合任务和检索技能。
## ai_cs_context
- **训练/推理**：涉及推理时动作生成（基于 LLM 策略 $\pi_{\theta}$ 和技能检索）和训练时元模型演化（通过支持数据适应技能库，查询数据优化策略参数）。
- **数据管线**：定义了轨迹数据的收集和分离协议（支持数据 $\mathcal{D}^{\text{sup}}$ 和查询数据 $\mathcal{D}^{\text{qry}}$），强调严格分离以避免优化偏差。
- **可复现要素**：元模型结构定义（参数 $\theta$ 和技能库 $\mathcal{S}$）、动作生成公式、数据分离原则（支持与查询数据不混合），为实验复现提供基础框架。