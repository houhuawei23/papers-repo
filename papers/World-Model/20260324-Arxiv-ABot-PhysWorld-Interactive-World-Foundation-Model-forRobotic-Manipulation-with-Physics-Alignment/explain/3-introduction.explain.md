## 说明

- **来源**：原论文 Markdown 中标题为「1 Introduction」的小节（对应标准章节：Introduction (引言)）。
- **解析模板**：`prompts/paper/section-introduction.md`
- **提示摘要**：你是 **计算机与人工智能领域**的论文解读助手。请针对下列 **Introduction（引言）** 原文进行中文解析。

---

## 问题背景与领域动机
### 领域痛点
具身世界模型需要生成符合真实世界物理规律的未来预测，才能有效用于仿真、规划和策略学习。当前最先进的视频生成模型（如Veo 3.1和Sora v2 Pro）在生成机器人操作序列时，经常产生违反基本物理规律的现象，包括物体穿透、无接触运动和异常形变。这些不是渲染伪影，而是物理推理的根本性失败，限制了其在机器人下游应用中的可靠性。

### 任务形式化
开发一个物理基础且动作可控的世界模型，能够生成视觉逼真、物理合理且高度可控的操作序列，作为机器人仿真和规划的高保真接口。

### 相关主线
与**机器学习（ML）**主线相关，具体涉及具身人工智能、视频生成、物理推理和强化学习。

## 现状与缺口（Research Gap）
### 现有方法局限
1. 训练数据缺乏丰富的具身交互信号，阻碍了模型学习细粒度物理动力学（如摩擦、碰撞响应、质量分布）。2. 微调时依赖标准的最大似然目标，对所有预测误差一视同仁，无法区分物理有效和无效的过渡。

### 可研究的问题
如何构建一个世界模型，能够同时实现视觉逼真性、物理合理性和动作可控性，以克服当前模型在物理推理上的系统性缺陷。

## 本文贡献声明
### 数据类贡献
设计了一个原则性的数据整理流程，通过精选采样和物理感知标注，提高了具身视频数据的多样性和平衡性，实现了对真实世界交互的可扩展和鲁棒训练。

### 方法类贡献
提出了ABot-PhysWorld，一个通过物理感知DPO和并行空间动作注入，联合优化视觉逼真性、物理合理性和动作可控性的统一框架。

### 工程类贡献
构建了ABot-PhysWorld模型，基于14B扩散变换器，并集成了并行上下文块以实现多通道空间动作注入，支持精确的跨具身控制和动作对齐的运动合成。

### 评估类贡献
引入了EZSbench，这是首个用于具身视频生成的、独立于训练的零样本基准测试，采用解耦协议来评估分布偏移下的物理保真度和动作对齐。

## 论文结构
### 章节对应
- Section 2: 相关工作（Related Work）
- Section 3: 方法（Method）——描述数据整理流程和模型架构
- Section 4: 实验（Experiments）——介绍评估基准和实验结果
- Section 5: 结果与分析（Results and Analysis）——提供详细结果和讨论
## 与Related Work的分工
### 引言内简要对比
引言中简要对比了现有视频生成模型（如Veo 3.1, Sora v2 Pro）在物理合理性方面的局限性，并提到了相关VLA策略和WAMs工作作为背景。

### 与后文分工
引言部分主要概述了领域背景、问题缺口和本文核心贡献。后文的“相关工作”章节（Section 2）预计会进行更系统、更全面的文献综述，详细分类和对比现有世界模型、视频生成方法、物理推理技术以及评估基准，从而更深入地定位本文工作的创新性。

## 作者观点标注
### 强主张
- “These are not mere rendering artifacts but fundamental failures in physical reasoning, limiting their reliability in downstream robotic applications.” —— **作者观点**，强调物理推理失败是根本性问题。
- “Our model achieves new state-of-the-art results on both PBench and EZSbench, surpassing Veo 3.1 and Sora v2 Pro in physical plausibility and action trajectory consistency.” —— **作者观点**，声明其模型的优越性能。