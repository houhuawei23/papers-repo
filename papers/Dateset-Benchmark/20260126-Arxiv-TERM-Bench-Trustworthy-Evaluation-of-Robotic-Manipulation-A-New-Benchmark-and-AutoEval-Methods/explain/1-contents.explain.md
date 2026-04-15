## 说明

- **来源**：原论文 Markdown 中标题为「Contents」的小节（非标准章节名，使用通用解析模板）。
- **解析模板**：`prompts/paper/section-generic.md`
- **提示摘要**：你是学术论文解读助手。下面小节在原论文中的标题为「{section_heading}」，可能无法简单归入常见的 Abstract / Introduction / Methods 等固定结构。请对该小节正文进行中文结构化解析。

---

## 对该小节（Contents）的结构化解析

### 1. 小节角色概括
该小节为论文的目录（Contents），其角色是**全文的结构总览**，系统地列出了论文从引言到参考文献的所有主要章节、子章节及其层级关系，为读者提供了导航全文的框架。

### 2. 关键技术点、定义与结构列表（按原文顺序）
1.  **论文核心结构**：遵循经典的学术论文组织方式，包含引言（I）、相关工作（II）、方法（IV）、实验（V）、结论（VI）和参考文献。
2.  **核心贡献章节**：
    *   **III. The Eval-Actions Dataset**：介绍了本文提出的新基准数据集，包含其构建、数据标注与细粒度动作质量的定义、以及数据集统计信息。
    *   **IV. Method**：介绍了本文提出的核心方法，包含两个部分：
        *   **IV-A Rank-Guided Weight Optimization**：一种优化方法。
        *   **IV-B AutoEval**：本文核心的自动评估方法。
3.  **实验部分详述**：
    *   **V-A Evaluation Metrics**：定义了两种评估指标：
        *   **V-A 1**：用于**细粒度动作质量评估（Fine-Grained Action Quality Assessment）** 的指标。
        *   **V-A 2**：**成功分类（Success Classification）** 指标。
    *   **V-B Experimental Details**：实验设置详情。
    *   **V-C Experimental Results**：主要实验结果。
    *   **V-D Cross-Embodiment Generalization**：**跨具身泛化**实验，验证方法的泛化能力。
    *   **V-E Ablation Study**：消融实验，分析各组件贡献。

### 3. 与相邻章节或全文的呼应关系
*   目录本身是全文的缩影，**II. Related Work** 中列出的三个子方向（机器人学习数据集、VA/VLA模型、动作质量评估）为**III. 数据集构建**和**IV. 方法设计**提供了研究背景和定位。
*   **IV. Method** 中提出的方法将在 **V. Experiment** 中，使用 **III. The Eval-Actions Dataset** 和 **V-A** 定义的指标进行全面的验证与分析。

### 4. 公式、图或表编号说明
*   原文未展开。本小节仅为标题列表，未包含具体的公式、图或表编号。