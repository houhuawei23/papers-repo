## 说明

- **来源**：原论文 Markdown 中标题为「Contents」的小节（非标准章节名，使用通用解析模板）。
- **解析模板**：`prompts/paper/section-generic.md`
- **提示摘要**：你是 **计算机与人工智能领域**的论文解读助手。下面小节在原论文中的标题为「{section_heading}」，可能无法简单归入 Abstract / Introduction / Methods 等固定结构（例如：威胁模型、伦理讨论、扩展实验、某节模型细节、Benchmark 说明等）。请对该小节正文进行**中文**结构化解析。

---

## role_positioning
该小节为论文的目录（Contents），功能是展示全文的结构框架，为读者提供导航，不属于论证或技术内容部分。

## key_technical_points
- 列出论文所有章节和子章节的标题及其在文档中的锚点链接。
- 章节结构依次为：摘要（Abstract）、引言（Introduction）、问题设定（Problem Setup）、MetaClaw主体方法（包含概述、技能驱动的快速适应、机会主义策略优化、技能生成版本控制、机会主义元学习调度器）、实验（包含实验设置、主要结果、分析）、相关工作（Related Work）、结论（Conclusion）。
## connection_with_context
原文未明确衔接（此为独立目录页）。

## formulas_figures_tables
无公式、图或表。

## ai_cs_context
无相关训练、推理等可复现要素。