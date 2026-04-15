## 说明

- **来源**：原论文 Markdown 中标题为「Contents」的小节（非标准章节名，使用通用解析模板）。
- **解析模板**：`prompts/paper/section-generic.md`
- **提示摘要**：你是 **计算机与人工智能领域**的论文解读助手。下面小节在原论文中的标题为「{section_heading}」，可能无法简单归入 Abstract / Introduction / Methods 等固定结构（例如：威胁模型、伦理讨论、扩展实验、某节模型细节、Benchmark 说明等）。请对该小节正文进行**中文**结构化解析。

---

## role_positioning
该小节为论文的目录（Contents），功能是展示全文的结构框架，为读者提供导航，不属于固定章节结构（如Abstract/Introduction等），而是对全文内容的概括性索引。

## key_technical_points
- 定义：目录条目，包括章节标题和子节标题（如'2 Data Curation'、'2.1 Embodied-Specific Data Filtering'）。
- 符号：使用数字编号（如1, 2, 3）和点号（如2.1, 3.2.1）表示层级结构。
- 假设：无技术性假设，仅为组织结构。
- 算法步骤或实验协议：无，目录不包含具体技术细节。
## connection_with_adjacent_sections
原文未明确衔接（目录独立于正文章节，但逻辑上按顺序列出所有章节，从Abstract到Contributions）。

## formulas_figures_tables
无公式、图或表；目录本身作为文本列表，用于指引读者定位内容。

## ai_cs_context
无训练、推理、复杂度、数据管线或评测协议等可复现要素；目录仅反映论文结构，不涉及具体技术实现。