# paper.yml 字段说明

本文档描述 `paper.yml` 的字段结构与兼容写法，与 arxiv2md-beta、academic-extension 等工具生成的格式兼容。

## 顶层结构

```yaml
paper:
  id: arxiv:2307.15818v1
  title: "论文标题"
  authors: [...]
  publication: {...}  # 或 [...]
  identifiers: {...}
  urls: {...}
  content: {...}
  workflow: {...}
  relations: {...}
  bibtex: "..."
```

## 核心字段

### paper.id

论文唯一标识，常见格式：`arxiv:2307.15818v1`。

### paper.title

论文标题，字符串。

### paper.authors

作者列表：

```yaml
authors:
  - name: Author One
  - name: Author Two
```

### paper.publication（兼容两种结构）

**单对象**（常见于 arXiv 预印本）：

```yaml
publication:
  type: preprint
  venue: arXiv
  date_published: "2023-07-28"
  year: 2023
```

**列表**（常见于有会议版本的论文）：

```yaml
publication:
  - type: preprint
    venue: arXiv
    date_published: "2021-02-26"
    year: 2021
  - type: conference
    venue: ICML 2021
    year: 2021
    note: oral
```

分析脚本会统一处理两种形式，提取 `year`、`venue`、`date_published` 等字段。

### paper.identifiers

外部标识符：

```yaml
identifiers:
  arxiv: 2307.15818v1
```

### paper.urls

链接：

```yaml
urls:
  pdf: https://arxiv.org/pdf/2307.15818v1
  abstract: https://arxiv.org/abs/2307.15818v1
  website: https://example.com/
```

### paper.content

内容相关：

```yaml
content:
  abstract: "摘要文本..."
  keywords: [cs.RO, cs.CL]
  language: en
  citation: "简短引用..."
```

### paper.workflow

工作流状态：

```yaml
workflow:
  status: unread  # unread | reading | read
  priority: normal
  date_added: "2023-07-28"
```

### paper.relations

分类与标签：

```yaml
relations:
  tags: [VLA, World Model]
  categories: [cs.RO, cs.CL]
  primary_category: cs.RO
  related: []
```

### paper.bibtex

BibTeX 字符串，用于引用。

## 可选扩展

- `notes/reading-notes.md`：阅读笔记，与 `paper.yml` 同目录或在其 `notes/` 子目录下。
- `*_trans.md`：译文 Markdown，分析脚本会统计其存在性。
