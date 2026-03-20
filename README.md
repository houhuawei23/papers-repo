# papers-repo

个人论文仓库，用于整理、阅读与追踪学术文献。采用 Markdown + YAML 元数据形式，与 [arxiv2md-beta](https://github.com/your-username/arxiv2md-beta) 和 [ask_llm](https://github.com/your-username/ask_llm) 等工具配合使用。

## 目录结构

```
papers-repo/
├── papers/                    # 论文目录
│   └── YYYYMMDD-Source-Short-Paper-Title/
│       ├── paper.yml          # 元数据（必需）
│       ├── *.md               # 正文 Markdown
│       ├── *_trans.md         # 译文（可选）
│       └── notes/
│           └── reading-notes.md   # 阅读笔记（可选）
├── scripts/
│   └── analyze_repo.py        # 仓库分析与报告脚本
├── reports/                   # 分析报告输出
└── docs/
    └── paper_schema.md        # paper.yml 字段说明
```

## 目录命名规范

推荐格式：`YYYYMMDD-Source-Short-Paper-Title`

- **Date**：添加日期（YYYYMMDD）
- **Source**：来源（Arxiv、ICML2021、PMLR 等）
- **Short**：简短标识（可选，如 arXiv ID 或论文简称）
- **Paper-Title**：论文标题关键词，用连字符连接

当前采用**非破坏式整理**：不自动重命名现有目录，命名问题由分析脚本在报告中给出建议。

## 添加新论文

1. 使用 arxiv2md-beta 等工具下载论文并生成 Markdown 与 `paper.yml`
2. 将输出目录放入 `papers/`，确保目录名符合规范
3. 可选：在 `notes/reading-notes.md` 中记录阅读笔记
4. 运行 `python scripts/analyze_repo.py` 更新仓库报告

## paper.yml 字段

详见 [docs/paper_schema.md](docs/paper_schema.md)。核心字段包括：

- `paper.title`、`paper.authors`、`paper.publication`
- `paper.relations.tags`、`paper.relations.categories`
- `paper.urls.pdf`、`paper.content.abstract`、`paper.bibtex`

## 日常维护与报告使用

### 运行分析脚本

```bash
# 安装依赖（首次）
pip install -r requirements.txt

# 默认扫描 papers/，输出到 reports/latest/
python scripts/analyze_repo.py

# 自定义路径
python scripts/analyze_repo.py --root ./papers --out ./reports/latest
```

### 报告输出说明

| 文件 | 用途 |
|------|------|
| `report.md` | 可读汇总：统计、年份/venue/标签分布、质量检查、命名问题、疑似重复、改进建议 |
| `report.json` | 结构化明细：供脚本或前端二次处理 |
| `papers_index.csv` | 扁平索引：便于 Excel/统计工具分析、筛选、可视化 |

### 如何解读报告

1. **基础统计**：论文总数、含 paper.yml 比例、笔记/译文覆盖情况
2. **质量检查**：缺失字段、解析失败、目录命名问题、疑似重复
3. **改进建议**：按优先级处理（合并重复、修正命名、补充元数据、确认隐藏目录）

### 增量维护建议

- 添加新论文后运行一次脚本，确认无新增问题
- 定期查看「疑似重复」与「命名问题」，按需手动修正
- 可将 `papers_index.csv` 导入表格工具，按年份/标签筛选与统计

## License

本仓库结构、脚本与文档采用 [MIT License](LICENSE)。论文内容版权归原作者所有。
