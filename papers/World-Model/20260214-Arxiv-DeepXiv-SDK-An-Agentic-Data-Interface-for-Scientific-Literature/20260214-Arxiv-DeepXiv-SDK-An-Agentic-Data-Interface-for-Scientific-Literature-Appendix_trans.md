[A.1 arXiv 规模处理流水线（A.1 ArXiv-Scale Processing Pipeline）](#a1-arxiv-scale-processing-pipeline)

- [A.2 服务、索引与 T+0 更新（A.2 Serving, Indexing, and T+0 Updates）](#a2-serving-indexing-and-t0-updates)

### A.1 arXiv 规模处理流水线（A.1 ArXiv-Scale Processing Pipeline）

我们通过 **OAI-PMH 接口（OAI-PMH interface）** 获取完整的 arXiv 元数据流，并以 **arXiv ID** 为键。对于初始的完整爬取，我们下载所有论文的 PDF 文件，并使用 **MinerU** Wang 等人 ([2024a]) 将其转换为 **Markdown** ，从而将异构的布局规范化为以文本为中心的表示。这个 **PDF$\rightarrow$MD** 阶段在 8 个节点上运行，每个节点配备 8$\times$ **H100 GPU** ，耗时约 $\sim$72 小时。从 Markdown 中，我们利用格式规律恢复章节结构，并为每篇论文物化一个规范的 **JSON** ，其中每个章节都是一个显式字段，以支持确定性的章节级访问。

在结构化 JSON 的基础上，我们使用 **Qwen3-4B-Instruct-2507** 运行轻量级的 **基于 LLM 的增强（LLM-based enrichment）** 。给定前 2048 个词元（tokens）和元数据作者列表，该模型推断作者-所属机构关系，生成一句话的 **TL;DR** ，并提取最多五个关键词。我们进一步使用正则表达式提取 **GitHub 仓库链接** ，并过滤常见的非论文特定仓库（例如 pytorch、vllm、sklearn）；此外，该模型通过将仓库标识（例如，所有者/名称）与论文上下文进行交叉检查，来验证每个提取的 GitHub URL 是否是 **论文特定的** 。为了支持预算感知的访问，我们使用 **tiktoken** 计算总词元数和各章节词元数，并为每个章节生成一个 TL;DR。增强阶段在相同的 8$\times$(8$\times$H100) 集群上运行，处理整个语料库耗时约 $\sim$100 小时。

<a id="table-3"></a>

> 表 3：评估项示例：以 arXiv ID 作为标准答案的论文识别问题。

| 问题                                                                                                                                     | 答案 (arXiv ID) |
| ---------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| 问题                                                                                                                                     | 答案 (arXiv ID) |
| 哪篇论文提出了一种想象力驱动的智能体，在冲突的网格世界任务中平衡救援他人和最小化环境危害，并在多种场景变体中优于标准和基于同理心的基线？ | 2501.00320      |
| 哪篇论文提出了一种想象力驱动的智能体，在冲突的网格世界任务中平衡救援他人和最小化环境危害，并在多种场景变体中优于标准和基于同理心的基线？ | 2501.00320      |
| 哪篇论文利用课程文档构建了一个课程特定的聊天机器人来引导对话模型，在数据库问题和策略方面优于通用模型，并引用来源，且无需教师训练？       | 2401.00052      |
| 哪篇论文利用课程文档构建了一个课程特定的聊天机器人来引导对话模型，在数据库问题和策略方面优于通用模型，并引用来源，且无需教师训练？       | 2401.00052      |
| 哪篇论文开发了一种通用算法，用于在具有任意边界数据的条带上构造最小对角凹函数，包括具有水平人字形叶理和裂隙的情况？                       | 2401.00053      |
| 哪篇论文开发了一种通用算法，用于在具有任意边界数据的条带上构造最小对角凹函数，包括具有水平人字形叶理和裂隙的情况？                       | 2401.00053      |

### A.2 服务、索引与 T+0 更新（A.2 Serving, Indexing, and T+0 Updates）

处理后的 JSON 和 Markdown 存储在对象存储中，而元数据和提取的属性存储在 **PostgreSQL** 中。我们在属性和内容代理上构建 **Elasticsearch** 索引，并在紧凑表示（标题 + 摘要 + 章节 TL;DRs）上使用 **BGE-m3** 计算密集嵌入，以支持混合检索而无需索引全文。服务部署在一个 64 核/256GB 的节点上（带有一个热备副本），使用 **Gunicorn** 提供 API， **Caddy** 进行反向代理和负载均衡， **Redis** 用于缓存高频端点。

对于增量维护， **DeepXiv-SDK** 通过拉取 OAI-PMH 列表增量并处理新的/更新的 ID，执行工作日定时同步（周一至周五）。由于许多 arXiv 条目提供 HTML 页面，我们采用 **HTML 优先策略（HTML-first strategy）** 进行解析，当 HTML 解析失败时回退到 PDF；下游的结构化和增强步骤保持不变。这实现了与新发布的 **T+0 同步** ，并保持检索和渐进式访问视图的新鲜度。最后，该流水线设计为可通过仅替换摄取连接器扩展到其他开放获取语料库（例如， **PMC** ），同时复用相同的规范化模式、增强、索引和服务协议。

<a id="appendix-b"></a>

## 附录 B：统计信息（Appendix B Statistics）

截至目前， **DeepXiv-SDK** 索引了 2,949,129 篇 arXiv 论文；其中，2,712,378 篇被成功解析为具有各章节 TL;DRs 的显式章节结构，而其余论文通常缺乏可用的章节线索（无章节标记、结构弱/模糊或内容以图表为主），但仍支持其他 API 视图和检索功能。在整个语料库中，我们提取了 219,717 个 GitHub URL；对 200 个抽样链接的手动审核未发现不匹配。对于社交关注度，我们每天查询 **X.com 搜索 API** 以获取 arxiv.org 的提及，并聚合曝光信号，迄今已为 448,825 篇论文生成趋势数据。最后，我们通过将 arXiv ID 链接到 **Semantic Scholar** 记录，为论文增加了引用次数和会议/期刊元数据。

<a id="table-4"></a>

> 表 4：DeepXiv-SDK API 接口。渐进式访问控制成本，而混合检索在证据级检查前筛选集合。表 [1](#LST1) 展示了一个头部响应示例。详细文档见[此页面](https://data.rag.ac.cn/api/docs)。

| 功能                  | 端点（模板）                                         | 返回内容（角色）                                          |
| --------------------- | ---------------------------------------------------- | --------------------------------------------------------- |
| 功能                  | 端点（模板）                                         | 返回内容（角色）                                          |
| 功能                  | 端点（模板）                                         | 返回内容（角色）                                          |
| 渐进式访问 (arXiv)    |                                                      |                                                           |
| 快速简报              | /arxiv (type=brief, id={id})                         | 简要元数据。                                              |
| 快速简报              | /arxiv (type=brief, id={id})                         | 简要元数据。                                              |
| 快速简报              | /arxiv (type=brief, id={id})                         | 简要元数据。                                              |
| 概览（头部优先）      | /arxiv (type=head, id={id})                          | 元数据 + 章节清单 + 全局预算提示。                        |
| 概览（头部优先）      | /arxiv (type=head, id={id})                          | 元数据 + 章节清单 + 全局预算提示。                        |
| 概览（头部优先）      | /arxiv (type=head, id={id})                          | 元数据 + 章节清单 + 全局预算提示。                        |
| 预览                  | /arxiv (type=preview, id={id})                       | 固定长度前缀（10k 字符） + 截断标志。                     |
| 预览                  | /arxiv (type=preview, id={id})                       | 固定长度前缀（10k 字符） + 截断标志。                     |
| 预览                  | /arxiv (type=preview, id={id})                       | 固定长度前缀（10k 字符） + 截断标志。                     |
| 章节读取              | /arxiv (type=section, id={id}, section={sec})        | 指定章节内容 + 本地预算提示。                             |
| 章节读取              | /arxiv (type=section, id={id}, section={sec})        | 指定章节内容 + 本地预算提示。                             |
| 章节读取              | /arxiv (type=section, id={id}, section={sec})        | 指定章节内容 + 本地预算提示。                             |
| 全文（Markdown）      | /arxiv (type=raw, id={id})                           | 用于验证的完整 Markdown。                                 |
| 全文（Markdown）      | /arxiv (type=raw, id={id})                           | 用于验证的完整 Markdown。                                 |
| 全文（Markdown）      | /arxiv (type=raw, id={id})                           | 用于验证的完整 Markdown。                                 |
| 完整对象（JSON）      | /arxiv (type=json, id={id})                          | 用于确定性处理的完整结构化 JSON。                         |
| 完整对象（JSON）      | /arxiv (type=json, id={id})                          | 用于确定性处理的完整结构化 JSON。                         |
| 完整对象（JSON）      | /arxiv (type=json, id={id})                          | 用于确定性处理的完整结构化 JSON。                         |
| 检索与实用功能        |                                                      |                                                           |
| 混合检索与过滤        | /arxiv (type=retrieve, q={…})                        | 带属性过滤和分页的 BM25 / 向量 / 混合搜索。               |
| 混合检索与过滤        | /arxiv (type=retrieve, q={…})                        | 带属性过滤和分页的 BM25 / 向量 / 混合搜索。               |
| 混合检索与过滤        | /arxiv (type=retrieve, q={…})                        | 带属性过滤和分页的 BM25 / 向量 / 混合搜索。               |
| 社交关注度（X）       | /arxiv/trending_signal (id={id})                     | 可选的曝光统计：总浏览量、总点赞数、总转发数。            |
| 社交关注度（X）       | /arxiv/trending_signal (id={id})                     | 可选的曝光统计：总浏览量、总点赞数、总转发数。            |
| 社交关注度（X）       | /arxiv/trending_signal (id={id})                     | 可选的曝光统计：总浏览量、总点赞数、总转发数。            |
| 使用量与配额审计      | /stats/usage (days={n})                              | 用于配额感知工具规划的使用量摘要。                        |
| 使用量与配额审计      | /stats/usage (days={n})                              | 用于配额感知工具规划的使用量摘要。                        |
| 使用量与配额审计      | /stats/usage (days={n})                              | 用于配额感知工具规划的使用量摘要。                        |
| 语料库可扩展性（PMC） |                                                      |                                                           |
| PMC 基础访问（当前）  | /pmc (type=head, id={id}), /pmc (type=json, id={id}) | 当前提供基本元信息和完整 JSON；章节解析/TL;DRs 尚未物化。 |
| PMC 基础访问（当前）  | /pmc (type=head, id={id}), /pmc (type=json, id={id}) | 当前提供基本元信息和完整 JSON；章节解析/TL;DRs 尚未物化。 |
| PMC 基础访问（当前）  | /pmc (type=head, id={id}), /pmc (type=json, id={id}) | 当前提供基本元信息和完整 JSON；章节解析/TL;DRs 尚未物化。 |

> 表：清单 1：元数据示例（JSON），可访问于 [https://data.rag.ac.cn/arxiv/?arxiv_id=2409.05591&type=head](https://data.rag.ac.cn/arxiv/?arxiv_id=2409.05591&type=head)。此 ID 无需令牌；其他请求需要 API 令牌。
