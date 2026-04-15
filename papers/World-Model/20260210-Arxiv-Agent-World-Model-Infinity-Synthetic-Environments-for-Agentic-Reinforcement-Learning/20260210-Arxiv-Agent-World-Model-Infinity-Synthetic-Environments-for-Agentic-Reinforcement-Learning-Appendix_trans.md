<a id="appendix-a"></a>

## 附录 A：实现细节（Appendix A Implementation Details）

- [A.1 场景生成（Scenario Generation）](#a1-scenario-generation)
- [A.2 任务生成（Task Generation）](#a2-task-generation)
- [A.3 环境合成（Environment Synthesis）](#a3-environment-synthesis)
- [A.4 工具调用格式与验证（Tool Calling Format and Validation）](#a4-tool-calling-format-and-validation)
- [A.5 训练与评估细节（Training and Evaluation Details）](#a5-training-and-evaluation-details)

### A.1 场景生成（Scenario Generation）

我们采用 **自指令式扩展（Self-Instruct style expansion）** （Wang 等人，2023）方法，从 100 个种子网站扩展到 1,000 个多样化场景。种子集来源于热门域名(^1^11[https://www.similarweb.com/top-websites/](https://www.similarweb.com/top-websites/))， **大型语言模型（Large Language Model, LLM）** 根据少量示例生成新的候选场景，这些示例强调有状态、基于数据库的交互。为了保持多样性，我们应用了两个互补的过滤器：(1) 一个 **LLM 分类器** ，根据对 **CRUD 操作（增、查、改、删）** 的适用性对每个候选进行评分，拒绝以内容为中心或只读的场景（例如新闻）；(2) 基于嵌入的去重，使用余弦相似度，阈值为 0.85，确保新生成的场景与现有池中的场景有足够差异。我们还设置了类别上限，以防止电子商务等主导类型过度出现。用于生成和分类的提示词见图 [10](https://arxiv.org/html/2602.10090v2#A2.F10) 和 [11](https://arxiv.org/html/2602.10090v2#A2.F11)。表 [9](#table-9) 展示了随机抽取的 100 个生成场景。我们还在图 [7](https://arxiv.org/html/2602.10090v2#A2.F7) 中展示了合成场景的分布，以及图 [7](https://arxiv.org/html/2602.10090v2#A2.F7) 中场景描述的 **词云（wordcloud）** 。分布情况表明，合成场景涵盖了多个领域，并未坍缩为少数主导类型。

### A.2 任务生成（Task Generation）

对于每个场景，我们提示 LLM 生成 $k=10$ 个不同的用户任务，作为下游合成的功能需求。这个适中的数量使得数据库模式和工具集实现对 LLM 而言是可处理的。提示词强制执行两个约束：(1) 任务必须可通过 **API（Application Programming Interface）** 解决，不依赖用户界面（UI），如点击或页面导航；(2) 任务假设处于认证后上下文，以解锁更深层次的功能。每个任务都要求具体且自包含，包括具体参数（例如产品 ID、用户名），以便无需额外澄清即可执行。这种需求驱动的设计确保随后生成的数据库模式和工具集与实际用户需求保持一致，而不是过度指定或不完整。提示词模板见图 [12](https://arxiv.org/html/2602.10090v2#A2.F12)。生成的任务见表 [11](#table-11)。

### A.3 环境合成（Environment Synthesis）

本节详细介绍了三个核心 **POMDP（部分可观测马尔可夫决策过程，Partially Observable Markov Decision Process）** 组件的合成：状态空间（数据库）、动作与观测空间（接口）以及奖励函数（验证）。

**数据库模式生成（Database Schema Generation）** 。给定场景描述和任务集 $\mathcal{T}_{E_{i}}$，LLM 推断出使所有任务可行所需的最小关系模式。输出是一组定义表、列、类型、主键、外键和索引的 **SQLite DDL（数据定义语言，Data Definition Language）** 语句。我们指示 LLM 根据任务推断隐含的实体关系（例如，“取消订单”任务意味着存在一个包含状态列的 Orders 表），并避免与认证相关的字段。基于引入的 **自校正机制（self-correction mechanism）** ，我们尝试在隔离环境中运行生成的 DDL 语句并测试其功能。如果任何 DDL 语句执行失败，我们会捕获错误信息，通过单独的 LLM 调用进行总结，并将摘要附加到提示词中以进行重新生成。此反馈循环重复进行，错误阈值为 10%：如果失败的表少于 10%，我们则接受该模式。提示词见图 [13](https://arxiv.org/html/2602.10090v2#A2.F13)。我们在图 [8](https://arxiv.org/html/2602.10090v2#A2.F8) 中可视化了部分带有示例数据的 SQLite 数据库模式。

**示例数据合成（Sample Data Synthesis）** 。空模式不足以执行任务，因为许多任务需要查询或更新现有记录。我们通过提示 LLM 分析任务前提条件并生成满足这些条件的 **INSERT** 语句来合成示例数据。例如，如果一个任务需要“更新产品库存”，则合成数据必须至少包含一个库存水平非零的产品。LLM 输出一个包含表名和相应 INSERT 语句的 **JSON（JavaScript Object Notation）** 结构，这些语句按照外键约束的依赖顺序执行。我们应用与模式生成相同的错误反馈循环，可接受的插入失败错误阈值为 10%。提示词见图 [14](https://arxiv.org/html/2602.10090v2#A2.F14) 和 [15](https://arxiv.org/html/2602.10090v2#A2.F15)。

**接口规范设计（Interface Specification Design）** 。在生成实现代码之前，我们首先合成一个定义工具集模式的接口规范。这种两阶段方法（先模式后代码）减少了长代码生成中的幻觉，因为试点实验表明，直接为具有 30 多个工具的环境生成代码通常会产生不一致的接口。给定任务集和数据库模式，LLM 推断出使每个任务可执行所需的最小端点集。每个端点规范包括：名称、方法、路径、摘要、类型化输入参数和响应模式。该规范还作为推理时代理人的文档，见表 [B.2](https://arxiv.org/html/2602.10090v2#A2.SS2)。提示词见图 [16](https://arxiv.org/html/2602.10090v2#A2.F16) 和 [17](https://arxiv.org/html/2602.10090v2#A2.F17)，合成 **MCP（模型上下文协议，Model Context Protocol）** 工具集的示例见图 [25](https://arxiv.org/html/2602.10090v2#A2.F25)。

**环境实现（Environment Implementation）** 。利用 **API** 规范和数据库模式，我们生成一个实现 **MCP** 服务器的完整 **Python** 文件。生成的代码包括：反映数据库模式的 **SQLAlchemy ORM（对象关系映射，Object-Relational Mapping）** 模型、 **Pydantic** 请求/响应模型以及执行数据库操作的 **FastAPI** 端点处理器。每个环境平均约有 2,000 行代码，平均暴露 35 个工具。生成后，我们通过启动并检查 MCP 服务器是否成功启动并响应健康检查和“列出工具”调用来验证环境。失败的环境进入自校正循环以修复错误。提示词见图 [18](https://arxiv.org/html/2602.10090v2#A2.F18)、[19](https://arxiv.org/html/2602.10090v2#A2.F19) 和 [20](https://arxiv.org/html/2602.10090v2#A2.F20)。

**验证代码合成（Verification Code Synthesis）** 。对于每个任务，我们生成一个 **Python** 验证函数，用于比较代理执行前后的数据库状态。该函数以两个数据库路径（初始和最终）作为输入，执行 **SQL（Structured Query Language）** 查询以提取任务相关信息，并返回一个结构化字典，包含：更改的记录、预期结果和诊断信号。我们还生成自然语言的成功和失败标准，以指导作为评判者的 LLM 解释验证结果。在 **RL（强化学习，Reinforcement Learning）** 训练期间，这种代码增强的验证为评判者提供了基于事实的证据，减少了奖励分配中的幻觉。生成验证代码的提示词见表 [21](https://arxiv.org/html/2602.10090v2#A2.F21)，合成验证代码的示例见图 [26](https://arxiv.org/html/2602.10090v2#A2.F26) 和 [27](https://arxiv.org/html/2602.10090v2#A2.F27)。

**自校正机制（Self-Correction Mechanism）** 。所有合成阶段共享一个共同的错误恢复模式。当生成的代码执行失败时，我们捕获完整的错误回溯，并调用单独的 LLM 来生成简洁的错误摘要（通常为 200-500 个词元），该摘要识别根本原因并提出修复建议。此摘要被附加到原始提示词中，用于下一次生成尝试。我们将每个组件的重试次数限制为 5 次迭代。如果所有重试后错误率超过阈值（模式/数据为 10%，环境启动为 0%），我们根据错误率选择最佳尝试并继续。这种方法在所有阶段实现了超过 85% 的首次尝试成功率，失败案例平均进行 1.13 次校正迭代。

### A.4 工具调用格式与验证

我们采用 **通义千问3（Qwen3）** 的工具调用格式（Yang 等人，2025），该格式将每次工具调用封装在 XML 标签中：`<tool_call>{name: …, arguments: …}</tool_call>`。我们并未将所有环境特定的工具直接暴露给智能体（agent），而是设计了一个两级抽象，将智能体与环境细节解耦。

**两级工具抽象** 智能体通过两个元工具（meta-tools）与 **模型上下文协议（Model Context Protocol, MCP）** 环境交互：(1) `list_tools`，用于查询 MCP 服务器以检索当前环境中所有可用工具及其元数据（输入/输出模式、描述）；(2) `call_tool`，通过工具名称调用环境特定工具，所需参数以 JSON 字符串形式传递。这种设计使智能体能够动态发现和调用不同环境中的工具集，而无需硬编码任何工具信息。完整的系统提示如图 [9](https://arxiv.org/html/2602.10090v2#A2.F9) 所示。

**格式验证规则** 在 **强化学习（Reinforcement Learning, RL）** 训练期间，我们强制执行格式约束，以确保工具调用格式正确，并防止产生幻觉或格式错误的交互。我们在轨迹的每一步都应用基于规则的验证，将其分类为有效、格式错误或环境/服务器错误。验证检查以下条件：

1.  **推理格式** ：所有助手消息必须在 `<think>...</think>` 标签内包含非空的推理内容。
2.  **工具名称有效性** ：智能体不得调用幻觉工具（即 `list_tools` 未返回的工具）或使用无效的工具名称。
3.  **参数有效性** ：工具参数必须是格式良好的 JSON，并符合工具模式。
4.  **协议遵守** ：智能体必须恰好调用一次 `list_tools`，且该调用必须是轨迹中的第一个工具调用。
5.  **交互一致性** ：如果智能体产生多次交互轮次，则必须在初始的 `list_tools` 之外至少进行一次成功的工具调用。
6.  **服务器响应** ：每次工具调用必须从 MCP 服务器收到非错误响应。例如，如果服务器返回指示超时或内部故障的错误消息，我们将其归类为环境错误。

对于每一步 $t$，如果违反条件 1-5 中的任何一条，我们将其归类为格式错误，并分配负奖励 $r_t = -1$。如果违反条件 6，我们将其归类为环境错误，并根据第 [4.1](#section-4-1) 节的奖励设定，分配奖励 $r_t = 0$。

**提前终止** 为了提高训练效率，我们对不可恢复的错误实施了提前停止机制。当验证器在推演过程中检测到格式错误或服务器错误时，我们会立即终止轨迹，而不是继续生成额外的步骤。这节省了计算资源，并防止智能体从根本已损坏的轨迹中获得奖励信号。

### A.5 训练与评估细节

**训练超参数** 表 [8](#table-8) 总结了用于 **分组相对策略优化（Group Relative Policy Optimization, GRPO）** 训练的超参数（Shao 等人，2024）。根据常见的智能体强化学习训练设置（Li 等人，2025b；Wang 等人，2025b），我们将 **KL 散度（Kullback-Leibler divergence）** 系数设置为 0.001。除了标准的 GRPO 设置外，我们采用了更高的裁剪比率 0.28，遵循 **去中心化异步策略优化（Decentralized Asynchronous Policy Optimization, DAPO）** （Yu 等人，2025）的建议，以允许智能体进行更多探索。此外，我们使用序列级重要性采样来缓解推演引擎与模型训练引擎之间的分布偏移问题（Yao 等人，2025）。

**环境管理** 我们在 AgentFly（Wang 等人，2025a）和 verl（Sheng 等人，2024）之上实现了多轮次强化学习训练。每个训练步骤并行启动 1024 个隔离的环境实例，每个实例作为一个独立的 MCP 服务器运行，并由其自己的 SQLite 数据库副本支持。这种隔离确保了并发推演不会相互干扰状态转换。每次推演完成后，通过将数据库恢复到初始状态来重置环境实例。环境启动（生成 MCP 服务器、复制数据库）是在线强化学习中的一个瓶颈，因为它会阻塞推演收集。为了将环境准备与策略训练重叠，我们进一步实现了预取机制：在当前批次进行梯度更新的同时，后台线程为下一个批次预配置环境。与顺序环境启动相比，这显著减少了每步的挂钟时间。

**用于历史感知训练的样本拆分** 如第 [4.2](#section-4-2) 节所述，我们通过在优化过程中应用历史截断来使训练与推理保持一致。我们使用简单的历史上下文管理，特别是滑动窗口截断，来研究训练与推理之间的不匹配问题。具体来说，给定一个包含 $T$ 个助手轮次的完整推演，我们将其拆分为 $T$ 个独立的训练样本。对于样本 $t$，输入包括系统提示、初始用户消息、第一个助手-工具交互（包含 `list_tools` 调用）以及轮次 $t$ 之前的 $w=3$ 个最近轮次。损失仅针对轮次 $t$ 的词元计算，而所有先前上下文词元的损失掩码设置为零。这种“样本拆分”方法确保每个训练样本都反映了智能体在推理时将看到的截断上下文。尽管这使每次推演的前向传播次数增加了 $T$ 倍，但它消除了在完整历史上进行训练但在截断上下文中部署时可能发生的分布偏移。更复杂的历史上下文管理是可能的，但这超出了本文的范围。

**奖励计算** 在每次推演结束时，我们调用 **代码增强的 LLM 即法官（code-augmented LLM-as-a-Judge）** 来确定任务完成状态。验证代码在最终数据库状态上执行，将其与初始状态进行比较，并将其结构化输出（更改的记录、成功标准）与智能体轨迹一起提供给 GPT-5（OpenAI，2025）。法官返回四种分类之一：已完成、部分完成、智能体错误或环境错误。我们按照第 [4.1](#section-4-1) 节的规定将这些分类映射为奖励。为了减少延迟，验证调用被批处理并在收集下一个推演批次时异步执行，为训练循环增加了可忽略的开销。

**评估协议** 我们在三个与训练分布差异很大的基准上评估训练后的智能体。一个关键挑战是每个基准使用不同的工具调用格式：$\tau^{2}$-bench（Barres 等人，2025；Cuadron 等人，2025）期望直接的工具名称（例如 `get_user_details`），BFCLv3（Patil 等人，2025）使用函数调用语法，而 MCP-Universe（Luo 等人，2025）使用 MCP 协议。为了弥合这一差距，我们实现了格式转换器，在我们的统一 `call_tool` 抽象与每个基准的本地格式之间进行转换。对于 $\tau^{2}$-bench，我们将每个工具调用包装为 `call_tool(tool_name="mcp_tool_{name}", arguments={...})` 并相应地解包响应。对于 BFCLv3，我们将多轮次轨迹聚合为预期的单轮次或并行函数调用格式，跳过 `list_tools` 元调用。这确保了评估衡量的是智能体的实际任务解决能力，而非格式合规性。

**解码与上下文配置** 在评估期间，我们使用温度 0.6、top-$k$=20 和 top-$p$=0.95，遵循 Qwen3（Yang 等人，2025）推荐的解码设置。我们通过 **旋转位置编码（Rotary Position Embedding, RoPE）** 缩放（Su 等人，2024）将所有评估的上下文窗口扩展到 131,072 个词元，因为一些基准任务（特别是 BFCLv3 多轮次类别）需要处理冗长的交互历史。在处理长上下文任务时，评估期间的历史限制放宽到 $w=10$ 个轮次，允许智能体利用更多的上下文信息。

<a id="table-8"></a>

> 表 8：GRPO 训练的超参数。

| 超参数             | 值                |
| :----------------- | :---------------- |
| **GRPO**           |                   |
| 学习率             | $7\times 10^{-7}$ |
| 批次大小           | 64                |
| 小批次大小         | 16                |
| 每任务推演次数 $G$ | 16                |
| 每步实例数         | 1,024             |
| 最大优化步数       | 96                |
| KL 系数            | 0.001             |
| 熵系数             | 0.0               |
| 裁剪比率（高）     | 0.28              |
| **推演**           |                   |
| 温度               | 1.0               |
| 最大响应长度       | 2,048             |
| 最大模型上下文     | 32,000            |
| **智能体**         |                   |
| 最大交互轮次       | 20                |
| 历史窗口大小       | 3                 |

<a id="appendix-b"></a>

## 附录 B 分析（Appendix B Analysis）

- [B.1 步骤级格式正确性奖励分析（B.1 Analysis on Step-level Format Correctness Reward）](#b1-analysis-on-step-level-format-correctness-reward)
- [B.2 验证案例研究（B.2 Case Study for Verification）](#b2-case-study-for-verification)

### B.1 步骤级格式正确性奖励分析（B.1 Analysis on Step-level Format Correctness Reward）

<a id="figure-5"></a>

![步骤奖励对比](images/step_reward_cmp.png)

> 图 5：AWM 的格式错误率对比。“w/o Format”表示禁用步骤级格式正确性奖励。

我们在图 [5](https://arxiv.org/html/2602.10090v2#A2.F5) 中研究了 **步骤级格式正确性奖励（step-level format correctness reward）** 的影响。使用此奖励后，智能体（agent）能快速学会遵循工具接口约定，格式错误率迅速收敛至较低水平。它还通过将平均运行时间减少约 $27\%$ 来提高训练效率。相比之下，没有此奖励时，即使在 50 个优化步骤后，格式错误率仍保持在 $20\%$ 以上，导致频繁的无效操作和噪声更大的学习信号。这种退化进一步转化为更低的任务完成率，该完成率在 $40\%$ 以下趋于饱和。总体而言， **步骤级格式正确性奖励同时提升了智能体性能和训练效率** 。

### B.2 验证案例研究（B.2 Case Study for Verification）

我们在图 [28](https://arxiv.org/html/2602.10090v2#A2.F28)、[29](https://arxiv.org/html/2602.10090v2#A2.F29) 和 [30](https://arxiv.org/html/2602.10090v2#A2.F30) 中提供了三个代表性案例，以说明为什么我们 **基于代码增强的 LLM 作为评判者（code-augmented LLM-as-a-Judge）** 比仅使用僵化的纯代码验证或仅从智能体轨迹进行纯 LLM 评判更为鲁棒。对于第一个案例（图 [28](https://arxiv.org/html/2602.10090v2#A2.F28)），任务是一个清晰、基于数据库的查询：智能体调用正确的工具来检索项目的完整出价历史并总结发现；验证器（verifier）可以从底层状态/返回的记录中确定性地确认这些统计数据，并且评判者（judge）与之保持一致，这说明了结构化验证证据起决定性作用的理想情况。对于第二个案例（图 [29](https://arxiv.org/html/2602.10090v2#A2.F29)），当智能体尝试创建例行程序时，环境表现出不完美性；一个期望状态增量的严格纯代码验证器会错误地标记失败，因为初始和最终快照看起来相同，然而轨迹显示了一条幂等的成功路径，而基于代码增强的评判者利用此上下文正确地标记为“已完成”，减少了在工具/基础设施短暂性问题下的假阴性。对于最后一个案例（图 [30](https://arxiv.org/html/2602.10090v2#A2.F30)），一个 API/工具调用错误误导智能体创建了一个重复事件，然后将会话添加到了错误的事件 ID 下；工具调用在局部是成功的，因此没有验证器基础的评判者可能会被欺骗而标记为“已完成”，但验证器揭示了真实目标事件保持不变，使得基于代码增强的评判者能够识别出错误操作并正确拒绝虚假的成功声明。

总体而言，关键主题是： **合成的交互环境是不完美的** ：短暂的工具故障、幂等任务和模糊的工具调用行为都可能打破僵化验证的假设。我们的设计通过让验证器从数据库快照和基于规则的检查中提取 **结构化证据（structured evidence）** ，同时让评判者基于 **轨迹上下文（trajectory context）** 进行推理以解决歧义，从而减少假阴性和假阳性，以此缓解了这个问题。

<a id="figure-6"></a>

![步骤奖励对比](images/step_reward_cmp.png)

> 图 6：AWM 合成场景的分布。

<a id="figure-8"></a>

![数据库视图](images/database_view.png)

> 图 8：“Spotify”环境部分 SQLite 数据库模式的可视化。

<a id="figure-9"></a>

![数据库视图](images/database_view.png)

> 图 9：MCP 环境中智能体的系统提示。

<a id="table-9"></a>

> 表 9：从 AWM 中随机抽取的 100 个生成场景样本。该集合包括代表常见企业和消费者工具使用模式的合成应用程序和真实世界服务。

| 场景名称（Scenario Name）                        | 场景名称（Scenario Name）                       | 场景名称（Scenario Name）                               |
| ------------------------------------------------ | ----------------------------------------------- | ------------------------------------------------------- |
| ArenaPlay Competitive Gaming Hub                 | AT&T                                            | AutoCareLane Service Scheduler                          |
| AutoGrid Dealer Inventory Manager                | Basecamp                                        | Bill.com                                                |
| BlockBridge HOA & Condo Community Manager        | CampusEnroll Admissions Registration Suite      | Canva                                                   |
| Charles Schwab                                   | CineRealm Digital Cinema                        | ClaimWise Property & Auto Insurance Claims Desk         |
| ClassBench Music School Scheduling & Attendance  | ClassTrack Micro-LMS                            | ClinicConnect Multi-Specialty Appointment Center        |
| ClinicPaws Veterinary Practice Suite             | ClubNest Membership Hub                         | CodeMentor Classroom LMS                                |
| CraftBench Custom Manufacturing Job Tracker      | DeskQueue IT Support                            | DevForum Hub (Technical Q&A Forum)                      |
| DeviceFleet Cloud Appliance Manager              | Discord                                         | Duo Security (Cisco Duo)                                |
| Eventbrite                                       | EventHarbor Conference & Session Manager        | FleetAxis Commercial Fleet Command                      |
| Fleetio                                          | FleetPath Manager                               | FlexDesk Workspace Reservations                         |
| FlowMesh Workflow Automation Hub                 | FlowPay Subscription Billing Engine             | FlowTrack Team Workflow Hub                             |
| FormPilot Enterprise Surveys & Workflows         | FundHarbor Retail Investing & Portfolio Tracker | GhostKitchenOS Virtual Brand Manager                    |
| GitHub                                           | GiveStream Community Grants Hub                 | GiveWell-style Donation Portal                          |
| Gmail                                            | GrantPath Scholarship Application Portal        | GreenLedger Utility & Sustainability Dashboard          |
| GuildForge Esports Hub                           | HelpLane Support Desk                           | HireBench Coding Graduate Pipeline                      |
| HireBoard Remote Jobs Marketplace                | HomeFlow Smart Device Console                   | HomeGrid Smart Panel                                    |
| HostEase Property Bookings                       | InsightLoop Feedback & NPS Tracker              | Jira                                                    |
| KitchenGrid Restaurant Ops Hub                   | LeaseLink Corporate Workspace                   | ListNest Shared Collections & Bookmarks                 |
| LootPlaza Virtual Goods Marketplace              | MacroMate Corporate Meal Program Portal         | Microsoft 365 (Office)                                  |
| Microsoft Account (Live.com)                     | Mindbody                                        | NeighborHands Volunteer Network                         |
| NeighborLink Mutual Aid Exchange                 | Notion                                          | OpenTable                                               |
| PantryPicks Grocery & Recipe Wishlist            | PawSitter Network & Booking Hub                 | PayBridge Merchant Payments Gateway                     |
| PayStream Merchant Payment Hub                   | PetCrate Supply Subscriptions                   | PinHarbor Visual Bookmarks                              |
| PrepLine Kitchen Display & Routing               | QuickBite Campus Food Ordering                  | QuickLift Auto Service Booking                          |
| RegShield Policy Compliance & Attestation Center | Restaurant Backoffice Pro                       | Reverb                                                  |
| Salesforce                                       | ShelfStack Personal Media Library               | SignSure Enterprise E-Signature & Policy Acknowledgment |
| SitBuddy Pet Sitting & Home Visit Marketplace    | SkinMart Virtual Goods Marketplace              | SlotBand Event & Vendor Scheduler                       |
| StackList Personal Collections Manager           | StockCrate Warehouse Inventory Cloud            | Stripe Dashboard                                        |
| SubScript Subscription & Billing Manager         | SubStream Creator Membership Hub                | SyncRoom Remote Collaboration Suite                     |
| TailCart Pet Supplies Marketplace                | TalentLoop Recruiting CRM                       | TeleVisit360 Virtual Care                               |
| Toast POS                                        | TutorLane Subject Sessions                      | VendorVault Procurement & Vendor Portal                 |
| VenueLane Event Space Booker                     | VetConnect Telehealth Hub                       | VolunteerGrid Scheduling & Shift Manager                |
| VolunteerShift Scheduler                         | WalkLoop Pet Sitting Network                    | WishCart Social Shopping Lists                          |
| WishLoom Multi-Store Gift Registry               |                                                 |                                                         |

<a id="table-10"></a>

> 表 10：具有丰富注释和显式数据库约束的接口模式片段示例。

| 场景（Scenario）                                                                                                                                                                   | 生成的任务（Generated Tasks）                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 音乐流媒体（Music Streaming）                                                                                                                                                      | 创建一个名为“Morning Focus 2025”、描述为“Upbeat but not distracting”的新播放列表，并将“Daft Punk”最受欢迎的 10 首曲目添加到其中。                                                                                |
| 根据我最近的收听历史，生成一个包含 30 首匹配“chill”心情歌曲的个性化播放列表，并将其保存为“Chill Evening Mix”。                                                                     |                                                                                                                                                                                                                  |
| 创建一个名为“Road Trip to Yosemite”的协作播放列表，并添加 1990 年代排名前 5 的摇滚曲目和 2010 年代排名前 5 的流行曲目。                                                            |                                                                                                                                                                                                                  |
| 电子商务平台（E-commerce Platform）                                                                                                                                                | 搜索“wireless noise cancelling headphones”，按平均客户评分排序结果，并将评分最高且价格低于 200 美元的商品以数量 1 添加到我的购物车。                                                                             |
| 查找我在过去 6 个月内下的“Instant Pot Duo 7-in-1”订单，并启动退货请求，选择“商品有缺陷或无法使用”作为原因，并要求退款至我的原始付款方式。                                          |                                                                                                                                                                                                                  |
| 使用“订阅并节省”功能订阅一个至少 4 星评级的“household paper towels”产品，每 2 个月向我的默认地址配送一个 12 卷装包裹。                                                             |                                                                                                                                                                                                                  |
| 旅行预订（Travel Booking）                                                                                                                                                         | 创建并保存一个多目的地旅行行程，包括 2025 年 8 月 5 日至 8 日在意大利罗马的一家酒店，以及 2025 年 8 月 8 日至 11 日在意大利佛罗伦萨的一家酒店，为 2 位成人选择中等价位（3 或 4 星）、客人评分至少为 8.5 的酒店。 |
| 搜索从柏林（BER）到巴塞罗那（BCN）的酒店和航班捆绑度假套餐，时间为 2025 年 4 月 12 日至 17 日，2 位成人，并返回包含市中心 2 公里范围内至少 4 星级酒店的最便宜套餐。                |                                                                                                                                                                                                                  |
| 搜索葡萄牙里斯本的宠物友好型公寓，入住时间为 2026 年 1 月 5 日至 2026 年 2 月 5 日，为期一个月，为 2 位成人和 1 名儿童，预算最高为每晚 80 欧元，并返回按客人评分排序的前三个选项。 |                                                                                                                                                                                                                  |
