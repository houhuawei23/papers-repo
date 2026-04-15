# 标题：ResearchArcade：面向学术任务的图接口（ResearchArcade : Graph Interface for Academic Tasks）

- ArXiv: 2511.22036
- 作者：Jingjun Xu, Chongshan Lin, Haofei Yu, Tao Feng, Jiaxuan You
- 章节数：83
- 估计词元数：12.9k

## 目录

- 1 引言
- 2 相关工作
- 3 ResearchArcade 数据描述
  - 3.1 多源与多模态
  - 3.2 高度关联与异构性
  - 3.3 动态演化
- 4 ResearchArcade 上的学术任务
  - 4.1 学术图作为异构图
  - 4.2 统一的学术任务定义
    - 4.2.1 引用预测
    - 4.2.2 学术任务 2：段落生成
    - 4.2.3 学术任务 3：修订检索
    - 4.2.4 学术任务 4：修订生成
    - 4.2.5 学术任务 5：录用预测
    - 4.2.6 学术任务 6：反驳生成
  - 4.3 ResearchArcade 支持的有前景的新任务
- 5 实验
  - 5.1 实验设置
  - 5.2 实验结果
    - 5.2.1 ResearchArcade 具有通用性
    - 5.2.2 ResearchArcade 建模动态演化
    - 5.2.3 关系图结构带来一致的性能提升
    - 5.2.4 多模态信息至关重要
    - 5.2.5 评审信息可能存在歧义
- 6 结论
- 伦理声明
- 可复现性声明
- 致谢

## 摘要（Abstract）

###### 摘要

学术研究产生了多样化的数据源。随着研究人员越来越多地使用机器学习来辅助研究任务，一个关键问题随之产生： **我们能否构建一个统一的数据接口，以支持为各种学术任务开发机器学习模型？** 在此类统一接口上训练的模型，能够更好地在整个研究过程中支持人类研究者，并最终加速知识发现。在本工作中，我们介绍了 **ResearchArcade** ，一个基于图的接口，它连接了多个学术数据源，统一了任务定义，并支持广泛的基模型以应对关键的学术挑战。ResearchArcade 利用具有图结构的连贯多表格式来组织来自不同来源的数据，包括来自 ArXiv 的学术语料库和来自 OpenReview 的同行评审，同时捕获文本、图表等多模态信息。ResearchArcade 还保留了稿件层面和社区层面的时间演化，支持研究论文修订以及更广泛的研究趋势随时间的变化。此外，ResearchArcade 统一了多样化的学术任务定义，并支持具有不同输入要求的各种模型。我们在六个学术任务上的实验表明，结合跨源和多模态信息能够支持更广泛的任务，而融入图结构则持续地提升了相对于基线方法的性能。这凸显了 ResearchArcade 的有效性及其推动研究进展的潜力。我们的代码可在 [https://github.com/ulab-uiuc/research-arcade](https://github.com/ulab-uiuc/research-arcade) 获取。

<a id="section-1"></a>

## 1 引言（Introduction）

学术研究代表了人类知识发现的顶峰。诸如预测研究趋势和调试科学论文（sundar2024cpapers; paper_copilot; tian2025mmcr; agi; feng2025grapheval; guide）等多样化的研究任务，都需要获取来自多个来源的全面数据。为了完成这些任务，需要采用各种模型。这些复杂性引发了一个重要的研究问题： **我们能否构建一个统一的数据接口，以支持为各种学术任务开发机器学习模型？**

为研究任务构建这样的接口具有挑战性。在数据方面，首先，学术数据来源于 **ArXiv** 和 **OpenReview** 等多样化平台，涵盖了作者、论文、引用和审稿意见等实体之间复杂的关系。这需要一个能够管理高度关系型数据的灵活框架。其次，数据表征本身跨越多种模态——从文本内容到视觉和表格数据。整体整合这些多样化的表征是一项重大挑战。此外，学术数据的动态性和不断演化的特性进一步使任务复杂化，因为需要框架的持续增长和维护，以跟上不断发展的研究进展。在任务方面，定义不同的学术任务需要在数据预处理和任务制定上付出巨大努力。在模型方面，不同类型的模型需要不同的接口。例如， **大型语言模型（Large Language Models, LLMs）** 需要基于文本的数据作为输入，而 **图神经网络（Graph Neural Networks, GNNs）** 则利用图结构数据。

尽管已有工作对科学研究进行基准测试，但开发一个统一且动态的研究活动表征仍然是一个开放的挑战。虽然现有的学术数据集已系统性地收集和组织了学术数据（kang2018dataset; lo2019s2orc），但它们主要关注单一来源的数据，例如学术语料库或同行评审对话。尽管多模态数据（例如科学论文中的图表和表格）已被纳入以构建有价值的数据集（xia2024docgenome; tian2025mmcr），但这些方法并未充分利用不同数据类型之间的多模态关系。近期工作已使用图来建模学术数据并定义学术任务（li2023scigraphqa; zhang2024oag）。然而，每个学术任务仍然是单独制定的，需要重复的开发工作。

在本文中，我们提出了 **ResearchArcade** ，这是一个基于图的接口，它连接了多样化的学术数据源，具有统一的任务定义，并支持多种基础模型来解决有价值的学术任务。总体而言，ResearchArcade 展现了四个核心特性，使其成为解决学术任务的理想选择： **多源（Multi-Source）** 、 **多模态（Multi-Modal）** 、 **高度结构化与异构（Highly Structural and Heterogeneous）** 以及 **动态演化（Dynamically Evolving）** 。ResearchArcade 整合了来自多个来源的学术数据，包括来自 ArXiv 的研究论文以及来自 OpenReview 的带有修订版本的同行评审，同时收集了多模态信息，包括文本、图表和表格。这些不同的实体以连贯的多表格式组织，选定的表被指定为节点和边，使 ResearchArcade 能够高效地将学术社区中高度相关和异构的数据作为图来处理。此外，ResearchArcade 在两个尺度上建模学术演化：在微观层面，它保留了带有时间信息的论文修订版本，以追踪单个手稿的发展；在宏观层面，其可扩展的框架支持持续的数据纳入，支持随时间推移分析研究趋势。此外，我们在 ResearchArcade 的学术图中统一了多样化的学术任务，使得能够直接在预测和生成两种范式中制定新任务。另外，ResearchArcade 中的结构化知识可以轻松导出为标准格式，如 CSV 和 JSON，便于与包括 LLMs 和 GNNs 在内的各种模型集成。

为了展示 ResearchArcade 的关键优势，我们定义了六项学术任务：图表/表格插入、段落生成、修订检索、修订生成、录用预测和反驳生成。大量实验表明，模型能够受益于 ResearchArcade 中的多源、多模态、异构和动态信息。

总体而言，我们的主要贡献包括：第一，ResearchArcade 通过整合多个数据源、多模态信息，并支持纳入时间和最新数据，实现了多样化的任务定义。第二，ResearchArcade 通过统一任务制定和支持各种模型的训练，促进了学术任务的解决。最后，ResearchArcade 表明，与基线方法相比，纳入图结构能持续提升模型的性能。

<a id="section-2"></a>

## 2 相关工作（Related Work）

**作为图的学术数据。** 现有的学术图研究对学术数据采用了各种分解方法。 **OAG-Bench（zhang2024oag）** 定义了作者、论文和机构等节点，将学术社区建模为异构图。 **unarXive（Saier_2023）** 和 **DocGenome（xia2024docgenome）** 通过将学术语料库进一步分解为段落，创建了更细粒度的图。unarXive 关注段落级别的引用，而 DocGenome 则考虑多模态元素（例如图表和表格）。在 ResearchArcade 中，我们整合了所有这些异构实体，并通过全面和精细的图对其进行了扩展。

**学术数据的动态建模。** 学术数据是动态演化的，其演化大致分为两部分：社区研究趋势和个体手稿演化。对于论文间的演化，gollapalli-li-2015-emnlp 使用主题分布分析了二十年的 ACL 和 EMNLP 会议录，以追踪会议场所的趋同与分化，而 tian2023predicting 将科学子社区的演化建模为事件预测，检测合作图中的增长、分裂和合并。对于论文内的演化，kuznetsov2022revise; d2024aries 在句子级别对齐修订版本，而 jourdan2025pararev 则关注段落级别。在 ResearchArcade 中，这两种演化被同时建模。

**用深度学习解决学术任务。** 各种深度学习模型被用来解决学术任务。yu2025researchtownsimulatorhumanresearch; tinyscientist 基于 LLMs 进行了端到端的科学发现。zhang2024oag 利用 CNN、GNN 和 LLM 来解决多样化的学术任务。然而，他们的工作是分散的，并且需要为不同模型准备专门的数据。ResearchArcade 提供了一个通用的图接口，以统一学术任务的输入数据和任务定义。

<a id="section-3"></a>

## 3 ResearchArcade 数据描述（ResearchArcade Data Description）

ResearchArcade 是对现实世界研究知识的包容性映射，具有四个关键属性：(1) **多源（multi-source）** ，(2) **多模态（multi-modal）** ，(3) **高度相关且异构（highly relational and heterogeneous）** ，(4) **动态演化（dynamically evolving）** 。图 [1](#figure-1) 展示了一个概览，更多细节见附录图 [3](https://arxiv.org/html/2511.22036v1#A1.F3)。

<a id="figure-1"></a>

![dataset_overview_v2](images/dataset_overview_v2.png)

> 图 1：ResearchArcade 使用具有图结构的多表格式，从不同来源收集具有多种模态的数据。表被分类为节点表（彩色）或边表（黑白）。蓝色（代表 OpenReview 部分）或红色（代表 ArXiv 部分）列表示每个节点或边的唯一标识，其余列表示节点或边的特征。黑色加粗列由 LLM 生成。从多表到异构图的转换是直接的。

<a id="section-3-1"></a>

### 3.1 多源与多模态（Multi-source & Multi-Modal）

ResearchArcade 主要来源于 ArXiv 中的计算机科学论文以及 OpenReview 中可用会议的同行评审数据。除了基于文本的数据，ResearchArcade 还整合了多模态数据（例如图表和表格），支持更复杂的多模态任务。

**ArXiv：** ResearchArcade 包含来自 ArXiv 的 66,918 篇论文，涵盖 11 个科学领域，包含 569,501 个章节、8,014,095 个段落、876,636 个图表、324,648 个表格。这些实体之间的相关连接也被 ResearchArcade 捕获。详细统计数据见表 [5](#table-5)，数据收集过程见附录 [A.3.1](https://arxiv.org/html/2511.22036v1#A1.SS3.SSS1)。ResearchArcade 还支持持续爬取，定期（例如每周、每天）更新 ArXiv 数据集。详细描述包含在附录 [A.3.2](https://arxiv.org/html/2511.22036v1#A1.SS3.SSS2) 中。

**OpenReview：** ResearchArcade 还包含来自 OpenReview 的数据，其中包括来自 ICLR、NeurIPS、ICML 和 EMNLP 会议的 57,278 篇投稿，由 189,038 位作者贡献。我们还探索了 CVPR、ECCV、AAAI、IJCAI、ACL 和 NAACL 会议，但它们的同行评审数据不可用。此外，还包括了相应的 884,875 条评审意见和反驳过程中的 54,467 次投稿修订。这些实体通过有价值的连接得以丰富。详细统计数据见表 [6](#table-6) 至表 [9](#table-9)，逐步的数据收集过程描述见附录 [A.3.3](https://arxiv.org/html/2511.22036v1#A1.SS3.SSS3)。

**连接 ArXiv 和 OpenReview：** 连接来自 ArXiv 和 OpenReview 的数据有助于构建更全面的学术图，从而能够定义更多样化的学术任务。为了实现这一目标，OpenReview 中的每篇投稿都根据标题与其在 ArXiv 中的对应论文相关联。请注意，来自 OpenReview 的 25,969 篇（约 45.34%）投稿成功连接到 ArXiv 中的论文。统计数据见表 [10](#table-10) 至表 [13](#table-13)。

**LLM 生成的内容：** 为了便于将 ResearchArcade 用于更多学术任务（例如，矛盾检测、理论综合），我们使用 **Llama-3.1-70B-Instruct（grattafiori2024llama）** 进行数据预处理。投稿修订根据 jourdan2025pararev 概述的类别进行分类。这些类别的详细描述见表 [14](#table-14)。此外，我们应用同一模型为 ArXiv 论文生成研究问题摘要和方法描述。使用的具体提示词见表 [15](#table-15) 至表 [17](#table-17)。

<a id="section-3-2"></a>

### 3.2 高度相关且异构（Highly Relational and Heterogeneous）

学术社区中的研究活动通过类型化实体之间的交互来建模。ResearchArcade 以多表节点-边模式存储数据，由节点表和边表组成，可直接映射到异构图。图 [1](#figure-1) 展示了示意图。

使用来自 ArXiv 的数据，ResearchArcade 构建了文献的双尺度图表示。在论文内层面，每篇论文被分解为一个段落尺度的内容图，包括论文、段落、图表和表格节点，通过类型化的边（例如，论文-段落、段落-图表/表格）连接。在宏观的论文间层面，我们包含作者、学科类别和引用链接，添加了作者身份、类别分配和论文间引用的边。

基于 OpenReview 数据构建的学术图主要对同行评审过程中发生的学术活动进行建模。它包含多样化的节点类型，例如论文、作者、段落、评审意见和修订版本。还包括一些关键关系：连接论文和作者的作者身份关系；连接论文和评审意见的论文下评论关系；连接论文和修订版本的论文修订关系；连接评审意见和修订版本的由评审引起的修订关系等。

<a id="section-3-3"></a>

### 3.3 动态演化（Dynamically Evolving）

随着学术社区的持续演化，ResearchArcade 记录了时间信息（例如，论文上传日期和论文修订时间戳），从而能够真实地模拟学术动态。这包括追踪研究趋势的演化，以及建模由反驳过程驱动的论文更新。此外，ResearchArcade 可以持续更新，以反映学术社区的持续发展。

<a id="section-4"></a>

## 4 ResearchArcade 上的学术任务（Academic Tasks on ResearchArcade）

<a id="figure-2"></a>

![academic_task_language_v1](images/academic_task_language_v1.png)

> **图 2** ：ResearchArcade 通过一个两步方案统一了学术任务定义：(i) **标注（Label）** ：识别任务的目标实体，并将其属性指定为标签；(ii) **输入（Input）** ：检索目标实体的邻域以构建支持任务求解的学术图。

定义不同的学术任务通常需要重复性工作，例如数据收集、清洗和任务规范。借助 ResearchArcade，这些任务可以在我们的学术图上得到统一和便捷的定义。

<a id="section-4-1"></a>

### 4.1 作为异质图的学术图（Academic Graph as A Heterogeneous Graph）

一个 **异质图（Heterogeneous Graph）** 可以定义为 $\mathcal{G}=(\mathcal{V},\,\mathcal{E})$，其中每个节点 $v\in\mathcal{V}$ 和每条边 $e\in\mathcal{E}$ 都通过映射函数被分配一个类型。具体来说，节点类型由 $\tau(v):\mathcal{V}\rightarrow\mathcal{C}$ 定义，边类型由 $\phi(e):\mathcal{E}\rightarrow\mathcal{D}$ 定义，其中 $c\in\mathcal{C}$ 和 $d\in\mathcal{D}$ 分别代表节点类型集合和边类型集合。连接一对节点的边 $e$ 记作 $e=(v,\,u)$。

来自 ResearchArcade 的数据可以表示为一个学术图 $\mathcal{G}=(\mathcal{V},\,\mathcal{E})$，它是异质的。在此上下文中，每个节点 $v\in\mathcal{V}$ 对应节点表中的一行，而每条边 $e$ 对应边表中的一行。此外，每个节点表 $V_{c}$ 与一个唯一的节点类型 $c$ 相关联，每个边表 $E_{d}$ 与一个唯一的边类型 $d$ 相关联。

<a id="section-4-2"></a>

### 4.2 统一的学术任务定义（Unified Academic Task Definition）

如图 [2](#figure-2) 所示，ResearchArcade 通过以下两个步骤统一学术任务定义：(1) 识别目标实体；(2) 检索目标实体的邻域。

**步骤 1：识别学术任务的目标实体。** 目标实体可以是一个节点 $v$ 或一条边 $e$，其属性定义了任务的标签。令 $t$ 表示具有属性 $\mathbf{a}_{t}$ 的目标实体。其某些属性，记作 $\mathbf{y}_{t}\subseteq\mathbf{a}_{t}$，是任务中隐含的标签。

**步骤 2：检索目标实体的邻域。** 为了支持学术任务求解，检索目标实体 $t$ 的多跳邻域，构建一个以 $t$ 为中心的学术图 $\mathcal{G}_{t}$。$t$ 的一跳邻域 $\mathcal{N}_{t}^{(1)}$ 由直接连接到 $t$ 的实体组成。如果 $t\in\mathcal{V}$，则 $\mathcal{N}_{t}^{(1)}=\{k\,|\,k\in\mathcal{V},\,(t,\,k)\in\mathcal{E}\}$。如果 $t\in\mathcal{E}$，则 $\mathcal{N}_{t}^{(1)}=\{k,\,u\,|\,k,\,u\in\mathcal{V},\,t=(k,\,u)\}$。对于 $i>1$，$i$ 跳邻域定义为 $\mathcal{N}_{t}^{(i)}=\{k\,|\,k\in\mathcal{V},\,k^{\prime}\in\mathcal{N}_{t}^{(i-1)},\,(k,k^{\prime})\in\mathcal{E}\}$，这通过额外一跳扩展了 $(i-1)$ 跳邻域。因此，学术图构建为 $\mathcal{G}_{t}=(\mathcal{V}_{t},\,\mathcal{E}_{t})$，其中 $\mathcal{V}_{t}$ 包含 $t$ 的多跳邻域中的节点，$\mathcal{E}_{t}$ 表示这些节点之间的边。因此，一个学术任务定义如下：

$$
f_{\theta}(\mathcal{G}_{t})\rightarrow\mathbf{y}_{t},(1)
$$

其中 $f_{\theta}$ 表示一个具有参数 $\theta$ 的模型。此外，学术任务大致分为 **预测性任务（Predictive tasks）** 和 **生成性任务（Generative tasks）** 。如果标签 $\mathbf{y}_{t}$ 来自有限的结果集，则该任务归类为预测性任务；如果标签 $\mathbf{y}_{t}$ 处于开放式的输出空间，则该任务归类为生成性任务。对于预测性任务，模型（在章节 [5.1](#section-5-1) 中详述）被认为是基于 MLP、基于嵌入、基于 GNN 或基于 GWM 的，其中 GWM 框架有效地将图结构数据与 LLM 集成 (feng2025graph)。不同预测性任务的训练损失各不相同。对于生成性任务，模型主要基于 LLM。使用 **监督微调（Supervised Fine-Tuning, SFT）** 进行训练，其损失定义如下：

$$
\mathcal{L}_{\rm SFT}(\theta)=-\frac{1}{\sum_{t=1}^{T}L_{t}}\sum_{t=1}^{T}\sum_{i=1}^{L_{t}}\log p_{\theta}\!\big(y_{t,i}\,\big|\,y_{t,<i},\,\mathcal{G}_{t}\big),(2)
$$

其中 $L_{t}$ 是 $\mathbf{y}_{t}=[y_{t,\,1},\,...,\,y_{t,\,L_{t}}]$ 的长度，$\log p$ 是对数似然。本文定义了六个学术任务来展示 ResearchArcade 的四个关键特性。表 [1](#table-1) 在两步方案下总结了这些任务，详细的任务定义见附录 [A.4](https://arxiv.org/html/2511.22036v1#A1.SS4)。

<a id="table-1"></a>

> **表 1** ：使用 ResearchArcade 研究的六个学术任务总结。缩写包括：“or”：OpenReview，“ar”：ArXiv，CE：交叉熵损失，BCE：二元交叉熵损失。

| 任务                                  | 目标实体（步骤 1）                      | 邻域（步骤 2）                                                                                      | 损失    | 类型   |
| :------------------------------------ | :-------------------------------------- | :-------------------------------------------------------------------------------------------------- | :------ | :----- |
| **引文预测（Citation Prediction）**   | ar_citation 边：引用段落的内容          | ar_section 节点、ar_paragraph 节点、ar_figure 节点、ar_table 节点、ar_paragraph 节点、ar_paper 节点 | CE      | 预测性 |
| **段落生成（Paragraph Generation）**  | ar_paragraph 节点：段落的文本内容       | ar_paragraph 节点、ar_table 节点、ar_figure 节点、ar_citation 边                                    | SFT     | 生成性 |
| **修订检索（Revision Retrieval）**    | or_revision 节点：被修改段落的索引列表  | 来自原始论文的 or_paragraph 节点、or_review 节点                                                    | InfoNCE | 预测性 |
| **修订生成（Revision Generation）**   | or_paragraph 节点：修订后段落的文本内容 | 原始论文的 or_paragraph 节点、or_review 节点                                                        | SFT     | 生成性 |
| **录用预测（Acceptance Prediction）** | or_paper 节点：论文决定                 | or_paper 节点、ar_paper 节点、ar_paragraph 节点、ar_figure 节点、ar_table 节点                      | BCE     | 预测性 |
| **反驳生成（Rebuttal Generation）**   | or_review 节点：作者回复的文本内容      | 被回复的官方评审的 or_review 节点、ar_paper 节点、ar_paragraph 节点、ar_figure 节点、ar_table 节点  | SFT     | 生成性 |

#### 4.2.1 引文预测（Citation Prediction）

引文预测要求模型为给定段落识别应引用的合适论文，这反映了诸如参考文献推荐等现实需求。虽然先前的工作 (citation*recommendation) 侧重于论文级别的引用，但我们利用 ResearchArcade 中的细粒度学术图在段落级别执行此任务。我们将此任务表述为一个多类分类问题。给定一个包含论文所有段落和现有引用的学术图 $\mathcal{G}*{t}$，以及一个候选段落，模型预测目标段落 $\mathbf{y}_{t}$ 应引用的论文 $\hat{{y}}_{t}$。真实标签 $\mathbf{y}_{t}$ 对应于目标段落中实际引用的论文。在此，我们使用对比交叉熵损失来优化模型：

$$
\mathcal{L}_{\rm CE}(\theta)=-\log\frac{\exp\big(\mathrm{sim}({h}_{t},{z}_{\mathbf{y}_{t}})/\tau\big)}{\sum_{j=1}^{M}\exp\big(\mathrm{sim}({h}_{t},{z}_{j})/\tau\big)},(3)
$$

其中 $\theta$ 表示模型参数，${h}_{t}$ 是目标段落的嵌入向量，${z}_{j}$ 是第 $j$ 个候选引用论文的嵌入向量，$M$ 是候选引用论文的总数，$\mathbf{y}_{t}$ 是真实引用段落的索引，$\tau$ 是温度超参数，$\mathrm{sim}(\cdot,\cdot)$ 表示 $\ell_{2}$ 归一化嵌入向量之间的余弦相似度。该目标函数鼓励模型为真实引用的论文分配比其它候选论文更高的相似度。

#### 4.2.2 学术任务 2：段落生成（Paragraph Generation）

理解如何在学术语料库的适当上下文中生成特定段落，对于辅助科学写作至关重要。ResearchArcade 中固有的图结构提供了段落之间的关系信号，这对于模型理解语料库内的结构依赖关系非常有价值。此生成任务定义如下：给定输入，即一个包含周围段落、引用图表和参考文献的学术图 $\mathcal{G}_{t}$，生成缺失的段落内容 $\hat{\mathbf{y}}_{t}$。原始段落内容作为真实标签 $\mathbf{y}_{t}$。为了训练 LLM，使用了 SFT 损失（公式 [2](https://arxiv.org/html/2511.22036v1#S4.E2)）。为帮助 LLM 更好地理解文档补全任务而设计的提示词如表 [22](#table-22) 所示。

#### 4.2.3 学术任务 3：修订检索（Revision Retrieval）

从审稿人的评论中识别修订的确切位置对于论文润色至关重要。这捕捉了同行评审过程中的论文内部动态，并展示了 ResearchArcade 基于图结构对演进内容进行建模的能力。我们将其表述为一个 top-$k$ 排序任务：给定一个包含论文段落和审稿意见的学术图 $\mathcal{G}_{t}$，预测 top-$k$ 个被修改的段落 $\hat{\mathbf{y}}_{t}$，其中真实标签 $\mathbf{y}_{t}$ 表示实际被修订的段落。训练采用 InfoNCE 损失 (he2020momentum)，该损失最小化审稿意见与修订段落之间的嵌入距离，同时最大化其与未修改段落之间的距离：

$$
\mathcal{L}_{\rm InfoNCE}(\theta)=-\frac{1}{R}\sum_{r=1}^{R}\log\frac{\sum_{i=1}^{M^{+}}\exp\big(\mathrm{sim}(q_{r},k_{i}^{+})/\tau\big)}{\sum_{i=1}^{M^{+}}\exp\big(\mathrm{sim}(q_{r},k_{i}^{+})/\tau\big)+\sum_{j=1}^{M^{-}}\exp\big(\mathrm{sim}(q_{r},k_{j}^{-})/\tau\big)},(4)
$$

其中 $\theta$ 表示模型参数；$q_{r}$ 是模型生成的第 $r$ 条审稿意见的嵌入向量（$r=1,...,R$）；$k_{i}^{+}$ 和 $k_{j}^{-}$ 分别是第 $i$ 个被修改段落和第 $j$ 个未修改段落的嵌入向量；$M^{+}$ 和 $M^{-}$ 是它们的数量；${\rm sim}(\cdot,\cdot)$ 是相似度函数；$\tau$ 是 InfoNCE 损失中的温度参数。

#### 4.2.4 学术任务 4：修订生成（Revision Generation）

基于第 [4.2.3](https://arxiv.org/html/2511.22036v1#S4.SS2.SSS3) 节，本任务侧重于根据审稿人反馈生成针对局部段落的质量提升修订，进一步展示了 ResearchArcade 基于图结构的动态演进能力。与之前关于修订生成的工作 (d2024aries; jourdan2025pararev) 不同，基于 ResearchArcade 中的学术图，我们可以方便地检索审稿人的相应评论以促进任务执行。形式上，给定一个包含原始段落及其审稿意见的学术图 $\mathcal{G}_{t}$，目标是生成一个修订后的段落 $\hat{\mathbf{y}}_{t}$，以实际修订 $\mathbf{y}_{t}$ 作为标签。训练使用 SFT 损失（公式 [2](https://arxiv.org/html/2511.22036v1#S4.E2)），并辅以特定任务提示词（表 [25](#table-25)）来指导 LLM 利用图结构。由于 LLM 的上下文长度有限，首先使用 Qwen3-8B 模型和表 [24](#table-24) 中的提示词对审稿意见进行总结。

#### 4.2.5 学术任务 5：录用预测（Acceptance Prediction）

预测学术论文的录用是一个有意义但具有挑战性的任务。与之前仅关注基于文本的学术图的工作 (feng2025grapheval) 不同，我们将 ArXiv 的全面多模态论文图与 OpenReview 的真实录用标签及时间信息融合来定义该任务。这反映了 ResearchArcade 的多源、多模态和动态演进的特性。我们将该任务设计为一个二分类问题：给定输入，即一个包含往年会议论文及其对应段落、图表和表格的学术图 $\mathcal{G}_{t}$，预测未来年份论文的录用情况 $\hat{\mathbf{y}}_{t}$（接受或拒绝）。真实的论文录用情况是标签 $\mathbf{y}_{t}$。使用二元交叉熵损失作为训练损失：

$$
\mathcal{L}_{\rm BCE}(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\Big[\mathbf{y}_{t}\log\hat{\mathbf{y}}_{t}+(1-\mathbf{y}_{t})\log(1-\hat{\mathbf{y}}_{t})\Big].(5)
$$

其中 $\theta$ 代表模型的参数，$T$ 是论文总数。

#### 4.2.6 学术任务 6：反驳生成（Rebuttal Generation）

针对官方审稿意见生成反驳回应至关重要，因为回应质量强烈影响论文录用。基于 ResearchArcade 中的学术图，此任务可以方便地利用来自 ArXiv 的文本和多模态信息以及来自 OpenReview 的官方审稿意见。形式上，给定一个包含审稿意见及其来自 ArXiv 的相关段落、图表和表格的学术图 $\mathcal{G}_{t}$，目标是生成作者的回应 $\hat{\mathbf{y}}_{t}$，以真实回应 $\mathbf{y}_{t}$ 作为标签。训练使用 SFT 损失（公式 [2](https://arxiv.org/html/2511.22036v1#S4.E2)），并由特定任务提示词（表 [27](#table-27)）指导，以帮助 LLM 捕捉图结构和任务要求。为了解决令牌长度限制，仅包含通过使用 Qwen3-Embedding-0.6B 计算审稿意见与段落嵌入向量之间的余弦相似度选择出的 top-$3$ 个相关段落。

<a id="section-4-3"></a>

### 4.3 ResearchArcade 支持的有前景的新任务（Promising New Tasks Enabled by ResearchArcade）

ResearchArcade 的通用性超越了上述定义的任务，支持研究流程的其它阶段，例如 **想法构思（idea brainstorming）、实验规划（experiment planning）、科学写作（scientific writing）和同行评审（peer reviewing）** ——这些是学术过程中的核心活动。这些有前景的新任务在表 [2](#table-2) 中进行了说明，详细规范见附录 [A.5](https://arxiv.org/html/2511.22036v1#A1.SS5)。

<a id="table-2"></a>

> 表 2：ResearchArcade 为未来工作支持的有前景的新任务。

| 任务                            | 目标实体（步骤 1）                 | 邻域（步骤 2）                                                                   | 损失 | 类型   |
| :------------------------------ | :--------------------------------- | :------------------------------------------------------------------------------- | :--- | :----- |
| 想法生成（Idea Generation）     | ar_paper 节点：摘要                | ar_citation 边，ar_paper 节点                                                    | SFT  | 生成式 |
| 实验规划（Experiment Planning） | ar_table 节点：实验部分的表格文本  | ar_paper 节点，ar_section 节点，ar_paragraph 节点，ar_figure 节点，ar_table 节点 | SFT  | 生成式 |
| 摘要撰写（Abstract Writing）    | ar_paper 节点：摘要                | ar_paper 节点，ar_section 节点，ar_paragraph 节点，ar_figure 节点，ar_table 节点 | SFT  | 生成式 |
| 审稿生成（Review Generation）   | or_review 节点：官方审稿的文本内容 | or_paper 节点，or_paragraph 节点                                                 | SFT  | 生成式 |

<a id="section-5"></a>

## 5 实验（Experiment）

<a id="section-5-1"></a>

### 5.1 实验设置（Experiment Setup）

**数据集** ：我们基于 ResearchArcade 中的一个数据子集进行实验。对于来自 ArXiv 的数据，我们主要关注计算机科学领域且在过去两年内发表的论文。对于从 OpenReview 收集的数据，我们主要关注过去五年内的 ICLR 会议。更多详细信息见附录 [A.6](https://arxiv.org/html/2511.22036v1#A1.SS6)。

**基础模型** ：为了展示 ResearchArcade 与不同模型的兼容性，我们在多种基础模型上进行了实验。

1.  **嵌入模型（Embedding model, EMB）** ：考虑到我们的学术任务输入令牌相对较长，我们使用 Longformer (beltagy2020longformer)，这是一个为处理长文档设计的模型。
2.  **图神经网络（Graph neural network, GNN）** ：由于从我们数据库构建的学术图具有高度关联性和异质性，我们考虑 HANConv (wang2019heterogeneous)，一种异构图注意力神经网络，作为我们基于 GNN 的模型。
3.  **大语言模型（Large language model, LLM）** ：我们主要利用 Qwen3-0.6B 和 Qwen3-8B (qwen3technicalreport) 作为我们基于 LLM 的模型，因为它们在近似参数数量的模型中表现优异，并且在各种评估任务中可与更大的模型相媲美。我们还在 GPTOSS-120B (agarwal2025gpt) 上验证了我们的任务，这是一个更大的最先进模型。
4.  **图世界模型（Graph world model, GWM）** ：为了有效地将图结构数据与 LLM 集成，我们采用了基于嵌入的 GWM (feng2025graph)。它采用多跳聚合来执行嵌入级别的消息传递，产生增强的图表示，这有助于 LLM 更好地理解图结构数据。Qwen3-0.6B (qwen3technicalreport) 被用作基于 GWM 模型的 LLM 模块。

**编码器** ：对于文本模态，我们将文本数据表示为向量嵌入，以便与基于 GNN 和基于 GWM 的模型集成。具体来说，Longformer (beltagy2020longformer) 用于下游 GNN，而 Qwen3-Embedding-0.6B (qwen3embedding) 被用于基于 GWM 的模型中，以与 Qwen3 LLM 模块对齐。对于视觉模态，LLaVA-1.5-7B (liu2024improved) 将图表转换为文本描述，然后使用相同的文本编码器进行编码。虽然我们也尝试过 CLIP，但我们当前的方法更有效且实现更简单。此编码框架保持灵活，可以容纳替代的多模态编码器。

**评估指标** ：为了系统地评估不同模型在我们的学术任务上的性能，为每个任务考虑了不同的评估指标。

1.  **预测任务** ：对于 top-$k$ 排序任务，我们报告 top-$5$ 精确率、top-$5$ 召回率和 top-$5$ F-1 分数来评估模型的性能。对于分类任务，计算准确率、AUC-ROC 分数和马修斯相关系数（Matthews correlation coefficient, MCC）进行评估。
2.  **生成任务** ：使用 SBERT 相似度分数 (reimers2019sentence) 测量生成答案与参考答案之间的语义相似度。使用 Rouge-L (lin2004rouge) 评估词汇重叠度。此外，我们利用 GPT-4o-mini (hurst2024gpt) 来判断输出的清晰度和适当性。我们不使用手工制作的评估指标，而是要求 LLM 表达对生成输出与真实答案之间的成对偏好，并将定量分数定义为生成输出优于（包括平局）参考答案的偏好比例。

$$
\text{LLM-as-a-judge 分数}=\frac{N_{\rm 生成}+N_{\rm 平局}}{N_{\rm 生成}+N_{\rm 平局}+N_{\rm 真实}}.(6)
$$

具体的提示词用法见附录 [A.7.1](https://arxiv.org/html/2511.22036v1#A1.SS7.SSS1)。

<a id="section-5-2"></a>

### 5.2 实验结果（Experiment Results）

**实验结果的最终分析如下** ，各项任务的详细分析见附录 [A.8](https://arxiv.org/html/2511.22036v1#A1.SS8)，案例研究见附录 [A.9](https://arxiv.org/html/2511.22036v1#A1.SS9)。

<a id="table-3"></a>

> **表 3：** 六项学术任务的评估结果。每个基础模型遵循 ($\rm Backbone$, $\rm Training$, $\rm Hop$) 格式，其中 $\rm Backbone$ 为具体模型，$\rm Training$ 为固定（Fixed）或训练（Trained），#-hop 为模型可观察的邻居跳数（0-hop 表示不观察任何邻居）。

| 引文预测（Citation Prediction）  | 段落生成（Paragraph Generation）  |                      |                   |                                  |                    |           |             |
| -------------------------------- | --------------------------------- | -------------------- | ----------------- | -------------------------------- | ------------------ | --------- | ----------- |
| 模型\指标（Model\Metric）        | 准确率（Accuracy）                | AUC - ROC            | MCC               | 模型\指标（Model\Metric）        | SBERT              | Rouge-L   | GPT-4o-mini |
| EMB (Longformer, Fixed, 1-hop)   | 0.970                             | 0.427                | 0.050             | GWM (Qwen3-0.6B, Trained, 0-hop) | 0.581              | 0.163     | 0.009       |
| GNN (HANConv, Trained, 1-hop)    | 0.989                             | 0.995                | 0.396             | GWM (Qwen3-0.6B, Trained, 1-hop) | 0.624              | 0.167     | 0.244       |
| GNN (HANConv, Trained, 3-hop)    | 0.987                             | 0.993                | 0.705             | GWM (Qwen3-0.6B, Trained, 3-hop) | 0.638              | 0.166     | 0.404       |
| GNN (HANConv, Trained, 5-hop)    | 0.989                             | 0.993                | 0.705             | GWM (Qwen3-0.6B, Trained, 5-hop) | 0.642              | 0.165     | 0.344       |
| 修订检索（Revision Retrieval）   | 录用预测（Acceptance Prediction） |                      |                   |                                  |                    |           |             |
| 模型\指标（Model\Metric）        | 精确率@5（Precision@5）           | 召回率@5（Recall@5） | F1分数@5（F-1@5） | 模型\指标（Model\Metric）        | 准确率（Accuracy） | AUC - ROC | MCC         |
| EMB (Longformer, Fixed, 1-hop)   | 0.183                             | 0.154                | 0.145             | MLP (Linear, Trained, 1-hop)     | 0.513              | 0.479     | 0.025       |
| GNN (HANConv, Trained, 1-hop)    | 0.307                             | 0.325                | 0.265             | GNN (HANConv, Trained, 1-hop)    | 0.507              | 0.465     | 0.000       |
| GNN (HANConv, Trained, 3-hop)    | 0.307                             | 0.324                | 0.265             | GNN (HANConv, Trained, 3-hop)    | 0.550              | 0.526     | 0.115       |
| GWM (Qwen3-0.6B, Trained, 1-hop) | 0.304                             | 0.325                | 0.264             | GWM (Qwen3-0.6B, Trained, 1-hop) | 0.470              | 0.478     | -0.063      |
| GWM (Qwen3-0.6B, Trained, 3-hop) | 0.306                             | 0.326                | 0.265             | GWM (Qwen3-0.6B, Trained, 3-hop) | 0.527              | 0.524     | 0.052       |
| 修订生成（Revision Generation）  | 反驳生成（Rebuttal Generation）   |                      |                   |                                  |                    |           |             |
| 模型\指标（Model\Metric）        | SBERT                             | Rouge-L              | GPT-4o-mini       | 模型\指标（Model\Metric）        | SBERT              | Rouge-L   | GPT-4o-mini |
| LLM (Qwen3-0.6B, Fixed, 1-hop)   | 0.321                             | 0.210                | 0.447             | LLM (Qwen3-0.6B, Fixed, 1-hop)   | 0.604              | 0.125     | 0.011       |
| LLM (Qwen3-0.6B, Trained, 1-hop) | 0.733                             | 0.554                | 0.572             | LLM (Qwen3-0.6B, Trained, 1-hop) | 0.638              | 0.131     | 0.022       |
| LLM (Qwen3-8B, Fixed, 1-hop)     | 0.704                             | 0.446                | 0.889             | LLM (Qwen3-8B, Fixed, 1-hop)     | 0.700              | 0.154     | 0.208       |
| LLM (GPTOSS-120B, Fixed, 1-hop)  | 0.669                             | 0.265                | 0.999             | LLM (GPTOSS-120B, Fixed, 1-hop)  | 0.703              | 0.152     | 0.884       |

#### 5.2.1 ResearchArcade 具有通用性（ResearchArcade is General）

表 [3](#table-3) 表明， **ResearchArcade** 通过整合来自 ArXiv 的学术语料与多模态信息以及来自 OpenReview 的带修订的同行评审，同时通过将数据转换为 CSV 或 JSON 格式来支持多种模型，从而实现了多样化任务。基于 **嵌入（Embedding, EMB）** 、基于 **图神经网络（Graph Neural Network, GNN）** 和基于 **图权重模型（Graph Weight Model, GWM）** 的模型能够执行预测性任务，而基于 **大语言模型（Large Language Model, LLM）** 的模型则处理生成性任务。此外，ResearchArcade 的数据质量得到了验证，经过训练的小型 LLM 性能接近大型 LLM。在修订生成任务中，Qwen3-0.6B 的 SBERT 相似度得分和 **LLM 作为评判者（LLM-as-a-judge）** 得分（公式 [6](https://arxiv.org/html/2511.22036v1#S5.E6)）分别从 $0.321$ 提升至 $0.733$ 和 $0.447$ 提升至 $0.572$，接近 Qwen3-8B 和 GPTOSS-120B 的得分。在反驳生成任务中，Qwen3-0.6B 的 SBERT 相似度得分和 LLM 作为评判者得分分别从 $0.604$ 提升至 $0.638$ 和 $0.011$ 提升至 $0.022$，接近 Qwen3-8B 和 GPTOSS-120B 的得分。

#### 5.2.2 ResearchArcade 建模动态演化（ResearchArcade Models Dynamic Evolution）

如表 [3](#table-3) 所示，ResearchArcade 通过整合来自 ArXiv 和 OpenReview 的时间数据，有效捕捉了 **论文内（intra-paper）** 和 **论文间（inter-paper）** 两个层面的动态演化。修订检索和修订生成任务突显了 ResearchArcade 建模论文内演化的能力，能够预测和生成反映稿件持续发展的修订。特别是，基于 GNN 和 GWM 的模型达到的 top-$5$ F1 分数（均为 $0.265$）优于基于 EMB 的模型（$0.145$），这凸显了该框架的有效性。相比之下，录用预测任务反映了论文间演化，旨在通过学习历史数据来识别有录用前景的论文。在此任务中，性能要差得多，最佳准确率仅达到 $0.55$，仅略高于随机水平。这强调了预测研究趋势的内在难度。

#### 5.2.3 关系图结构带来持续增益（Relational Graph Structure Delivers Consistent Gains）

为评估 **ResearchArcade** 以图为中心的设计有效性，我们在两项任务中比较了基于图的模型（基于 **图神经网络（Graph Neural Network, GNN）** 和基于 **图感知大语言模型（Graph-aware Large Language Model, GWM）** ）与非图模型（基于 **嵌入（Embedding, EMB）** 和基于 **多层感知机（Multilayer Perceptron, MLP）** ）的性能。如表 [3](#table-3) 所示，在 **修订检索（Revision Retrieval）** 和 **录用预测（Acceptance Prediction）** 任务中，基于图的模型分别取得了 $67\%$ 和 $7.2\%$ 的性能提升。 **多跳聚合（Multi-hop aggregation）** 进一步提升了性能，尤其在录用预测任务中：虽然 $1$ 跳聚合效果较弱（准确率分别为 $0.507$ 和 $0.47$），但扩展到 $3$ 跳后，基于 GNN 和基于 GWM 的模型准确率均提升至 $0.55$，超过了 MLP 基线（$0.513$）。这表明录用决策依赖于更高阶的上下文信息，例如会议/期刊关联性和时间趋势，这些信息通过多跳邻域得以捕获。

**引用预测（Citation Prediction）** 任务也探究了不同聚合跳数的影响。对于引用预测，尽管 $1$ 跳性能已很高，但扩展邻域范围显著提升了鲁棒性， **马修斯相关系数（Matthews Correlation Coefficient, MCC）** 得分提高了 $30.9\%$。然而，对于其他任务（如修订检索、段落生成），增加跳数带来的收益甚微，甚至导致性能波动。在 **段落生成（Paragraph Generation）** 任务中，GPT-4o-mini 评分从 $40.4\%$（$3$ 跳）下降至 $34.4\%$（$5$ 跳），因为更大的邻域可能引入不相关或噪声信息。

#### 5.2.4 多模态信息至关重要

<a id="table-4"></a>

> 表 4：多模态与审稿信息的消融研究。每个基础模型遵循（$\rm Backbone$, $\rm Training$, $\rm Modality$）格式，其中 $\rm Backbone$ 为具体模型，$\rm Training$ 为固定（Fixed）或训练（Trained），$\rm Modality$ 为包含图与表（with Figure & Table）、仅包含图（with Figure）、仅包含表（with Table）、不包含图与表（without Figure & Table）、包含审稿（with Review）或不包含审稿（without Review）。

| 反驳生成（Rebuttal Generation）  | 引用预测（Citation Prediction） |          |             |                                        |                    |         |             |
| -------------------------------- | ------------------------------- | -------- | ----------- | -------------------------------------- | ------------------ | ------- | ----------- |
| 模型\指标（Model\Metric）        | SBERT                           | Rouge-L  | GPT-4o-mini | 模型\指标（Model\Metric）              | 准确率（Accuracy） | AUC-ROC | MCC         |
| LLM (Qwen3-8B, Fixed, w/o F&T)   | 0.671                           | 0.140    | 0.134       | GNN (HANConv, Trained, 5-hop, w/o F&T) | 0.977              | 0.990   | 0.542       |
| LLM (Qwen3-8B, Fixed, w F)       | 0.692                           | 0.150    | 0.178       | GNN (HANConv, Trained, 5-hop, w F)     | 0.977              | 0.990   | 0.542       |
| LLM (Qwen3-8B, Fixed, w T)       | 0.693                           | 0.152    | 0.191       | GNN (HANConv, Trained, 5-hop, w T)     | 0.980              | 0.990   | 0.564       |
| LLM (Qwen3-8B, Fixed, w F&T)     | 0.700                           | 0.154    | 0.208       | GNN (HANConv, Trained, 5-hop, w F&T)   | 0.989              | 0.993   | 0.705       |
| 修订检索（Revision Retrieval）   | 修订生成（Revision Generation） |          |             |                                        |                    |         |             |
| 模型\指标（Model\Metric）        | Precision@5                     | Recall@5 | F-1@5       | 模型\指标（Model\Metric）              | SBERT              | Rouge-L | GPT-4o-mini |
| EMB (Longformer, Fixed, w/o R)   | 0.067                           | 0.043    | 0.046       | LLM (Qwen3-0.6B, Fixed, w/o R)         | 0.570              | 0.401   | 0.596       |
| EMB (Longformer, Fixed, w R)     | 0.183                           | 0.154    | 0.145       | LLM (Qwen3-0.6B, Fixed, w R)           | 0.321              | 0.210   | 0.447       |
| GNN (HANConv, Trained, w/o R)    | 0.290                           | 0.329    | 0.260       | LLM (Qwen3-8B, Fixed, w/o R)           | 0.712              | 0.473   | 0.873       |
| GNN (HANConv, Trained, w R)      | 0.307                           | 0.324    | 0.265       | LLM (Qwen3-8B, Fixed, w R)             | 0.704              | 0.446   | 0.889       |
| GWM (Qwen3-0.6B, Trained, w/o R) | 0.301                           | 0.320    | 0.260       | LLM (GPTOSS-120B, Fixed, w/o R)        | 0.672              | 0.369   | 0.924       |
| GWM (Qwen3-0.6B, Trained, w R)   | 0.306                           | 0.326    | 0.265       | LLM (GPTOSS-120B, Fixed, w R)          | 0.669              | 0.265   | 0.999       |

表 [4](#table-4) 显示，在反驳生成和引用预测任务中，与纯文本基线相比， **纳入图表信息持续提升了模型性能** 。具体而言，图和表都至关重要，单独添加任一项均能带来一致的性能提升，而同时使用两者则能获得最佳性能。这表明，视觉和表格数据的加入增强了模型对文本内容的理解，从而带来了明显的性能增益。对于反驳生成， **SBERT 相似度得分** 和 **大语言模型即评委（LLM-as-a-judge）** 得分（公式 [6](https://arxiv.org/html/2511.22036v1#S5.E6)）分别从 $0.671$ 提升至 $0.700$，从 $0.134$ 提升至 $0.208$。类似地，在引用预测中，当包含完整模态时，MCC 从 $0.542$ 提升至 $0.705$。这些结果验证了 ResearchArcade 的多模态设计，并突显了其编码多模态信息方法的有效性。

#### 5.2.5 审稿信息可能具有模糊性

我们对修订检索和修订生成任务中的审稿信息进行了消融研究。对于修订检索，我们通过将所有审稿内容替换为表 [28](#table-28) 所列的相同提示来移除审稿信息。根据表 [4](#table-4) 的结果，纳入具体的审稿内容为所有模型带来了性能提升。特别是，与基于 GNN 和基于 GWM 的模型相比，基于 EMB 的模型表现出更大的性能增益。基于 GNN 的模型可以利用审稿图结构进行更好的预测，而基于 GWM 的模型可以进一步利用其 LLM 模块的推理能力以获得更高的绝对性能。对于修订生成的消融研究，我们直接提示模型根据原始段落生成修订后的段落，而不纳入审稿信息。令人惊讶的是，Qwen3-0.6B 在包含审稿时表现甚至更差，可能是因为小模型难以处理更长的上下文。而更大的模型，如 Qwen3-8B 和 GPTOSS-120B，仅显示出适度的改进。一个原因是许多审稿缺乏明确的修订指令，因此模型倾向于进行表面编辑，而非做出能显著改善段落的实质性修改。此外，一些要求的修订需要作者添加特定领域的内容，这对模型而言难以生成。

<a id="section-6"></a>

## 6 结论（Conclusion）

我们提出了 **ResearchArcade** ，这是一个基于图的界面，它将多源（ArXiv、OpenReview）、多模态（文本、图、表）且随时间演化的学术数据统一为连贯的多表格式。此外，ResearchArcade 展现出强大的可扩展性，并支持定期持续爬取新数据。基于一个简单的两步方案：（i）识别目标实体（标签）和（ii）检索任务特定的学术图（邻域），ResearchArcade 标准化了预测性和生成性学术任务的定义。ResearchArcade 兼容多种模型，是研究科研进展和开发促进自动化科研模型的宝贵平台。在六个代表性任务上的实验表明，图结构带来了持续的性能增益。

## 伦理声明（Ethics Statement）

我们依据 ICLR 伦理准则开展此项工作，并仔细考虑了其对学术研究社区的广泛影响。我们的系统旨在通过提供论文发现、审稿辅助和研究趋势分析等工具，为研究自动化做出积极贡献，这些工具有助于普及对学术见解的获取，并支持不同资源水平的研究人员。

潜在风险与缓解措施：我们认识到在学术任务自动化能力方面存在若干值得关注的领域。诸如论文完成和回复草稿生成等自动化功能可能被滥用于学术不端行为。我们强调，我们的系统旨在作为增强人类判断的研究辅助工具，而非替代学术思考或写作。此外，我们对现有学术数据源（ArXiv、OpenReview）的依赖可能会延续出版物模式和审稿过程中现有的偏见。录用预测功能可能无意中以预测录用优先于科学价值的方式影响投稿策略，而不是鼓励方法的严谨性和新颖性。

数据与隐私：我们的系统仅使用来自 ArXiv 和 OpenReview 平台的公开学术数据。我们尊重这些平台的现有使用条款，不试图去匿名化审稿过程或访问私人信息。我们的研究过程不直接涉及人类受试者，因此无需额外的伦理批准。

透明度与负责任使用：我们承认，我们的图构建和任务制定选择嵌入了关于学术工作流程的假设，这些假设可能无法推广到所有研究领域。我们鼓励用户将我们的系统用作探索和辅助工具，而非用于自动化决策，特别是对于高风险的学术决策。所提供的任何研究辅助都应受到适当的人工监督和验证，以保持研究的完整性。

## 可复现性声明（Reproducibility Statement）

为确保我们结果的可复现性，我们已付出大量努力记录方法并提供必要资源。我们图构建过程的完整实现细节，包括来自 ArXiv 和 OpenReview 的多源数据集成，已在 [A.3.1](https://arxiv.org/html/2511.22036v1#A1.SS3.SSS1) 和 [A.3.3](https://arxiv.org/html/2511.22036v1#A1.SS3.SSS3) 中提供。两步任务制定方案在第 4 节中结合具体示例进行了完整说明。在六个代表性任务中使用的所有实验配置、超参数和模型架构均在 [5.1](#section-5-1) 和 [A.6](https://arxiv.org/html/2511.22036v1#A1.SS6) 中详述。我们在 [5.2](#section-5-2) 中提供了全面的消融研究和统计显著性检验程序。数据处理、图构建、模型实现和评估的代码将在发表时提供。构建的异构图数据集，连同任务特定的划分和评估协议，也将一并发布以促进未来研究。

## 致谢（Acknowledgments）

我们衷心感谢亚马逊研究基金以及 Meta 和联想的研究资助的支持。
