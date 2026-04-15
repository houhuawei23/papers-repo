# Title: Understanding World or Predicting Future? A Comprehensive Survey of World Models （理解世界还是预测未来？世界模型综述）

- ArXiv: 2411.14499
- 作者（Authors）: 丁景涛（Jingtao Ding），张云柯（Yunke Zhang），尚宇（Yu Shang），冯杰（Jie Feng），张宇恒（Yuheng Zhang），宗泽方（Zefang Zong），袁源（Yuan Yuan），苏宏远（Hongyuan Su），李念（Nian Li），朴京华（Jinghua Piao），邓雨成（Yucheng Deng），Nicholas Sukiennik，高辰（Chen Gao），许方莉（Fengli Xu），李勇（Yong Li），中国（China）
- 章节（Sections）: 47
- 估计词元数（Estimated tokens）: 73.8k

## 目录（Contents）

- 1. 引言（Introduction）
- 2. 背景与分类（Background and Categorization）
  - 2.1. 历史与当前发展（History and Current Development）
  - 2.2. 来自多领域的演进概念（Evolving Concept from Multiple Domains）
  - 2.3. 分类（Categorization）
- 3. 外部世界的隐式表示（Implicit Representation of the External World）
  - 3.1. 决策中的世界模型（World Model in Decision-Making）
    - 3.1.1. 基于模型的强化学习中的世界模型（World model in model-based RL）
    - 3.1.2. 基于语言主干网络的世界模型（World model with language backbone）
  - 3.2. 模型学习到的世界知识（World Knowledge Learned by Models）
    - 3.2.1. 全球物理世界的知识（Knowledge of the Global Physical World）
    - 3.2.2. 局部物理世界的知识（Knowledge of the Local Physical World）
    - 3.2.3. 人类社会的知识（Knowledge of the Human Society）
- 4. 物理世界的未来预测（Future Prediction of the Physical World）
  - 4.1. 作为视频生成的世界模型（World Model as Video Generation）
    - 4.1.1. 迈向视频世界模型（Towards Video World Models）
    - 4.1.2. 视频世界模型的能力（Capabilities of Video World Models）
  - 4.2. 作为具身环境的世界模型（World Model as Embodied Environment）
    - 4.2.1. 室内环境（Indoor Environments）
    - 4.2.2. 室外环境（Outdoor Environments）
    - 4.2.3. 动态环境（Dynamic Environments）
- 5. 应用领域（Application Domains）
  - 5.1. 游戏智能（Game Intelligence）
  - 5.2. 具身智能（Embodied Intelligence）
    - 5.2.1. 学习隐式表示（Learning Implicit Representation）
    - 5.2.2. 预测环境未来状态（Predicting Future States of the Environment）
    - 5.2.3. 从仿真到现实世界（From Simulation to Real World）
  - 5.3. 城市智能（Urban Intelligence）
    - 5.3.1. 自动驾驶（Autonomous Driving）
    - 5.3.2. 自主物流（Autonomous Logistics）
    - 5.3.3. 城市分析（Urban Analytics）
  - 5.4. 社会智能（Societal Intelligence）
    - 5.4.1. 构建映射现实社会的社会模拟器（Building Social Simulacra Mirroring Real-world Society）
    - 5.4.2. 智能体在社会模拟器中对世界的理解（Agent’s Understanding of External World in Social Simulacra）
  - 5.5. 世界模型的功能（Functions of World Models）
- 6. 开放问题与未来方向（Open Problems and Future Directions）
  - 6.1. 物理规则与反事实仿真（Physical Rules and Counterfactual Simulation）
  - 6.2. 丰富社会维度（Enriching the Social Dimension）
  - 6.3. 基准测试（Benchmarks）
  - 6.4. 通过具身智能连接仿真与现实（Bridging Simulation and Reality with Embodied Intelligence）
  - 6.5. 仿真效率（Simulation Efficiency）
  - 6.6. 伦理与安全问题（Ethical and Safety Concerns）
- 7. 结论（Conclusion）
- 参考文献（References）
- 附录 A 相关综述（Appendix A Related survey）
- 附录 B 图与表（Appendix B Figures and tables）
- 附录 C 更新历史（Appendix C Update History）

## Abstract

###### Abstract.

**世界模型（World Models）** 的概念，由于诸如 GPT-4 等多模态大语言模型（Multimodal Large Language Models, MLLMs）和 Sora 等视频生成模型的进展，已获得广泛关注，这些模型是追求 **人工通用智能（Artificial General Intelligence, AGI）** 的核心。本综述对世界模型的相关文献进行了全面回顾。通常，世界模型被视为理解世界当前状态或预测其未来动态的工具。本综述提出了世界模型的系统性分类，强调其两个主要功能：(1) 构建内部表征以理解世界的运行机制；(2) 预测未来状态以模拟和指导决策。首先，我们审视了这两类功能的当前进展。接着，我们探讨了世界模型在关键领域的应用，包括生成式游戏、自动驾驶、机器人学和 **社会模拟（Social Simulacra）** ，重点关注每个领域如何利用这些方面。最后，我们概述了关键挑战，并对潜在的未来研究方向提供了见解。我们在 [https://github.com/tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model) 中总结了代表性论文及其代码仓库。

## 1. Introduction（引言）

科学界长期以来一直渴望开发一个能够 **复现世界基本动力学（replicate its fundamental dynamics of the world）** 的统一模型，以追求 **人工通用智能（Artificial General Intelligence, AGI）** [1]。2024 年， **多模态大语言模型（multimodal large language models, LLMs）** 和诸如 Sora [2] 等 **视频生成模型（video generation models）** 的出现，加剧了围绕此类 **世界模型（World Models）** 的讨论。尽管这些模型展现出了捕捉世界知识某些方面的新兴能力——例如 Sora 生成的视频似乎完美地遵循物理定律——但关于它们是否真正有资格成为全面的世界模型，疑问依然存在。因此，在我们展望人工智能（Artificial Intelligence, AI）时代新突破之际，对世界模型研究的最新进展、应用和未来方向进行一次系统性综述，既是及时的，也是至关重要的。

世界模型的定义仍然是一个持续争论的话题，通常分为两个主要视角： **理解世界（understanding the world）** 和 **预测未来（predicting the future）** 。如图 [2](#figure-2) 所示，Ha 和 Schmidhuber [3] 的早期工作侧重于 **抽象外部世界（abstracting the external world）** ，以获得对其底层机制的深刻理解。相比之下，LeCun [1] 认为，世界模型不仅应感知和建模真实世界，还应具备 **设想可能的未来状态（envision possible future states）** 以为决策提供信息的能力。诸如 Sora 这样的视频生成模型代表了一种专注于 **模拟未来世界演化（simulating future world evolution）** 的方法，因此更贴近世界模型的预测层面。这就引出了一个关键问题： **世界模型应该优先理解当下，还是预测未来状态？** 在本文中，我们从这两个视角对相关文献进行了全面综述，并重点阐述了关键方法和挑战。

**世界模型（World Models）** 的潜在应用遍及众多领域，每个领域对理解和预测能力都有不同的要求。例如，在 **自动驾驶（Autonomous Driving）** 领域，世界模型需要实时感知路况 [1, 2] 并准确预测其演变 [3, 4, 5]，尤其侧重于即时环境感知和复杂趋势的预测。对于 **机器人（Robotics）** 领域，世界模型对于 **导航（Navigation）** [6]、 **目标检测（Object Detection）** [7] 和 **任务规划（Task Planning）** [8] 等任务至关重要，需要精确理解外部动态 [9] 并具备生成交互式和具身环境的能力 [10]。在 **虚拟社会系统仿真（Simulation of Virtual Social Systems）** 领域，世界模型必须捕捉和预测更抽象的行为动态，例如社会互动和人类决策过程。因此，对这些能力的进展进行全面综述，并探索未来的研究方向和趋势，既是及时的，也是必要的。

现有的关于世界模型的综述通常可分为两类，如表 [S1](#table-1) 所示。第一类主要侧重于描述世界模型在特定领域的应用，例如 **视频处理与生成（Video Processing and Generation）** [11, 12]、 **自动驾驶（Autonomous Driving）** [13, 14, 15] 和 **基于智能体（Agent-based）** 的应用 [12]。第二类 [16] 则关注从能够处理多种模态数据的 **多模态模型（Multi-modal Models）** 到世界模型的技术转变。然而，这些论文往往缺乏对世界模型确切构成内容以及不同现实世界应用对这些模型具体需求的系统性考察。在本文中，我们旨在正式定义和分类世界模型，回顾近期的技术进展，并探索其广泛的应用。

本综述的主要贡献可总结如下：(1) 我们提出了一种围绕两个主要功能构建的世界模型新颖分类体系：构建 **隐式表征（Implicit Representations）** 以理解外部世界机制，以及预测外部世界的未来状态。第一类侧重于开发学习和内化世界知识以支持后续决策的模型，而后者则强调从视觉感知出发，增强在物理世界中的预测和模拟能力。(2) 基于此分类，我们分析了包括 **生成式游戏（Generative Games）** 、自动驾驶、机器人和 **社会模拟（Social Simulacra）** 在内的各种关键应用领域如何强调世界模型的不同方面。(3) 我们强调了世界模型未来能够适应更广泛实际应用的研究方向和趋势。

本文的其余部分组织如下。在第 [2](#section-2) 节中，我们介绍了世界模型的背景并提出了我们的分类体系。第 [3](#section-3) 节和第 [4](#section-4) 节分别详细阐述了两类世界模型的当前研究进展细节。第 [5](#section-5) 节涵盖了世界模型在三个关键研究领域的应用。第 [6](#section-6) 节概述了世界模型的开放问题和未来方向。

## 2. 背景与分类（Background and Categorization）

<a id="figure-1"></a>

![roadmap](images/roadmap.png)

> 图 1. 深度学习时代的世界模型发展路线图。

### 2.1. 历史与当前发展（History and Current Development）

在本节中，我们将探讨文献中不断演化的 **世界模型（World Model）** 概念，并将构建世界模型的努力归类为两个不同的分支： **内部表征（Internal Representation）** 与 **未来预测（Future Prediction）** 。

**深度学习时代之前（Pre Deep Learning Era）** 。构建世界内部模型的概念在 **人工智能（Artificial Intelligence, AI）** 领域有着悠久的历史，可追溯至基础性工作，例如 20 世纪 60 年代 **马文·明斯基（Marvin Minsky）** 的 **框架表示（frame representation）** [1]，其设计旨在系统地捕捉关于世界的结构化知识。在 **强化学习（Reinforcement Learning）** 的背景下， **世界模型（world models）** 作为 **基于模型的方法（model-based approaches）** 的一个基本组成部分出现，其中 **智能体（agents）** 构建对其环境动态的显式表示。该领域的早期工作侧重于学习 **转移模型（transition models）** ，这些模型能够在给定当前状态和动作的情况下预测下一个状态 [2]，使智能体能够在执行前进行规划并模拟潜在的动作序列。这些环境模型通常使用 **表格方法（tabular methods）** 或简单的 **参数化函数（parametric functions）** 来表示，为随着深度学习出现而产生的更复杂的世界建模方法奠定了基础。

**基于模型的强化学习（Model-based Reinforcement Learning）**

Ha 等人（ha2018recurrent,; ha2018world,）在 2018 年通过提出一种基于 **循环神经网络（Recurrent Neural Network, RNN）** 的隐式模型来学习潜在表示，显著地复兴并普及了“ **世界模型（World Model）** ”这一术语。这一研究方向与“ **心理模型（Mental Models）** ” [johnson1983mental,](https://plato.stanford.edu/entries/mental-representation/) 的心理学理论相一致，该理论认为人类通过将外部世界抽象为简化的元素和关系来感知它——这是一个同时体现在 **框架（Frames）** 和 **世界模型（World Models）** 中的基本哲学原理。

这一原理表明，从认知视角来看，我们对世界的理解通常涉及构建能够捕捉基本模式而无需穷尽所有细节的抽象表示。基于这一概念框架，作者引入了一个受人类认知系统启发的 **智能体（Agent）** 模块，如图 [2](#figure-2) 所示。在这个 **循环世界模型（Recurrent World Model, RWM）** 中，智能体接收来自真实环境的反馈，这些反馈随后被转化为一系列用于训练模型的输入。该模型擅长模拟在外部环境中执行特定动作后的潜在结果。本质上，它创建了对潜在未来世界演变的心理模拟，决策则基于对这些状态预测结果的分析而做出。这种方法与 **基于模型的强化学习（Model-based Reinforcement Learning）** 方法高度相似，两者都涉及模型生成外部世界的内部表示，以促进对各种决策任务的导航和解决。

在此概念基础之上，后续的发展进一步推进了世界模型的架构，包括 Google DeepMind 的 **Dreamer 系列（Dreamer series）** （hafner2019dream,; hafner2020mastering,; hafner2025mastering,），该系列已经证明了习得的世界表示在日益复杂的领域中具有可扩展性和有效性。

**自监督学习（Self-supervised Learning）**
在 2022 年那篇关于自主机器智能发展的前瞻性文章中 [1]， **杨立昆（Yann LeCun）** 提出了 **联合嵌入预测架构（Joint Embedding Predictive Architecture, JEPA）** ，这是一个模拟人脑结构的框架。如图 [2](#figure-2) 所示，JEPA 包含一个处理感官数据的 **感知模块（Perception module）** （即编码器），其后是一个评估该信息的 **认知模块（Cognitive module）** （即预测器），后者有效地体现了 **世界模型（World model）** 。这个模型使得大脑能够评估行动，并为现实应用确定最合适的响应。

JEPA 的一个关键创新在于其 **自监督学习（Self-supervised learning）** 范式，它使系统能够在不依赖大量标注数据的情况下学习世界的丰富表征。JEPA 并非预测像素空间中的原始感官输入，而是学习预测 **潜在嵌入空间（Latent embedding space）** 中的抽象表征，这使得学习过程更加高效和鲁棒。这种方法使模型能够捕捉数据中的语义关系和因果结构，同时避免了像素级预测的计算负担和潜在缺陷，例如关注无关细节或噪声。

LeCun 的框架特别引人入胜，因为它融合了 **双系统概念（Dual-system concept）** ，模拟了“快”与“慢”的思考方式。 **系统 1（System 1）** 涉及直觉、本能的反应：即在不明确咨询世界模型的情况下做出的快速决策，例如本能地躲避迎面走来的人。相比之下， **系统 2（System 2）** 则运用审慎、有计划的推理，利用已学习的世界模型来考虑未来状态。它超越了即时的感官输入，通过自监督表征来模拟潜在的未来场景，例如预测房间内未来十分钟的事件并相应地调整行动。这种前瞻性水平要求构建一个能够基于环境的预期动态和演变来有效指导决策的世界模型。

在此框架中，世界模型对于通过 **潜在变量（Latent variables）** 的自监督学习来理解和表征外部世界至关重要，这些变量在过滤冗余信息的同时捕获关键信息。这种方法允许对世界进行高度高效、极简的表征，从而促进对未来场景的最优决策和规划。基于这些原则，最近的实现如 **V-JEPA** [2] 和 **V-JEPA2** [3] 已经证明了基于视频的自监督学习的实际可行性，展示了 JEPA 架构如何能够从未标注的视频数据中学习丰富的时空表征，以用于下游视觉任务。

**参考文献**

1.  Yann LeCun 等. (2022). _A Path Towards Autonomous Machine Intelligence_. 原文链接。
2.  Adrien Bardes 等. (2024). _Revisiting the Joint Embedding Predictive Architecture for Visual Representation Learning_. 原文链接。
3.  Mahmoud Assran 等. (2025). _V-JEPA2: Scaling Video Joint Embedding Predictive Architectures_. 原文链接。

**大型语言模型（Large Language Models）**
“我的语言的界限意味着我的世界的界限。”——路德维希·维特根斯坦（Ludwig Wittgenstein）。这一深刻的观察在 **大型语言模型（Large Language Models, LLMs）** 的语境下尤为贴切，它们通过文本数据学习世界运作的基本原理，这些数据可被用来构建全面的世界模型。近期研究表明，在庞大语料库上训练的 LLMs 会自然地习得潜在的世界知识，包括空间和时间理解，使它们能够对现实世界场景做出复杂的预测 [gurnee2023language, manvi2023geollm]。这一能力已被用于基于模型的任务规划，其中预训练的语言模型作为构建能够推理复杂序列任务的世界模型的基础 [guan2023leveraging]。多模态能力的整合进一步增强了世界建模的潜力。 **多模态大型语言模型（Multimodal Large Language Models, MLLMs）** 能够处理和整合视觉、文本及其他感官模态的信息，从而创建更丰富、更全面的世界表征 [ge2024worldgpt]。
理解这些模型如何处理、表征和利用世界知识，对于开发能够弥合语言知识与现实世界理解之间差距的更有效的世界模型仍然至关重要 [yang2025thinking]。

**视频生成（Video Generation）**
视频生成已成为当代人工智能研究中世界建模的主导方法。与早期隐式的世界表征不同，这些模型显式地生成视觉序列，展示了对时间动态、空间一致性和物理定律的理解。借助 **扩散建模（Diffusion modeling）** 和 **Transformer架构（Transformer architectures）** 等先进生成技术的推动，近期的视频生成模型，包括 Sora [sora2024]、Keling [keling2024] 和 Gen-2 [runway2023]，以文本指令或真实世界视觉数据作为输入，并生成高质量的视频序列。这些模型展示了卓越的世界建模能力，包括在 3D 视频模拟中保持一致性、产生物理上合理的结果以及模拟复杂的数字环境。
这些方法的复杂性表明，它们建模的是底层的真实世界动态，而不仅仅是生成视觉上吸引人的内容。这代表了向能够主动模拟和预测环境如何随时间演变的世界模型的一个根本性转变。最近的进展进一步推进了这一范式，Cosmos [agarwal2025cosmos] 在物理定律遵循方面取得了突破性性能，而 Genie 3 [genie3] 则为可控世界模拟实现了实时交互能力。

**交互式 3D 环境（Interactive 3D Environments）**
交互式 3D 场景生成代表了世界建模的另一个重要范式，其重点是创建沉浸式的 3D 世界，使用户能够在虚拟环境中进行空间探索和交互。代表性工作如 Wonderworld 展示了从单张 2D 图像生成交互式 3D 场景的能力 [yu2025wonderworld]，展现了从最少输入创建可探索虚拟世界的潜力。这种方法强调空间一致性、几何理解以及对用户导航和交互的实时响应能力。最近的进展显著扩展了这些能力，Matrix-3D 通过全景 3D 重建实现了广覆盖、全向可探索的 3D 世界生成 [yang2025matrix]，而 HunyuanWorld 1.0 则通过语义分层的 3D 网格表示实现了沉浸式的 360° 体验，该表示与现有的计算机图形管线提供了无缝兼容性 [team2025hunyuanworld]。

**应用（Applications）**
自 2023 年以来， **世界模型（World models）** 已迅速扩展到不同的应用领域。在 **自动驾驶（Autonomous driving）** 领域，诸如 GAIA-1 和 Drive-WM 等开创性工作，为在复杂交通场景中建模车辆交互与环境动态建立了方法 [hu2023gaia,; wang2023drivingfuturemultiviewvisual,]。机器人领域也取得了类似的进展，2023 年的 DayDreamer 便是例证 [wu2023daydreamer,]，并延续至 2025 年针对机器人操作任务的最新发展 [lu2025gwm,]。

导航应用已随着机器人路径规划 [bar2025navigation,] 而兴起，并扩展到 **六自由度（six-degree-of-freedom）** 空中智能体 [zhao2025airscape,]。游戏是一个特别有前景的领域，关于世界与人类动作模型（World and Human Action Models, WHAM）的里程碑式工作，展示了世界模型如何能创建动态、响应式的虚拟环境 [kanervisto2025world,]。在更大尺度上，基于智能体的社会模拟利用世界模型来理解复杂的社会动态和人类互动，为现实世界的社会现象提供计算层面的见解 [piao2025agentsociety,]。

### 2.2. 多领域演化的概念（Evolving Concept from Multiple Domains）

人工智能中的 **世界模型（World models）** 概念具有深厚的心理学根源，其渊源远早于当代的机器学习。理解这些基础性的联系，揭示了现代 AI 世界模型如何代表了跨越多个学科、被研究数十年的基本认知原理在计算层面的实现。

**心理模型（Mental models）** 这一心理学概念，最初由苏格兰心理学家 **肯尼斯·克雷克（Kenneth Craik）** 在其开创性著作《解释的本质》（"The Nature of Explanation", 1943）中阐明。他在书中提出，“心智构建了现实的小规模模型”，用以预测和理解外部事件 [craik1943nature,]。克雷克的洞见在于，人类认知从根本上是通过创建内部表征来运作的，这些表征捕捉了外部世界的基本结构和动态，从而实现了预测性推理和适应性行为。

这一基础性概念在 20 世纪 80 年代，由英国心理学家 **菲利普·尼古拉斯·约翰逊-莱尔德（Philip Nicholas Johnson-Laird）** 通过其 **心理模型理论（Mental Models Theory）** 进行了系统性的发展和形式化。在其影响深远的著作《心理模型：走向语言、推理和意识的认知科学》（"Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness", 1983）中，约翰逊-莱尔德论证了人类推理是通过构建和操作心理模型——即保留其所表征情境结构关系的内部表征——来进行的 [johnson1983mental,]。根据该理论，当人类进行演绎推理、归纳推理或反事实思考时，他们会通过构建和测试可能世界的替代模型，在心理上模拟不同的场景。

约翰逊-莱尔德的框架确立了几个关键原则，这些原则直接与当代 AI 世界模型相呼应：心理模型是潜在无限领域的有限表征；它们捕捉的是结构关系而非表面细节；它们能够实现对替代场景的预测性模拟。这些原则已成为理解人类和人工智能体如何有效地表征复杂环境并进行推理的基础。

<a id="figure-2"></a>

![structure](images/structure.png)

> 图 2. 本综述的整体框架。我们系统地定义了 **世界模型（World Model）** 的本质目的是理解外部世界的动态并预测未来场景。时间线展示了关键定义和应用的发展历程。

### 2.3. 分类（Categorization）

无论是侧重于学习外部世界的内部表征，还是模拟其运行原理，这些概念都汇聚成一个共同共识： **世界模型（World Model）** 的本质目的是理解世界的动态，并以确定性（或某种保证）计算下一个状态，这使模型能够推断更长周期的演化，并支持下游的决策与规划。基于此视角，我们对世界模型的最新进展进行了全面审视，并通过以下视角进行分析，如图 [2](#figure-2) 所示。

- **外部世界的隐式表征（Implicit representation of the external world）** （第 [3](#section-3) 节）：这一研究类别构建环境变化的模型，以实现更明智的决策，其最终目标是预测未来状态的演化。它通过将外部现实转化为一个将这些要素表示为 **潜变量（Latent Variables）** 的模型，从而培养一种隐式的理解。此外，随着 **大型语言模型（Large Language Models, LLMs）** 的出现，这些模型关于世界知识的详细描述能力，极大地增强了先前集中于传统决策任务的努力。我们进一步关注将世界知识整合到现有模型中的方法。
- **外部世界的未来预测（Future predictions of the external world）** （第 [4](#section-4) 节）：我们首先探索模拟外部世界的 **生成模型（Generative Models）** ，主要使用视觉视频数据。这些工作强调生成的、反映物理世界未来状态的视频的真实性。随着近期进展将焦点转向开发真正交互式的物理世界，我们进一步研究了从视觉表征到空间表征、以及从视频到具身化的转变。这包括全面涵盖与生成反映外部世界的具身环境相关的研究。
- **世界模型的应用（Applications of world models）** （第 [5](#section-5) 节）：世界模型已在多个不同领域展现出广泛的应用，涵盖游戏智能、 **具身智能体（Embodied Agents）** 、城市系统和社会建模。这些领域——分别以生成式游戏、机器人学、自动驾驶和社会模拟为代表——说明了世界模型如何在虚拟和物理环境中连接感知、推理与想象。我们探讨了在这些领域中整合世界模型如何同时推进理论理解和实践创新，强调了它们在塑造智能系统方面的变革潜力。

## 3. 外部世界的隐式表示（Implicit Representation of the External World）

本节探讨 **世界模型（World Model）** 如何通过将环境表示为 **隐变量（Latent Variables）** 来实现明智的决策。第 [3.1](#section-3-1) 节聚焦于 **基于模型的强化学习（Model-Based Reinforcement Learning, MBRL）** 中的世界模型，而第 [3.2](#section-3-2) 节则探讨如何将世界知识整合到先进的人工智能模型，特别是 **大型语言模型（Large Language Models, LLMs）** 中，以提升其在现实世界任务中的性能。

### 3.1. 决策中的世界模型（World Model in Decision-Making）

在决策任务中，理解环境是为生成优化策略奠定基础的主要任务。因此，决策中的世界模型应包含对环境的全面理解。它使我们能够在不影响真实环境的情况下采取假设性行动，从而降低试错成本。在文献中，关于如何学习和利用世界模型的研究最初是在基于模型的强化学习领域提出的。此外，近期在 LLM 和 **多模态大语言模型（Multimodal Large Language Models, MLLMs）** 方面的进展也为世界模型的构建提供了全面的骨干网络。由于语言是一种更通用的表示形式，基于语言的世界模型可以适应更广泛的任务。在决策任务中利用世界模型的两种方案如图 [3](#figure-3) 所示。

<a id="figure-3"></a>

![decisionmaking](images/decisionmaking.png)

> 图 3. 在决策中利用世界模型的两种方案。

#### 3.1.1. 基于模型的强化学习中的世界模型（World model in model-based RL）

在决策中，世界模型的概念在很大程度上指的是 MBRL 中的 **环境模型（Environment Model）** 。一个决策问题通常被表述为一个 **马尔可夫决策过程（Markov Decision Process, MDP）** ，用一个元组 $(S,A,M,R,\gamma)$ 表示，其中 $S, A, \gamma$ 分别表示 **状态空间（State Space）** 、 **动作空间（Action Space）** 和 **折扣因子（Discount Factor）** 。此处的世界模型由 $M$（状态转移动态）和 $R$（奖励函数）组成。由于在大多数情况下奖励函数是已定义的，MBRL 的关键任务是学习和利用转移动态，这可以进一步支持策略优化。

**世界模型学习（World Model Learning）** 。为了学习一个准确的世界模型，最直接的方法是利用每个一步转移的 **均方预测误差（Mean Squared Prediction Error）** [1, 2, 3, 4, 5]，

$$
(1) \min_{\theta}\mathbb{E}_{s^{\prime}\sim M^{*}(\cdot|s,a)}[||s^{\prime}-M_{\theta}(s,a)||^{2}_{2}],
$$

其中 $M^{*}$ 是用于收集轨迹数据的真实转移动态，$M_{\theta}$ 是要学习的参数化转移。除了直接使用确定性转移模型外，Chua 等人 (chua2018deep,) 进一步使用概率转移模型对 **偶然不确定性（Aleatoric Uncertainty）** 进行建模。其目标是最小化转移模型之间的 KL 散度（Kullback–Leibler Divergence），

$$
(2) \min_{\theta}\mathbb{E}_{s^{\prime}\sim M{*}(\cdot|s,a)}[log(\frac{M^{*}(s^{\prime}|s,a)}{M_{\theta}(s^{\prime}|s,a)})].
$$

在这两种设定下， **世界模型（World Model）** 学习任务的阶段都可以转化为一个 **监督学习（Supervised Learning）** 任务。学习标签是来自真实交互环境的轨迹，也称为 **仿真数据（Simulation Data）** (luo2024survey,)。

对于高维环境， **表示学习（Representation Learning）** 对于在 **基于模型的强化学习（Model-Based Reinforcement Learning, MBRL）** 中进行有效的世界模型训练至关重要。Ha 和 Schmidhuber (ha2018recurrent,) 的早期工作通过 **自编码器-潜在状态（Autoencoder–Latent-State）** 流程重建图像，而 Hafner 等人 (hafner2019dream,; hafner2020mastering,) 则将视觉编码器与潜在动态耦合，以掌握基于像素的控制任务。他们的最新版本 DreamerV3 (hafner2025mastering,) 增加了鲁棒的归一化和平衡技术，无需人类数据或领域特定调优，就解决了超过 150 项任务——包括《我的世界（Minecraft）》中的钻石收集。以记忆为中心的扩展，如 Samsami 等人 (samsami2024mastering,) 的 **Recall-to-Imaging** ，进一步增强了长时程推理能力。一个互补的趋势是通过 **下一词元预测（Next-token Prediction）** 与 **Transformer 架构（Transformer Architecture）** 进行统一模型学习，如 Janner 等人 (janner2021offline,) 所示，并由 Schubert 等人 (schubert2023generalist,) 扩展。此外，Georgiev 等人 (georgievpwm,) 训练了一个大型的 **离策略（Off-policy）** 多任务世界模型，其平滑的潜在动态使得能够使用一阶梯度进行高效的每任务策略学习，在不进行在线规划的情况下实现了强大的可扩展性和性能。
Jonathan Richens 等人 (richens2025general,) 最近的工作进一步强化了世界模型的必要性，表明任何能够泛化到多步目标导向任务的智能体，都必须已经学习了一个对其环境的预测模型，而世界模型正是从智能体的策略中产生的。这一见解与将预测建模融入强化学习以处理更复杂、更面向目标的任务的持续趋势相一致。

**使用世界模型生成策略（Policy Generation with World Model）** 。对于一个理想优化的世界模型，生成相应策略最直接的方法是 **模型预测控制（Model Predictive Control, MPC）** (kouvaritakis2016model,)。MPC 在给定模型的情况下规划一个优化的动作序列，如下所示：

$$
(3) \max_{a_{t:t+\tau}}\mathbb{E}_{s_{t^{\prime}+1}\sim p(s_{t^{\prime}+1}|s_{t^{\prime}},a_{t^{\prime}})}[\sum^{t+\tau}_{t^{\prime}=t}r(s_{t^{\prime}},a_{t^{\prime}})].
$$

其中 $\tau$ 表示规划时域（planning horizon）。Nagabandi 等人 (nagabandi2018neural,) 采用了一种简单的蒙特卡洛方法（Monte Carlo method）来采样动作序列。与均匀采样动作不同，Chua 等人 (chua2018deep,) 提出了一种新的概率算法，该算法与轨迹采样（trajectory sampling）进行集成。后续文献也通过利用世界模型（world model）的用法来提高优化效率 (hafner2019dream,; yu2016derivative,; hu2017sequential,; wang2019exploring,)。Hansen 等人 (hansentd,) 提出了一种改进的基于模型的强化学习（Model-based Reinforcement Learning）算法，称为 TD-MPC2，它将轨迹优化集成在所学隐式世界模型（implicit world model）的潜在空间（latent space）中。该算法在多种连续控制任务中实现了强大的性能，并通过在多个领域训练具有数亿参数的大型智能体（agent）展示了可扩展性（scalability）。

另一种生成世界模型策略的流行方法是蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）。通过维护一棵搜索树，其中每个节点对应一个由预定义价值函数（value function）评估的状态（state），算法将选择能使智能体转移到具有更高价值状态的动作。AlphaGo 和 AlphaGo Zero 是在离散动作空间（discrete action space）中使用 MCTS 的两个重要应用 (silver2016mastering,; silver2017mastering,)。Moerland 等人 (moerland2018a0c,) 将 MCTS 扩展到解决连续动作空间（continuous action space）中的决策问题。Oh 等人 (oh2017value,) 提出了一个价值预测网络（value prediction network），将 MCTS 应用于学习到的模型，以基于价值和奖励预测来搜索动作。

#### 3.1.2. 基于语言主干的世界模型（World model with language backbone）

语言模型，尤其是大型语言模型（Large Language Model, LLM）和多模态大语言模型（Multimodal Large Language Model, MLLM）的快速发展，惠及了许多相关应用的发展。以语言作为通用表示主干（universal representation backbone），基于语言的世界模型已在许多决策任务中展现出潜力。

**通过 LLM 世界模型直接生成动作（Direct Action Generation via LLM World Models）**
LLM 能够基于相应构建的世界模型，在决策任务中直接生成动作。例如，在导航场景中，Yang 等人 (yang2023probabilistic,) 将预训练的文本到视频模型（text-to-video model）迁移到特定领域的机器人控制任务中，成功地将 LLM 输出的文本指令用于标注机器人操作。Zhou 等人 (zhou2024robodreamer,) 进一步通过对视频生成过程进行分解，学习了一个组合式世界模型（compositional world model）。这种方法使其对未见任务具有很强的少样本迁移（few-shot transfer）能力。

除了训练或微调专门的基于语言的世界模型外，LLMs 和 MLLMs 也可以直接部署用于理解决策任务中的世界环境。例如，Long 等人 (long2024discuss,) 提出了一种多专家（multi-expert）方案来处理视觉语言导航（visual language navigation）任务。他们构建了一个标准化的讨论过程，其中八个基于 LLM 的专家参与其中，以生成最终的运动决策。一个抽象的世界模型从专家们的讨论和进一步的（对未来状态的）想象中构建出来，以支持动作生成。Zhao 等人 (zhao2024over,) 进一步结合 LLMs 和开放词汇检测（open-vocabulary detection）来构建导航中多模态信号与关键信息之间的关系。他们提出了一个全向图（omni-graph）来捕获局部空间的结构，作为导航任务的世界模型。同时，Yang 等人 (yang2024rila,) 利用一个基于 LLM 的想象助手（imaginative assistant）根据环境感知推断出全局语义图（global semantic graph）作为世界模型，并使用另一个反思规划器（reflective planner）来直接生成动作。

**近期研究（Recent works）** 持续通过解决特定挑战来增强这一范式，例如在网络导航中发现的挑战，相关研究以网络智能体（Web agents）为例 [chae2024web,; qiao2024agent,]。Chae 等人 [chae2024web,] 开发了一种 **世界模型增强（World-model-augmented, WMA）** 智能体，它通过一种新颖的、专注于状态转移的观察抽象来预测动作结果，从而改善网络导航，解决了诸如购买不可退款机票等不可逆动作的复杂问题。类似地，Qiao 等人 [qiao2024agent,] 提出了一种参数化的 **世界知识模型（World Knowledge Model, WKM）** ，该模型为智能体提供先验的全局知识和动态的局部知识，有效缓解了盲目试错和幻觉动作等常见问题。

**LLM 世界模型的模块化使用（Modular Usage of LLM World Models）**
虽然直接将 **大型语言模型（Large Language Model, LLM）** 的输出作为动作在应用和部署上很直接，但在此类方案中，决策质量在很大程度上依赖于 LLM 自身的推理能力。尽管今年见证了 LLM 推理能力的巨大潜力 [xu2025towards,]，但通过将基于 LLM 的世界模型作为模块与外部基于模型的验证器或其他有效的规划算法集成，可以进一步提升其性能 [kambhampati2024position,]。

Guan 等人 [guan2023leveraging,] 通过提示 GPT-4 生成并迭代精炼 **规划领域定义语言（Planning Domain Definition Language, PDDL）** 领域描述来提取显式的世界模型，然后将这些模型与现成的规划器配对，以更少的人工干预获得了良好的规划性能。Xiang 等人 [xiang2024language,] 在世界模型（即 VirtualHome 的模拟器 [puig2018virtualhome,]）中部署了一个具身智能体，并将相应的具身知识注入到 LLM 中。为了更好地规划和完成特定目标，他们提出了一种目标条件规划方案，其中利用 **蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）** 来搜索真实的具身任务目标。Lin 等人 [lin2024learningmodelworldlanguage,] 介绍了一个名为 Dynalang 的智能体，它学习一个多模态世界模型来预测未来的文本和图像表征，并从想象的模型推演中学习如何行动。策略学习阶段利用一个纯粹基于先前生成的多模态表征的 **演员-评论家（Actor-Critic）** 算法。Liu 等人 [liu2024reasonfutureactnow,] 进一步将 LLM 中的推理视为 **贝叶斯自适应马尔可夫决策过程（Bayesian adaptive Markov Decision Processes, MDPs）** 中的学习和规划。LLM 像世界模型一样，在 MDP 的演员-评论家更新中以 **上下文学习（In-context Learning）** 的方式执行。所提出的 RAFA 框架在多个复杂推理任务和环境（如 ALFWorld [shridhar2020alfworld,]）中显示出显著提升的性能。

这种模块化方法也已成功应用于特定领域，例如网络导航。Gu 等人 [gu2024your,] 提出了 WebDreamer，这是一个基于模型的规划框架，它使用一个专门的 LLM 作为世界模型来模拟动作，在网络任务上实现了有竞争力的性能，且效率显著高于树搜索方法。Tang 等人 [tang2024worldcoder,] 则采用了一种不同的方法，引入了 WorldCoder，这是一个基于模型的智能体，通过编写和编辑 Python 程序来构建和精炼其世界模型，展示了比现有方法更高的样本效率和计算效率。

### 3.2. 模型习得的世界知识（World Knowledge Learned by Models）

<a id="table-1"></a>

表 1 | 近期关于模型习得世界知识的研究工作概览。

表 1 | 近期关于模型习得世界知识的研究工作概览。

| 类别（Category）                                   | 方法/模型（Methods/Model）                                      | 年份与会议（Year&Venue） | 模态（Modality） | 内容（Content）       |
| :------------------------------------------------- | :-------------------------------------------------------------- | :----------------------- | :--------------- | :-------------------- |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | PIGEON (lan2025open,)                                           | 2025 ACL Findings        | 语言（Language） | 预测（Prediction）    |
|                                                    | PIGEON (lan2025open,)                                           | 2025 ACL Findings        | 语言（Language） | 预测（Prediction）    |

| 类别（Category）                                   | 方法/模型（Methods/Model）                                      | 年份与会议（Year&Venue） | 模态（Modality） | 内容（Content）       |
| :------------------------------------------------- | :-------------------------------------------------------------- | :----------------------- | :--------------- | :-------------------- |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
| 常识与通用知识（Common Sense & General Knowledge） | KoLA (yu2023kola,)                                              | 2024 ICLR                | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | EWOK (ivanova2024elements,)                                     | 2024 arXiv               | 语言（Language） | 基准测试（Benchmark） |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | Geometry of Concepts (li2024geometryconceptssparseautoencoder,) | 2024 arXiv               | 语言（Language） | 分析（Analysis）      |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | BLEnD (myung2024blend,)                                         | 2024 NeurIPS             | 语言（Language） | 基准测试（Benchmark） |
|                                                    | PIGEON (lan2025open,)                                           | 2025 ACL Findings        | 语言（Language） | 预测（Prediction）    |
|                                                    | PIGEON (lan2025open,)                                           | 2025 ACL Findings        | 语言（Language） | 预测（Prediction）    |

| 模型/研究名称                      | 发表年份与会议                   | 模态             | 任务类型              |
| :--------------------------------- | :------------------------------- | :--------------- | :-------------------- | ---------------- |
| PIGEON (lan2025open,)              | 2025 ACL Findings                | 语言（Language） | 预测（Prediction）    |
| LocalGPT (lan2025benchmarking,)    | 2025 KDD                         | 语言（Language） | 基准测试（Benchmark） |
| Knowledge of Global Physical World | Space&Time (gurnee2023language,) | 2024 ICLR        | 语言（Language）      | 分析（Analysis） |
| GeoLLM (manvi2023geollm,)          | 2024 ICLR                        | 语言（Language） | 理解（Understanding） |
| GeoLLM-Bias (manvi2024large,)      | 2024 ICML                        | 语言（Language） | 理解（Understanding） |
| GPT4GEO (roberts2023gpt4geo,)      | 2023 NeurIPS(FMDM)               | 语言（Language） | 基准测试（Benchmark） |
| CityGPT (feng2024citygpt,)         | 2025 KDD                         | 语言（Language） | 理解（Understanding） |

表 1 | 相关研究工作概览。
| 模型/方法（Model/Method） | 发表年份/会议（Year/Venue） | 模态（Modality） | 任务（Task） |
| :--- | :--- | :--- | :--- |
| CityGPT (feng2024citygpt,) | 2025 KDD | 语言（Language） | 理解（Understanding） |
| CityBench (feng2024citybench,) | 2025 KDD | 语言与视觉（Language&Vision） | 基准测试（Benchmark） |
| UrbanLLaVA (feng2025urbanllava,) | 2025 ICCV | 语言与视觉（Language&Vision） | 理解（Understanding） |
| GPS-To-Image (feng2025gps,) | 2025 CVPR | 视觉（Vision） | 生成（Generation） |
| Ai’s Blind Spots (beneduce2025ai,) | 2025 arXiv | 视觉（Vision） | 生成（Generation） |
| AgentMove (feng2025agentmove,) | 2025 NAACL | 语言（Language） | 预测（Prediction） |
| GLOBE (li2025recognition,) | 2025 arXiv | 语言与视觉（Language&Vision） | 理解（Understanding） |

|                                               | GLOBE (li2025recognition,)        | 2025 arxiv      | Language&Vision | Understanding |
| --------------------------------------------- | --------------------------------- | --------------- | --------------- | ------------- |
|                                               | GLOBE (li2025recognition,)        | 2025 arxiv      | Language&Vision | Understanding |
| Knowledge of Local Physical World             | Predictive (gornet2024automated,) | 2024 NMI        | Vision          | Learning      |
| Knowledge of Local Physical World             | Predictive (gornet2024automated,) | 2024 NMI        | Vision          | Learning      |
| Knowledge of Local Physical World             | Predictive (gornet2024automated,) | 2024 NMI        | Vision          | Learning      |
| Knowledge of Local Physical World             | Predictive (gornet2024automated,) | 2024 NMI        | Vision          | Learning      |
| Knowledge of Local Physical World             | Predictive (gornet2024automated,) | 2024 NMI        | Vision          | Learning      |
| Emergent (jinemergent,)                       | 2024 ICML                         | Language        | Learning        |               |
| Emergent (jinemergent,)                       | 2024 ICML                         | Language        | Learning        |               |
| Emergent (jinemergent,)                       | 2024 ICML                         | Language        | Learning        |               |
| Emergent (jinemergent,)                       | 2024 ICML                         | Language        | Learning        |               |
| E2WM (xiang2024language,)                     | 2023 NeurIPS                      | Language        | Learning        |               |
| E2WM (xiang2024language,)                     | 2023 NeurIPS                      | Language        | Learning        |               |
| E2WM (xiang2024language,)                     | 2023 NeurIPS                      | Language        | Learning        |               |
| E2WM (xiang2024language,)                     | 2023 NeurIPS                      | Language        | Learning        |               |
| Dynalang (lin2024learningmodelworldlanguage,) | 2024 ICML                         | Language&Vision | Learning        |               |
| Dynalang (lin2024learningmodelworldlanguage,) | 2024 ICML                         | Language&Vision | Learning        |               |
| Dynalang (lin2024learningmodelworldlanguage,) | 2024 ICML                         | Language&Vision | Learning        |               |
| Dynalang (lin2024learningmodelworldlanguage,) | 2024 ICML                         | Language&Vision | Learning        |               |
| WM-ABench (gao2025vision,)                    | 2025 ACL Findings                 | Vision          | Benchmark       |               |
| WM-ABench (gao2025vision,)                    | 2025 ACL Findings                 | Vision          | Benchmark       |               |
| WM-ABench (gao2025vision,)                    | 2025 ACL Findings                 | Vision          | Benchmark       |               |
| WM-ABench (gao2025vision,)                    | 2025 ACL Findings                 | Vision          | Benchmark       |               |
| Spatial457 (spatiallm,)                       | 2025 CVPR                         | Vision          | Benchmark       |               |
| Spatial457 (spatiallm,)                       | 2025 CVPR                         | Vision          | Benchmark       |               |

表 1 | 表格描述。
| 模型/数据集/任务名称（Model/Dataset/Task Name） | 发表年份/出处（Year/Venue） | 模态（Modality） | 类型（Type） |
| :--- | :--- | :--- | :--- |
| Spatial457（spatiallm,） | 2025 CVPR | 视觉（Vision） | 基准测试（Benchmark） |
| Thinking in Space（yang2025thinking,） | 2025 CVPR | 视觉（Vision） | 基准测试（Benchmark） |
| Knowledge of Human Society | Testing ToM（strachan2024testing,） | 2024 NHB | 语言（Language） | 基准测试（Benchmark） |
| High-order ToM（street2024llms,） | 2024 arxiv | 语言（Language） | 基准测试（Benchmark） |
| COKE（wu-etal-2024-coke,） | 2024 ACL | 语言（Language） | 学习（Learning） |
| MuMA-ToM（shi2024muma,） | 2024 ACL | 语言与视觉（Language&Vision） | 基准测试（Benchmark） |
| SimToM（wilf2023thinktwiceperspectivetakingimproves,） | 2024 ACL | 语言（Language） | 学习（Learning） |

| SimToM (wilf2023thinktwiceperspectivetakingimproves,) | 2024 ACL | Language | Learning | |
| SimToM (wilf2023thinktwiceperspectivetakingimproves,) | 2024 ACL | Language | Learning | |
| SimToM (wilf2023thinktwiceperspectivetakingimproves,) | 2024 ACL | Language | Learning | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |

| SimToM (wilf2023thinktwiceperspectivetakingimproves,) | 2024 ACL | Language | Learning | |
| SimToM (wilf2023thinktwiceperspectivetakingimproves,) | 2024 ACL | Language | Learning | |
| SimToM (wilf2023thinktwiceperspectivetakingimproves,) | 2024 ACL | Language | Learning | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| EAI (mozikov2024eai,) | 2024 NeurIPS | Language | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| LLM-ToM (kosinski2024evaluating,) | 2024 PNAS | Lanauge | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| SafeWorld (yin2024safeworld,) | 2024 NeurIPS | Lanuage | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |
| 100 languages (vayani2025all,) | 2025 CVPR | Language | Benchmark | |

<a id="figure-4"></a>

![InnerWorldModel-Framework](images/InnerWorldModel-Framework.png)

> 图 4. 用于世界模型的大型语言模型中的世界知识。

在 **大规模网络文本和书籍（large-scale web text and books）** 上进行预训练后（touvron2023llama,; chatgpt,）， **大型语言模型（Large Language Models, LLMs）** 获得了关于现实世界和日常生活相关 **常识（common sense）** 的广泛知识。这种内嵌的知识被认为是其能够在现实世界任务中出色地 **泛化（generalize）** 和有效执行的关键。例如，研究人员利用大型语言模型的常识进行 **任务规划（task planning）** （zhao2024large,）、 **机器人控制（robot control）** （huang2022inner,）和 **图像理解（image understanding）** （liu2024visual,）。此外，Li 等人（li2024geometryconceptssparseautoencoder,）发现了嵌入在大型语言模型中代表概念宇宙的 **高维向量（high-dimensional vectors）** 内的、类似大脑结构的世界知识。同时，Li 等人（li2024vision,）证明，语言模型部分地收敛于与视觉模型 **同构（isomorphic）** 的表征。基于这种关于人类日常生活的广泛知识（myung2024blend,），LLMs 已成功应用于现实世界场景。例如，通过利用这种先验知识为人类的日常活动提供语义信息，它们已在 **本地生活服务（local life services）** 等领域被证明是有效的（lan2025benchmarking,; lan2025open,）。

与常识和通用知识不同，我们从 **世界模型（World Model）** 的视角来关注 **大型语言模型（Large Language Models, LLMs）** 中的世界知识。如图 [4](#figure-4) 所示，根据对象和空间范围，大型语言模型中的世界知识可分为三部分：1） **全球物理世界知识（Knowledge of the Global Physical World）** ；2） **局部物理世界知识（Knowledge of the Local Physical World）** ；以及 3） **人类社会知识（Knowledge of Human Society）** 。我们在表 [1](#table-1) 中总结了近期相关工作。

#### 3.2.1. 全球物理世界知识（Knowledge of the Global Physical World）

我们首先介绍专注于分析和理解全球物理世界知识的研究。
Gurnee 等人 [gurnee2023language] 首次提供了证据，表明大型语言模型确实习得了世界的空间和时间知识，而不仅仅是收集了表面的统计数据。他们在 LLama2 [touvron2023llama] 中识别出独特的“空间神经元（spatial neurons）”和“时间神经元（temporal neurons）”，这表明模型学习了跨多个尺度的空间和时间的线性表示。与先前专注于嵌入空间的观察不同，Manvi 等人 [manvi2023geollm; manvi2024large] 开发了关于文本地址的有效提示，以提取关于地理空间空间的直观现实世界知识，并成功提升了模型在各种下游地理空间预测任务中的性能。

尽管大型语言模型确实获得了一些关于现实世界的隐性知识 [gurnee2023language; li2024geometryconceptssparseautoencoder]，但这些知识的质量仍然存疑 [roberts2023gpt4geo; feng2024citygpt]。例如，Feng 等人 [feng2024citygpt; feng2025urbanllava] 发现，嵌入大型语言模型中的城市知识通常是粗略且不准确的。为了解决这个问题，他们提出了一个有效的框架，以改进大型语言模型对特定城市知识的获取。基于嵌入 LLMs 中的全球地理空间知识，研究人员现在正在应用这种先验世界知识来克服先前方法面临的泛化挑战，例如用于全球移动性预测的 AgentMove [feng2025agentmove]、用于生成具有地理感知和风格可控图像的 GPS-to-Image [feng2025gps; beneduce2025ai]，以及用于基于知识的图像地理定位的 GLOBE [li2025recognition]。

从长远来看，我们可以看到，尽管大型语言模型已经展现出捕获现实世界知识某些方面的能力 [gurnee2023language; li2024geometryconceptssparseautoencoder; roberts2023gpt4geo]，但显然需要进一步的努力来增强这些知识，以实现更广泛、更可靠的现实世界应用 [feng2025survey]。

#### 3.2.2. 局部物理世界知识（Knowledge of the Local Physical World）

#### 3.2.3. 人类社会知识（Knowledge of the Human Society）

与全球物理世界知识不同， **局部物理世界（Local Physical World）** 代表了人类日常生活和大多数现实世界任务的主要环境。因此，理解和建模局部物理世界对于构建一个全面的 **世界模型（World Model）** 而言，是一个更为关键的主题。我们首先引入 **认知地图（Cognitive Map）** 的概念 [tolman1948cognitive]，它指的是人类为了导航和理解其环境（包括空间关系和地标）而形成的心理表征。虽然最初是为了解释人类学习过程而发展出来的，但研究人员已在 **大型语言模型（Large Language Models, LLMs）** 中发现了类似的结构 [li2024geometryconceptssparseautoencoder]，并利用这些洞见来提升人工智能模型在学习和理解物理世界时的效率和性能。

最近的研究探索了如何通过类似认知地图的过程，在各种环境中主动鼓励模型学习抽象知识。例如，Cornet 等人 [gornet2024automated] 表明，在一个简化的 Minecraft 世界中， **视觉预测编码（Visual Predictive Coding）** 可以让智能体仅从像素构建空间认知地图。一旦训练完成，这个 **潜在地图（Latent Map）** 就能编码其到任何目标的度量距离，从而实现对未来观测的准确推演。Lin 等人 [lin2024learningmodelworldlanguage] 研究了如何通过世界模型学习过程（特别是通过预测环境的后续帧）来教导模型理解游戏环境。通过这种方式，模型可以在动态环境中生成更好的行动。此外，Jin 等人 [jinemergent] 发现，语言模型可以通过预测下一个 **词元（Token）** 来学习程序语义的 **涌现表征（Emergent Representations）** 。最近，研究人员 [gao2025vision; spatiallm; yang2025thinking] 将这些研究扩展到更现实的场景，揭示了基于大型语言模型的方法在构建精确模型（即使是针对简单的局部环境）的能力上存在显著差距。

**超越物理世界（Beyond the physical world）** ，理解人类社会是 **世界模型（World Models）** 的另一个关键方面。戴维·普雷马克（David Premack）和盖伊·伍德拉夫（Guy Woodruff）提出了 **心理理论（Theory of Mind, ToM）** [premack1978does]，该理论后来被发展用于解释个体如何推断周围他人的心理状态。近期的研究广泛探讨了 **大型语言模型（Large Language Models, LLMs）** 如何发展并展示这种 **社会世界模型（Social World Model）** [sap2022neural; strachan2024testing; kosinski2024evaluating]。Sap 等人 [sap2022neural] 进行了一项调查，重点评估大型语言模型在各种心理理论任务上的表现，以确定其类人行为是否反映了对社会规则和隐性知识的真正理解。Strachan 等人 [strachan2024testing] 则在理解错误信念和识别讽刺等多样化的心理理论能力上，对人类和 LLM 的表现进行了比较分析。他们的发现虽然展示了 GPT-4 在这些任务上的潜力，但也指出了其局限性，尤其是在察觉失礼行为方面。

除了推断个体心理状态，研究人员也在探索 LLMs 如何建模人类社会更广泛、更底层的规则。例如，Mozikov 等人 [mozikov2024eai] 研究了情感因素如何影响 LLMs 中的伦理判断和决策，强调了需要稳健的机制来确保一致的伦理标准。其他工作则探索了这些模型驾驭全球化世界复杂性的能力。例如，Yin 等人 [yin2024safeworld] 评估了 LLMs 生成回应的能力，要求这些回应不仅要有帮助，还要在不同的全球背景下具备文化敏感性和法律合规性。类似地，Vayani [vayani2025all] 对 LLMs 在百种文化多样语言上进行了大规模评估，强调了语言多样性在开发真正全球性社会模型中的重要性。

尽管这些研究展示了 LLMs 在建模社会世界方面的巨大潜力，但它们也揭示了当这些模型必须处理复杂社会情境时的显著局限性。为了弥补这些不足，并增强 LLMs 在复杂现实应用中的心理理论能力，研究人员提出了几种创新方法。例如，Wu 等人 [wu-etal-2024-coke] 引入了 **COKE** 框架，该框架构建了一个 **知识图谱（Knowledge Graph）** ，通过 **认知链（Cognitive Chains）** 帮助 LLMs 显式地应用心理理论。此外，Alex 等人 [wilf2023thinktwiceperspectivetakingimproves] 开发了 **SimToM** ，这是一个旨在提升大型语言模型在心理理论任务上表现的 **两阶段提示框架（Two-stage Prompting Framework）** 。

## 4. 物理世界的未来预测（Future Prediction of the Physical World）

### 4.1. 作为视频生成的世界模型（World Model as Video Generation）

将 **视频生成（Video Generation）** 技术整合到 **世界模型（World Model）** 中，标志着 **环境建模（Environment Modeling）** 领域的一次重大飞跃 [sora2024]。传统的世界模型主要侧重于预测离散或静态的未来状态 [ha2018world, lecun2022path]。然而，通过生成能够捕捉连续时空动态的类视频模拟，世界模型 [sora2024, yang2024worldgpt] 已经发展到能够处理更复杂、动态的环境。视频生成领域的这一突破，将世界模型的能力推向了新的高度。

#### 4.1.1. 迈向视频世界模型（Towards Video World Models）

**视频世界模型（Video world model）** 是一种计算框架，旨在通过处理视觉上下文中的过往观测和潜在动作，来模拟和预测世界的未来状态 [sora2024]。这一概念建立在更广泛的 **世界模型（World models）** 理念之上，后者致力于捕捉环境的动态特性，并使机器能够预测世界如何随时间演变。对于视频世界模型而言，其重点是生成代表这些演变状态的视觉帧序列。

**Sora** [sora2024] 是一个大规模视频生成模型，旨在基于各种多模态输入，生成长达一分钟的高质量、时间一致的视频序列。它利用神经网络架构来产生视觉上连贯的模拟，这些模拟通常与现实世界的物理原理（如光的反射或融化）相符。这些能力表明 Sora 有潜力作为世界模拟器，基于初始条件和参数预测未来状态。然而，尽管其视频生成能力令人印象深刻，Sora 在完全理解和模拟外部世界方面存在显著局限。一个关键局限是其 **因果推理能力（Causal reasoning ability）** [zhu2024sora; cho2024sora]，这限制了它只能被动生成序列，而无法主动预测动作可能如何改变事件。此外，Sora 难以一致地复现正确的物理定律 [kang2024far]，无法准确模拟复杂的物理现象，如物体在力作用下的行为、流体动力学或光相互作用。

继 Sora 在生成高质量视频方面取得成功之后，过去两年见证了多个大规模视频生成基础模型的涌现，例如 **OpenSora** [zheng2024open]、 **CogVideoX** [yang2024cogvideox] 和 **Wan** [wan2025wan]。这些模型通过对更高效的 **变分自编码器（Variational Autoencoders, VAEs）** 进行预训练，并在大规模视频数据集上进行广泛的预训练，展现出强大的视觉生成能力，并可作为世界模型的基础组件。为了进一步推动该领域发展， **Cosmos** [agarwal2025cosmos] 引入了一个专用于物理世界模拟的视频生成基础模型，通过在海量真实世界物理视频上进行预训练，并探索扩散和自回归架构，在物理定律遵循和理解方面取得了新的突破。与此同时， **Genie 2** [parkerholder2024genie2] 和 **Genie 3** [genie3] 专注于游戏场景中的视频生成，其中 Genie 2 专门设计了一种用于交互式视频生成的自回归扩散架构，支持遵循外部动作指令。除了这些特定模型，在关键技术挑战方面也持续取得进展，包括 **长时生成（Long-duration generation）** [yin2023nuwa; liu2024world; hu2023gaia; henschel2025streamingt2v]、 **交互式生成（Interactive generation）** [zhen20243d; xiang2024pandora; yang2023learning; yang2024video; wu2024ivideogpt; zhang2025physdreamer; jain2024peekaboo; xiang2024pandora; team2025aether; mao2025yume; bar2025navigation] 以及 **物理定律遵循（Physical law adherence）** [yang2024worldgpt; cai2023diffdreamer; ren2024consisti2v; shang2025roboscape]。

研究人员正日益将焦点从基本的、用户不可控的视频生成，转向能够复制现实世界决策空间以辅助决策的交互式模拟。此外，世界模型的概念已扩展到超越纯粹的想象，在多样化的特定场景模拟中找到了应用 [liu2024world; wang2024worlddreamer; bruce2024genie; mendonca2023structured; hu2023gaia; wang2023drivedreamer; bogdoll2023muvo; min2023uniworld]，涵盖了自然环境、游戏、自动驾驶和机器人技术等领域。

<a id="table-2"></a>

表 2. 近期视频生成模型概览（Overview of recent models in video generation across various categories），该表总结了 **长期视频生成（Long-term video generation）** 、 **多模态学习（Multimodal learning）** 、 **交互式视频生成（Interactive video generation）** 、 **时序一致性（Temporal consistency）** 以及 **多样化环境建模（Diverse environment modeling）** 等各类别中的关键模型。

| 类别（Category）  | 模型（Model）                            | 描述（Description）                                                | 技术（Technique）                         |
| :---------------- | :--------------------------------------- | :----------------------------------------------------------------- | :---------------------------------------- |
| 长期（Long-term） | NUWA-XL (yin2023nuwa,)                   | 用于生成长视频的“由粗到精（Coarse-to-fine）”扩散叠加扩散架构。     | 扩散（Diffusion）                         |
|                   | LWM (liu2024world,)                      | 在长视频和语言序列上训练大型 **变换器（Transformer）** 。          | 变换器（Transformer）                     |
|                   | GAIA-1 (hu2023gaia,)                     | 预测自动驾驶场景的 **生成式世界模型（Generative world model）** 。 | 变换器（Transformer）， 扩散（Diffusion） |
|                   | StreamingT2V (henschel2025streamingt2v,) | 配备长/短期记忆模块的自回归 **文本到视频（Text-to-video）** 模型。 |                                           |

| 类别（Category） | 模型（Model）                             | 描述（Description）                                                       | 核心架构（Core Architecture）              |
| :--------------- | :---------------------------------------- | :------------------------------------------------------------------------ | :----------------------------------------- |
| Diffusion        | StreamingT2V（henschel2025streamingt2v,） | 一个配备了长/短期记忆块的自回归文生视频模型。                             | Diffusion                                  |
| Multimodal       | 3D-VLA（zhen20243d,）                     | 在具身人工智能（Embodied AI）的世界模型中整合了3D感知、推理与行动。       | Diffusion                                  |
|                  | Pandora（xiang2024pandora,）              | 支持自由文本动作的世界状态模拟与实时控制。                                | LLM（Large Language Model， 大型语言模型） |
|                  | Genie（bruce2024genie,）                  | 基于文本、图像和草图的生成模型。                                          | Transformer                                |
| Interactive      | UniSim（yang2023learning,）               | 为视觉-语言和强化学习（Reinforcement Learning, RL）训练模拟真实世界交互。 | Diffusion, RL                              |

| 类别                                                                     | 模型                                 | 描述                                                               | 核心技术                                     |
| :----------------------------------------------------------------------- | :----------------------------------- | :----------------------------------------------------------------- | :------------------------------------------- |
| **扩散模型（Diffusion Models）、强化学习（Reinforcement Learning, RL）** |                                      |                                                                    |                                              |
| **交互式（Interactive）**                                                | **UniSim** （yang2023learning,）     | 为视觉-语言和强化学习训练模拟真实世界交互。                        | 扩散模型（Diffusion Models）、强化学习（RL） |
|                                                                          | **VideoDecision** （yang2024video,） | 将视频模型扩展到规划和强化学习等真实世界任务。                     | 变换器（Transformer）、扩散模型（Diffusion） |
|                                                                          | **VideoDecision** （yang2024video,） | 将视频模型扩展到规划和强化学习等真实世界任务。                     | 变换器（Transformer）、扩散模型（Diffusion） |
|                                                                          | **VideoDecision** （yang2024video,） | 将视频模型扩展到规划和强化学习等真实世界任务。                     | 变换器（Transformer）、扩散模型（Diffusion） |
|                                                                          | **VideoDecision** （yang2024video,） | 将视频模型扩展到规划和强化学习等真实世界任务。                     | 变换器（Transformer）、扩散模型（Diffusion） |
|                                                                          | **iVideoGPT** （wu2024ivideogpt,）   | 结合视觉、动作和奖励信号进行交互式世界建模。                       | 变换器（Transformer）                        |
|                                                                          | **iVideoGPT** （wu2024ivideogpt,）   | 结合视觉、动作和奖励信号进行交互式世界建模。                       | 变换器（Transformer）                        |
|                                                                          | **iVideoGPT** （wu2024ivideogpt,）   | 结合视觉、动作和奖励信号进行交互式世界建模。                       | 变换器（Transformer）                        |
|                                                                          | **iVideoGPT** （wu2024ivideogpt,）   | 结合视觉、动作和奖励信号进行交互式世界建模。                       | 变换器（Transformer）                        |
|                                                                          | **PEEKABOO** （jain2024peekaboo,）   | 通过时空控制增强交互性，无需额外训练。                             | 扩散变换器（Diffusion Transformer）          |
|                                                                          | **PEEKABOO** （jain2024peekaboo,）   | 通过时空控制增强交互性，无需额外训练。                             | 扩散变换器（Diffusion Transformer）          |
|                                                                          | **PEEKABOO** （jain2024peekaboo,）   | 通过时空控制增强交互性，无需额外训练。                             | 扩散变换器（Diffusion Transformer）          |
|                                                                          | **PEEKABOO** （jain2024peekaboo,）   | 通过时空控制增强交互性，无需额外训练。                             | 扩散变换器（Diffusion Transformer）          |
|                                                                          | **Aether** （team2025aether,）       | 利用相机轨迹作为几何感知的动作，实现准确的动作条件预测和视觉规划。 | 扩散模型（Diffusion）                        |
|                                                                          | **Aether** （team2025aether,）       | 利用相机轨迹作为几何感知的动作，实现准确的动作条件预测和视觉规划。 | 扩散模型（Diffusion）                        |
|                                                                          | **Aether** （team2025aether,）       | 利用相机轨迹作为几何感知的动作，实现准确的动作条件预测和视觉规划。 | 扩散模型（Diffusion）                        |
|                                                                          | **Aether** （team2025aether,）       | 利用相机轨迹作为几何感知的动作，实现准确的动作条件预测和视觉规划。 | 扩散模型（Diffusion）                        |
|                                                                          | **Yume** （mao2025yume,）            | 根据连续的键盘输入，增强流式交互式世界生成。                       |                                              |

| 方法类别    | 模型名称                          | 核心贡献                                                         | 技术基础  |
| :---------- | :-------------------------------- | :--------------------------------------------------------------- | :-------- |
| Diffusion   | Yume (mao2025yume,)               | 增强了对连续键盘输入的流式交互世界生成能力。                     | Diffusion |
| Diffusion   | Yume (mao2025yume,)               | 增强了对连续键盘输入的流式交互世界生成能力。                     | Diffusion |
| Diffusion   | Yume (mao2025yume,)               | 增强了对连续键盘输入的流式交互世界生成能力。                     | Diffusion |
| Diffusion   | NWM (bar2025navigation,)          | 实现了基于历史观察和导航动作的可控视频生成。                     | Diffusion |
| Diffusion   | NWM (bar2025navigation,)          | 实现了基于历史观察和导航动作的可控视频生成。                     | Diffusion |
| Diffusion   | NWM (bar2025navigation,)          | 实现了基于历史观察和导航动作的可控视频生成。                     | Diffusion |
| Diffusion   | NWM (bar2025navigation,)          | 实现了基于历史观察和导航动作的可控视频生成。                     | Diffusion |
| Consistency | WorldGPT (yang2024worldgpt,)      | 通过多模态学习和优化的关键帧生成，提升了时间一致性和动作平滑度。 | Diffusion |
| Consistency | WorldGPT (yang2024worldgpt,)      | 通过多模态学习和优化的关键帧生成，提升了时间一致性和动作平滑度。 | Diffusion |
| Consistency | WorldGPT (yang2024worldgpt,)      | 通过多模态学习和优化的关键帧生成，提升了时间一致性和动作平滑度。 | Diffusion |
| Consistency | WorldGPT (yang2024worldgpt,)      | 通过多模态学习和优化的关键帧生成，提升了时间一致性和动作平滑度。 | Diffusion |
| Consistency | DiffDreamer (cai2023diffdreamer,) | 实现了具有改进一致性的长距离场景外推。                           | Diffusion |
| Consistency | DiffDreamer (cai2023diffdreamer,) | 实现了具有改进一致性的长距离场景外推。                           | Diffusion |
| Consistency | DiffDreamer (cai2023diffdreamer,) | 实现了具有改进一致性的长距离场景外推。                           | Diffusion |
| Consistency | DiffDreamer (cai2023diffdreamer,) | 实现了具有改进一致性的长距离场景外推。                           | Diffusion |
| Consistency | ConsistI2V (ren2024consisti2v,)   | 增强了图像到视频生成中的视觉一致性。                             | Diffusion |
| Consistency | ConsistI2V (ren2024consisti2v,)   | 增强了图像到视频生成中的视觉一致性。                             | Diffusion |
| Consistency | ConsistI2V (ren2024consisti2v,)   | 增强了图像到视频生成中的视觉一致性。                             | Diffusion |

| 方法类别             | 模型名称                             | 核心贡献                                   | 技术基础    |
| :------------------- | :----------------------------------- | :----------------------------------------- | :---------- |
| Diffusion            | ConsistI2V (ren2024consisti2v,)      | 增强图像到视频生成中的视觉一致性。         | Diffusion   |
|                      | WorldMem (xiao2025worldmem,)         | 通过集成的记忆机制增强长期一致的世界模拟。 | Diffusion   |
|                      | WorldMem (xiao2025worldmem,)         | 通过集成的记忆机制增强长期一致的世界模拟。 | Diffusion   |
|                      | WorldMem (xiao2025worldmem,)         | 通过集成的记忆机制增强长期一致的世界模拟。 | Diffusion   |
|                      | WorldMem (xiao2025worldmem,)         | 通过集成的记忆机制增强长期一致的世界模拟。 | Diffusion   |
| Diverse environments | WorldDreamer (wang2024worlddreamer,) | 捕捉多样化场景中动态元素的世界模型。       | Transformer |
| Diverse environments | WorldDreamer (wang2024worlddreamer,) | 捕捉多样化场景中动态元素的世界模型。       | Transformer |
| Diverse environments | WorldDreamer (wang2024worlddreamer,) | 捕捉多样化场景中动态元素的世界模型。       | Transformer |
| Diverse environments | WorldDreamer (wang2024worlddreamer,) | 捕捉多样化场景中动态元素的世界模型。       | Transformer |
|                      | Genie (bruce2024genie,)              | 用于动作可控虚拟环境的无监督生成模型。     | Transformer |
|                      | Genie (bruce2024genie,)              | 用于动作可控虚拟环境的无监督生成模型。     | Transformer |
|                      | Genie (bruce2024genie,)              | 用于动作可控虚拟环境的无监督生成模型。     | Transformer |
|                      | Genie (bruce2024genie,)              | 用于动作可控虚拟环境的无监督生成模型。     | Transformer |
|                      | MUVO (bogdoll2023muvo,)              | 使用相机和激光雷达数据的多模态世界模型。   | Transformer |
|                      | MUVO (bogdoll2023muvo,)              | 使用相机和激光雷达数据的多模态世界模型。   | Transformer |
|                      | MUVO (bogdoll2023muvo,)              | 使用相机和激光雷达数据的多模态世界模型。   | Transformer |
|                      | MUVO (bogdoll2023muvo,)              | 使用相机和激光雷达数据的多模态世界模型。   | Transformer |
|                      | UniWorld (min2023uniworld,)          | 用于自动驾驶的 3D 检测和运动预测。         | Transformer |

| Transformer |
| :---------- | :-------------------------- | :------------------------------- | :---------- |
|             | UniWorld (min2023uniworld,) | 自动驾驶中的 3D 检测与运动预测。 | Transformer |
|             | UniWorld (min2023uniworld,) | 自动驾驶中的 3D 检测与运动预测。 | Transformer |
|             | UniWorld (min2023uniworld,) | 自动驾驶中的 3D 检测与运动预测。 |             |
| Transformer |                             |                                  |             |
|             | UniWorld (min2023uniworld,) | 自动驾驶中的 3D 检测与运动预测。 | Transformer |
|             | UniWorld (min2023uniworld,) | 自动驾驶中的 3D 检测与运动预测。 | Transformer |
|             | UniWorld (min2023uniworld,) | 自动驾驶中的 3D 检测与运动预测。 | Transformer |

#### 4.1.2. 视频世界模型的能力（Capabilities of Video World Models）

尽管关于像 Sora 这样的模型是否能被视为成熟的世界模型（World Models）的争论仍在继续，但毫无疑问，视频世界模型（Video World Models）在推进环境模拟与预测方面具有巨大潜力 [1, 2, 3]。这些模型能够通过生成逼真、动态的视频序列，为理解和交互复杂环境提供一种强大的方法。为了达到这种复杂程度，本节概述了视频世界模型必须具备的关键能力，以使其区别于传统的视频生成模型（Video Generation Models）。

**长期预测能力（Long-Term Predictive Ability）** 。一个鲁棒的 **视频世界模型（Video World Model）** 应该能够进行长期预测，这些预测需要在长时间跨度内遵循环境的动态规则。这种能力使模型能够模拟场景如何演变，确保生成的视频序列与现实世界的时间进程保持一致。尽管 Sora 已经实现了具有高质量时间一致性的分钟级视频序列生成，但它仍然远未能够模拟现实世界环境中复杂的长期动态。最近的研究工作探索了扩展视频长度以捕获更长期的依赖关系并改善时间一致性 [1, 2, 3]。

**多模态集成（Multi-Modal Integration）** 。除了语言引导的视频生成，视频世界模型正越来越多地集成其他模态，例如图像和动作，以增强真实感和交互性 [4, 5]。多模态的集成允许进行更丰富的模拟，从而更好地捕捉现实世界环境的复杂性，提高生成场景的准确性和多样性。

**交互性（Interactivity）** 。视频世界模型的另一个关键能力是其可控性和交互性的潜力。一个理想的模型不仅应能生成逼真的模拟，还应允许与环境进行交互。这种交互性涉及模拟不同行动的后果并提供反馈，使模型能够用于需要动态决策的应用中。最近的研究工作正专注于增强对模拟的控制，允许更多用户引导的场景探索 [6, 7]。

**多样化环境（Diverse Environments）** 。最后，视频世界模型正在被调整以适应各种特定场景的模拟，包括自然环境、自动驾驶和游戏。这些模型正在超越基本的视频生成，以复制现实世界的动态并支持广泛的应用 [2, 8, 9]。

**长期预测能力（Long-Term Predictive Ability）** 。一个鲁棒的 **视频世界模型（Video World Model）** 应该能够进行长期预测，这些预测需要在长时间跨度内遵循环境的动态规则。这种能力使模型能够模拟场景如何演变，确保生成的视频序列与现实世界的时间进程保持一致。尽管 Sora 已经实现了具有高质量时间一致性的分钟级视频序列生成，但它仍然远未能够模拟现实世界环境中复杂的长期动态。最近的研究工作探索了扩展视频长度以捕获更长期的依赖关系并改善时间一致性 [1, 2, 3]。

**多模态集成（Multi-Modal Integration）** 。除了语言引导的视频生成，视频世界模型正越来越多地集成其他模态，例如图像和动作，以增强真实感和交互性 [4, 5]。多模态的集成允许进行更丰富的模拟，从而更好地捕捉现实世界环境的复杂性，提高生成场景的准确性和多样性。

**交互性（Interactivity）** 。视频世界模型的另一个关键能力是其可控性和交互性的潜力。一个理想的模型不仅应能生成逼真的模拟，还应允许与环境进行交互。这种交互性涉及模拟不同行动的后果并提供反馈，使模型能够用于需要动态决策的应用中。最近的研究工作正专注于增强对模拟的控制，允许更多用户引导的场景探索 [6, 7]。

**多样化环境（Diverse Environments）** 。最后，视频世界模型正在被调整以适应各种特定场景的模拟，包括自然环境、自动驾驶和游戏。这些模型正在超越基本的视频生成，以复制现实世界的动态并支持广泛的应用 [2, 8, 9]。

**参考文献**

1. yin2023nuwa
2. liu2024world
3. hu2023gaia
4. zhen20243d
5. xiang2024pandora
6. yang2024video
7. wu2024ivideogpt
8. wang2024worlddreamer
9. bruce2024genie

### 4.2. World Model as Embodied Environment （世界模型作为具身环境）

为 **具身环境（Embodied Environment）** 开发 **世界模型（World Model）** 对于模拟和预测 **智能体（Agent）** 如何与外部世界交互并适应至关重要。最初， **生成模型（Generative Model）** 侧重于模拟世界的视觉方面，利用视频数据来捕捉环境的动态变化。最近，焦点已转向创建完全交互式和具身的模拟。这些模型不仅表示世界的视觉元素，还融合了空间和物理交互，从而更准确地反映真实世界的动态。通过整合 **空间表征（Spatial Representation）** ，并从基于视频的模拟过渡到沉浸式、具身的环境，世界模型现在可以为开发能够与复杂真实世界环境交互的智能体提供一个更全面的平台。

作为具身环境的世界模型可分为三类：室内环境、室外环境和动态环境，如图 [5](#figure-5) 所示，相关工作总结于表 [3](#table-3)。可以总结出，当前大多数工作侧重于开发静态的、现有的室内和室外具身环境。一个新兴趋势是通过生成模型预测动态的未来世界，这些模型能产生基于视频的、第一人称的动态模拟环境。此类环境可以为训练具身智能体提供灵活且真实的反馈，使其能够与不断变化的环境交互，并提高其 **泛化能力（Generalization Ability）** 。

<a id="table-3"></a>

**表 3. 现有将世界模型作为具身环境的研究对比，包括室内、室外和动态环境。** 在“模态（Modality）”列中，“V”代表视觉（Vision），“L”代表激光雷达（Lidar），“T”代表文本（Text），“A”代表音频（Audio）。在“场景数量（Num of Scenes）”列中，“-”表示无报告数据，“任意（Arbitrary）”表示该方法支持生成任意数量的场景。

表 3 | 现有将世界模型作为具身环境的研究对比，包括室内、室外和动态环境。
| 方法 | 模态 | 场景数量 | 环境类型 |
| :--- | :--- | :--- | :--- |
| 方法 1 | V, L | 10 | 室内 |
| 方法 2 | V, T | 任意 | 室外 |
| 方法 3 | V, A | - | 动态 |

表 1 | 表格描述。
| 类型（Type） | 名称（Name） | 环境（Environment） | 年份（Year） | 场景数量（Num of Scenes） | 模态（Modality） | 物理（Physics） | 3D 资产（3D Assets） |
| :-----------: | :-----------: | :------------------: | :----------: | :----------------------: | :--------------: | :-------------: | :------------------: |
| 室内（Indoor） | AI2-THOR (kolve2017ai2,) | 家庭（Home） | 2017 | 120 | V | ✓ | ✓ |
| 室内（Indoor） | Matterport 3D (Matterport3D,) | 家庭（Home） | 2018 | 90 | V | ✗ | ✗ |
| 室内（Indoor） | Virtual Home (puig2018virtualhome,) | 家庭（Home） | 2018 | 50 | V | ✓ | ✓ |

| 室内 | Virtual Home (puig2018virtualhome,) | 家庭 | 2018 | 50 | V | ✓ | ✓ |
| 室内 | Virtual Home (puig2018virtualhome,) | 家庭 | 2018 | 50 | V | ✓ | ✓ |
| 室内 | Virtual Home (puig2018virtualhome,) | 家庭 | 2018 | 50 | V | ✓ | ✓ |
| 室内 | Virtual Home (puig2018virtualhome,) | 家庭 | 2018 | 50 | V | ✓ | ✓ |
| 室内 | Virtual Home (puig2018virtualhome,) | 家庭 | 2018 | 50 | V | ✓ | ✓ |
| 室内 | Virtual Home (puig2018virtualhome,) | 家庭 | 2018 | 50 | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | Habitat (savva2019habitat,) | 家庭 | 2019 | - | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | SAPIEN (xiang2020sapien,) | 家庭 | 2020 | 46 | V | ✓ | ✓ |
| 室内 | iGibson (shen2021igibson,) | 家庭 | 2021 | 15 | V, L | ✓ | ✓ |
| 室内 | iGibson (shen2021igibson,) | 家庭 | 2021 | 15 | V, L | ✓ | ✓ |
| 室内 | iGibson (shen2021igibson,) | 家庭 | 2021 | 15 | V, L | ✓ | ✓ |
| 室内 | iGibson (shen2021igibson,) | 家庭 | 2021 | 15 | V, L | ✓ | ✓ |
| 室内 | iGibson (shen2021igibson,) | 家庭 | 2021 | 15 | V, L | ✓ | ✓ |

| 环境类型 | 数据集/模拟器                | 场景类型 | 年份 | 场景数量 | 模态    | 交互性 | 可导航性 |
| :------- | :--------------------------- | :------- | :--- | :------- | :------ | :----- | :------- |
| 室内     | iGibson (shen2021igibson,)   | 家庭     | 2021 | 15       | V, L    | ✓      | ✓        |
| 室内     | iGibson (shen2021igibson,)   | 家庭     | 2021 | 15       | V, L    | ✓      | ✓        |
| 室内     | iGibson (shen2021igibson,)   | 家庭     | 2021 | 15       | V, L    | ✓      | ✓        |
| 室内     | iGibson (shen2021igibson,)   | 家庭     | 2021 | 15       | V, L    | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | AVLEN (paul2022avlen,)       | 家庭     | 2022 | 85       | V, T, A | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | ProcTHOR (deitke2022,)       | 家庭     | 2022 | 任意     | V       | ✓      | ✓        |
| 室内     | Holodeck (yang2024holodeck,) | 家庭     | 2024 | 任意     | V       | ✓      | ✓        |
| 室内     | Holodeck (yang2024holodeck,) | 家庭     | 2024 | 任意     | V       | ✓      | ✓        |
| 室内     | Holodeck (yang2024holodeck,) | 家庭     | 2024 | 任意     | V       | ✓      | ✓        |
| 室内     | Holodeck (yang2024holodeck,) | 家庭     | 2024 | 任意     | V       | ✓      | ✓        |
| 室内     | Holodeck (yang2024holodeck,) | 家庭     | 2024 | 任意     | V       | ✓      | ✓        |
| 室内     | Holodeck (yang2024holodeck,) | 家庭     | 2024 | 任意     | V       | ✓      | ✓        |
| 室内     | Holodeck (yang2024holodeck,) | 家庭     | 2024 | 任意     | V       | ✓      | ✓        |

| 环境类型 | 平台                         | 场景        | 年份 | 场景数量 | 模态 | 交互 | 可编程 |
| :------- | :--------------------------- | :---------- | :--- | :------- | :--- | :--- | :----- |
| 室内     | Holodeck (yang2024holodeck,) | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | AnyHome (fu2025anyhome,)     | 家庭        | 2024 | 任意     | V    | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | LEGENT (cheng2024legent,)    | 家庭        | 2024 | 任意     | V, T | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内     | TDW (gan2020threedworld,)    | 家庭        | 2021 | -        | V, A | ✓    | ✓      |
| 室内外   | GRUTopia (wang2024grutopia,) | 家庭， 城市 | 2024 | 100k     | V, T | ✓    | ✓      |
| 室内外   | GRUTopia (wang2024grutopia,) | 家庭， 城市 | 2024 | 100k     | V, T | ✓    | ✓      |

| 环境类型 | 数据集名称                        | 场景       | 年份 | 规模  | 模态 | 支持导航 | 支持交互 |
| :------- | :-------------------------------- | :--------- | :--- | :---- | :--- | :------- | :------- |
| 室内外   | GRUTopia (wang2024grutopia,)      | 家庭、城市 | 2024 | 100k  | V, T | ✓        | ✓        |
| 室内外   | GRUTopia (wang2024grutopia,)      | 家庭、城市 | 2024 | 100k  | V, T | ✓        | ✓        |
| 室内外   | GRUTopia (wang2024grutopia,)      | 家庭、城市 | 2024 | 100k  | V, T | ✓        | ✓        |
| 室内外   | GRUTopia (wang2024grutopia,)      | 家庭、城市 | 2024 | 100k  | V, T | ✓        | ✓        |
| 室内外   | GRUTopia (wang2024grutopia,)      | 家庭、城市 | 2024 | 100k  | V, T | ✓        | ✓        |
| 室内外   | GRUTopia (wang2024grutopia,)      | 家庭、城市 | 2024 | 100k  | V, T | ✓        | ✓        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MineDOJO (fan2022minedojo,)       | 游戏       | 2022 | -     | V    | ✗        | ✗        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | MetaUrban (wu2024metaurban,)      | 城市       | 2024 | 13800 | V, L | ✓        | ✓        |
| 室外     | UrbanWorld (shang2024urbanworld,) | 城市建筑   | 2024 | 任意  | V    | ✓        | ✓        |
| 室外     | UrbanWorld (shang2024urbanworld,) | 城市建筑   | 2024 | 任意  | V    | ✓        | ✓        |
| 室外     | UrbanWorld (shang2024urbanworld,) | 城市建筑   | 2024 | 任意  | V    | ✓        | ✓        |
| 室外     | UrbanWorld (shang2024urbanworld,) | 城市建筑   | 2024 | 任意  | V    | ✓        | ✓        |
| 室外     | UrbanWorld (shang2024urbanworld,) | 城市建筑   | 2024 | 任意  | V    | ✓        | ✓        |

| 环境类型 | 数据集名称（引用）                   | 场景描述         | 年份 | 规模  | 模态 | 3D 标注 | 动作标注 |
| :------: | :----------------------------------- | :--------------- | :--- | :---- | :--- | :-----: | :------: |
|   户外   | UrbanWorld (shang2024urbanworld,)    | 城市建筑         | 2024 | 任意  | V    |    ✓    |    ✓     |
|   户外   | UrbanWorld (shang2024urbanworld,)    | 城市建筑         | 2024 | 任意  | V    |    ✓    |    ✓     |
|   户外   | UrbanWorld (shang2024urbanworld,)    | 城市建筑         | 2024 | 任意  | V    |    ✓    |    ✓     |
|   户外   | UrbanWorld (shang2024urbanworld,)    | 城市建筑         | 2024 | 任意  | V    |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   户外   | EmbodiedCity (gao2024embodiedcity,)  | 城市             | 2024 | 87.1k | V, T |    ✓    |    ✓     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | UniSim (yanglearning,)               | 家庭、城市、仿真 | 2023 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | Streetscapes (deng2024streetscapes,) | 街景             | 2024 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | Streetscapes (deng2024streetscapes,) | 街景             | 2024 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | Streetscapes (deng2024streetscapes,) | 街景             | 2024 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | Streetscapes (deng2024streetscapes,) | 街景             | 2024 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | Streetscapes (deng2024streetscapes,) | 街景             | 2024 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | Streetscapes (deng2024streetscapes,) | 街景             | 2024 | 任意  | V, T |    ✗    |    ✗     |
|   动态   | Streetscapes (deng2024streetscapes,) | 街景             | 2024 | 任意  | V, T |    ✗    |    ✗     |

| 动态性 | Streetscapes（deng2024streetscapes，） | 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | AVID（rigter2024avid，） | 家庭， 游戏 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | EVA（chi2024eva，） | 家庭， 仿真 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Pandora（xiang2024pandora，） | 家庭， 游戏， 仿真， 街景视图 | 2024 | 任意 | V， T | ✗ | ✗ |
| 动态性 | Roboscape（shang2025roboscape，） | 家庭 | 2025 | 任意 | V， T | ✓ | ✗ |
| 动态性 | Roboscape（shang2025roboscape，） | 家庭 |

| 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Roboscape（shang2025roboscape，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Roboscape（shang2025roboscape，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Roboscape（shang2025roboscape，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Roboscape（shang2025roboscape，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Roboscape（shang2025roboscape，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Roboscape（shang2025roboscape，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | TesserAct（zhen2025tesseract，） | 家庭（Home） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Aether（team2025aether，） | 家庭（Home）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Deepverse（chen2025deepverse，） | 家庭（Home）、仿真（Simulation）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Deepverse（chen2025deepverse，） | 家庭（Home）、仿真（Simulation）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Deepverse（chen2025deepverse，） | 家庭（Home）、仿真（Simulation）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Deepverse（chen2025deepverse，） | 家庭（Home）、仿真（Simulation）、街景（Street View） | 2025 | 任意（Arbitrary） | V, T | ✓ | ✗ |
| 动态（Dynamic） | Deepverse（chen2025deepve

| 动态性 | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |
| :----- | :----------------------------- | :--------------- | :--- | :--- | :--- | :-- | :-- |
| 动态   | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |
| 动态   | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |
| 动态   | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |
| 动态   | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |
| 动态   | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |
| 动态   | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |
| 动态   | Deepverse (chen2025deepverse,) | 家庭、模拟、街景 | 2025 | 任意 | V, T | ✓   | ✗   |

#### 4.2.1. 室内环境（Indoor Environments）

室内环境提供了受控、结构化的场景，使得智能体（Agents）能够执行详细、任务特定的动作，例如 **物体操控（Object Manipulation）** 、 **导航（Navigation）** 以及与用户的 **实时交互（Real-time Interaction）** [1]。

早期建立室内环境的工作，如 **AI2-THOR** [2] 和 **Matterport 3D** [3]，主要专注于提供视觉信息。这些工作通过提供照片般逼真的场景来构建室内环境，智能体可以在其中练习 **视觉导航（Visual Navigation）** 并参与模拟现实家庭活动的交互式任务。这些环境强调了使用 **基于视觉的强化学习（Visual-based Reinforcement Learning）** 技术的重要性，该技术允许智能体根据环境线索优化其决策。通过模拟烹饪或清洁等现实世界任务，这些平台评估了智能体在不同类型空间和物体间 **泛化（Generalize）** 所学行为的能力。

一系列后续工作致力于扩展所提供环境的数据模态。其中， **iGibson** [4] 引入了 **激光雷达（Lidar）** 观测作为额外的信号反馈，有助于智能体实现更准确的环境感知。

**AVLEN** [5] 进一步补充了音频信号，使智能体能够在类似家庭的环境中执行物体操控和导航等任务。这里的挑战在于使智能体能够在受限空间内理解并基于包含视觉、语言和声音的 **多模态输入（Multimodal Input）** 采取行动。

在增加社交维度方面，像 **GRUtopia** [6] 这样的环境将智能体引入到必须与物体和 **非玩家角色（Non-Player Characters, NPCs）** 进行导航和交互的空间中。在这里，智能体需要理解社交动态，例如定位和任务共享，这需要更高级的交互建模形式。在这些设置中加入社交交互模块，展示了如何训练智能体在类人的社交行为与任务性能之间取得平衡。

最近，随着 **大型语言模型（Large Language Models, LLMs）** 的发展，一些工作 [7] 试图提供一个灵活的环境生成流程，支持通过语言指令生成任意的室内环境。

[1] gao2024alexa,; paul2022avlen,; kolve2017ai2,; shen2021igibson,; Matterport3D,; puig2018virtualhome,; savva2019habitat,; xiang2020sapien,
[2] kolve2017ai2,
[3] Matterport3D,
[4] shen2021igibson,
[5] paul2022avlen,
[6] wang2024grutopia,
[7] cheng2024legent,; yang2024holodeck,; fu2025anyhome,

<a id="figure-5"></a>

![embodied_env](images/embodied_env.png)

> 图 5. 作为交互式具身环境的世界模型分类，包括室内、室外和动态环境。对外部世界的建模正从构建静态的、当前的环境，向预测动态的、未来的环境演变。

#### 4.2.2. 室外环境（Outdoor Environments）

与室内环境相比，创建室外环境（wang2024grutopia,; gan2020threedworld,; wu2024metaurban,; shang2024urbanworld,; fan2022minedojo,）面临着更大的挑战，因为其规模更大、可变性更高。一些现有工作专注于城市环境，例如 **MetaUrban（MetaUrban）** [wu2024metaurban,]，其中智能体被部署在大型城市环境中进行导航，并面临诸如动态变化的交通、多样的建筑结构以及与其他实体的社交互动等挑战。这些任务通常需要使用 **上下文感知导航算法（Context-aware navigation algorithms）** ，使智能体能够根据环境的布局和条件调整其轨迹和行为。
然而，MetaUrban 中的环境是通过从现有库中检索和组织 3D 资产创建的。
最近，利用先进的生成技术， **UrbanWorld（UrbanWorld）** [shang2024urbanworld,] 显著扩展了室外环境的范围，它使用 3D 生成模型来创建复杂、可定制的城市空间，从而允许更丰富的城市场景。这种从基于静态资产的环境向生成式环境的转变，确保了智能体能够接触到更多样化的任务，从导航陌生的街道布局到与新型物体或结构互动。
除了上述真实的开放世界生成工作外，还有一些虚拟开放世界平台，如 **MineDOJO（MineDOJO）** [fan2022minedojo,]，它们通过模拟程序化生成的、类似沙盒的环境，进一步扩展了这些挑战。这些平台受《我的世界（Minecraft）》开放世界的启发，推动智能体参与资源收集、建造和生存等任务，要求其进行持续探索和适应性学习。在这样的环境中，智能体被激励去寻求新信息并调整其行为以完成给定任务。在此类环境中进行训练，可以帮助智能体学习跨广泛任务和地形的知识，使其能够在各种室外环境中有效运作。

#### 4.2.3. 动态环境（Dynamic Environments）

与室内环境相比，创建室外环境（wang2024grutopia,; gan2020threedworld,; wu2024metaurban,; shang2024urbanworld,; fan2022minedojo,）面临着更大的挑战，因为其规模更大、可变性更高。一些现有工作专注于城市环境，例如 **MetaUrban（MetaUrban）** [wu2024metaurban,]，其中智能体被部署在大型城市环境中进行导航，并面临诸如动态变化的交通、多样的建筑结构以及与其他实体的社交互动等挑战。这些任务通常需要使用 **上下文感知导航算法（Context-aware navigation algorithms）** ，使智能体能够根据环境的布局和条件调整其轨迹和行为。
然而，MetaUrban 中的环境是通过从现有库中检索和组织 3D 资产创建的。
最近，利用先进的生成技术， **UrbanWorld（UrbanWorld）** [shang2024urbanworld,] 显著扩展了室外环境的范围，它使用 3D 生成模型来创建复杂、可定制的城市空间，从而允许更丰富的城市场景。这种从基于静态资产的环境向生成式环境的转变，确保了智能体能够接触到更多样化的任务，从导航陌生的街道布局到与新型物体或结构互动。
除了上述真实的开放世界生成工作外，还有一些虚拟开放世界平台，如 **MineDOJO（MineDOJO）** [fan2022minedojo,]，它们通过模拟程序化生成的、类似沙盒的环境，进一步扩展了这些挑战。这些平台受《我的世界（Minecraft）》开放世界的启发，推动智能体参与资源收集、建造和生存等任务，要求其进行持续探索和适应性学习。在这样的环境中，智能体被激励去寻求新信息并调整其行为以完成给定任务。在此类环境中进行训练，可以帮助智能体学习跨广泛任务和地形的知识，使其能够在各种室外环境中有效运作。

**动态环境（Dynamic environments）** 标志着对传统静态模拟器的一次重大演进，其通过利用 **生成模型（Generative models）** 来创建灵活、实时的模拟。与需要手动调整的预定义环境不同，这些模型允许动态创建多种多样的场景，使 **智能体（Agents）** 能够体验多样化的第一人称视角。这种转变为智能体提供了更丰富、更多样的训练体验，从而提升了它们在复杂、不可预测的现实世界情境中的适应性和泛化能力。

一个代表性工作是 **UniSim（yanglearning）** [1]，它能够基于空间运动、文本指令和相机参数等输入条件，动态生成机器人操作视频序列。该系统利用来自 3D 仿真、真实世界机器人动作和互联网媒体的多模态数据，生成多样且逼真的环境，使智能体可以在其中练习物体操作和导航等任务。该方法的关键优势在于其灵活性，允许智能体适应各种场景，而不受静态物理环境的限制。

**Pandora（xiang2024pandora）** [2] 将动态环境生成的范围从 UniSim 中的机器人动作扩展到更广泛的领域，包括室内外场景中的人类和机器人动作。

另一项后续工作 **AVID（rigter2024avid）** [3] 在 UniSim 的基础上，通过对动作进行条件化处理并修改来自预训练 **扩散模型（Diffusion model）** 的噪声预测，来生成用于动态环境生成的动作驱动视觉序列。

除了 UniSim 基于视频扩散的框架， **EVA（chi2024eva）** [4] 引入了一个额外的 **视觉-语言模型（Vision-Language Model, VLM）** 用于具身视频预测，从而产生更一致的具身视频预测结果。

至于开放世界动态环境的生成， **Streetscapes（deng2024streetscapes）** [5] 采用 **自回归视频扩散模型（Autoregressive Video Diffusion Models）** 来模拟城市环境，智能体必须在其中应对诸如天气变化和交通等动态挑战。这些环境提供了持续连贯且灵活的城市设置，使智能体暴露于类似现实世界的可变性中。

动态环境的核心趋势是使用 **生成世界模型（Generative World Models）** 来提供可扩展、可适应的模拟。这种方法显著减少了环境设置所需的人工工作量，使智能体能够快速地在各种场景中进行训练。此外，对第一人称训练的专注密切模拟了现实世界的决策过程，增强了智能体适应不断变化情况的能力。这些进展对于开发支持智能体在复杂、动态场景中学习的具身环境至关重要。

鉴于上述发展，显而易见，作为具身环境（embodied environments）的 **世界模型（World Models）** 在模拟真实世界的环境变迁方面已取得显著进展。当前研究主要集中于开发室内、静态环境，同时值得注意的努力正扩展到大规模户外和动态仿真环境。一个前景广阔的方向是构建 **动态环境（dynamic environments）** ，它能够提供第一人称视角、以动作为条件的未来世界预测，使智能体（agents）能更好地适应未见过的条件。与此同时，近期在构建动态具身世界方面的进展强调了对 **物理约束（physical constraints）** 的整合。例如，Aether（team2025aether）通过将相机轨迹作为动作驱动的RGB-D视频生成，来增强几何知识学习。TesserAct（zhen2025tesseract）进一步将法线贴图（normal maps）作为视频生成的物理约束。Roboscape（shang2025roboscape）在视频生成过程中整合深度图（depth maps）和关键点动力学（keypoint dynamics），以学习并产生更逼真的运动和空间结构。此外，Deepverse（chen2025deepverse）提出将先前时间步的几何预测整合到以当前动作为条件的预测中。这些方法共同增强了动态世界的真实感和物理遵循性，从而为具身智能体（embodied agents）创造了更真实、更可靠的仿真环境。

## 5. 应用领域（Application Domains）

### 5.1. 游戏智能（Game Intelligence）

游戏环境是 **世界模型（World Model）** 研究的理想试验场，它们提供了受控但复杂的领域，要求对 **物理学（Physics）** 、 **因果关系（Causality）** 和 **交互动力学（Interactive Dynamics）** 有深刻的理解。与现实世界的应用不同，现实世界中的 **真实情况（Ground Truth）** 常常是模糊或难以获取的，而游戏则提供了定义明确的规则系统和清晰的动作-结果关系，这使得能够对世界模型的能力进行精确评估。

更重要的是，世界模型技术正在以前所未有的方式从根本上改变游戏开发和玩家体验。传统的游戏开发依赖于手动编码的规则、预先设计的资产和脚本化的交互，这些限制了创作的可能性并需要大量的开发资源。世界模型提供了一种向 **生成式游戏系统（Generative Game Systems）** 的范式转变，这些系统能够自主创建新内容、动态适应玩家行为，并实现那些通过传统编程方法以前无法实现的 **涌现式游戏体验（Emergent Gameplay Experiences）** 。

最近的发展展示了游戏应用中的三个关键能力维度：

**交互性（Interactivity）** 。对用户输入做出适当响应的能力是游戏世界模型（Gaming World Models）的基本要求。GameNGen 通过创建一个完全神经的游戏引擎来展示这种能力，该引擎能够实现与复杂环境的实时交互，以每秒 20 帧的速度运行，同时在长时间会话中保持稳定的游戏玩法 [1]。类似地，GameGen-X 引入了一个专门的模块来整合与游戏相关的多模态控制信号，首次在视频生成中统一了角色交互和场景内容控制 [2]。Matrix-Game 通过训练一个超过 170 亿参数的模型进一步推进了这一点，该模型能够通过细粒度的键盘和鼠标动作标注，精确控制角色动作和摄像机运动 [3]。

**一致性（Consistency）** 。在时间序列中保持连贯的游戏状态对生成模型构成了重大挑战。最近的研究解决了游戏中的数值一致性（确保游戏机制正确反映分数变化和定量元素）和空间一致性（防止突兀的场景转换）问题 [4]。MineWorld 通过视觉-动作自回归变换器（Visual-Action Autoregressive Transformers）来解决一致性问题，该变换器同时学习游戏状态和动作-状态关系的丰富表示 [5]。该模型的并行解码算法能够在保持长时间游戏序列时间连贯性的同时实现实时生成。WHAM（世界与人类动作模型，World and Human Action Model）进一步例证了这一进展，它能生成一致且多样的游戏序列，同时保持用户的修改——这些能力被认为对支持游戏开发中的创意实践至关重要 [6]。

**跨多样化环境的泛化能力（Generalization Across Diverse Environments）** 。适应不同游戏场景和环境的能力或许代表了最具挑战性的维度。GameFactory 通过场景可泛化的动作控制来解决这个问题，它利用来自预训练视频扩散模型（Video Diffusion Models）的开放域生成先验，创造出超越固定风格和场景的全新游戏 [7]。最近的工作进一步创建了一个“生成式无限游戏”，它超越了传统的有限、硬编码系统 [8]。该系统使用专门的蒸馏大型语言模型（Distilled Large Language Models, LLMs）进行动态游戏机制生成，并使用动态区域图像提示适配器（Dynamic Regional Image Prompt Adapters）进行一致的视觉生成，从而实现了可以从底层生成模型中自然涌现的开放式机制。虚拟环境中的探索驱动方法展示了通向泛化的另一条路径，其中探索智能体完全依赖世界模型的不确定性来提供多样化的训练数据，无需环境特定的奖励即可轻松适应新环境 [9]。

---

**参考文献**

1.  valevski2024diffusion
2.  che2024gamegen
3.  zhang2025matrix
4.  chen2025model
5.  guo2025mineworld
6.  kanervisto2025world
7.  yu2025gamefactory
8.  li2024unbounded
9.  savov2025exploration

### 5.2. 具身智能（Embodied Intelligence）

**具身智能（Embodied Intelligence）** 致力于创造能够有效感知、理解并与复杂物理世界互动的智能体。该领域的一个核心挑战是赋予机器人对其环境动态进行推理的能力，以支持鲁棒、实时的决策。 **世界模型（World models）** 已成为一种变革性的范式，直接应对这一需求，赋予机器人感知、预测和有效行动的关键能力。这一进展部分得益于 **神经架构（Neural architectures）** [1, 2] 和 **学习算法（Learning algorithms）** [3, 4] 的进步，它们使机器人能够构建捕捉外部世界关键方面的 **隐式表征（Implicit representations）** 。作为补充， **预测模型（Prediction models）** [5, 6] 提供了预测未来环境状态的能力，超越了静态抽象，以支持预见性和适应性行为。这些能力共同使得机器人直接从现实世界交互中学习变得越来越可行。在表 [4](#table-4) 中，我们总结了为机器人构建世界模型所涉及的核心学习任务，并根据上述三个主要视角进行了分类（典型示例如图 [S1](https://arxiv.org/html/2411.14499v4#A2.F1) 所示）。

#### 5.2.1. 学习隐式表征（Learning Implicit Representation）

传统的机器人任务（例如，物体抓取）通常在高度结构化的环境中执行，其中关键组件已被显式建模 [7, 8]，从而无需机器人独立学习或调整其对世界的理解。
然而，当机器人被部署到不熟悉的环境中，尤其是那些关键特征或动态未被显式建模的环境时，先前成功的任务可能会失败，因为机器人难以将这些未知特征进行泛化 [9, 10]。
因此，使机器人能够学习其环境的隐式表征，是实现智能的关键第一步。

**参考文献**

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In _Advances in neural information processing systems_ (pp. 5998-6008).
2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. In _Advances in neural information processing systems_ (Vol. 33, pp. 6840-6851).
3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. _arXiv preprint arXiv:1707.06347_.
4. Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2024). Direct preference optimization: Your language model is secretly a reward model. In _Advances in neural information processing systems_ (Vol. 36).
5. Finn, C., Goodfellow, I., & Levine, S. (2016). Unsupervised learning for physical interaction through video prediction. In _Advances in neural information processing systems_ (pp. 64-72).
6. Finn, C., & Levine, S. (2017). Deep visual foresight for planning robot motion. In _2017 IEEE International Conference on Robotics and Automation (ICRA)_ (pp. 2786-2793). IEEE.
7. Kleeberger, K., Bormann, R., Kraus, W., & Huber, M. F. (2020). A survey on learning-based robotic grasping. _Current Robotics Reports_, 1(4), 239-249.
8. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous localization and mapping: part I. _IEEE Robotics & Automation Magazine_, 13(2), 99-110.
9. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. _Nature_, 518(7540), 529-533.
10. Kahn, G., Villaflor, A., Ding, B., Abbeel, P., & Levine, S. (2017). Uncertainty-aware reinforcement learning for collision avoidance. _arXiv preprint arXiv:1702.01182_.

为了帮助机器人理解世界中的物体， **卷积神经网络（Convolutional Neural Networks, CNNs）** (lecun1998gradient,; krizhevsky2012imagenet,; girshick2014rich,) 和 **视觉变换器（Vision Transformers, ViT）** (dosovitskiy2020image,; wang2024repvit,) 等视觉模型将实体的视觉特征整合到表征中，使机器人能够识别任务所需的关键物体。
**RoboCraft** (shi2024robocraft,) 将视觉观察转换为粒子，并通过 **图神经网络（Graph Neural Network, GNN）** 捕捉底层系统的结构。
此外，还有其他针对物理三维空间感知的尝试。
**PointNet** (qi2017pointnet,; qian2022pointnext,) 首次使用非对称函数对非结构化的三维点云进行编码，捕捉环境的空间特征。
一项近期工作 (gornet2024automated,) 将沿局部探索路径获取的观测结果在其潜在空间内组装成物理空间的全局表征，使机器人能够追踪并接近特定目标。
**SpatialLM** (spatiallm,) 通过将原始三维点云处理成带有语义标签的结构化三维场景表征，进一步推进了这一方向，从而增强了机器人和自动驾驶中复杂任务的空间推理能力。
随着 **大型语言模型（Large Language Models, LLMs）** (touvron2023llama,; brown2020language,; du2022glam,) 语言理解能力的进步，一种使机器人捕捉任务意图的新范式应运而生：即以文本形式描述任务，然后通过 LLMs 获得文本表征 (mu2023clarifygpt,; gestrin2024nl2plan,; hua2024gensim2,; wang2023gensim,)。
**BC-Z** (jang2022bc,) 利用语言表征作为任务表征，从而提升了机器人的多任务性能。
**Text2Motion** (lin2023text2motion,) 则利用 LLM 将自然语言指令分解为任务级和运动级计划，以处理复杂的顺序操作任务。

<a id="table-4"></a>

**表 4. 构建机器人世界模型所涉及的核心学习任务（Table 4. Core learning tasks involved in constructing world models for robotics）**

|  列 1  |  列 2  |  列 3  |
| :----: | :----: | :----: |
| 数据 1 | 数据 2 | 数据 3 |
| 数据 4 | 数据 5 | 数据 6 |

表 1 | 表格描述。
| | 任务（Task） | 模型（Model） | 年份（Year） | 输入（Input） | 骨干网络（Backbone） |
| :--------------------------------------- | :----------------------------- | :-------------------------------- | :---------------- | :---------- | :---------- |
| | 任务（Task） | 模型（Model） | 年份（Year） | 输入（Input） | 骨干网络（Backbone） |
| | 任务（Task） | 模型（Model） | 年份（Year） | 输入（Input） | 骨干网络（Backbone） |
| | 任务（Task） | 模型（Model） | 年份（Year） | 输入（Input） | 骨干网络（Backbone） |
| | 任务（Task） | 模型（Model） | 年份（Year） | 输入（Input） | 骨干网络（Backbone） |
| | 任务（Task） | 模型（Model） | 年份（Year） | 输入（Input） | 骨干网络（Backbone） |
| 学习<br>内部<br>表征（Learning<br>Inner<br>Representation） | 视觉<br>表征（Visual<br>Representation） | CNN（lecun1998gradient，） | 1998 | 图像（Image） | CNN |
| 学习<br>内部<br>表征（Learning<br>Inner<br>Representation） | 视觉<br>表征（Visual<br>Representation） | CNN（lecun1998gradient，） | 1998 | 图像（Image） | CNN |
| 学习<br>内部<br>表征（Learning<br>Inner<br>Representation） | 视觉<br>表征（Visual<br>Representation） | CNN（lecun1998gradient，） | 1998 | 图像（Image） | CNN |
| 学习<br>内部<br>表征（Learning<br>Inner<br>Representation） | 视觉<br>表征（Visual<br>Representation） | CNN（lecun1998gradient，） | 1998 | 图像（Image） | CNN |
| 学习<br>内部<br>表征（Learning<br>Inner<br>Representation） | 视觉<br>表征（Visual<br>Representation） | CNN（lecun1998gradient，） | 1998 | 图像（Image） | CNN |
| 学习<br>内部<br>表征（Learning<br>Inner<br>Representation） | 视觉<br>表征（Visual<br>Representation） | CNN（lecun1998gradient，） | 1998 | 图像（Image） | CNN |
| ViT（dosovitskiy2020image，） | 2020 | 图像（Image） | Transformer | | |
| ViT（dosovitskiy2020image，） | 2020 | 图像（Image） | Transformer | | |
| ViT（dosovitskiy2020image，） | 2020 | 图像（Image） | Transformer | | |
| ViT（dosovitskiy2020image，） | 2020 | 图像（Image） | Transformer | | |
| RoboCraft（shi2024robocraft，） | 2024 | 图像（Image） | GNN | | |
| RoboCraft（shi2024robocraft，） | 2024 | 图像（Image） | GNN | | |
| RoboCraft（shi2024robocraft，） | 2024 | 图像（Image） | GNN | | |
| RoboCraft（shi2024robocraft，） | 2024 | 图像（Image） | GNN | | |
| 3D<br>表征（3D<br>Representation） | PointNet（qi2017pointnet，） | 2017 | 3D 点云（3D point clouds） | MLP | |
| 3D<br>表征（3D<br>Representation） | PointNet（qi2017pointnet，） | 2017 | 3D 点云（3D point clouds） | MLP | |
| 3D<br>表征（3D<br>Representation） | PointNet（qi2017pointnet，） | 2017 | 3D 点云（3D point clouds） | MLP | |
| 3D<br>表征（3D<br>Representation） | PointNet（qi2017pointnet，） | 2017 | 3D 点云（3D point clouds） | MLP | |

表 1 | 相关研究概览。
| 3D 表示（3D Representation） | 方法（Method） | 年份（Year） | 模态（Modality） | 模型（Model） | 备注（Notes） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3D 表示（3D Representation） | PointNet (qi2017pointnet) | 2017 | 3D 点云（3D point clouds） | MLP | |
| 预测编码（Predictive Coding） | Predictive Coding (gornet2024automated) | 2024 | 图像（Image） | ResNet | |
| 预测编码（Predictive Coding） | Predictive Coding (gornet2024automated) | 2024 | 图像（Image） | ResNet | |
| 预测编码（Predictive Coding） | Predictive Coding (gornet2024automated) | 2024 | 图像（Image） | ResNet | |
| 预测编码（Predictive Coding） | Predictive Coding (gornet2024automated) | 2024 | 图像（Image） | ResNet | |
| 空间语言模型（Spatial Language Model） | SpatialLM (spatiallm) | 2025 | 3D 点云（3D point clouds） | MLLM | |
| 空间语言模型（Spatial Language Model） | SpatialLM (spatiallm) | 2025 | 3D 点云（3D point clouds） | MLLM | |
| 空间语言模型（Spatial Language Model） | SpatialLM (spatiallm) | 2025 | 3D 点云（3D point clouds） | MLLM | |
| 空间语言模型（Spatial Language Model） | SpatialLM (spatiallm) | 2025 | 3D 点云（3D point clouds） | MLLM | |
| 任务表示（Task Representation） | BC-Z (jang2022bc) | 2022 | 文本与视频（Text & Video） | LLM & ResNet | |
| 任务表示（Task Representation） | BC-Z (jang2022bc) | 2022 | 文本与视频（Text & Video） | LLM & ResNet | |
| 任务表示（Task Representation） | BC-Z (jang2022bc) | 2022 | 文本与视频（Text & Video） | LLM & ResNet | |
| 任务表示（Task Representation） | BC-Z (jang2022bc) | 2022 | 文本与视频（Text & Video） | LLM & ResNet | |
| 任务表示（Task Representation） | BC-Z (jang2022bc) | 2022 | 文本与视频（Text & Video） | LLM & ResNet | |
| 文本到动作（Text to Motion） | Text2Motion (lin2023text2motion) | 2023 | 文本（Text） | LLM | |
| 文本到动作（Text to Motion） | Text2Motion (lin2023text2motion) | 2023 | 文本（Text） | LLM | |
| 文本到动作（Text to Motion） | Text2Motion (lin2023text2motion) | 2023 | 文本（Text） | LLM | |
| 文本到动作（Text to Motion） | Text2Motion (lin2023text2motion) | 2023 | 文本（Text） | LLM | |
| 生成式模拟（Generative Simulation） | Gensim (wang2023gensim) | 2023 | 文本（Text） | LLM | |
| 生成式模拟（Generative Simulation） | Gensim (wang2023gensim) | 2023 | 文本（Text） | LLM | |
| 生成式模拟（Generative Simulation） | Gensim (wang2023gensim) | 2023 | 文本（Text） | LLM | |
| 生成式模拟（Generative Simulation） | Gensim (wang2023gensim) | 2023 | 文本（Text） | LLM | |
| 预测未来环境（Predicting Future Environment） | 视频预测（Video Prediction） | UniPi (du2024learning) | 2024 | 视频（Video） | 扩散模型（Diffusion） |
| 预测未来环境（Predicting Future Environment） | 视频预测（Video Prediction） | UniPi (du2024learning) | 2024 | 视频（Video） | 扩散模型（Diffusion） |
| 预测未来环境（Predicting Future Environment） | 视频预测（Video Prediction） | UniPi (du2024learning) | 2024 | 视频（Video） | 扩散模型（Diffusion） |

| 任务（Task）                                  | 方法（Method）               | 年份（Year）                 | 模态（Modality）           | 架构（Architecture）  |
| :-------------------------------------------- | :--------------------------- | :--------------------------- | :------------------------- | :-------------------- | :-------------------- |
| 预测未来环境（Predicting Future Environment） | 视频预测（Video Prediction） | UniPi (du2024learning,)      | 2024                       | 视频（Video）         | 扩散模型（Diffusion） |
| 预测未来环境（Predicting Future Environment） | 视频预测（Video Prediction） | UniPi (du2024learning,)      | 2024                       | 视频（Video）         | 扩散模型（Diffusion） |
| 预测未来环境（Predicting Future Environment） | 视频预测（Video Prediction） | UniPi (du2024learning,)      | 2024                       | 视频（Video）         | 扩散模型（Diffusion） |
| 预测未来环境（Predicting Future Environment） | 视频预测（Video Prediction） | UniPi (du2024learning,)      | 2024                       | 视频（Video）         | 扩散模型（Diffusion） |
|                                               | VIPER (escontrela2024video,) | 2024                         | 视频（Video）              | Transformer           |
|                                               | VIPER (escontrela2024video,) | 2024                         | 视频（Video）              | Transformer           |
|                                               | VIPER (escontrela2024video,) | 2024                         | 视频（Video）              | Transformer           |
|                                               | VIPER (escontrela2024video,) | 2024                         | 视频（Video）              | Transformer           |
|                                               | GR-2 (cheang2024gr,)         | 2024                         | 文本与视频（Text & Video） | Transformer           |
|                                               | GR-2 (cheang2024gr,)         | 2024                         | 文本与视频（Text & Video） | Transformer           |
|                                               | GR-2 (cheang2024gr,)         | 2024                         | 文本与视频（Text & Video） | Transformer           |
|                                               | GR-2 (cheang2024gr,)         | 2024                         | 文本与视频（Text & Video） | Transformer           |
|                                               | IRASim (zhu2024irasim,)      | 2024                         | 轨迹（Trajectory）         | 扩散模型（Diffusion） |
|                                               | IRASim (zhu2024irasim,)      | 2024                         | 轨迹（Trajectory）         | 扩散模型（Diffusion） |
|                                               | IRASim (zhu2024irasim,)      | 2024                         | 轨迹（Trajectory）         | 扩散模型（Diffusion） |
|                                               | IRASim (zhu2024irasim,)      | 2024                         | 轨迹（Trajectory）         | 扩散模型（Diffusion） |
|                                               | IRASim (zhu2024irasim,)      | 2024                         | 轨迹（Trajectory）         | 扩散模型（Diffusion） |
|                                               |                              | VPP (hu2024video,)           | 2024                       | 文本（Text）          | 扩散模型（Diffusion） |
|                                               |                              | VPP (hu2024video,)           | 2024                       | 文本（Text）          | 扩散模型（Diffusion） |
|                                               |                              | VPP (hu2024video,)           | 2024                       | 文本（Text）          | 扩散模型（Diffusion） |
|                                               |                              | VPP (hu2024video,)           | 2024                       | 文本（Text）          | 扩散模型（Diffusion） |
|                                               |                              | VPP (hu2024video,)           | 2024                       | 文本（Text）          | 扩散模型（Diffusion） |
|                                               |                              | VPP (hu2024video,)           | 2024                       | 文本（Text）          | 扩散模型（Diffusion） |
|                                               |                              | DreamGen (jang2025dreamgen,) | 2025                       | 文本（Text）          | 扩散模型（Diffusion） |
|                                               |                              | DreamGen (jang2025dreamgen,) | 2025                       | 文本（Text）          | 扩散模型（Diffusion） |

| | | DreamGen (jang2025dreamgen,) | 2025 | Text | Diffusion |
| | | DreamGen (jang2025dreamgen,) | 2025 | Text | Diffusion |
| | | DreamGen (jang2025dreamgen,) | 2025 | Text | Diffusion |
| | | DreamGen (jang2025dreamgen,) | 2025 | Text | Diffusion |
| | | Roboscape (shang2025roboscape,) | 2025 | Trajectory | Transformer |
| | | Roboscape (shang2025roboscape,) | 2025 | Trajectory | Transformer |
| | | Roboscape (shang2025roboscape,) | 2025 | Trajectory | Transformer |
| | | Roboscape (shang2025roboscape,) | 2025 | Trajectory | Transformer |
| | | Roboscape (shang2025roboscape,) | 2025 | Trajectory | Transformer |
| | | Roboscape (shang2025roboscape,) | 2025 | Trajectory | Transformer |
| | | EVAC (jiang2025enerverse,) | 2025 | Trajectory | Diffusion |
| | | EVAC (jiang2025enerverse,) | 2025 | Trajectory | Diffusion |
| | | EVAC (jiang2025enerverse,) | 2025 | Trajectory | Diffusion |
| | | EVAC (jiang2025enerverse,) | 2025 | Trajectory | Diffusion |
| | | EVAC (jiang2025enerverse,) | 2025 | Trajectory | Diffusion |
| | | EVAC (jiang2025enerverse,) | 2025 | Trajectory | Diffusion |
| | | Genie Envisioner (liao2025genie,) | 2025 | Text | Diffusion |
| | | Genie Envisioner (liao2025genie,) | 2025 | Text | Diffusion |
| | | Genie Envisioner (liao2025genie,) | 2025 | Text | Diffusion |
| | | Genie Envisioner (liao2025genie,) | 2025 | Text | Diffusion |
| | | Genie Envisioner (liao2025genie,) | 2025 | Text | Diffusion |
| | | Genie Envisioner (liao2025genie,) | 2025 | Text | Diffusion |
| | | Vidar (feng2025generalist,) | 2025 | Text | Diffusion |
| | | Vidar (feng2025generalist,) | 2025 | Text | Diffusion |
| | | Vidar (feng2025generalist,) | 2025 | Text | Diffusion |

| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |
| :----------------- | :------------------------------ | :--- | :------------------------ | :------------------ | :-- |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | OpenEQA（majumdar2024openeqa,） | 2024 | 图像与文本（Image& Text） | 大型语言模型（LLM） |     |
| 评估（Evaluation） | Open                            |

| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |
| :----------------- | :----------------------------- | :--- | :------------------------ | :---------------------------------------- | :-- |
| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |
| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |
| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |

| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |
| :----------------- | :----------------------------- | :--- | :------------------------ | :---------------------------------------- | :-- |
| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |
| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |
| 评估（Evaluation） | OpenEQA (majumdar2024openeqa,) | 2024 | 图像与文本（Image& Text） | 大型语言模型（Large Language Model, LLM） |     |

#### 5.2.2. 环境未来状态预测（Predicting Future States of the Environment）

**世界模型（World models）** 处于机器人研究的前沿，主要推动着三个应用领域的进展： **合成数据生成（synthetic data generation）** 、通过想象未来进行 **动作引导（action guidance）** ，以及用于 **策略评估（policy evaluation）** 的环境模拟。

首先， **具身世界模型（embodied world models）** 能够合成高质量的机器人动作视频，以补充现实世界收集的数据，从而增强下游机器人策略模型的训练。例如，DreamGen [1] 提出了一种四阶段流程来生成 **神经轨迹（neural trajectories）** ，这是一种源自视频世界模型的合成机器人数据形式。这种方法显著提升了 **视觉-语言-动作（Vision-Language-Action, VLA）** 模型训练中的机器人操作成功率和泛化能力，特别是在接触密集型任务中。Roboscape [2] 在视频生成过程中整合了物理定律，从而生成了运动合理性和空间准确性更高的合成数据，当这些数据被纳入 VLA 训练时，带来了显著的性能提升。EVAC [3] 通过利用多样化的失败轨迹来扩展训练数据，从而改进了泛化能力；它采用了一种多级动作条件机制和光线图编码来进行动态多视角图像生成，通过增强人工收集的轨迹，有效地充当了数据引擎和评估器的双重角色。

其次， **具身世界模型（Embodied World Models）** 通过利用想象的未来观测来指导机器人动作生成。近期的一个关键见解是使用 **生成式视频模型（Generative Video Models）** ——特别是那些利用 **扩散模型（Diffusion Models）** [1, 2, 3, 4] 和 **Transformer架构（Transformer Architectures）** [5, 6] 的模型——直接从视觉数据中隐式学习环境动态。例如，UniPi [7] 明确地将动作预测构建为一个视频生成问题，通过将当前状态作为条件输入一个受限的扩散模型来可视化未来场景。类似地，VIPER [8] 采用一个预训练的 **自回归Transformer（Autoregressive Transformer）** 来指导机器人动作，有效地利用了从专家演示视频中学到的丰富表征。此外，像 GR-2 [9] 这样的模型受益于互联网视频的巨大规模来建立鲁棒的先验，随后在特定的机器人任务上进行微调，以生成准确的图像预测和动作轨迹。VPP [10] 通过基于文本指令微调一个视频生成模型来学习机器人动作，随后从一个以视觉表征为条件的 **逆动力学模型（Inverse Dynamics Model）** 中推导出动作。类似地，Genie Envisioner [11] 利用一个预训练的具身视频生成基础模型，连接一个轻量级的 **并行流匹配动作模型（Parallel Flow-Matching Action Model）** ，该模型将语言条件的视觉潜在特征转换为细粒度、低延迟的运动指令。Vidar [12] 提出了一个用于机器人动作预测的两阶段框架，结合了大规模、基于扩散的视频预训练和一个新颖的 **掩码逆动力学模型（Masked Inverse Dynamics Model）** 。与之不同的是，V-JEPA 2 [13] 在潜在空间中建模世界状态转换，并通过 **模型预测控制（Model Predictive Control, MPC）** 执行动作规划，这涉及对可能的动作轨迹进行广泛采样，并基于能量优化选择最优轨迹。

第三，具身世界模型可以作为 **策略评估（Policy Evaluation）** 的环境模拟器。IRASim [14] 和 Roboscape [15] 都利用世界模型进行从轨迹到视频的生成任务，从一个给定的初始帧开始。它们所展示的在世界模型中和在真实环境中对策略模型评估结果之间的高度相关性表明，学习到的世界模型准确地捕捉了世界转换动态。GE-Sim [11] 也通过设计策略与世界模型之间的 **闭环交互（Closed-loop Interaction）** 来例证这一点，从而实现了可扩展且灵活的模拟，而无需手动进行环境建模。

[1] esser2023structure,
[2] chi2023diffusion,
[3] black2023zero,
[4] helearning,
[5] yu2023magvit,
[6] yan2021videogpt,
[7] du2024learning,
[8] escontrela2024video,
[9] cheang2024gr,
[10] hu2024video,
[11] liao2025genie,
[12] feng2025generalist,
[13] assran2025v,
[14] zhu2024irasim,
[15] shang2025roboscape

总而言之，这些方法展示了 **生成式、以视觉为中心的世界建模（Generative, vision-centric world modeling）** 作为 **预测性机器人控制（Anticipatory robotic control）** 与仿真基础的巨大潜力。这显著增强了机器人对未来状态进行推理并提升长期任务性能的能力。

#### 5.2.3. 从仿真到现实世界（From Simulation to Real World）

**深度强化学习（Deep Reinforcement Learning, DRL）** 已在机器人学领域展现出卓越的能力，使其能够在复杂任务中实现自主操作，例如 **稳定运动（Stable locomotion）** [smith2022walk; kumar2021rma]、 **精确物体操控（Precise object manipulation）** [yu2022se; dogar2019multi] 以及像系鞋带这样的精细活动 [aldaco2024aloha]。然而，其实际应用仍因 **样本效率（Sample efficiency）** 低下而受到极大限制。例如，在现实世界中训练一个机器人解魔方可能需要数万模拟年 [akkaya2019solving]。因此，大多数机器人训练都在仿真环境中进行，并利用 **分布式训练技术（Distributed training techniques）** 来提高效率 [rudin2022learning; ha2020learning]。不幸的是，由于仿真条件与现实世界条件之间存在差异，在仿真中训练出的 **策略（Policies）** 在直接迁移到物理机器人上时常常失败，尤其是在复杂或新颖的环境中。

近期一个关键见解是， **世界模型（World models）** 可以通过学习现实世界动态的 **泛化表征（Generalized representations）** ，有效弥合这种 **仿真与现实间的鸿沟（Simulation-to-reality gap）** 。例如，NeBula [agha2021nebula] 构建了一个 **结构化信念空间（Structured belief space）** ，使其能够在不同的机器人形态和非结构化环境中进行推理和快速适应。DayDreamer [wu2023daydreamer] 进一步展示了泛化世界模型的能力，使机器人能够在数小时内直接在现实环境中学习运动，显著减少了对大量仿真的依赖。此外，SWIM [mendonca2023structured] 凸显了 **基于人类视频的学习（Human-video-based learning）** 结合极少量现实世界 **微调（Fine-tuning）** 的强大之处，能够在少于 30 分钟的交互时间内实现 **任务泛化（Task generalization）** 。这些例子表明，通过构建鲁棒的、面向现实世界的内部表征，世界模型极大地缩小了仿真与现实之间的差距，促进了机器人学中的快速适应与泛化。

### 5.3. 城市智能（Urban Intelligence）

#### 5.3.1. 自动驾驶（Autonomous Driving）

近年来，随着基于视觉的 **生成模型（Generative models）** [ho2020denoising; song2020score; videoworldsimulators2024] 和 **多模态大语言模型（Multimodal Large Language Models, MLLMs）** [liu2023llava; achiam2023gpt4] 的快速发展， **世界模型（World models）** 在自动驾驶领域引起了越来越多的关注。

现代自动驾驶流程通常分为四个关键组成部分： **感知（Perception）** 、 **预测（Prediction）** 、 **规划（Planning）** 和 **控制（Control）** 。其中，感知和预测阶段对应于驾驶场景理解——即学习车辆外部环境的 **隐式表示（Implicit representation）** 。与此同时，最近的综述 [guan2024world] 强调了 **端到端世界模拟器（End-to-end world simulators）** 的出现，它们学习基于多模态输入（如图像、点云、轨迹和语言）模拟真实的驾驶环境，然后生成未来状态以支持规划和决策等下游任务。这两种视角与我们之前对世界模型的分类非常吻合，下文我们将据此详细阐述它们在自动驾驶领域内的应用与发展。

<a id="figure-6"></a>

![autonomous](images/autonomous.png)

> 图 6. 世界模型在自动驾驶中的应用。

**学习隐式表示（Learning Implicit Representations）** 。
自动驾驶车辆通常利用摄像头、雷达和 **激光雷达（LiDAR）** 来感知现实世界，通过图像、视频数据和 **点云数据（Point cloud data）** 收集信息。在早期的决策范式 [chen2019modelfreedeepreinforcementlearning; Saxena_2020] 中，模型通常以感知数据作为输入，直接输出自动驾驶车辆的运动规划结果。相反，当人类驾驶车辆时，他们通常会观察并预测其他交通参与者的当前和未来状态，以确定自己的驾驶策略 [survey_trajpre_2022]。因此，通过感知数据学习世界的隐式表示并预测周围环境的未来状态，是提升自动驾驶车辆决策可靠性的关键一步。我们将此过程视为自动驾驶车辆如何在 **潜在空间（Latent space）** 中学习世界模型的具体体现。

如图 [6](#figure-6) 左半部分所示，在多模态大模型和端到端自动驾驶技术 [hu2023planningorientedautonomousdriving] 出现之前，自动驾驶车辆的感知和预测任务通常被分配给不同的模块，每个模块在其各自的任务和数据集上进行训练。感知模块处理来自图像、点云等来源的数据，以完成诸如 **目标检测（Object detection）** 和 **地图分割（Map segmentation）** 等任务，将感知到的世界投影到一个抽象的几何空间中。此外，预测模块通常在这些几何空间内运行，以预测周围环境的未来状态，包括交通参与者的轨迹和运动。

<a id="table-5"></a>

**表 5. 场景理解与世界模拟现有工作的比较。**

**表 5. 场景理解与世界模拟现有工作的比较。**

|                                                        | 任务（Task）       | 工作（Work）                      | 年份（Year）                                      | 数据模态（Data Modality）                      | 技术（Technique）                                 | 任务描述（Task Description） |
| ------------------------------------------------------ | ------------------ | --------------------------------- | ------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------- | ---------------------------- |
|                                                        | 任务（Task）       | 工作（Work）                      | 年份（Year）                                      | 数据模态（Data Modality）                      | 技术（Technique）                                 | 任务描述（Task Description） |
|                                                        | 任务（Task）       | 工作（Work）                      | 年份（Year）                                      | 数据模态（Data Modality）                      | 技术（Technique）                                 | 任务描述（Task Description） |
|                                                        | 任务（Task）       | 工作（Work）                      | 年份（Year）                                      | 数据模态（Data Modality）                      | 技术（Technique）                                 | 任务描述（Task Description） |
|                                                        | 任务（Task）       | 工作（Work）                      | 年份（Year）                                      | 数据模态（Data Modality）                      | 技术（Technique）                                 | 任务描述（Task Description） |
|                                                        | 任务（Task）       | 工作（Work）                      | 年份（Year）                                      | 数据模态（Data Modality）                      | 技术（Technique）                                 | 任务描述（Task Description） |
|                                                        | 任务（Task）       | 工作（Work）                      | 年份（Year）                                      | 数据模态（Data Modality）                      | 技术（Technique）                                 | 任务描述（Task Description） |
| 驾驶场景理解（Driving Scene<br>Understanding）         | 感知（Perception） | Faster R-CNN (NIPS2015_14bfa6bb,) | 2015                                              | 摄像头（Camera）                               | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection） |
| 驾驶场景理解（Driving Scene<br>Understanding）         | 感知（Perception） | Faster R-CNN (NIPS2015_14bfa6bb,) | 2015                                              | 摄像头（Camera）                               | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection） |
| 驾驶场景理解（Driving Scene<br>Understanding）         | 感知（Perception） | Faster R-CNN (NIPS2015_14bfa6bb,) | 2015                                              | 摄像头（Camera）                               | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection） |
| 驾驶场景理解（Driving Scene<br>Understanding）         | 感知（Perception） | Faster R-CNN (NIPS2015_14bfa6bb,) | 2015                                              | 摄像头（Camera）                               | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection） |
| 驾驶场景理解（Driving Scene<br>Understanding）         | 感知（Perception） | Faster R-CNN (NIPS2015_14bfa6bb,) | 2015                                              | 摄像头（Camera）                               | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection） |
| 驾驶场景理解（Driving Scene<br>Understanding）         | 感知（Perception） | Faster R-CNN (NIPS2015_14bfa6bb,) | 2015                                              | 摄像头（Camera）                               | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection） |
| 驾驶场景理解（Driving Scene<br>Understanding）         | 感知（Perception） | Faster R-CNN (NIPS2015_14bfa6bb,) | 2015                                              | 摄像头（Camera）                               | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection） |
| PointNet (qi2017pointnetdeephierarchicalfeature,)      | 2017               | 激光雷达（Lidar）                 | 多层感知机（Multilayer Perceptron, MLP）          | 三维分类（3D Classification）                  |                                                   |                              |
| PointNet (qi2017pointnetdeephierarchicalfeature,)      | 2017               | 激光雷达（Lidar）                 | 多层感知机（Multilayer Perceptron, MLP）          | 三维分类（3D Classification）                  |                                                   |                              |
| PointNet (qi2017pointnetdeephierarchicalfeature,)      | 2017               | 激光雷达（Lidar）                 | 多层感知机（Multilayer Perceptron, MLP）          | 三维分类（3D Classification）                  |                                                   |                              |
| PointNet (qi2017pointnetdeephierarchicalfeature,)      | 2017               | 激光雷达（Lidar）                 | 多层感知机（Multilayer Perceptron, MLP）          | 三维分类（3D Classification）                  |                                                   |                              |
| PointNet (qi2017pointnetdeephierarchicalfeature,)      | 2017               | 激光雷达（Lidar）                 | 多层感知机（Multilayer Perceptron, MLP）          | 三维分类（3D Classification）                  |                                                   |                              |
| MultiNet (teichmann2018multinetrealtimejointsemantic,) | 2018               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 语义分割（Semantic Segmentation）              |                                                   |                              |
| MultiNet (teichmann2018multinetrealtimejointsemantic,) | 2018               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 语义分割（Semantic Segmentation）              |                                                   |                              |
| MultiNet (teichmann2018multinetrealtimejointsemantic,) | 2018               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 语义分割（Semantic Segmentation）              |                                                   |                              |
| MultiNet (teichmann2018multinetrealtimejointsemantic,) | 2018               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 语义分割（Semantic Segmentation）              |                                                   |                              |
| MultiNet (teichmann2018multinetrealtimejointsemantic,) | 2018               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 语义分割（Semantic Segmentation）              |                                                   |                              |
| OmniDet (kumar2023omnidetsurroundviewcameras,)         | 2021               | 摄像头（Camera）                  | 卷积神经网络与注意力机制（CNN & Attention）       | 多任务视觉感知（Multi-task Visual Perception） |                                                   |                              |
| OmniDet (kumar2023omnidetsurroundviewcameras,)         | 2021               | 摄像头（Camera）                  | 卷积神经网络与注意力机制（CNN & Attention）       | 多任务视觉感知（Multi-task Visual Perception） |                                                   |                              |
| OmniDet (kumar2023omnidetsurroundviewcameras,)         | 2021               | 摄像头（Camera）                  | 卷积神经网络与注意力机制（CNN & Attention）       | 多任务视觉感知（Multi-task Visual Perception） |                                                   |                              |
| OmniDet (kumar2023omnidetsurroundviewcameras,)         | 2021               | 摄像头（Camera）                  | 卷积神经网络与注意力机制（CNN & Attention）       | 多任务视觉感知（Multi-task Visual Perception） |                                                   |                              |
| OmniDet (kumar2023omnidetsurroundviewcameras,)         | 2021               | 摄像头（Camera）                  | 卷积神经网络与注意力机制（CNN & Attention）       | 多任务视觉感知（Multi-task Visual Perception） |                                                   |                              |
| YOLOP (yolop_2022,)                                    | 2022               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection）                   |                                                   |                              |
| YOLOP (yolop_2022,)                                    | 2022               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection）                   |                                                   |                              |
| YOLOP (yolop_2022,)                                    | 2022               | 摄像头（Camera）                  | 卷积神经网络（Convolutional Neural Network, CNN） | 目标检测（Object Detection                     |

| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| :------------------------------------------------------------------ | :--- | :-------------- | :-------- | :-------------------- | :-- | :-- |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |
| Jiang et al. (jiang2023motiondiffusercontrollablemultiagentmotion,) | 2023 | Geometric Space | Diffusion | Trajectory Prediction |     |     |

**感知数据处理（Perceptual Data Processing）** 与深度学习技术的演进紧密相连，如表 [5](#table-5) 所示。2017 年提出的 **PointNet（PointNet）** [1] 首次采用深度学习方法处理点云数据。随着卷积神经网络（Convolutional Neural Networks, CNNs）的发展，基于图像数据的感知技术应运而生，以 **YOLOP（YOLOP）** [2] 和 **MultiNet（MultiNet）** [3] 为代表，并在驾驶场景理解任务中表现出色 [4-7]。近年来， **Transformer 架构（Transformer architecture）** 在自然语言处理领域崭露头角，该技术也被应用于图像数据理解。 **BEVFormer（BEVFormer）** [8] 利用注意力机制（Attention Mechanism）整合来自多个相机角度的图像，构建鸟瞰视角的抽象几何空间，并在包括目标检测在内的多项任务中取得了最先进的结果。此外， **TransFusion（TransFusion）** [9] 通过跨注意力方法融合激光雷达（LiDAR）与相机数据，从而提升了感知精度。基于感知结果，一系列技术，如 **循环神经网络（Recurrent Neural Networks, RNNs）** [10-12]、 **卷积神经网络（Convolutional Neural Networks, CNNs）** [13-15] 和 **Transformer（Transformer）** [16-19]，已被用于编码历史场景信息并预测交通参与者的未来行为。

随着近年来 **多模态大语言模型（Multimodal Large Language Models, MLLMs）** 的出现与快速发展，许多研究致力于将这些模型的通用场景理解能力应用于自动驾驶领域。 **TOKEN（TOKEN）** [20] 将整个交通场景 **标记化（Tokenize）** 为对象级知识，利用语言模型的推理能力处理长尾预测与规划问题； **OmniDrive（OmniDrive）** [6] 设置了基于大语言模型（Large Language Model, LLM）的智能体，并通过视觉问答覆盖了包括场景描述、反事实推理和决策制定在内的多项任务。

**世界模拟器（World Simulators）**
如表 [5](#table-5) 所示，在 **多模态大模型（Multimodal Large Models）** 和基于视觉的生成模型出现之前，交通场景模拟通常在几何空间中进行。这些模拟所依赖的场景数据通常由自动驾驶车辆的感知模块采集或人工构建。这些模拟以几何轨迹的形式表示场景的未来状态 [1, 2, 3, 4]，需要进一步的建模和渲染才能产生适合车辆感知的输出。多个模块的级联常常导致信息丢失，并增加了模拟的复杂性，使得场景控制更具挑战性。此外，逼真的场景渲染通常需要大量的计算资源，这限制了虚拟交通场景生成的效率。

使用基于 **扩散模型（Diffusion models）** 的视频生成模型作为 **世界模型（World Model）** ，部分解决了上述问题。通过在 **大规模交通场景数据集（Large-scale Traffic Scenario Datasets）** 上进行训练，扩散模型可以直接生成与现实高度相似的 **相机感知数据（Camera Perception Data）** 。此外，扩散模型固有的可控性，结合如 CLIP [5] 等 **文本-图像对齐方法（Text-Image Alignment Methods）** ，使用户能够以直接的方式对场景生成施加控制。GAIA-1 [6] 和 DriveDreamer 系列 [7, 8] 是首批采用此方法构建世界模型的代表。在此基础上，Drive-WM [9] 为规划任务引入了 **闭环控制（Closed-loop Control）** ，而 Vista [10] 则专注于提高生成结果的分辨率和延长预测时长。除了在视频空间中预测未来状态的方法外，还有许多其他工作探索了不同形式的车辆感知数据。OccWorld [11] 和 OccSora [12] 通过预测 **三维占据栅格（3D Occupancy Grids）** 来预测世界的未来状态，而 Copilot4D [13] 则通过预测 **雷达点云数据（Radar Point Cloud Data）** 的变化来构建世界模型。与视频数据相比，这些类型的特征能更好地反映交通场景的空间特性。

<a id="table-6"></a>

> 表 6. 世界模型在自主物流与城市分析中的应用

**世界模拟器（World Simulators）**
如表 [5](#table-5) 所示，在 **多模态大模型（Multimodal Large Models）** 和基于视觉的生成模型出现之前，交通场景模拟通常在几何空间中进行。这些模拟所依赖的场景数据通常由自动驾驶车辆的感知模块采集或人工构建。这些模拟以几何轨迹的形式表示场景的未来状态 [1, 2, 3, 4]，需要进一步的建模和渲染才能产生适合车辆感知的输出。多个模块的级联常常导致信息丢失，并增加了模拟的复杂性，使得场景控制更具挑战性。此外，逼真的场景渲染通常需要大量的计算资源，这限制了虚拟交通场景生成的效率。

使用基于 **扩散模型（Diffusion models）** 的视频生成模型作为 **世界模型（World Model）** ，部分解决了上述问题。通过在 **大规模交通场景数据集（Large-scale Traffic Scenario Datasets）** 上进行训练，扩散模型可以直接生成与现实高度相似的 **相机感知数据（Camera Perception Data）** 。此外，扩散模型固有的可控性，结合如 CLIP [5] 等 **文本-图像对齐方法（Text-Image Alignment Methods）** ，使用户能够以直接的方式对场景生成施加控制。GAIA-1 [6] 和 DriveDreamer 系列 [7, 8] 是首批采用此方法构建世界模型的代表。在此基础上，Drive-WM [9] 为规划任务引入了 **闭环控制（Closed-loop Control）** ，而 Vista [10] 则专注于提高生成结果的分辨率和延长预测时长。除了在视频空间中预测未来状态的方法外，还有许多其他工作探索了不同形式的车辆感知数据。OccWorld [11] 和 OccSora [12] 通过预测 **三维占据栅格（3D Occupancy Grids）** 来预测世界的未来状态，而 Copilot4D [13] 则通过预测 **雷达点云数据（Radar Point Cloud Data）** 的变化来构建世界模型。与视频数据相比，这些类型的特征能更好地反映交通场景的空间特性。

<a id="table-6"></a>

> 表 6. 世界模型在自主物流与城市分析中的应用

| 类别（Category）                       | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| -------------------------------------- | ---------------------------- | ------------------------ | ------------ | ------------- |
| 类别（Category）                       | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 类别（Category）                       | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 类别（Category）                       | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 类别（Category）                       | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 自主物流（Autonomous Logistics）       | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）       | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）       | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）       | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）       | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| URBAN-SIM (wu2025towards,)             | 2025                         | CVPR                     |              |               |
| URBAN-SIM (wu2025towards,)             | 2025                         | CVPR                     |              |               |
| URBAN-SIM (wu2025towards,)             | 2025                         | CVPR                     |              |               |
| Vid2Sim (xie2025vid2sim,)              | 2025                         | CVPR                     |              |               |
| Vid2Sim (xie2025vid2sim,)              | 2025                         | CVPR                     |              |               |
| Vid2Sim (xie2025vid2sim,)              | 2025                         | CVPR                     |              |               |
| CityWalker (liu2025citywalker,)        | 2025                         | CVPR                     |              |               |
| CityWalker (liu2025citywalker,)        | 2025                         | CVPR                     |              |               |
| CityWalker (liu2025citywalker,)        | 2025                         | CVPR                     |              |               |
| 航空（Aerial）                         | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| 航空（Aerial）                         | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| 航空（Aerial）                         | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| 航空（Aerial）                         | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| CityNavAgent (zhang2025citynavagent,)  | 2025                         | ACL                      |              |               |
| CityNavAgent (zhang2025citynavagent,)  | 2025                         | ACL                      |              |               |
| CityNavAgent (zhang2025citynavagent,)  | 2025                         | ACL                      |              |               |
| CityEQA (zhao2025cityeqa,)             | 2025                         | EMNLP                    |              |               |
| CityEQA (zhao2025cityeqa,)             | 2025                         | EMNLP                    |              |               |
| CityEQA (zhao2025cityeqa,)             | 2025                         | EMNLP                    |              |               |
| UrbanVideo-Bench (zhao2025urbanvideo,) | 2025                         | ACL                      |              |               |
| UrbanVideo-Bench (zhao2025urbanvideo,) | 2025                         | ACL                      |              |               |
| UrbanVideo-Bench (zhao2025urbanvideo,) | 2025                         | ACL                      |              |               |

| 类别（Category）                      | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| ------------------------------------- | ---------------------------- | ------------------------ | ------------ | ------------- |
| 类别（Category）                      | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 类别（Category）                      | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 类别（Category）                      | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 类别（Category）                      | 子类别（Sub-category）       | 论文（Paper）            | 年份（Year） | 会议（Venue） |
| 自主物流（Autonomous Logistics）      | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）      | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）      | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）      | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| 自主物流（Autonomous Logistics）      | 微出行（Micromobility）      | NWM (bar2025navigation,) | 2025         | CVPR          |
| URBAN-SIM (wu2025towards,)            | 2025                         | CVPR                     |              |               |
| URBAN-SIM (wu2025towards,)            | 2025                         | CVPR                     |              |               |
| URBAN-SIM (wu2025towards,)            | 2025                         | CVPR                     |              |               |
| Vid2Sim (xie2025vid2sim,)             | 2025                         | CVPR                     |              |               |
| Vid2Sim (xie2025vid2sim,)             | 2025                         | CVPR                     |              |               |
| Vid2Sim (xie2025vid2sim,)             | 2025                         | CVPR                     |              |               |
| CityWalker (liu2025citywalker,)       | 2025                         | CVPR                     |              |               |
| CityWalker (liu2025citywalker,)       | 2025                         | CVPR                     |              |               |
| CityWalker (liu2025citywalker,)       | 2025                         | CVPR                     |              |               |
| 航空（Aerial）                        | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| 航空（Aerial）                        | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| 航空（Aerial）                        | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| 航空（Aerial）                        | AirScape (zhao2025airscape,) | 2025                     | ACM MM       |               |
| CityNavAgent (zhang2025citynavagent,) | 2025                         | ACL                      |              |               |
| CityNavAgent (zhang2025citynavagent,) | 2025                         | ACL                      |              |               |
| CityNavAgent (zhang2025citynavagent,) | 2025                         | ACL                      |              |               |
| CityEQA (zhao2025cityeqa,)            | 2025                         | EMNLP                    |              |               |
| CityEQA (zhao2025cityeqa,)            | 2025                         | EMNLP                    |              |               |
| CityEQA (zhao2025cityeqa,)            | 2025                         | EM                       |

表 1 | 相关研究总结。
| 领域（Domain） | 类别（Category） | 模型/方法（Model/Method） | 年份（Year） | 会议/期刊（Venue） |
| :--- | :--- | :--- | :--- | :--- |
| 城市分析（Urban Analytics） | 知识（Knowledge） | CityGPT（feng2024citygpt，） | 2025 | KDD |
| 城市分析（Urban Analytics） | 知识（Knowledge） | CityGPT（feng2024citygpt，） | 2025 | KDD |
| 城市分析（Urban Analytics） | 知识（Knowledge） | CityGPT（feng2024citygpt，） | 2025 | KDD |
| 城市分析（Urban Analytics） | 知识（Knowledge） | CityGPT（feng2024citygpt，） | 2025 | KDD |
| 城市分析（Urban Analytics） | 知识（Knowledge） | CityGPT（feng2024citygpt，） | 2025 | KDD |
| 城市分析（Urban Analytics） | 知识（Knowledge） | UrbanLLaVA（feng2025urbanllava，） | 2025 | ICCV |
| 城市分析（Urban Analytics） | 知识（Knowledge） | UrbanLLaVA（feng2025urbanllava，） | 2025 | ICCV |
| 城市分析（Urban Analytics） | 知识（Knowledge） | UrbanLLaVA（feng2025urbanllava，） | 2025 | ICCV |
| 城市分析（Urban Analytics） | 知识（Knowledge） | GeoLLM（manvi2023geollm，） | 2024 | ICLR |
| 城市分析（Urban Analytics） | 知识（Knowledge） | GeoLLM（manvi2023geollm，） | 2024 | ICLR |
| 城市分析（Urban Analytics） | 知识（Knowledge） | GeoLLM（manvi2023geollm，） | 2024 | ICLR |
| 预测（Prediction） | 知识（Knowledge） | GPS-to-Image（feng2025gps，） | 2025 | CVPR |
| 预测（Prediction） | 知识（Knowledge） | GPS-to-Image（feng2025gps，） | 2025 | CVPR |
| 预测（Prediction） | 知识（Knowledge） | GPS-to-Image（feng2025gps，） | 2025 | CVPR |
| 预测（Prediction） | 知识（Knowledge） | GPS-to-Image（feng2025gps，） | 2025 | CVPR |
| 预测（Prediction） | 知识（Knowledge） | AI’s Blind Spots（beneduce2025ai，） | 2025 | arXiv |
| 预测（Prediction） | 知识（Knowledge） | AI’s Blind Spots（beneduce2025ai，） | 2025 | arXiv |
| 预测（Prediction） | 知识（Knowledge） | AI’s Blind Spots（beneduce2025ai，） | 2025 | arXiv |
| 理解（Understanding） | 知识（Knowledge） | AgentMove（feng2025agentmove，） | 2025 | NAACL |
| 理解（Understanding） | 知识（Knowledge） | AgentMove（feng2025agentmove，） | 2025 | NAACL |
| 理解（Understanding） | 知识（Knowledge） | AgentMove（feng2025agentmove，） | 2025 | NAACL |
| 理解（Understanding） | 知识（Knowledge） | AgentMove（feng2025agentmove，） | 2025 | NAACL |
| 理解（Understanding） | 知识（Knowledge） | CAMS（du2025cams，） | 2025 | arXiv |
| 理解（Understanding） | 知识（Knowledge） | CAMS（du2025cams，） | 2025 | arXiv |
| 理解（Understanding） | 知识（Knowledge） | CAMS（du2025cams，） | 2025 | arXiv |
| 理解（Understanding） | 知识（Knowledge） | PIGEON（lan2025open，） | 2025 | ACL |
| 理解（Understanding） | 知识（Knowledge） | PIGEON（lan2025open，） | 2025 | ACL |
| 理解（Understanding） | 知识（Knowledge） | PIGEON（lan2025open，） | 2025 | ACL |

#### 5.3.2. Autonomous Logistics （自主物流）

本节介绍 **世界模型（World Models）** 在城市场景自主物流中的应用，主要聚焦于两个领域：1) **微型移动物流车辆（Miniature Mobile Logistics Vehicles）** 和 2) **低空飞行器（Low-altitude Aerial Vehicles）** 。在这两个领域的研究中，我们将从 **理解（Understanding）** 和 **预测（Prediction）** 两个角度介绍具体工作，如表 [6](#table-6) 所示。

在 **微型移动物流车辆（Miniature Mobile Logistics Vehicles）** 的背景下，这代表了 **自动驾驶（Autonomous Driving）** 向 **具身智能（Embodied Intelligence）** 场景的延伸，需要应对更复杂的周围环境和人机交互。其核心任务是 **导航（Navigation）** ，这涉及理解周围世界，以实现更高效、更安全的移动与交互。在理解方面，诸如 Vid2Sim (xie2025vid2sim) 和 CityWalker (liu2025citywalker) 等研究，利用互联网上大量的视频来理解运动中的多样化环境与交互场景。这种理解随后被用于训练机器人的导航策略，旨在实现强大的 **泛化能力（Generalization）** 和 **可控性（Controllability）** 。在预测未来方面，存在两种典型范式。第一种涉及构建逼真的 **物理模拟器（Physical Simulator）** (dosovitskiy2017carla; wu2025towards) 以生成多样化场景。通过在此虚拟环境中的大规模训练，机器人的能力得到增强，随后应用于现实世界。另一种范式则基于源自 **视频生成模型（Video Generation Models）** 的 **交互式可控世界模型（Interactive and Controllable World Model）** (bar2025navigation)。该模型根据机器人的动作（具体而言是其 **轨迹（Trajectory）** ）生成其可能遇到的潜在未来场景，从而提升其在各种场景下的泛化导航能力。

关于 **低空飞行器（Low-altitude Aerial Vehicles）** ，当前绝大多数工作集中于 **理解（Understanding）** 方面的应用，特别是在 **场景理解（Scene Comprehension）** 与 **导航（Navigation）** 方面 [zhao2025cityeqa,; zhao2025urbanvideo,; zhang2025citynavagent,]，而 **预测（Prediction）** 方面的应用 [zhao2025airscape,] 仍处于起步阶段。在理解领域，一个典型的范式涉及从图像 [zhao2025cityeqa,] 或视频 [zhao2025urbanvideo,] 中理解当前城市场景及其关键要素。这为后续任务（如导航 [zhang2025citynavagent,]）提供了充分的支持。理解多样且复杂的城市场景依赖于对城市环境的强大先验知识，尤其是源自 **大型语言模型（Large Language Models, LLMs）** 的丰富世界知识与常识 [achiam2023gpt4,]。在 **生成（Generation）** 领域， **AirScape** [zhao2025airscape,] 是首个针对低空飞行器的 **世界模型（World Model）** 。它能够基于飞行器的动作预测未来场景，同时保持物理与时空一致性。这为未来低空飞行器的高效训练与任务解决提供了新的途径与环境。

#### 5.3.3. 城市分析（Urban Analytics）

在本节中，我们主要介绍世界模型在城市分析中的应用。基于第 3.2 节中提到的大型语言模型已习得世界性地理知识的前提 [feng2024citygpt,; manvi2023geollm,]，我们将从理解和预测两个角度介绍相关工作。相关工作总结于表 [6](#table-6)。

在理解方面，一方面，关于环境，诸如 **UrbanLLaVA** [feng2025urbanllava,] 等 **多模态大型语言模型（Multimodal Large Language Models）** 利用模型内嵌的丰富世界知识，执行城市场景识别与理解 [feng2024citybench,] 等可泛化的任务。另一方面，关于环境中的人类行为， **AgentMove** [feng2025agentmove,] 和 **CAMS** [du2025cams,] 利用模型内现有的或专门增强的城市地理空间知识 [feng2024citygpt,] 来建模人类移动模式。同时， **PIGEON** [lan2025benchmarking,] 借鉴大型语言模型的常识知识来理解人类的日常需求及其对应行为，从而即使对于不常见的场景也能实现准确的理解与预测。

在预测方面， **GPS-to-Image** [feng2025gps,] 尝试使用 GPS 信号来控制生成图像的风格与场景，证明了利用模型把握地理位置与城市景观之间关系的可行性。然而，研究人员 [beneduce2025ai,] 进一步发现，现有的图像生成模型在准确区分不同地理位置的文化风格与景观特征方面仍有很大的改进空间。

总体而言， **世界模型（World Models）** 在城市分析中的应用仍然相对有限，这表明其未来应用具有巨大潜力。

### 5.4. 社会智能（Societal Intelligence）

**社会智能（Societal Intelligence）** 是一个社会感知其环境、推理可能的未来并协调行动以实现共同目标的集体能力 [1, 2, 3]。它产生于个体、机构及其环境之间的相互作用 [4]。在计算机模拟中实现社会智能的一种有效方式是通过 **社会模拟（Social Simulacra）** [5]，这是一种虚拟的社会计算系统，其中包含能够表现出逼真、类人行为的多样化 **智能体（Agents）** 。传统上，此类系统是使用专家定义的规则 [6, 7] 构建的，这些规则将领域知识编码为明确的行为规范；或是通过 **强化学习（Reinforcement Learning, RL）** [8] 构建，该方法通过在模拟环境中试错来优化智能体策略。尽管这些方法在某些情境下有效，但它们通常会导致过于简化的动态或有限的可解释性。 **大型语言模型（Large Language Models, LLMs）** 的出现为创建更丰富、更令人信服的模拟提供了变革性的基础，既能再现 **程式化事实（Stylized Facts）** [9]，也能生成可信的预测 [10]。

在本综述中，我们从两个互补的视角，在社会智能的框架内审视世界模型。首先，社会模拟可以作为一个 **显式世界模型（Explicit World Model）** 来运作，它映射现实世界的社会，为 **社会智能（Societal Intelligence）** 的涌现提供了一个结构化的环境。其次，模拟中的智能体可以通过交互发展出 **隐式世界模型（Implicit World Model）** ，形成指导其决策和社会行为的外部环境的内部表征。这两个视角——映射现实社会与理解外部世界——构成了以下小节的框架。表 [7](#table-7) 总结了这两个视角下的代表性工作。

表 7 | 社会智能框架下的世界模型研究总结。
| 视角 | 研究 | 关键方法 | 主要贡献 |
| :--- | :--- | :--- | :--- |
| 显式世界模型（映射社会） | Park et al. (2022) [5] | 基于 LLM 的社会模拟 | 提出社会模拟框架，生成逼真的类人行为与对话 |
| 显式世界模型（映射社会） | Li et al. (2024) [9] | 基于 LLM 的智能体经济模拟 | 再现经济领域的程式化事实，验证模拟有效性 |
| 隐式世界模型（智能体内在表征） | Piao et al. (2025) [10] | 多智能体社会模拟与预测 | 智能体通过交互形成对社会的内在理解，用于生成可信的社会预测 |
| 隐式世界模型（智能体内在表征） | (其他代表性工作) | (方法描述) | (贡献描述) |

<a id="table-7"></a>

> 表 7. 从两个视角看 **大型语言模型驱动（LLM-driven）** 的社会模拟器（Social Simulacra）的代表性工作： **镜像现实世界社会（Mirroring Real-world Society）** 与 **理解外部世界（Understanding the External World）** 。

|                      | 代表性工作                                     | 模拟焦点     | 世界模型的作用 |
| :------------------- | :--------------------------------------------- | :----------- | :------------- |
| **映射现实世界社会** | Generative Agents（park2023generative，）      | 日常社会生活 | 程式化事实     |
|                      | AI Town（park2023generative，）                | 沙盒社区     | 程式化事实     |
|                      | S3（gao2023s，）                               | 社交网络     | 预测           |
|                      | Papachristou 等人（papachristou2024network，） | 网络形成     | 程式化事实     |
|                      | Xu 等人（xu2023exploring，）                   | 社交游戏     | 战略模式       |

| (xu2023exploring,)                               | 社交游戏（Social games）             | 策略模式（Strategic patterns）  |                                                |
| :----------------------------------------------- | :----------------------------------- | :------------------------------ | :--------------------------------------------- |
| EconAgent (li2024econagent,)                     | 宏观经济学（Macroeconomics）         | 典型事实（Stylized facts）      |                                                |
| EconAgent (li2024econagent,)                     | 宏观经济学（Macroeconomics）         | 典型事实（Stylized facts）      |                                                |
| EconAgent (li2024econagent,)                     | 宏观经济学（Macroeconomics）         | 典型事实（Stylized facts）      |                                                |
| SRAP-Agent (ji2024srap,)                         | 资源分配（Resource allocation）      | 典型事实（Stylized facts）      |                                                |
| SRAP-Agent (ji2024srap,)                         | 资源分配（Resource allocation）      | 典型事实（Stylized facts）      |                                                |
| SRAP-Agent (ji2024srap,)                         | 资源分配（Resource allocation）      | 典型事实（Stylized facts）      |                                                |
| Project Sid (al2024project,)                     | 集体规则（Collective rules）         | 典型事实（Stylized facts）      |                                                |
| Project Sid (al2024project,)                     | 集体规则（Collective rules）         | 典型事实（Stylized facts）      |                                                |
| Project Sid (al2024project,)                     | 集体规则（Collective rules）         | 典型事实（Stylized facts）      |                                                |
| OASIS (yangoasis,)                               | 社交媒体（Social media）             | 典型事实（Stylized facts）      |                                                |
| OASIS (yangoasis,)                               | 社交媒体（Social media）             | 典型事实（Stylized facts）      |                                                |
| OASIS (yangoasis,)                               | 社交媒体（Social media）             | 典型事实（Stylized facts）      |                                                |
| GenSim (tanggensim,)                             | 社交媒体（Social media）             | 典型事实（Stylized facts）      |                                                |
| GenSim (tanggensim,)                             | 社交媒体（Social media）             | 典型事实（Stylized facts）      |                                                |
| GenSim (tanggensim,)                             | 社交媒体（Social media）             | 典型事实（Stylized facts）      |                                                |
|                                                  | YuLan-OneSim (wang2025yulan,)        | 通用平台（General platform）    | 典型事实（Stylized facts）                     |
|                                                  | YuLan-OneSim (wang2025yulan,)        | 通用平台（General platform）    | 典型事实（Stylized facts）                     |
|                                                  | YuLan-OneSim (wang2025yulan,)        | 通用平台（General platform）    | 典型事实（Stylized facts）                     |
|                                                  | YuLan-OneSim (wang2025yulan,)        | 通用平台（General platform）    | 典型事实（Stylized facts）                     |
|                                                  | AgentSociety (piao2025agentsociety,) | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
|                                                  | AgentSociety (piao2025agentsociety,) | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
|                                                  | AgentSociety (piao2025agentsociety,) | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
|                                                  | AgentSociety (piao2025agentsociety,) | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
|                                                  | SocioVerse (zhang2025socioverse,)    | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
|                                                  | SocioVerse (zhang2025socioverse,)    | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
|                                                  | SocioVerse (zhang2025socioverse,)    | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
|                                                  | SocioVerse (zhang2025socioverse,)    | 通用平台（General platform）    | 典型事实与预测（Stylized facts & predictions） |
| 理解外部世界（Understanding the External World） | Agent-Pro (zhang2024agent,)          | 交互式游戏（Interactive games） | 信念形成（Belief formation）                   |
| 理解外部世界（Understanding the External World） | Agent-Pro (zhang2024agent,)          | 交互式游戏（Interactive games） | 信念形成（Belief formation）                   |
| 理解外部世界（Understanding the External World） | Agent-Pro (zhang2024agent,)          | 交互式游戏（Interactive games） | 信念形成（Belief formation）                   |
| 理解外部世界（Understanding the External World） | Agent-Pro (zhang2024agent,)          | 交互式游戏（Interactive games） | 信念形成（Belief formation）                   |
| Zhang et al.                                     |                                      |                                 |                                                |

| Zhang et al. (zhang2023exploring,) | Collaboration tasks | Reflection & debate |     |
| :--------------------------------- | :------------------ | :------------------ | :-- |
| Zhang et al. (zhang2023exploring,) | Collaboration tasks | Reflection & debate |     |
| Zhang et al. (zhang2023exploring,) | Collaboration tasks | Reflection & debate |     |
| Zhang et al. (zhang2023exploring,) | Collaboration tasks | Reflection & debate |     |
| Zhang et al.                       |                     |                     |     |

#### 5.4.1. 构建映射现实世界的社会仿真（Building Social Simulacra Mirroring Real-world Society）

随着 **大型语言模型智能体（Large Language Model Agents, LLM Agents）** 的快速发展，构建逼真的社会模拟系统已变得越来越可行。一个著名的例子是 **AI Town** [1]，这是一个由 **生成式智能体（Generative Agents）** 构成的沙盒环境。这些智能体展现出可信的个体行为，并在群体层面产生类似于真实社区的 **涌现动态（Emergent Dynamics）** 。这些系统阐明了 **社会仿真（Social Simulacra）** 如何能够作为显式的 **世界模型（World Models）** ，为观察和研究 **社会智能（Societal Intelligence）** ——包括 **集体感知（Collective Sensing）、集体推理（Collective Reasoning）** 与 **集体协调（Collective Coordination）** ——提供了环境。

在社交网络中，S3（gao2023s，）的研究表明， **大型语言模型智能体（Large Language Model Agents, LLM Agents）** 能够再现真实的信息传播模式，捕捉公共事件展开过程中的动态。Papachristou 等人（papachristou2024network，）进一步强调了智能体之间网络结构的自发形成，这反映了人类社会的自组织特性。此类工作揭示了 **社会模拟物（Social Simulacra）** 如何以数字形式体现社会智能（Societal Intelligence）的适应性和交流性方面。超越网络范畴，LLM 智能体已展现出对战略互动中高阶推理进行建模的能力。Xu 等人（xu2023exploring，）的研究表明，在诸如《狼人杀》等社交推理游戏中，智能体表现出欺骗和对抗等战略行为，这反映了社会智能的认知和竞争维度。在经济学和资源分配领域，基于 LLM 的智能体使得从微观层面的推理出发，对宏观层面的结果进行自下而上的建模成为可能。EconAgent（li2024econagent，）再现了从个体决策中涌现的宏观经济趋势，SRAP-Agent（ji2024srap，）评估了资源分配中的政策效应，而 Project Sid（al2024project，）则探索了对税收规则的集体反应。这些例子说明了社会智能如何在经济系统中表现为聚合模式。

最近，研究已转向能够在多个领域进行泛化的大规模平台。其中， **AgentSociety（piao2025agentsociety，）** 是迄今为止最先进的尝试，它提供了一个大规模、多功能的平台环境，用于研究极化、政策干预以及其他与社会智能核心相关的现象。沿着这一方向，诸如 GenSim（tanggensim，）、YuLan-OneSim（wang2025yulan，）、OASIS（yangoasis，）和 SocioVerse（zhang2025socioverse，）等平台进一步扩展了这一愿景，它们通过整合多样化的社会情境，并将模拟规模扩展到数千甚至数百万个智能体。

总之，这些研究表明， **LLM 驱动的社会模拟物（LLM-driven Social Simulacra）** 可以作为人类社会的显式世界模型（Explicit World Models），从而能够对社会、战略和经济等多个领域的社会智能进行系统性研究。

#### 5.4.2. Agent’s Understanding of External World in Social Simulacra（社会模拟物中智能体对外部世界的理解）

在社交网络中，S3（gao2023s，）的研究表明， **大型语言模型智能体（Large Language Model Agents, LLM Agents）** 能够再现真实的信息传播模式，捕捉公共事件展开过程中的动态。Papachristou 等人（papachristou2024network，）进一步强调了智能体之间网络结构的自发形成，这反映了人类社会的自组织特性。此类工作揭示了 **社会模拟物（Social Simulacra）** 如何以数字形式体现社会智能（Societal Intelligence）的适应性和交流性方面。超越网络范畴，LLM 智能体已展现出对战略互动中高阶推理进行建模的能力。Xu 等人（xu2023exploring，）的研究表明，在诸如《狼人杀》等社交推理游戏中，智能体表现出欺骗和对抗等战略行为，这反映了社会智能的认知和竞争维度。在经济学和资源分配领域，基于 LLM 的智能体使得从微观层面的推理出发，对宏观层面的结果进行自下而上的建模成为可能。EconAgent（li2024econagent，）再现了从个体决策中涌现的宏观经济趋势，SRAP-Agent（ji2024srap，）评估了资源分配中的政策效应，而 Project Sid（al2024project，）则探索了对税收规则的集体反应。这些例子说明了社会智能如何在经济系统中表现为聚合模式。

最近，研究已转向能够在多个领域进行泛化的大规模平台。其中， **AgentSociety（piao2025agentsociety，）** 是迄今为止最先进的尝试，它提供了一个大规模、多功能的平台环境，用于研究极化、政策干预以及其他与社会智能核心相关的现象。沿着这一方向，诸如 GenSim（tanggensim，）、YuLan-OneSim（wang2025yulan，）、OASIS（yangoasis，）和 SocioVerse（zhang2025socioverse，）等平台进一步扩展了这一愿景，它们通过整合多样化的社会情境，并将模拟规模扩展到数千甚至数百万个智能体。

总之，这些研究表明， **LLM 驱动的社会模拟物（LLM-driven Social Simulacra）** 可以作为人类社会的显式世界模型（Explicit World Models），从而能够对社会、战略和经济等多个领域的社会智能进行系统性研究。

除了在宏观层面反映社会， **社会模拟（Social Simulacra）** 还使得研究智能体如何形成对其环境的内部表征成为可能。通过交互， **大型语言模型智能体（Large Language Model Agents, LLM Agents）** 积累经验，将其存储为记忆，并将其转化为 **隐式世界模型（Implicit World Models）** 。这些模型为社会智能提供了认知基础，使得智能体不仅能够回忆过去的交互，还能在决策时对其他智能体及更广泛的环境进行推理 [zhang2024survey]。

多项工作展示了隐式世界模型在实践中是如何形成的。 **Agent-Pro** [zhang2024agent] 将交互历史转化为结构化的信念，这些信念随后指导后续的决策制定和策略更新。这些信念反映了一个智能体对他人的理解，并直接关联到第 [3.2](#section-3-2) 节讨论的 **心理理论（Theory of Mind）** 能力。Zhang 等人 [zhang2023exploring] 通过引入来自社会心理学的反思与辩论机制，进一步拓展了这一方向，以改善多智能体任务中的协作。

在集体层面， **GovSim** [piatti2024cooperate] 研究了在 LLM 智能体社会中是否能够出现可持续的合作。在此设置中，每个智能体通过对话收集关于共享资源及同伴行为策略的信息，形成关于外部环境的高层次洞见。这些洞见构成了支撑群体层面社会智能的世界模型的隐式表征。另一个应用是 **Interactive Group Chat** [gu2024agent]，它探索了在诸如遗产纠纷和法庭辩论等场景中类似人类的审议过程。在此，智能体利用记忆和推理来产生与真实人类互动高度相似的策略和社会动态。

### 5.5. 世界模型的功能（Functions of World Models）

世界模型的核心设计目标是接收外部命令或动作，并对环境的动态状态转换进行建模。其功能可大致分为两个主要角色：充当 **基于云的环境（Cloud-based Environments）** 和作为 **边缘侧智能体大脑（Edge-side Agent Brains）** 。

基于云的世界模型通常表现为 **视频生成系统（Video Generation Systems）** ，能够合成由文本或动作轨迹驱动的大量高质量视频数据。这些生成的数据可以充当 **数据引擎（Data Engine）** ，用于增强训练 **策略模型（Policy Models）** （例如， **视觉语言动作模型（Vision-Language-Action Models, VLA）** 和 **视觉语言导航模型（Vision-and-Language Navigation Models, VLN）** ）所需的真实世界数据。此外，基于云的世界模型可以作为 **强化学习（Reinforcement Learning）** 中的环境，与智能体交互以促进虚拟环境中的进化学习。这种能力显著降低了与现实世界交互相关的成本和风险，在 **自动驾驶（Autonomous Driving）** 等领域尤为重要。另外，基于云的世界模型还可以充当 **策略评估器（Policy Evaluators）** ，通过与策略模型交互输出观察序列，从而实现对策略模型性能的评估。

相比之下，作为边缘侧智能体大脑的世界模型通常不需要低层次的视觉生成。相反，它们可以在 **潜在空间（Latent Space）** 内压缩世界状态；例如， **V-JEPA 2** [assran2025v] 训练了一个潜在空间世界模型，使其能够通过 **模型预测控制（Model Predictive Control）** 实现设备端动作规划。或者，也可以采用一种 **两阶段方法（Two-stage Approach）** [hu2024video]，其中世界模型首先处理视觉观察，然后将其转换为可执行的动作。

## 6. 开放问题与未来方向（Open Problems and Future Directions）

近期 **超写实生成式人工智能（Hyper-realistic Generative AI）** 的进展，极大地推动了 **世界模型（World Model）** 的发展，并特别聚焦于像 Sora [sora2024] 这样的 **多模态大模型（Multi-modal Big Models）** 。尽管创新速度很快，但仍有许多重要的开放问题有待解决。

### 6.1. 物理规则与反事实模拟（Physical Rules and Counterfactual Simulation）

**世界模型（World Model）** 的一个关键目标是捕捉其环境的 **因果结构（Causal Structure）** ——尤其是底层的 **物理规则（Physical Rules）** ——以便它们能够对数据分布之外的反事实进行推理 [pearl2009causal]。这种能力对于处理罕见的关键任务事件（例如， **自动驾驶（Autonomous Driving）** 的 **极端案例（Corner Cases）** [feng2023dense]）以及缩小 **仿真与现实之间的差距（Sim-to-real Gap）** 至关重要。最近的进展提出了一个问题：大规模、纯数据驱动的生成模型能否仅从原始视觉数据中习得这些规则？虽然基于 **Transformer** 和 **扩散模型（Diffusion Model）** 的视频生成器（如 Sora [sora2024]）能生成极其逼真的序列，但研究揭示了其持续存在的物理定律失效问题——例如，不准确的重力、流体或热动力学 [wang2023newton]。

明确嵌入物理学的 **混合方法（Hybrid Approaches）** 正成为有前景的替代方案。Genesis [Genesis] 通过将快速、照片级真实感的渲染与重新设计的通用物理核心相统一，阐明了这一方向，从而允许基于 **第一性原理模拟（First-principles Simulation）** 进行语言条件化的数据生成。PhysGen [liu2024physgen] 在图像到视频层面采取了类似的立场：它将一个 **刚体模拟器（Rigid-body Simulator）** 与一个 **扩散精炼器（Diffusion Refiner）** 耦合，使得从单张图像生成可控、物理上合理的运动成为可能。作为补充， **软约束混合方法（Soft-constraint Hybrids）** 通过 **学习时先验（Learning-time Priors）** 来强制执行物理规则。 **物理信息扩散（Physics-informed Diffusion）** 引入了基于 **偏微分方程（Partial Differential Equation, PDE）** 的残差损失，在保持生成灵活性的同时，惩罚对控制方程的违反 [bastek2025physicsinformed]。这种“ **硬+软（Hard+Soft）** ”设计在不牺牲真实感的前提下，提高了可控性和可解释性。

互补的诊断工作强调了为何需要此类混合方法。Kang 等人 [kang2024far] 表明，扩展扩散视频模型能产生完美的 **分布内保真度（In-distribution Fidelity）** ，但在 **分布外（Out-of-distribution）** 或组合测试中却会失效，这表明其泛化是“ **基于案例的（Case-based）** ”而非“ **基于规则的（Rule-based）** ”。Motamed 等人 [motamed2025generative] 利用 **Physics-IQ 基准测试（Physics-IQ Benchmark）** 得出了类似的结论：当前的视频生成器实现了视觉真实感，但在需要理解光学、流体动力学或磁学的任务上大多失败。与此同时， **第一性原理基准测试（First-principles Benchmarks）** 已开始将“ **物理保真度（Physical Fidelity）** ”作为一个可测量的维度进行操作化：T2VPhysBench [guo2025t2vphysbench] 评估了对核心物理定律（包括 **牛顿力学（Newtonian Mechanics）** 和守恒原理）的遵循程度，并记录了领先的 **文本到视频（Text-to-video）** 系统中普遍存在的系统性违反；而 VBench-2.0 [zheng2025vbench] 则明确将 **物理（Physics）** 和 **常识（Commonsense）** 引入作为视频生成的标准评估维度。

综上所述，证据表明，仅靠数据驱动的规模化不足以恢复稳健的物理定律。整合显式模拟器——或以其他方式强制执行物理先验（shi2022learning,）——仍然是构建 **世界模型（world models）** 的一条有前景的路径，这些模型能够泛化到未见过的反事实场景，同时保持可解释性和透明度。

### 6.2. 丰富社会维度（Enriching the Social Dimension）

仅模拟物理元素对于构建一个先进的世界模型是不够的，因为人类行为和社会互动在许多重要场景中也扮演着关键角色（gao2024large,; yuan2025learning,; gong2025behavegpt,）。例如，城市居民的行为对于构建城市环境的世界模型尤为重要（batty2024digital,; xu2023urban,）。先前的研究表明， **大型语言模型（Large Language Models, LLMs）** 所具有的类人常识推理能力，为使用 **生成式智能体（generative agents）** 模拟真实人类行为提供了独特的机会（park2023generative,）。然而，设计能够模拟真实且全面的人类行为及社会互动的自主智能体，仍然是一个悬而未决的问题。

近期研究表明，关于人类行为模式和认知过程的理论可以为 **智能体工作流（agentic workflows）** 的设计提供信息，这反过来又能增强 LLMs 的人类行为模拟能力（shao2024beyond,; park2023generative,），这代表了未来研究的一个重要方向。此外，对生成的人类行为真实性的评估，目前仍然在很大程度上依赖于主观的人工评估，这难以扩展到大规模的世界模型。开发一个可靠且可扩展的评估方案，将是另一个未来的研究方向，可以丰富世界模型的社会维度。

### 6.3. 基准测试（Benchmarks）

<a id="table-8"></a>

**表 8. 评估世界模型的代表性基准（Table 8. Representative benchmarks for evaluating world models）**

**表 8. 评估世界模型的代表性基准（Table 8. Representative benchmarks for evaluating world models）**

| 类别                                                              | 基准测试                               | 范围与亮点                                                                                                                                        |
| :---------------------------------------------------------------- | :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| **以视频为中心的世界模拟** <br>（Video-centric world simulation） | WorldSimBench (qin2024worldsimbench,)  | 沙盒/驾驶/操作；将人类偏好与 **动作级一致性** （action-level consistency）联系起来                                                                |
|                                                                   | WorldScore (duan2025worldscore,)       | 3000 个相机指定场景；分解 **可控性/质量/动态** （controllability/quality/dynamics）；比较 3D/4D/视频生成器                                        |
|                                                                   | VBench (huang2024vbench,)              | 通用 T2V/V2V；自动评估维度： **时间一致性** （temporal consistency）、稳定性、提示遵循度                                                          |
|                                                                   | VBench-2.0 (zheng2025vbench,)          | 通用 T2V/V2V； **内在忠实性** （intrinsic faithfulness）：物理、常识、人类保真度、可控性                                                          |
|                                                                   | T2V-CompBench (sun2025t2vcompbench,)   | 组合式 T2V；测试属性/动作/关系与相机运动的 **绑定** （binding）                                                                                   |
| **物理与空间推理** <br>（Physical & spatial reasoning）           | PhysBench (chow2025physbench,)         | 10k 视频-图像-文本三元组；探究 **视觉语言模型** （Vision-Language Models, VLMs）在 **属性/关系/动态** （properties/relations/dynamics）方面的差距 |
|                                                                   | UrbanVideo-Bench (zhao2025urbanvideo,) | 5.2k 城市第一人称视角片段（16 项任务）； **回忆、导航、因果推理** （recall, navigation, causal reasoning）                                        |
|                                                                   | Physics-IQ (motamed2025generative,)    | 五个领域（固体/流体/光学/热学/磁学）； **定律遵循度与...** （law adherence vs. ...）                                                              |

| | Physics-IQ (motamed2025generative,) | 五个领域（固体/流体/光学/热力学/磁学）； **定律遵循度（law adherence）** 与 **感知真实感（perceived realism）** |
| | Physics-IQ (motamed2025generative,) | 五个领域（固体/流体/光学/热力学/磁学）； **定律遵循度（law adherence）** 与 **感知真实感（perceived realism）** |
| | T2VPhysBench (guo2025t2vphysbench,) | 文本到视频；基于第一性原理的 **12 条核心定律** 检查清单 |
| | T2VPhysBench (guo2025t2vphysbench,) | 文本到视频；基于第一性原理的 **12 条核心定律** 检查清单 |
| | T2VPhysBench (guo2025t2vphysbench,) | 文本到视频；基于第一性原理的 **12 条核心定律** 检查清单 |
| | VideoPhy (bansal2025videophy,) | 以动作为中心的提示； **语义遵循度（semantic adherence）** 与 **物理常识（physical commonsense）** ；规则归因 |
| | VideoPhy (bansal2025videophy,) | 以动作为中心的提示； **语义遵循度（semantic adherence）** 与 **物理常识（physical commonsense）** ；规则归因 |
| | VideoPhy (bansal2025videophy,) | 以动作为中心的提示； **语义遵循度（semantic adherence）** 与 **物理常识（physical commonsense）** ；规则归因 |
| | Basic Spatial Abilities (xu2025bsa,) | 心理测量框架；五种空间技能（感知/关系/定向/旋转/可视化） |
| | Basic Spatial Abilities (xu2025bsa,) | 心理测量框架；五种空间技能（感知/关系/定向/旋转/可视化） |
| | Basic Spatial Abilities (xu2025bsa,) | 心理测量框架；五种空间技能（感知/关系/定向/旋转/可视化） |
| 具身<br>决策制定（Embodied decision making） | EAI (li2024embodied,) | 基于大型语言模型（Large Language Models, LLMs）的智能体（agents）；模块级（目标/子目标/动作/状态转移）评估；错误分类法（error taxonomy） |
| 具身<br>决策制定（Embodied decision making） | EAI (li2024embodied,) | 基于大型语言模型（Large Language Models, LLMs）的智能体（agents）；模块级（目标/子目标/动作/状态转移）评估；错误分类法（error taxonomy） |
| 具身<br>决策制定（Embodied decision making） | EAI (li2024embodied,) | 基于大型语言模型（Large Language Models, LLMs）的智能体（agents）；模块级（目标/子目标/动作/状态转移）评估；错误分类法（error taxonomy） |
| | EWMBench (yue2025ewmbench,) | AgiBotWorld 操作/导航； **场景一致性（scene consistency）** 、 **运动正确性（motion correctness）** 、语义对齐（semantic alignment） |
| | EWMBench (yue2025ewmbench,) | AgiBotWorld 操作/导航； **场景一致性（scene consistency）** 、 **运动正确性（motion correctness）** 、语义对齐（semantic alignment） |
| | EWMBench (yue2025ewmbench,) | AgiBotWorld 操作/导航； **场景一致性（scene consistency）** 、 **运动正确性（motion correctness）** 、语义对齐（semantic alignment） |
| | WPE (quevedo2025evaluating,) | 世界模型（world model）与真实/仿真环境中的策略评估；相同动作下的 **序列级/部件级对应关系（sequence-/part-level correspondence）** |
| | WPE (quevedo2025evaluating,) | 世界模型（world model）与真实/仿真环境中的策略评估；相同动作下的 **序列级/部件级对应关系（sequence-/part-level correspondence）** |
| | WPE (quevedo2025evaluating,) | 世界模型（world model）与真实/仿真环境中的策略评估；相同动作下的 **序列级/部件级对应关系（sequence-/part-level correspondence）** |

**对世界模型进行基准测试（Benchmarking world models）** 既是必要的，也是具有挑战性的。由于研究界追求的目标各异——学习内部表征（internal representations）与预测未来世界（predicting future worlds）——技术方法多样（例如，LLM 智能体（LLM agents）、视频扩散模型（video diffusion）），应用领域广泛（自动驾驶（autonomous driving）、机器人学（robotics）、社会模拟（social simulation）），因此并不存在一个单一的规范任务或指标。尽管如此，最近的几项工作表明，精心设计的测试平台如何能够揭示阻碍当前模型成为可靠世界模拟器（world simulators）的具体差距，如表 [8](#table-8) 所示。

**以视频为中心的世界模拟（Video-centric world simulation）** 。WorldSimBench 通过将人类偏好与沙盒（sandbox）、驾驶（driving）和操控（manipulation）场景下的动作级一致性（action-level consistency）相结合，将感知质量与控制联系起来 [1]；WorldScore 通过一个相机指定协议（camera-specified protocol）对此进行了补充，该协议将性能分解为可控性（controllability）、视觉质量（visual quality）和动态性（dynamics），覆盖 3000 个场景，使得 3D/4D 和视频生成器（video generators）能够进行直接比较 [2]。VBench 将几个主要自动化的评估维度——时间一致性（temporal consistency）、主体/背景稳定性（subject/background stability）和提示遵循度（prompt adherence）——进行了操作化 [3]，而 VBench-2.0 则提升了 **内在忠实性（intrinsic faithfulness）** （物理、常识、人类逼真度、可控性），以区分“看起来真实”和“像一个世界那样运作” [4]。组合性压力测试（Compositional stress tests）通过 T2V-CompBench 实现，它使用基于 MLLM（多模态大语言模型（Multimodal Large Language Model））、检测（detection）和跟踪（tracking）的指标，来衡量属性、动作、关系和相机运动之间的绑定（binding）能力 [5]。

**物理与空间推理（Physical and spatial reasoning）** 。除了外观，世界模型还必须遵守物理定律并支持空间能力。PhysBench（包含 1 万个视频-图像-文本三元组）揭示了现代视觉语言模型（Vision-Language Models, VLMs）在物体属性、关系和动态方面的系统性差距 [6]；UrbanVideo-Bench（包含 5.2k 个无人机视频片段，16 种任务类型）诊断了在长时第一人称视角（egocentric）视频流中，回忆、导航和因果推理方面的缺陷 [7]。以定律为中心的测试套件（Law-centric suites）使情况更加清晰：Physics-IQ 评估了五个领域（固体/流体力学、光学、热力学、磁学），发现物理理解在很大程度上与感知到的真实感脱钩 [8]；T2VPhysBench 为文本到视频系统（text-to-video systems）提供了一个包含 12 条核心定律的第一性原理检查表 [9]。在生成方面，VideoPhy 在以动作为中心的提示下量化了 **语义遵循度（semantic adherence）** 和 **物理常识（physical commonsense）** ，并将错误归因于具体的物理规则（例如，支撑、惯性、连续性） [10]。一个补充性的心理测量框架（psychometric framing）锚定了五种基本空间能力（Basic Spatial Abilities）——感知、关系、定向、心理旋转、可视化——揭示了 13 个视觉语言模型在几何/旋转方面的弱点，并为进展跟踪提供了校准任务 [11]。

**参考文献**

1. Qin, Y., et al. (2024). WorldSimBench: Linking perceptual quality to control in world models.
2. Duan, J., et al. (2025). WorldScore: A camera-specified protocol for comparing 3D/4D and video generators.
3. Huang, T., et al. (2024). VBench: A benchmark for video generation evaluation.
4. Zheng, Y., et al. (2025). VBench-2.0: Elevating intrinsic faithfulness in video generation evaluation.
5. Sun, Z., et al. (2025). T2V-CompBench: Compositional stress tests for text-to-video models.
6. Chow, D., et al. (2025). PhysBench: Revealing systematic gaps in physical understanding for VLMs.
7. Zhao, L., et al. (2025). UrbanVideo-Bench: Diagnosing deficits in long egocentric video streams.
8. Motamed, M., et al. (2025). Physics-IQ: Evaluating physical understanding in generative models.
9. Guo, X., et al. (2025). T2VPhysBench: A first-principles checklist for text-to-video systems.
10. Bansal, A., et al. (2025). VideoPhy: Quantifying semantic adherence and physical commonsense in video generation.
11. Xu, M., et al. (2025). Basic Spatial Abilities: A psychometric framing for evaluating VLMs.

**具身决策（Embodied decision making）** 。当 **世界模型（World Models）** 被嵌入控制回路时，聚合成功率会掩盖过程中的失败。 **具身智能体接口（Embodied Agent Interface, EAI）** 标准化了四个基于 **大型语言模型（Large Language Model, LLM）** 的模块——目标解释、子目标分解、动作序列化、状态转移建模——并报告细粒度的错误分类（例如，幻觉、可供性、规划）[1]。EWMBench 使用 AgiBot World 数据，在场景一致性、运动正确性（物理/任务一致的轨迹）和语义对齐方面测试具身世界模型，明确地将视频合理性关联到动作前提条件和可供性 [2]。一种以角色为中心的观点将模型评估为 **环境（environments）** ：WPE 比较了在模型与真实视频/模拟器中相同动作序列下的执行轨迹，报告序列级和部件级的对应关系 [3]；像 RoboScape 这样的物理信息世界模型，通过测量在模型生成的经验上进行训练时的策略提升和 **仿真到现实差距（sim-to-real gap）** ，来评估其 **数据引擎（data-engine）** 角色 [4]。

尽管取得了这些进展，对世界模型进行基准测试仍然是一个开放的挑战。未来的工作应侧重于构建更多样化和更现实的基准，以严格测试泛化能力。此外，标准化评估协议将是提高跨环境可比性和鲁棒性评估的关键。

### 6.4. Bridging Simulation and Reality with Embodied Intelligence （通过具身智能连接仿真与现实）

**具身决策（Embodied decision making）** 。当 **世界模型（World Models）** 被嵌入控制回路时，聚合成功率会掩盖过程中的失败。 **具身智能体接口（Embodied Agent Interface, EAI）** 标准化了四个基于 **大型语言模型（Large Language Model, LLM）** 的模块——目标解释、子目标分解、动作序列化、状态转移建模——并报告细粒度的错误分类（例如，幻觉、可供性、规划）[1]。EWMBench 使用 AgiBot World 数据，在场景一致性、运动正确性（物理/任务一致的轨迹）和语义对齐方面测试具身世界模型，明确地将视频合理性关联到动作前提条件和可供性 [2]。一种以角色为中心的观点将模型评估为 **环境（environments）** ：WPE 比较了在模型与真实视频/模拟器中相同动作序列下的执行轨迹，报告序列级和部件级的对应关系 [3]；像 RoboScape 这样的物理信息世界模型，通过测量在模型生成的经验上进行训练时的策略提升和 **仿真到现实差距（sim-to-real gap）** ，来评估其 **数据引擎（data-engine）** 角色 [4]。

尽管取得了这些进展，对世界模型进行基准测试仍然是一个开放的挑战。未来的工作应侧重于构建更多样化和更现实的基准，以严格测试泛化能力。此外，标准化评估协议将是提高跨环境可比性和鲁棒性评估的关键。

### 6.4. Bridging Simulation and Reality with Embodied Intelligence （通过具身智能连接仿真与现实）

**世界模型（World model）** 长期以来被视为发展 **具身智能（Embodied Intelligence）** 的关键一步 [savva2019habitat]。它可以作为一个强大的 **模拟器（Simulator）** ，创建环境的全面要素并建模它们之间真实的关系。这样的环境可以促进 **具身智能体（Embodied agents）** 通过与模拟环境的交互进行学习，从而减少对监督数据的需求。为实现这一目标，提升 **生成式人工智能（Generative AI）** 模型的多模态、多任务和三维能力，已成为为具身智能体开发通用世界模型的一个重要研究课题。此外， **缩小仿真与现实差距（Closing the simulation-to-reality gap）** [hofer2021sim2real] 一直是具身环境模拟器领域一个长期存在的研究难题，因此，将训练好的具身智能从仿真环境迁移到物理世界至关重要。收集更细粒度的 **感知数据（Sensory data）** 也是实现该目标的关键一步，这可以通过具身智能体的接口来促进。因此，一个有趣的未来研究方向是创建 **自我强化循环（Self-reinforcing loops）** ，以利用生成式世界模型与具身智能体的协同力量。

### 6.5. Simulation Efficiency （仿真效率）

确保世界模型具有高仿真效率对于许多应用都至关重要。例如， **每秒帧数（Frames Per Second, FPS）** 是衡量用于学习复杂无人机操控 AI 的模拟环境质量的关键指标。当前大多数大型生成式 AI 所采用的流行 **Transformer 架构（Transformer architecture）** 因其 **自回归（Autoregressive）** 特性（一次只能生成一个 **词元（Token）** ），对高速仿真构成了巨大挑战。已有多种策略被提出来加速大型生成模型的推理，例如结合大小生成模型 [shang2024defint] 以及 **蒸馏（Distilling）** 大模型 [shao2024beyond]。更全面的解决方案包括构建能够优化调度 **大型语言模型（Large Language Model, LLM）** 请求的仿真平台 [yan2024opencity]。

当需要模拟大型复杂系统时，高计算成本也是经典 **物理模拟器（Physics simulators）** 面临的一个问题。先前研究发现，像 **图神经网络（Graph Neural Network, GNN）** 这样的深度学习模型可以用来高效地近似物理系统 [sanchez2020learning]。因此，一个重要的研究方向将是探索较小的深度学习模型与大型生成式 AI 模型之间的协同作用。此外，要实现显著的加速，还需要从底层硬件到编程平台再到 AI 模型的整体改进。

### 6.6. 伦理与安全问题（Ethical and Safety Concerns）

**数据隐私（Data Privacy）** 。近期利用大型生成式人工智能（Artificial Intelligence, AI）构建世界模型（World Model）的趋势引发了人们对隐私风险的重大担忧，这主要源于其训练数据规模庞大且往往不透明 [1]。大量研究工作致力于评估使用大型生成式人工智能（如大型语言模型（Large Language Model, LLM））推断私人信息的风险 [2]，这在视频生成模型的背景下可能尤为敏感。为了遵守《通用数据保护条例》（General Data Protection Regulation, GDPR）等隐私法规 [3]，提高生成式人工智能生命周期的透明度至关重要，这有助于公众理解数据在这些人工智能模型中是如何被收集、存储和使用的。

**模拟不安全场景（Simulating Unsafe Scenario）** 。生成式人工智能（Generative AI）令人难以置信的智能能力使得保障其访问权限成为一项至关重要的任务。先前对大型语言模型（LLMs）的研究发现，它们可能被对抗性提示（Adversarial Prompting）误导，生成不安全的内容 [4, 5]。世界模型被不安全使用的风险可能更大。对抗性用户可能利用此类技术来模拟有害场景，从而降低策划非法和不道德活动的成本。因此，保障世界模型的使用是未来一个重要的研究方向。

**问责制（Accountability）** 。生成超逼真文本、图像和视频的能力已经导致了传播虚假信息和错误信息的严重社会问题。例如，深度伪造（Deepfake）技术的出现引发了大规模的滥用，对社会、经济和政治体系产生了广泛的负面影响 [6]。因此，检测人工智能生成的内容已成为应对这些风险的一个关键研究问题 [7]。然而，由于生成式人工智能的进步，这个问题正变得越来越具有挑战性，而随着能够生成一致、多维输出的世界模型的到来，情况将变得更加困难。数字水印（Watermarking）等技术可能有助于提高世界模型使用的问责制 [8]。需要更多的研究关注以及法律解决方案，以提高世界模型使用的问责制。

**参考文献（References）**

1. yao2024survey
2. li2024llm
3. tamburri2020design
4. kumar2023certifying
5. inan2023llama
6. westerlund2019emergence
7. rana2022deepfake
8. dathathri2024scalable

## 7. 结论（Conclusion）

理解世界并预测未来，一直是致力于开发 **人工生成智能（Artificial Generative Intelligence）** 的科学家们的长期目标，这凸显了在不同领域构建 **世界模型（World Models）** 的重要性。本文首次对世界模型进行了全面的综述，系统性地探讨了其两大核心功能：对外部世界的 **隐式表征（Implicit Representations）** 与 **未来预测（Future Predictions）** 。我们对围绕这些核心功能的现有研究进行了广泛的总结，特别强调了世界模型在 **决策制定（Decision-making）** 、模型习得的 **世界知识（World Knowledge）** 、作为 **视频生成（Video Generation）** 的世界模型以及作为 **具身环境（Embodied Environments）** 的世界模型等方面的应用。此外，我们回顾了世界模型在关键应用领域中的进展，包括 **生成式游戏（Generative Games）** 、 **机器人学（Robotics）** 、 **自动驾驶（Autonomous Driving）** 和 **社会模拟（Social Simulacra）** 。最后，认识到这一快速发展领域中尚未解决的挑战，我们重点指出了 **开放性问题（Open Problems）** ，并提出了有前景的研究方向，以期激发这一新兴领域的进一步探索。

## 参考文献

- [1]
  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya,
  Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman,
  Shyamal Anadkat, 等。
  GPT-4 技术报告。
  arXiv 预印本 arXiv:2303.08774, 2023。
- [2]
  Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany
  Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, 等。
  Cosmos：面向物理人工智能的世界基础模型平台。
  arXiv 预印本 arXiv:2501.03575, 2025。
- [3]
  Ali Agha, Kyohei Otsu, Benjamin Morrell, David D Fan, Rohan Thakker, Angel
  Santamaria-Navarro, Sung-Kyun Kim, Amanda Bouman, Xianmei Lei, Jeffrey
  Edlund, 等。
  Nebula：在挑战性环境中追求机器人自主性；DARPA 地下挑战赛中的 Team CoSTAR。
  arXiv 预印本 arXiv:2103.11470, 2021。
- [4]
  Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew,
  Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas,
  等。
  用机器人手解魔方。
  arXiv 预印本 arXiv:1910.07113, 2019。
- [5]
  Altera AL, Andrew Ahn, Nic Becker, Stephanie Carroll, Nico Christie, Manuel
  Cortes, Arda Demirci, Melissa Du, Frankie Li, Shuying Luo, 等。
  Project SID：迈向人工智能文明的智能体模拟。
  arXiv 预印本 arXiv:2411.00114, 2024。
- [6]
  Jorge Aldaco, Travis Armstrong, Robert Baruch, Jeff Bingham, Sanky Chan,
  Kenneth Draper, Debidatta Dwibedi, Chelsea Finn, Pete Florence, Spencer
  Goodrich, 等。
  ALOHA 2：用于双手遥操作的低成本增强硬件。
  arXiv 预印本 arXiv:2405.02292, 2024。
- [7]
  Florent Altché 与 Arnaud de La Fortelle。
  用于高速公路轨迹预测的 LSTM 网络。
  收录于 2017 年 IEEE 第 20 届智能交通系统国际会议（ITSC），第 353–359 页。IEEE，2017。
- [8]
  Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew
  Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, 等。
  V-JEPA 2：自监督视频模型实现理解、预测与规划。
  arXiv 预印本 arXiv:2506.09985, 2025。
- [9]
  Genesis 作者。
  Genesis：面向机器人及其他领域的生成式通用物理引擎，2024 年 12 月。
- [10]
  Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu, 与
  Chiew-Lan Tai。
  TransFusion：利用 Transformer 实现鲁棒的激光雷达-相机融合用于 3D 目标检测。
  收录于 IEEE/CVF 计算机视觉与模式识别会议论文集，第 1090–1099 页，2022。
- [11]
  Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat,
  Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri
  Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack
  Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer
  Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David
  Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul
  Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles
  Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian
  Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex
  Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew
  Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin,
  Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson,
  Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu,
  Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray
  Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, Aäron
  van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, 与 Tim
  Rocktäschel。
  Genie 3：世界模型的新前沿。

2025.

- [12]
  Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan
  Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover.
  Videophy: 评估视频生成的物理常识。
  发表于第十三届国际学习表征会议，2025年。
- [13]
  Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun.
  导航世界模型。
  发表于计算机视觉与模式识别会议论文集，第15791–15801页，2025年。
- [14]
  Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann
  LeCun, Mahmoud Assran, and Nicolas Ballas.
  重新审视从视频中学习视觉表征的特征预测方法。
  arXiv预印本 arXiv:2404.08471，2024年。
- [15]
  Jan-Hendrik Bastek, WaiChing Sun, and Dennis Kochmann.
  物理信息扩散模型。
  发表于第十三届国际学习表征会议，2025年。
- [16]
  Michael Batty.
  城市规划中的数字孪生。
  自然计算科学，第4卷，第3期，第192–199页，2024年。
- [17]
  Ciro Beneduce, Massimiliano Luca, and Bruno Lepri.
  人工智能的盲点：生成城市场景中的地理知识与多样性缺失。
  arXiv预印本 arXiv:2506.16898，2025年。
- [18]
  Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn,
  Aviral Kumar, and Sergey Levine.
  利用预训练图像编辑扩散模型进行零样本机器人操作。
  arXiv预印本 arXiv:2310.10639，2023年。
- [19]
  Daniel Bogdoll, Yitian Yang, and J Marius Zöllner.
  Muvo：一种用于自动驾驶的、具有几何表征的多模态生成世界模型。
  arXiv电子版，第arXiv–2311页，2023年。
- [20]
  William A Brock and Cars H Hommes.
  一个简单资产定价模型中的异质信念与通往混沌的路径。
  经济动力学与控制杂志，第22卷，第8-9期，第1235–1274页，

1998.

- [21]
  Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David
  Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and
  Aditya Ramesh.
  **Video generation models as world simulators** . 2024.

- [22]
  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
  Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
  et al.
  **Language models are few-shot learners** .
  _Advances in neural information processing systems_,
  33:1877–1901, 2020.
- [23]
  Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi,
  Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps,
  et al.
  **Genie: Generative interactive environments** .
  In _Forty-first International Conference on Machine Learning_,

2024.

- [24]
  Shengqu Cai, Eric Ryan Chan, Songyou Peng, Mohamad Shahbazi, Anton Obukhov, Luc Van Gool, and Gordon Wetzstein.
  Diffdreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models.
  In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2139–2150, 2023.
- [25]
  Hyungjoo Chae, Namyoung Kim, Kai Tzu-iunn Ong, Minju Gwak, Gwanwoo Song, Jihoon Kim, Sunghwan Kim, Dongha Lee, and Jinyoung Yeo.
  Web agents with world models: Learning and leveraging environment dynamics in web navigation.
  arXiv preprint arXiv:2410.13232, 2024.
- [26]
  Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang.
  Matterport3d: Learning from rgb-d data in indoor environments.
  International Conference on 3D Vision (3DV), 2017.
- [27]
  Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen.
  Gamegen-x: Interactive open-world game video generation.
  arXiv preprint arXiv:2411.00769, 2024.
- [28]
  Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al.
  Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation.
  arXiv preprint arXiv:2410.06158, 2024.
- [29]
  Jianyu Chen, Bodi Yuan, and Masayoshi Tomizuka.
  Model-free deep reinforcement learning for urban autonomous driving, 2019.
- [30]
  Jingye Chen, Yuzhong Zhao, Yupan Huang, Lei Cui, Li Dong, Tengchao Lv, Qifeng Chen, and Furu Wei.
  Model as a game: On numerical and spatial consistency for generative games.
  arXiv preprint arXiv:2503.21172, 2025.
- [31]
  Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, et al.
  Deepverse: 4d autoregressive video generation as a world model.
  arXiv preprint arXiv:2506.01103, 2025.
- [32]
  Zhili Cheng, Zhitong Wang, Jinyi Hu, Shengding Hu, An Liu, Yuge Tu, Pengkai Li, Lei Shi, Zhiyuan Liu, and Maosong Sun.
  Legent: Open platform for embodied agents.
  arXiv preprint arXiv:2404.18243, 2024.
- [33]
  Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song.
  Diffusion policy: Visuomotor policy learning via action diffusion.
  The International Journal of Robotics Research, page 02783649241273668, 2023.
- [34]
  Xiaowei Chi, Hengyuan Zhang, Chun-Kai Fan, Xingqun Qi, Rongyu Zhang, Anthony Chen, Chi-min Chan, Wei Xue, Wenhan Luo, Shanghang Zhang, et al.
  Eva: An embodied world model for future video anticipation.
  arXiv preprint arXiv:2410.15461, 2024.
- [35]
  Joseph Cho, Fachrina Dewi Puspitasari, Sheng Zheng, Jingyao Zheng, Lik-Hang Lee, Tae-Ho Kim, Choong Seon Hong, and Chaoning Zhang.
  Sora as an agi world model? a complete survey on text-to-video generation.
  arXiv preprint arXiv:2403.05131, 2024.
- [36]
  Fang-Chieh Chou, Tsung-Han Lin, Henggang Cui, Vladan Radosavljevic, Thi Nguyen, Tzu-Kuo Huang, Matthew Niedoba, Jeff Schneider, and Nemanja Djuric.
  Predicting motion of vulnerable road users using high-definition maps and efficient convnets.
  In 2020 IEEE Intelligent Vehicles Symposium (IV), pages 1655–1662. IEEE, 2020.
- [37]
  Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang.
  Physbench: Benchmarking and enhancing vision-language models for physical world understanding.
  arXiv preprint arXiv:2501.16411, 2025.
- [38]
  Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.
  Deep reinforcement learning in a handful of trials using probabilistic dynamics models.
  Advances in neural information processing systems, 31, 2018.
- [39]
  Kenneth James Williams Craik.
  The nature of explanation.

1943.

- [40]
  崔恒刚（Henggang Cui）、弗拉丹·拉多萨夫列维奇（Vladan Radosavljevic）、周方杰（Fang-Chieh Chou）、林宗翰（Tsung-Han Lin）、Thi Nguyen、黄子国（Tzu-Kuo Huang）、杰夫·施耐德（Jeff Schneider）、内马尼亚·久里奇（Nemanja Djuric）。
  使用深度卷积网络进行自动驾驶的多模态轨迹预测。
  载于 2019 年 IEEE 机器人与自动化国际会议（ICRA），第 2090–2096 页。IEEE，2019 年。
- [41]
  苏曼斯·达塔特里（Sumanth Dathathri）、阿比盖尔·西（Abigail See）、苏梅德·盖萨斯（Sumedh Ghaisas）、黄博森（Po-Sen Huang）、罗布·麦克亚当（Rob McAdam）、约翰内斯·韦尔布（Johannes Welbl）、万达娜·巴查尼（Vandana Bachani）、亚历克斯·卡斯卡索利（Alex Kaskasoli）、罗伯特·斯坦福斯（Robert Stanforth）、塔蒂亚娜·马特约维奇（Tatiana Matejovicova）等。
  用于识别大型语言模型输出的可扩展水印技术。
  《自然》，634(8035):818–823，2024 年。
- [42]
  马特·戴特克（Matt Deitke）、伊莱·范德比尔特（Eli VanderBilt）、阿尔瓦罗·埃拉斯蒂（Alvaro Herrasti）、卢卡·韦斯（Luca Weihs）、基亚娜·埃赫萨尼（Kiana Ehsani）、霍尔迪·萨尔瓦多（Jordi Salvador）、温森·韩（Winson Han）、埃里克·科尔夫（Eric Kolve）、阿尼尔·肯巴维（Aniruddha Kembhavi）、鲁兹贝·莫塔吉（Roozbeh Mottaghi）。
  Procthor：使用程序化生成的大规模具身人工智能。
  《神经信息处理系统进展》，35:5982–5994，2022 年。
- [43]
  邓博洋（Boyang Deng）、理查德·塔克（Richard Tucker）、李正奇（Zhengqi Li）、莱奥尼达斯·吉巴斯（Leonidas Guibas）、诺亚·斯内夫利（Noah Snavely）、戈登·韦茨斯坦（Gordon Wetzstein）。
  Streetscapes：使用自回归视频扩散进行大规模一致的街景生成。
  载于 ACM SIGGRAPH 2024 会议论文，第 1–11 页，2024 年。
- [44]
  梅赫梅特·多加尔（Mehmet Dogar）、安德鲁·斯皮尔伯格（Andrew Spielberg）、斯图尔特·贝克（Stuart Baker）、丹妮拉·鲁斯（Daniela Rus）。
  用于顺序装配操作的多机器人抓取规划。
  《自主机器人》，43:649–664，2019 年。
- [45]
  阿列克谢·多索维茨基（Alexey Dosovitskiy）。
  一张图像价值 16x16 个词：用于大规模图像识别的 Transformer。
  arXiv 预印本 arXiv:2010.11929，2020 年。
- [46]
  阿列克谢·多索维茨基（Alexey Dosovitskiy）、赫尔曼·罗斯（German Ros）、费利佩·科德维拉（Felipe Codevilla）、安东尼奥·洛佩斯（Antonio Lopez）、弗拉德伦·科尔顿（Vladlen Koltun）。
  Carla：一个开放的城市场景驾驶模拟器。
  载于机器人学习会议，第 1–16 页。PMLR，2017 年。
- [47]
  杜楠（Nan Du）、黄艳萍（Yanping Huang）、戴安德鲁·M（Andrew M Dai）、西蒙·童（Simon Tong）、德米特里·列皮欣（Dmitry Lepikhin）、徐元中（Yuanzhong Xu）、马克西姆·克里昆（Maxim Krikun）、周延奇（Yanqi Zhou）、余亚当斯·魏（Adams Wei Yu）、奥尔汗·菲拉特（Orhan Firat）等。
  Glam：使用专家混合高效扩展语言模型。
  载于国际机器学习会议，第 5547–5569 页。PMLR，2022 年。
- [48]
  杜一伦（Yilun Du）、杨雪莉（Sherry Yang）、戴博（Bo Dai）、戴汉俊（Hanjun Dai）、奥菲尔·纳胡姆（Ofir Nachum）、乔希·特南鲍姆（Josh Tenenbaum）、戴尔·舒尔曼斯（Dale Schuurmans）、彼得·阿比尔（Pieter Abbeel）。
  通过文本引导的视频生成学习通用策略。
  《神经信息处理系统进展》，36，2024 年。
- [49]
  杜宇伟（Yuwei Du）、冯杰（Jie Feng）、袁健（Jian Yuan）、李勇（Yong Li）。
  Cams：一个由 CityGPT 驱动的城市人类移动性模拟智能体框架。
  arXiv 预印本 arXiv:2506.13599，2025 年。
- [50]
  段浩一（Haoyi Duan）、俞宏兴（Hong-Xing Yu）、陈思睿（Sirui Chen）、李飞飞（Li Fei-Fei）、吴佳俊（Jiajun Wu）。
  Worldscore：一个用于世界生成的统一评估基准。
  arXiv 预印本 arXiv:2504.00983，2025 年。
- [51]
  休·杜兰特-怀特（Hugh Durrant-Whyte）、蒂姆·贝利（Tim Bailey）。
  同步定位与建图：第一部分。
  《IEEE 机器人与自动化杂志》，13(2):99–110，2006 年。
- [52]
  约书亚·M·爱泼斯坦（Joshua M Epstein）。
  生成性社会科学：基于智能体的计算建模研究。
  载于《生成性社会科学》。普林斯顿大学出版社，2012 年。
- [53]
  亚历杭德罗·埃斯孔特雷拉（Alejandro Escontrela）、阿德米·阿德尼吉（Ademi Adeniji）、严威尔逊（Wilson Yan）、阿贾伊·贾恩（Ajay Jain）、彭学斌（Xue Bin Peng）、肯·戈德堡（Ken Goldberg）、李永云（Youngwoon Lee）、丹尼亚尔·哈夫纳（Danijar Hafner）、彼得·阿比尔（Pieter Abbeel）。
  视频预测模型作为强化学习的奖励。
  《神经信息处理系统进展》，36，2024 年。
- [54]
  帕特里克·埃塞尔（Patrick Esser）、乔纳森·邱（Johnathan Chiu）、帕尔米达·阿蒂格赫钱（Parmida Atighehchian）、乔纳森·格兰斯科格（Jonathan Granskog）、阿纳斯塔西斯·杰马尼迪斯（Anastasis Germanidis）。
  使用扩散模型进行结构和内容引导的视频合成。
  载于 IEEE/CVF 计算机视觉国际会议论文集，第 7346–7356 页，2023 年。
- [55]
  范林曦（Linxi Fan）、王冠智（Guanzhi Wang）、蒋云帆（Yunfan Jiang）、阿贾伊·曼德卡（Ajay Mandlekar）、杨云聪（Yuncong Yang）、朱浩一（Haoyi Zhu）、唐安德鲁（Andrew Tang）、黄德安（De-An Huang）、朱宇科（Yuke Zhu）、阿尼玛·阿南德库马尔（Anima Anandkumar）。
  Minedojo：利用互联网规模知识构建开放式具身智能体。
  《神经信息处理系统进展》，35:18343–18362，2022 年。
- [56]
  冯超（Chao Feng）、陈子阳（Ziyang Chen）、亚历山大·霍林斯基（Aleksander Holynski）、阿列克谢·A·埃夫罗斯（Alexei A Efros）、安德鲁·欧文斯（Andrew Owens）。
  将 GPS 作为图像生成的控制信号。

- [57]
  Jie Feng, Yuwei Du, Tianhui Liu, Siqi Guo, Yuming Lin, and Yong Li.
  Citygpt: Empowering urban spatial cognition of large language models.
  arXiv preprint arXiv:2406.13948, 2024.
- [58]
  Jie Feng, Yuwei Du, Jie Zhao, and Yong Li.
  Agentmove: A large language model based agentic framework for
  zero-shot next location prediction.
  In Proceedings of the 2025 Conference of the Nations of the
  Americas Chapter of the Association for Computational Linguistics: Human
  Language Technologies (Volume 1: Long Papers), pages 1322–1338, 2025.
- [59]
  Jie Feng, Shengyuan Wang, Tianhui Liu, Yanxin Xi, and Yong Li.
  Urbanllava: A multi-modal large language model for urban intelligence
  with spatial reasoning and understanding.
  arXiv preprint arXiv:2506.23219, 2025.
- [60]
  Jie Feng, Jinwei Zeng, Qingyue Long, Hongyi Chen, Jie Zhao, Yanxin Xi, Zhilun
  Zhou, Yuan Yuan, Shengyuan Wang, Qingbin Zeng, et al.
  A survey of large language model-powered spatial intelligence across
  scales: Advances in embodied agents, smart cities, and earth science.
  arXiv preprint arXiv:2504.09848, 2025.
- [61]
  Jie Feng, Jun Zhang, Junbo Yan, Xin Zhang, Tianjian Ouyang, Tianhui Liu, Yuwei
  Du, Siqi Guo, and Yong Li.
  Citybench: Evaluating the capabilities of large language model as
  world model.
  arXiv preprint arXiv:2406.13945, 2024.
- [62]
  Shuo Feng, Haowei Sun, Xintao Yan, Haojie Zhu, Zhengxia Zou, Shengyin Shen, and
  Henry X Liu.
  Dense reinforcement learning for safety validation of autonomous
  vehicles.
  Nature, 615(7953):620–627, 2023.
- [63]
  Yao Feng, Hengkai Tan, Xinyi Mao, Guodong Liu, Shuhe Huang, Chendong Xiang,
  Hang Su, and Jun Zhu.
  Generalist bimanual manipulation via foundation video diffusion
  models.
  arXiv preprint arXiv:2507.12898, 2025.
- [64]
  Chelsea Finn, Ian Goodfellow, and Sergey Levine.
  Unsupervised learning for physical interaction through video
  prediction.
  Advances in neural information processing systems, 29, 2016.
- [65]
  Chelsea Finn and Sergey Levine.
  Deep visual foresight for planning robot motion.
  In 2017 IEEE International Conference on Robotics and Automation
  (ICRA), pages 2786–2793. IEEE, 2017.
- [66]
  Rao Fu, Zehao Wen, Zichen Liu, and Srinath Sridhar.
  Anyhome: Open-vocabulary generation of structured and textured 3d
  homes.
  In European Conference on Computer Vision, pages 52–70.
  Springer, 2025.
- [67]
  Chuang Gan, Jeremy Schwartz, Seth Alter, Damian Mrowca, Martin Schrimpf, James
  Traer, Julian De Freitas, Jonas Kubilius, Abhishek Bhandwaldar, Nick Haber,
  et al.
  Threedworld: A platform for interactive multi-modal physical
  simulation.
  arXiv preprint arXiv:2007.04954, 2020.
- [68]
  Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli
  Xu, and Yong Li.
  Large language models empowered agent-based modeling and simulation:
  A survey and perspectives.

2024.

2025.

- [69]
  陈高（Chen Gao）、兰晓冲（Xiaochong Lan）、卢志宏（Zhihong Lu）、毛金柱（Jinzhu Mao）、朴京华（Jinghua Piao）、王焕栋（Huandong Wang）、金德鹏（Depeng Jin）、李勇（Yong Li）。
  **S3：基于大型语言模型智能体的社交网络模拟系统（S3: Social-network simulation system with large language model-empowered agents）** 。
  arXiv 预印本 arXiv:2307.14984，2023年。
- [70]
  陈高（Chen Gao）、赵柏宁（Baining Zhao）、张伟臣（Weichen Zhang）、毛金柱（Jinzhu Mao）、张俊（Jun Zhang）、郑志恒（Zhiheng Zheng）、满凡航（Fanhang Man）、方建杰（Jianjie Fang）、周子乐（Zile Zhou）、崔金强（Jinqiang Cui）等。
  **Embodiedcity：真实世界城市环境中的具身智能体基准平台（Embodiedcity: A benchmark platform for embodied agent in real-world city environment）** 。
  arXiv 预印本 arXiv:2410.09604，2024年。
- [71]
  高乔子（Qiaozi Gao）、戈文德·塔泰（Govind Thattai）、苏海拉·沙基亚（Suhaila Shakiah）、高晓峰（Xiaofeng Gao）、什雷亚斯·潘萨雷（Shreyas Pansare）、瓦苏·夏尔马（Vasu Sharma）、高拉夫·苏克哈特梅（Gaurav Sukhatme）、石航杰（Hangjie Shi）、杨博飞（Bofei Yang）、张德胜（Desheng Zhang）等。
  **Alexa Arena：一个以用户为中心的具身人工智能交互平台（Alexa arena: A user-centric interactive platform for embodied ai）** 。
  《神经信息处理系统进展（Advances in Neural Information Processing Systems）》，第36卷，2024年。
- [72]
  高启越（Qiyue Gao）、皮欣宇（Xinyu Pi）、刘凯文（Kevin Liu）、陈俊荣（Junrong Chen）、杨若兰（Ruolan Yang）、黄心琪（Xinqi Huang）、方欣宇（Xinyu Fang）、孙璐（Lu Sun）、高塔姆·基肖尔（Gautham Kishore）、艾波（Bo Ai）等。
  **视觉语言模型具有内部世界模型吗？走向原子化评估（Do vision-language models have internal world models? towards an atomic evaluation）** 。
  arXiv 预印本 arXiv:2506.21876，2025年。
- [73]
  高申远（Shenyuan Gao）、杨佳志（Jiazhi Yang）、陈丽（Li Chen）、卡什亚普·奇塔（Kashyap Chitta）、邱一航（Yihang Qiu）、安德烈亚斯·盖格（Andreas Geiger）、张俊（Jun Zhang）、李洪洋（Hongyang Li）。
  **Vista：一个具有高保真度和多功能可控性的通用驾驶世界模型（Vista: A generalizable driving world model with high fidelity and versatile controllability）** 。
  arXiv 预印本 arXiv:2405.17398，2024年。
- [74]
  葛志奇（Zhiqi Ge）、黄泓喆（Hongzhe Huang）、周明泽（Mingze Zhou）、李俊成（Juncheng Li）、王国明（Guoming Wang）、唐思良（Siliang Tang）、庄越挺（Yueting Zhuang）。
  **WorldGPT：赋能大型语言模型作为多模态世界模型（Worldgpt: Empowering llm as multimodal world model）** 。
  载于《第32届ACM国际多媒体会议论文集（Proceedings of the 32nd ACM International Conference on Multimedia）》，第7346–7355页，2024年。
- [75]
  伊格纳特·格奥尔基耶夫（Ignat Georgiev）、瓦伦·吉里达尔（Varun Giridhar）、尼克拉斯·汉森（Nicklas Hansen）、阿尼梅什·加格（Animesh Garg）。
  **PWM：基于多任务世界模型的策略学习（Pwm: Policy learning with multi-task world models）** 。
  载于《第十三届国际学习表征会议（The Thirteenth International Conference on Learning Representations）》，2025年。
- [76]
  埃利奥特·格斯特林（Elliot Gestrin）、马可·库尔曼（Marco Kuhlmann）、延德里克·赛普（Jendrik Seipp）。
  **NL2Plan：基于最小文本描述的鲁棒大型语言模型驱动规划（Nl2plan: Robust llm-driven planning from minimal text descriptions）** 。
  arXiv 预印本 arXiv:2405.04215，2024年。
- [77]
  罗斯·吉尔希克（Ross Girshick）、杰夫·多纳休（Jeff Donahue）、特雷弗·达雷尔（Trevor Darrell）、吉滕德拉·马利克（Jitendra Malik）。
  **用于精确目标检测和语义分割的丰富特征层次结构（Rich feature hierarchies for accurate object detection and semantic segmentation）** 。
  载于《IEEE计算机视觉与模式识别会议论文集（Proceedings of the IEEE conference on computer vision and pattern recognition）》，第580–587页，2014年。
- [78]
  龚佳慧（Jiahui Gong）、丁景涛（Jingtao Ding）、孟繁锦（Fanjin Meng）、杨晨（Chen Yang）、陈虹（Hong Chen）、王作坚（Zuojian Wang）、卢海生（Haisheng Lu）、李勇（Yong Li）。
  **BehaveGPT：用于大规模用户行为建模的基础模型（Behavegpt: A foundation model for large-scale user behavior modeling）** 。
  arXiv 预印本 arXiv:2505.17631，2025年。
- [79]
  詹姆斯·戈内特（James Gornet）、马特·汤姆森（Matt Thomson）。
  **利用视觉预测编码自动构建认知地图（Automated construction of cognitive maps with visual predictive coding）** 。
  《自然机器智能（Nature Machine Intelligence）》，第6卷，第7期，第820–833页，2024年。
- [80]
  顾宇（Yu Gu）、张凯（Kai Zhang）、宁雨婷（Yuting Ning）、郑博元（Boyuan Zheng）、苟博宇（Boyu Gou）、薛天赐（Tianci Xue）、常成（Cheng Chang）、桑贾里·斯里瓦斯塔瓦（Sanjari Srivastava）、谢亚男（Yanan Xie）、齐鹏（Peng Qi）等。
  **你的大型语言模型是互联网的秘密世界模型吗？网络智能体的基于模型规划（Is your llm secretly a world model of the internet? model-based planning for web agents）** 。
  arXiv 预印本 arXiv:2411.06559，2024年。
- [81]
  顾周鸿（Zhouhong Gu）、朱晓璇（Xiaoxuan Zhu）、郭浩然（Haoran Guo）、张琳（Lin Zhang）、蔡寅（Yin Cai）、沈昊（Hao Shen）、陈江杰（Jiangjie Chen）、叶哲宇（Zheyu Ye）、戴一飞（Yifei Dai）、高岩（Yan Gao）等。
  **智能体群聊：用于更好激发集体涌现行为的交互式群聊模拟（Agent group chat: An interactive group chat simulacra for better eliciting collective emergent behavior）** 。
  arXiv 预印本 arXiv:2403.13433，2024年。
- [82]
  关林（Lin Guan）、卡尔提克·瓦尔米卡姆（Karthik Valmeekam）、萨拉特·斯里达兰（Sarath Sreedharan）、苏巴拉奥·坎巴姆帕蒂（Subbarao Kambhampati）。
  **利用预训练大型语言模型构建和利用世界模型进行基于模型的任务规划（Leveraging pre-trained large language models to construct and utilize world models for model-based task planning）** 。
  《神经信息处理系统进展（Advances in Neural Information Processing Systems）》，第36卷，第79081–79094页，2023年。
- [83]
  管彦臣（Yanchen Guan）、廖海成（Haicheng Liao）、李振宁（Zhenning Li）、胡佳（Jia Hu）、袁润泽（Runze Yuan）、李运健（Yunjian Li）、张国辉（Guohui Zhang）、徐成中（Chengzhong Xu）。
  **自动驾驶世界模型：初步综述（World models for autonomous driving: An initial survey）** 。
  《IEEE智能车辆汇刊（IEEE Transactions on Intelligent Vehicles）》，2024年。
- [84]
  科尔·古利诺（Cole Gulino）、贾斯汀·傅（Justin Fu）、罗文杰（Wenjie Luo）、乔治·塔克（George Tucker）、伊莱·布朗斯坦（Eli Bronstein）、卢一任（Yiren Lu

- [85]
  郭俊良（Junliang Guo）、叶阳（Yang Ye）、何天宇（Tianyu He）、吴昊宇（Haoyu Wu）、姜雨舒（Yushu Jiang）、蒂姆·皮尔斯（Tim Pearce）、边江（Jiang Bian）。
  **Mineworld：一个基于《我的世界》（Minecraft）的实时开源交互式世界模型（Interactive World Model）** 。
  arXiv 预印本 arXiv:2504.08388，2025年。
- [86]
  郭旭阳（Xuyang Guo）、霍佳岩（Jiayan Huo）、石珍梅（Zhenmei Shi）、宋钊（Zhao Song）、张嘉豪（Jiahao Zhang）、赵佳乐（Jiale Zhao）。
  **T2VPhysBench：一个用于文本到视频生成（Text-to-Video Generation）中物理一致性（Physical Consistency）的第一性原理基准（First-principles Benchmark）** 。
  arXiv 预印本 arXiv:2505.00337，2025年。
- [87]
  韦斯·格尼（Wes Gurnee）、马克斯·泰格马克（Max Tegmark）。
  **语言模型表征空间与时间（Language Models Represent Space and Time）** 。
  载于《第十二届国际学习表征会议（The Twelfth International Conference on Learning Representations）》，2024年。
- [88]
  大卫·哈（David Ha）、于尔根·施密德胡伯（Jürgen Schmidhuber）。
  **循环世界模型促进策略进化（Recurrent World Models Facilitate Policy Evolution）** 。
  《神经信息处理系统进展（Advances in Neural Information Processing Systems）》，第31卷，2018年。
- [89]
  大卫·哈（David Ha）、于尔根·施密德胡伯（Jürgen Schmidhuber）。
  **世界模型（World Models）** 。
  arXiv 预印本 arXiv:1803.10122，2018年。
- [90]
  河世勋（Sehoon Ha）、徐鹏（Peng Xu）、谭振宇（Zhenyu Tan）、谢尔盖·莱文（Sergey Levine）、谭杰（Jie Tan）。
  **以最少人力在现实世界中学习行走（Learning to Walk in the Real World with Minimal Human Effort）** 。
  arXiv 预印本 arXiv:2002.08550，2020年。
- [91]
  丹尼尔·哈夫纳（Danijar Hafner）、蒂莫西·利利克雷普（Timothy Lillicrap）、吉米·巴（Jimmy Ba）、穆罕默德·诺鲁齐（Mohammad Norouzi）。
  **梦想控制：通过潜在想象学习行为（Dream to Control: Learning Behaviors by Latent Imagination）** 。
  arXiv 预印本 arXiv:1912.01603，2019年。
- [92]
  丹尼尔·哈夫纳（Danijar Hafner）、蒂莫西·利利克雷普（Timothy Lillicrap）、伊恩·菲舍尔（Ian Fischer）、鲁本·比列加斯（Ruben Villegas）、大卫·哈（David Ha）、李洪燮（Honglak Lee）、詹姆斯·戴维森（James Davidson）。
  **从像素中学习用于规划的潜在动态（Learning Latent Dynamics for Planning from Pixels）** 。
  载于《国际机器学习会议（International Conference on Machine Learning）》，第2555–2565页。PMLR，2019年。
- [93]
  丹尼尔·哈夫纳（Danijar Hafner）、蒂莫西·利利克雷普（Timothy Lillicrap）、穆罕默德·诺鲁齐（Mohammad Norouzi）、吉米·巴（Jimmy Ba）。
  **用离散世界模型精通雅达利游戏（Mastering Atari with Discrete World Models）** 。
  arXiv 预印本 arXiv:2010.02193，2020年。
- [94]
  丹尼尔·哈夫纳（Danijar Hafner）、尤尔吉斯·帕斯库尼斯（Jurgis Pasukonis）、吉米·巴（Jimmy Ba）、蒂莫西·利利克雷普（Timothy Lillicrap）。
  **通过世界模型掌握多样化控制任务（Mastering Diverse Control Tasks through World Models）** 。
  《自然（Nature）》，第1–7页，2025年。
- [95]
  尼克拉斯·汉森（Nicklas Hansen）、苏浩（Hao Su）、王晓龙（Xiaolong Wang）。
  **TD-MPC2：用于连续控制的可扩展、鲁棒的世界模型（TD-MPC2: Scalable, Robust World Models for Continuous Control）** 。
  载于《第十二届国际学习表征会议（The Twelfth International Conference on Learning Representations）》，2024年。
- [96]
  何浩然（Haoran He）、白晨佳（Chenjia Bai）、潘凌（Ling Pan）、张伟楠（Weinan Zhang）、赵斌（Bin Zhao）、李学龙（Xuelong Li）。
  **通过大规模无动作视频预训练学习可执行的离散扩散策略（Learning an Actionable Discrete Diffusion Policy via Large-scale Actionless Video Pre-training）** 。
  载于《第三十八届神经信息处理系统年会（The Thirty-eighth Annual Conference on Neural Information Processing Systems）》，2024年。
- [97]
  何俊才（Juncai He）、许进超（Jinchao Xu）。
  **MGNet：多重网格与卷积神经网络的统一框架（MGNet: A Unified Framework of Multigrid and Convolutional Neural Network）** 。
  《中国科学：数学（Science China Mathematics）》，第62卷，第7期，第1331–1354页，2019年5月。
- [98]
  罗伯托·亨舍尔（Roberto Henschel）、列翁·哈恰特良（Levon Khachatryan）、哈伊克·波戈相（Hayk Poghosyan）、丹尼尔·海拉佩特扬（Daniil Hayrapetyan）、瓦赫拉姆·塔德沃相（Vahram Tadevosyan）、汪张扬（Zhangyang Wang）、尚特·纳瓦萨尔迪安（Shant Navasardyan）、石慧敏（Humphrey Shi）。
  **StreamingT2V：从文本生成一致、动态且可扩展的长视频（StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text）** 。
  载于《计算机视觉与模式识别会议论文集（Proceedings of the Computer Vision and Pattern Recognition Conference）》，第2568–2577页，2025年。
- [99]
  乔纳森·何（Jonathan Ho）、阿贾伊·贾因（Ajay Jain）、彼得·阿比尔（Pieter Abbeel）。
  **去噪扩散概率模型（Denoising Diffusion Probabilistic Models）** 。
  《神经信息处理系统进展（Advances in Neural Information Processing Systems）》，第33卷，第6840–6851页，2020年。
- [100]
  塞巴斯蒂安·赫费尔（Sebastian Höfer）、科斯塔斯·贝克里斯（Kostas Bekris）、安库尔·汉达（Ankur Handa）、胡安·卡米洛·甘博亚（Juan Camilo Gamboa）、梅丽莎·莫齐菲安（Melissa Mozifian）、弗洛里安·戈莱莫（Florian Golemo）、克里斯·阿特克森（Chris Atkeson）、迪特尔·福克斯（Dieter Fox）、肯·戈德堡（Ken Goldberg）、约翰·伦纳德（John Leonard）等。
  **机器人学与自动化中的仿真到现实：应用与挑战（Sim2Real in Robotics and Automation: Applications and Challenges）** 。
  《IEEE自动化科学与工程汇刊（IEEE Transactions on Automation Science and Engineering）》，第18卷，第2期，第398–400页，2021年。
- [101]
  安东尼·胡（Anthony Hu）、劳埃德·罗素（Lloyd Russell）、杨哈德森（Hudson Yeo）、扎克·穆雷兹（Zak Murez）、乔治·费多谢耶夫（George Fedoseev）、亚历克斯·肯德尔（Alex Kendall）、杰米·肖顿（Jamie Shotton）、詹卢卡·科拉多（Gianluca Corrado）。
  **Gaia-1：一个用于自动驾驶的生成式世界模型（Gaia-1: A Generative World Model for Autonomous Driving）** 。
  arXiv 预印本 arXiv:2309.17080，2023年。
- [102]
  安东尼·胡（Anthony Hu）、劳埃德·罗素（Lloyd Russell）、杨哈德森（Hudson Yeo）、扎克·穆雷兹（Zak Murez）、乔治·费多谢耶夫（George Fedoseev）、亚历克斯·肯德尔（Alex Kendall）、杰米·肖顿（Jamie Shotton）、詹卢卡·科拉多（Gianluca Corrado）。
  \*\*Gaia-1：一个用于自动驾驶的生成式世界模型（Gaia-1: A Generative

- [104]
  胡一涵（Yihan Hu）、杨嘉志（Jiazhi Yang）、陈力（Li Chen）、李科宇（Keyu Li）、司马崇浩（Chonghao Sima）、朱曦洲（Xizhou Zhu）、柴思琪（Siqi Chai）、杜森垚（Senyao Du）、林天威（Tianwei Lin）、王文海（Wenhai Wang）、卢乐为（Lewei Lu）、贾晓松（Xiaosong Jia）、刘强（Qiang Liu）、戴季峰（Jifeng Dai）、乔宇（Yu Qiao）、李宏阳（Hongyang Li）。
  Planning-oriented autonomous driving（面向规划的自动驾驶），2023。
- [105]
  胡宇成（Yucheng Hu）、郭彦江（Yanjiang Guo）、王鹏超（Pengchao Wang）、陈晓宇（Xiaoyu Chen）、王彦人（Yen-Jen Wang）、张建科（Jianke Zhang）、Koushil Sreenath、陆超超（Chaochao Lu）、陈建宇（Jianyu Chen）。
  Video prediction policy: A generalist robot policy with predictive visual representations（视频预测策略：一种具备预测性视觉表征的通用机器人策略）。
  arXiv preprint arXiv:2412.14803，2024。
- [106]
  华璞（Pu Hua）、刘明欢（Minghuan Liu）、Annabella Macaluso、林云峰（Yunfeng Lin）、张伟楠（Weinan Zhang）、徐华哲（Huazhe Xu）、王立睿（Lirui Wang）。
  Gensim2: Scaling robot data generation with multi-modal and reasoning llms（Gensim2：利用多模态与推理大型语言模型扩展机器人数据生成）。
  arXiv preprint arXiv:2410.03645，2024。
- [107]
  黄文龙（Wenlong Huang）、夏飞（Fei Xia）、Ted Xiao、Harris Chan、Jacky Liang、Pete Florence、Andy Zeng、Jonathan Tompson、Igor Mordatch、Yevgen Chebotar 等。
  Inner monologue: Embodied reasoning through planning with language models（内心独白：通过语言模型规划实现的具身推理）。
  arXiv preprint arXiv:2207.05608，2022。
- [108]
  黄彦君（Yanjun Huang）、杜佳桐（Jiatong Du）、杨子如（Ziru Yang）、周泽伟（Zewei Zhou）、张林（Lin Zhang）、陈红（Hong Chen）。
  A survey on trajectory-prediction methods for autonomous driving（自动驾驶轨迹预测方法综述）。
  IEEE Transactions on Intelligent Vehicles（IEEE智能车辆汇刊），7(3):652–674，2022。
- [109]
  黄志宇（Zhiyu Huang）、莫晓宇（Xiaoyu Mo）、吕辰（Chen Lv）。
  Multi-modal motion prediction with transformer-based neural network for autonomous driving（基于Transformer神经网络的自动驾驶多模态运动预测）。
  收录于 2022 International Conference on Robotics and Automation (ICRA)（2022年机器人与自动化国际会议），第2605–2611页。IEEE，2022。
- [110]
  黄子骐（Ziqi Huang）、何一楠（Yinan He）、于佳硕（Jiashuo Yu）、张帆（Fan Zhang）、司晨阳（Chenyang Si）、蒋宇明（Yuming Jiang）、张元翰（Yuanhan Zhang）、吴天行（Tianxing Wu）、金庆阳（Qingyang Jin）、Nattapol Chanpaisit、王耀辉（Yaohui Wang）、陈新元（Xinyuan Chen）、王利民（Limin Wang）、林达华（Dahua Lin）、乔宇（Yu Qiao）、刘子纬（Ziwei Liu）。
  Vbench: Comprehensive benchmark suite for video generative models（Vbench：视频生成模型的综合基准套件）。
  收录于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)（IEEE/CVF计算机视觉与模式识别会议论文集），第21807–21818页，2024年6月。
- [111]
  Hakan Inan、Kartikeya Upasani、迟剑锋（Jianfeng Chi）、Rashi Rungta、Krithika Iyer、毛宇宁（Yuning Mao）、Michael Tontchev、胡庆（Qing Hu）、Brian Fuller、Davide Testuggine 等。
  Llama guard: Llm-based input-output safeguard for human-ai conversations（Llama Guard：基于大型语言模型的人机对话输入输出安全防护）。
  arXiv preprint arXiv:2312.06674，2023。
- [112]
  Anna A Ivanova、Aalok Sathe、Benjamin Lipkin、Unnathi Kumar、Setayesh Radkani、Thomas H Clark、Carina Kauf、胡珍妮弗（Jennifer Hu）、RT Pramod、Gabriel Grand 等。
  Elements of world knowledge (ewok): A cognition-inspired framework for evaluating basic world knowledge in language models（世界知识要素（EWOK）：一个受认知启发的评估语言模型中基本世界知识的框架）。
  arXiv preprint arXiv:2405.09605，2024。
- [113]
  Yash Jain、Anshul Nasery、Vibhav Vineet、Harkirat Behl。
  Peekaboo: Interactive video generation via masked-diffusion（Peekaboo：通过掩码扩散实现的交互式视频生成）。
  收录于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（IEEE/CVF计算机视觉与模式识别会议论文集），第8079–8088页，2024。
- [114]
  Eric Jang、Alex Irpan、Mohi Khansari、Daniel Kappler、Frederik Ebert、Corey Lynch、Sergey Levine、Chelsea Finn。
  Bc-z: Zero-shot task generalization with robotic imitation learning（BC-Z：通过机器人模仿学习实现零样本任务泛化）。
  收录于 Conference on Robot Learning（机器人学习会议），第991–1002页。PMLR，2022。
- [115]
  Joel Jang、Seonghyeon Ye、林宗宇（Zongyu Lin）、项建楠（Jiannan Xiang）、Johan Bjorck、方宇（Yu Fang）、胡丰源（Fengyuan Hu）、Spencer Huang、Kaushil Kundalia、林彦辰（Yen-Chen Lin）等。
  Dreamgen: Unlocking generalization in robot learning through video world models（Dreamgen：通过视频世界模型解锁机器人学习中的泛化能力）。
  arXiv preprint arXiv:2505.12705，2025。
- [116]
  Michael Janner、Justin Fu、Marvin Zhang、Sergey Levine。
  When to trust your model: Model-based policy optimization（何时信任你的模型：基于模型的策略优化）。
  Advances in neural information processing systems（神经信息处理系统进展），32，2019。
- [117]
  Michael Janner、李启阳（Qiyang Li）、Sergey Levine。
  Offline reinforcement learning as one big sequence modeling problem（离线强化学习作为一个大型序列建模问题）。
  Advances in neural information processing systems（神经信息处理系统进展），34:1273–1286，2021。
- [118]
  纪佳睿（Jiarui Ji）、李洋（Yang Li）、刘洪涛（Hongtao Liu）、杜志诚（Zhicheng Du）、魏哲伟（Zhewei Wei）、沈蔚然（Weiran Shen）、齐琪（Qi Qi）、林衍凯（Yankai Lin）。
  Srap-agent: Simulating and optimizing scarce resource allocation policy with llm-based agent（SRAP-Agent：利用基于大型语言模型的智能体模拟与优化稀缺资源分配策略）。
  arXiv preprint arXiv:2410.14152，2024。

- [119]
  Chiyu Max Jiang, Andre Cornman, Cheolho Park, Ben Sapp, Yin Zhou, and Dragomir
  Anguelov.
  Motiondiffuser: Controllable multi-agent motion prediction using
  diffusion, 2023.
- [120]
  Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue
  Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, et al.
  Enerverse-ac: Envisioning embodied environments with action
  condition.
  arXiv preprint arXiv:2505.09723, 2025.
- [121]
  Charles Jin and Martin Rinard.
  Emergent representations of program semantics in language models
  trained on programs.

- [122]
  菲利普·尼古拉斯·约翰逊-莱尔德（Philip Nicholas Johnson-Laird）。
  《心智模型：走向语言、推理与意识的认知科学（Mental models: Towards a cognitive science of language, inference, and consciousness）》。
  第 6 号。哈佛大学出版社（Harvard University Press），1983 年。
- [123]
  格雷戈里·卡恩（Gregory Kahn）、亚当·维拉弗洛尔（Adam Villaflor）、维切尔·庞（Vitchyr Pong）、彼得·阿比尔（Pieter Abbeel）和谢尔盖·莱文（Sergey Levine）。
  《用于避撞的不确定性感知强化学习（Uncertainty-aware reinforcement learning for collision avoidance）》。
  arXiv 预印本 arXiv:1702.01182，2017 年。
- [124]
  苏巴拉奥·坎巴姆帕蒂（Subbarao Kambhampati）、卡尔提克·瓦尔米卡姆（Karthik Valmeekam）、关林（Lin Guan）、穆迪特·维尔马（Mudit Verma）、卡亚·斯特奇利（Kaya Stechly）、西丹特·巴姆布里（Siddhant Bhambri）、卢卡斯·保罗·萨尔迪特（Lucas Paul Saldyt）和阿尼尔·B·穆尔蒂（Anil B Murthy）。
  《立场：大型语言模型无法规划，但能在 LLM-modulo 框架中辅助规划（Position: Llms can’t plan, but can help planning in llm-modulo frameworks）》。
  载于第四十一届国际机器学习会议（Forty-first International Conference on Machine Learning），2024 年。

- [125]
  安西·卡内维斯托（Anssi Kanervisto）、戴夫·比格内尔（Dave Bignell）、温琳达·依琳（Linda Yilin Wen）、马丁·格雷森（Martin Grayson）、拉卢卡·乔治斯库（Raluca Georgescu）、塞尔吉奥·瓦尔卡塞尔·马库阿（Sergio Valcarcel Macua）、陈善正（Shan Zheng Tan）、塔比什·拉希德（Tabish Rashid）、蒂姆·皮尔斯（Tim Pearce）、曹宇涵（Yuhan Cao）等。
  《面向游戏玩法构思的世界与人类行为模型（World and human action models towards gameplay ideation）》。
  《自然（Nature）》，638(8051):656–663，2025 年。
- [126]
  康炳毅（Bingyi Kang）、岳洋（Yang Yue）、卢锐（Rui Lu）、林志杰（Zhijie Lin）、赵阳（Yang Zhao）、王凯欣（Kaixin Wang）、黄高（Gao Huang）和冯佳时（Jiashi Feng）。
  《视频生成离世界模型还有多远：物理定律视角（How far is video generation from world model: A physical law perspective）》。
  arXiv 预印本 arXiv:2411.02385，2024 年。
- [127]
  川崎笃（Atsushi Kawasaki）和关彰仁（Akihito Seki）。
  《利用车辆与车道间的几何关系进行城市环境多模态轨迹预测（Multimodal trajectory predictions for urban environments using geometric relationships between a vehicle and lanes）》。
  载于 2020 年 IEEE 机器人与自动化国际会议（2020 IEEE International Conference on Robotics and Automation (ICRA)），第 9203–9209 页。IEEE，2020 年。
- [128]
  基利安·克莱伯格（Kilian Kleeberger）、理查德·博尔曼（Richard Bormann）、沃纳·克劳斯（Werner Kraus）和马可·F·胡贝尔（Marco F Huber）。
  《基于学习的机器人抓取综述（A survey on learning-based robotic grasping）》。
  《机器人学现状报告（Current Robotics Reports）》，1:239–249，2020 年。
- [129]
  埃里克·科尔夫（Eric Kolve）、鲁兹贝·莫塔吉（Roozbeh Mottaghi）、韩文森（Winson Han）、埃利·范德比尔特（Eli VanderBilt）、卢卡·韦斯（Luca Weihs）、阿尔瓦罗·赫拉斯蒂（Alvaro Herrasti）、马特·戴特克（Matt Deitke）、基亚娜·埃赫萨尼（Kiana Ehsani）、丹尼尔·戈登（Daniel Gordon）、朱玉可（Yuke Zhu）等。
  《Ai2-thor：用于视觉人工智能的交互式 3D 环境（Ai2-thor: An interactive 3d environment for visual ai）》。
  arXiv 预印本 arXiv:1712.05474，2017 年。
- [130]
  米哈尔·科辛斯基（Michal Kosinski）。
  《在心智理论任务中评估大型语言模型（Evaluating large language models in theory of mind tasks）》。
  《美国国家科学院院刊（Proceedings of the National Academy of Sciences）》，121(45):e2405460121，2024 年。
- [131]
  巴兹尔·库瓦里塔基斯（Basil Kouvaritakis）和马克·坎农（Mark Cannon）。
  《模型预测控制（Model predictive control）》。
  瑞士：施普林格国际出版（Springer International Publishing），38:13–56，2024 年。

2016.

- [132]
  亚历克斯·克里泽夫斯基（Alex Krizhevsky）、伊利亚·苏茨克弗（Ilya Sutskever）和杰弗里·E·辛顿（Geoffrey E Hinton）。
  使用深度卷积神经网络进行 ImageNet 分类。
  神经信息处理系统进展，25，2012。
- [133]
  快手（Kuaishou）。
  Kling AI：下一代人工智能创意工作室。
  [https://www.klingai.com/global/](https://www.klingai.com/global/)，2024。
  （访问于 2025 年 6 月 5 日）。
- [134]
  奥农·库马尔（Aounon Kumar）、奇拉格·阿加瓦尔（Chirag Agarwal）、苏拉杰·斯里尼瓦斯（Suraj Srinivas）、索海尔·费齐（Soheil Feizi）和希马·拉卡茹（Hima Lakkaraju）。
  认证大型语言模型对抗对抗性提示的安全性。
  arXiv 预印本 arXiv:2309.02705，2023。
- [135]
  阿希什·库马尔（Ashish Kumar）、傅子鹏（Zipeng Fu）、迪帕克·帕塔克（Deepak Pathak）和吉滕德拉·马利克（Jitendra Malik）。
  RMA：腿式机器人的快速运动适应。
  arXiv 预印本 arXiv:2107.04034，2021。
- [136]
  瓦伦·拉维·库马尔（Varun Ravi Kumar）、森蒂尔·约加马尼（Senthil Yogamani）、哈泽姆·拉希德（Hazem Rashed）、加内什·西斯图（Ganesh Sistu）、克里斯蒂安·维特（Christian Witt）、伊莎贝尔·梁（Isabelle Leang）、斯特凡·米尔茨（Stefan Milz）和帕特里克·梅德（Patrick Mäder）。
  Omnidet：基于环视摄像头的自动驾驶多任务视觉感知网络，2023。
- [137]
  塔纳德·库鲁塔赫（Thanard Kurutach）、伊格纳西·克拉韦拉（Ignasi Clavera）、段岩（Yan Duan）、阿维夫·塔马尔（Aviv Tamar）和彼得·阿比尔（Pieter Abbeel）。
  模型集成信赖域策略优化。
  arXiv 预印本 arXiv:1802.10592，2018。
- [138]
  兰晓冲（Xiaochong Lan）、冯杰（Jie Feng）、雷佳欢（Jiahuan Lei）、石鑫磊（Xinlei Shi）和李勇（Yong Li）。
  LocalGPT：面向本地生活服务的大型语言模型基准测试与推进。
  发表于第 31 届 ACM SIGKDD 知识发现与数据挖掘会议 V. 2，第 4566–4577 页，2025。
- [139]
  兰晓冲（Xiaochong Lan）、冯杰（Jie Feng）、孙一舟（Yizhou Sun）、高晨（Chen Gao）、雷佳欢（Jiahuan Lei）、石鑫磊（Xinlei Shi）、罗恒亮（Hengliang Luo）和李勇（Yong Li）。
  使用大型语言模型进行开放集生活需求预测。
  计算语言学协会 2025 年 ACL 会议发现，2025。
- [140]
  杨立昆（Yann LeCun）。
  迈向自主机器智能之路 版本 0.9. 2，2022-06-27。
  开放评论，62(1):1–62，2022。
- [141]
  杨立昆（Yann LeCun）、莱昂·博图（Léon Bottou）、约书亚·本吉奥（Yoshua Bengio）和帕特里克·哈夫纳（Patrick Haffner）。
  基于梯度的学习在文档识别中的应用。
  IEEE 会刊，86(11):2278–2324，1998。
- [142]
  皮埃尔·莱维（Pierre Lévy）。
  集体智慧：人类在网络空间中的新兴世界。
  珀尔修斯图书，1997。
- [143]
  李嘉昂（Jiaang Li）、约瓦·凯门切吉耶娃（Yova Kementchedjhieva）、康斯坦扎·菲耶罗（Constanza Fierro）和安德斯·瑟高（Anders Søgaard）。
  视觉和语言模型共享概念吗？一项向量空间对齐研究。
  计算语言学协会汇刊，12:1232–1249，2024。
- [144]
  李佳璐（Jialu Li）、李远真（Yuanzhen Li）、尼尔·瓦德瓦（Neal Wadhwa）、耶尔·普里奇（Yael Pritch）、大卫·E·雅各布斯（David E Jacobs）、迈克尔·鲁宾斯坦（Michael Rubinstein）、莫希特·班萨尔（Mohit Bansal）和纳塔尼尔·鲁伊斯（Nataniel Ruiz）。
  Unbounded：一个生成式的无限角色生活模拟游戏。
  arXiv 预印本 arXiv:2410.18975，2024。
- [145]
  李林灿（Lincan Li）、邵伟（Wei Shao）、董伟（Wei Dong）、田一骏（Yijun Tian）、张启明（Qiming Zhang）、杨凯翔（Kaixiang Yang）和张文杰（Wenjie Zhang）。
  自动驾驶中以数据为中心的演进：大数据系统、数据挖掘和闭环技术综合综述。
  arXiv 预印本 arXiv:2401.12888，2024。
- [146]
  李玲（Ling Li）、周尧（Yao Zhou）、梁宇轩（Yuxuan Liang）、曾福基（Fugee Tsung）和魏嘉恒（Jiaheng Wei）。
  通过推理进行识别：利用大型视觉语言模型增强图像地理定位。
  arXiv 预印本 arXiv:2506.14674，2025。
- [147]
  李曼玲（Manling Li）、赵诗雨（Shiyu Zhao）、王启能（Qineng Wang）、王康睿（Kangrui Wang）、周宇（Yu Zhou）、桑贾纳·斯里瓦斯塔瓦（Sanjana Srivastava）、杰姆·格克门（Cem Gokmen）、托尼·李（Tony Lee）、李艾伦·李（Erran Li Li）、张若涵（Ruohan Zhang）等。
  具身智能体接口：为具身决策制定基准测试大型语言模型。
  神经信息处理系统进展，37:100428–100534，2024。
- [148]
  李念（Nian Li）、高晨（Chen Gao）、李明宇（Mingyu Li）、李勇（Yong Li）和廖庆敏（Qingmin Liao）。
  EconAgent：用于模拟宏观经济活动的大型语言模型赋能智能体。
  发表于第 62 届计算语言学协会年会（第 1 卷：长论文），第 15523–15536 页，2024。
- [149]
  李钦斌（Qinbin Li）、洪俊元（Junyuan Hong）、谢楚林（Chulin Xie）、陈杰夫（Jeffrey Tan）、辛瑞秋（Rachel Xin）、侯俊毅（Junyi Hou）、尹泽维尔（Xavier Yin）、王准（Zhun Wang）、丹·亨德里克斯（Dan Hendrycks）、王张洋（Zhangyang Wang）等。
  LLM-PBE：评估大型语言模型中的数据隐私。
  arXiv 预印本 arXiv:2408.12787，2024。

- [150]
  李全意（Quanyi Li）、彭正浩（Zhenghao Peng）、冯岚（Lan Feng）、张启航（Qihang Zhang）、薛正海（Zhenghai Xue）、周博磊（Bolei Zhou）。
  **Metadrive：为可泛化的强化学习组合多样化驾驶场景（Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning）** 。
  《IEEE 模式分析与机器智能汇刊（IEEE transactions on pattern analysis and machine intelligence）》，45(3):3461–3475，2022。
- [151]
  李宇霄（Yuxiao Li）、埃里克·J·米肖（Eric J. Michaud）、大卫·D·贝克（David D. Baek）、约书亚·恩格斯（Joshua Engels）、孙晓晴（Xiaoqing Sun）、马克斯·泰格马克（Max Tegmark）。
  **概念几何：稀疏自编码器特征结构（The geometry of concepts: Sparse autoencoder feature structure）** ，2024。
- [152]
  李志奇（Zhiqi Li）、王文海（Wenhai Wang）、李宏阳（Hongyang Li）、谢恩泽（Enze Xie）、司马崇浩（Chonghao Sima）、卢通（Tong Lu）、乔宇（Yu Qiao）、戴继峰（Jifeng Dai）。
  **Bevformer：通过时空变换器从多摄像头图像学习鸟瞰图表示（Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers）** 。
  arXiv 预印本 arXiv:2203.17270，2022。
- [153]
  廖悦（Yue Liao）、周鹏飞（Pengfei Zhou）、黄思远（Siyuan Huang）、杨东林（Donglin Yang）、陈胜聪（Shengcong Chen）、蒋雨欣（Yuxin Jiang）、胡越（Yue Hu）、蔡敬斌（Jingbin Cai）、刘思（Si Liu）、罗建兰（Jianlan Luo）等。
  **Genie Envisioner：用于机器人操作的统一世界基础平台（Genie envisioner: A unified world foundation platform for robotic manipulation）** 。
  arXiv 预印本 arXiv:2508.05635，2025。
- [154]
  杰西·林（Jessy Lin）、杜雨晴（Yuqing Du）、奥利维亚·沃特金斯（Olivia Watkins）、丹尼尔·哈夫纳（Danijar Hafner）、皮特·阿比尔（Pieter Abbeel）、丹·克莱因（Dan Klein）、安卡·德拉甘（Anca Dragan）。
  **学习用语言建模世界（Learning to model the world with language）** ，2024。
- [155]
  林凯文（Kevin Lin）、克里斯托弗·阿吉亚（Christopher Agia）、时木松（Toki Migimatsu）、马可·帕沃内（Marco Pavone）、珍妮特·博格（Jeannette Bohg）。
  **Text2motion：从自然语言指令到可行计划（Text2motion: From natural language instructions to feasible plans）** 。
  《自主机器人（Autonomous Robots）》，47(8):1345–1365，2023。
- [156]
  刘浩（Hao Liu）、严威尔逊（Wilson Yan）、马泰·扎哈里亚（Matei Zaharia）、皮特·阿比尔（Pieter Abbeel）。
  **基于百万长度视频和语言及环注意力（RingAttention）的世界模型（World model on million-length video and language with ringattention）** 。
  arXiv 预印本 arXiv:2402.08268，2024。
- [157]
  刘昊天（Haotian Liu）、李春元（Chunyuan Li）、吴庆阳（Qingyang Wu）、李永宰（Yong Jae Lee）。
  **视觉指令微调（Visual instruction tuning）** 。
  《神经信息处理系统进展（Advances in neural information processing systems）》，36，2024。
- [158]
  刘少伟（Shaowei Liu）、任中正（Zhongzheng Ren）、索拉布·古普塔（Saurabh Gupta）、王申龙（Shenlong Wang）。
  **Physgen：基于刚体物理的图像到视频生成（Physgen: Rigid-body physics-grounded image-to-video generation）** 。
  载于《欧洲计算机视觉会议（European Conference on Computer Vision）》，第 360–378 页。施普林格（Springer），2024。
- [159]
  刘鑫浩（Xinhao Liu）、李金桐（Jintong Li）、蒋一诚（Yicheng Jiang）、尼兰詹·苏贾伊（Niranjan Sujay）、杨志成（Zhicheng Yang）、张珏晓（Juexiao Zhang）、约翰·阿巴内斯（John Abanes）、张静（Jing Zhang）、冯晨（Chen Feng）。
  **Citywalker：从网络规模视频学习具身城市导航（Citywalker: Learning embodied urban navigation from web-scale videos）** 。
  载于《计算机视觉与模式识别会议论文集（Proceedings of the Computer Vision and Pattern Recognition Conference）》，第 6875–6885 页，2025。
- [160]
  刘志涵（Zhihan Liu）、胡浩（Hao Hu）、张申傲（Shenao Zhang）、郭弘毅（Hongyi Guo）、柯书琪（Shuqi Ke）、刘博一（Boyi Liu）、王昭然（Zhaoran Wang）。
  **为未来推理，为当下行动：具有可证明样本效率的自主 LLM 智能体原理性框架（Reason for future, act for now: A principled framework for autonomous llm agents with provable sample efficiency）** ，2024。
- [161]
  龙宇星（Yuxing Long）、李晓琪（Xiaoqi Li）、蔡文喆（Wenzhe Cai）、董豪（Hao Dong）。
  **动前讨论：通过多专家讨论实现视觉语言导航（Discuss before moving: Visual language navigation via multi-expert discussions）** 。
  载于《2024 年 IEEE 机器人与自动化国际会议（2024 IEEE International Conference on Robotics and Automation, ICRA）》，第 17380–17387 页。IEEE，2024。
- [162]
  巴勃罗·阿尔瓦雷斯·洛佩斯（Pablo Alvarez Lopez）、迈克尔·贝里施（Michael Behrisch）、劳拉·比克尔-瓦尔茨（Laura Bieker-Walz）、雅各布·埃德曼（Jakob Erdmann）、云-庞·弗勒特罗德（Yun-Pang Flötteröd）、罗伯特·希尔布里希（Robert Hilbrich）、莱昂哈德·吕肯（Leonhard Lücken）、约翰内斯·鲁梅尔（Johannes Rummel）、彼得·瓦格纳（Peter Wagner）、埃娃玛丽·维森纳（Evamarie Wießner）。
  **使用 SUMO 进行微观交通仿真（Microscopic traffic simulation using sumo）** 。
  载于《第 21 届 IEEE 智能交通系统国际会议（The 21st IEEE International Conference on Intelligent Transportation Systems）》。IEEE，2018。
- [163]
  卢冠星（Guanxing Lu）、贾宝雄（Baoxiong Jia）、李普浩（Puhao Li）、陈一心（Yixin Chen）、王子维（Ziwei Wang）、唐延松（Yansong Tang）、黄思远（Siyuan Huang）。
  **GWM：迈向用于机器人操作的可扩展高斯世界模型（Gwm: Towards scalable gaussian world models for robotic manipulation）** 。
  arXiv 预印本 arXiv:2508.17600，2025。
- [164]
  罗凡明（Fan-Ming Luo）、徐天（Tian Xu）、赖航（Hang Lai）、陈雄辉（Xiong-Hui Chen）、张伟楠（Weinan Zhang）、俞扬（Yang Yu）。
  **基于模型的强化学习综述（A survey on model-based reinforcement learning）** 。
  《中国科学：信息科学（Science China Information Sciences）》，67(2):121101，2024。
- [165]
  罗玉平（Yuping Luo）、徐华哲（Huazhe Xu）、李远志（Yuanzhi Li）、田元冬（Yuandong Tian）、特雷弗·达雷尔（Trevor Darrell）、马腾宇（Tengyu Ma）。
  **具有理论保证的基于模型的深度强化学习算法框架（Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees）** 。
  arXiv 预印本 arXiv:1807.03858，2018。
- [166]
  麦新基（Xinji Mai）、陶增（Zeng Tao）、林俊雄（Junxiong Lin）、王浩然（Haoran Wang）、常阳（Yang Chang）、康艳兰（Yanlan Kang）、王岩（Yan Wang）、张文强（Wenqiang Zhang）。
  **从高效多模态模型到世界模型：综述（From efficient multimodal models to world models: A survey）** 。
  arXiv 预印本 arXiv:2407

- [167]
  Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra,
  Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud,
  et al.
  **OpenEQA: Embodied question answering in the era of foundation models** （OpenEQA：基础模型时代的具身问答）。
  In Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, pages 16488–16498, 2024.
- [168]
  Thomas W Malone.
  **Superminds: The surprising power of people and computers thinking together** （超级心智：人与计算机共同思考的惊人力量）。
  Little, Brown Spark, 2018.
- [169]
  Thomas W Malone and Michael Bernstein.
  **Handbook of collective intelligence** （集体智能手册）。
  MIT press, 2015.
- [170]
  Rohin Manvi, Samar Khanna, Marshall Burke, David Lobell, and Stefano Ermon.
  **Large language models are geographically biased** （大型语言模型存在地理偏见）。
  arXiv preprint arXiv:2402.02680, 2024.
- [171]
  Rohin Manvi, Samar Khanna, Gengchen Mai, Marshall Burke, David Lobell, and
  Stefano Ermon.
  **GeoLLM: Extracting geospatial knowledge from large language models** （GeoLLM：从大型语言模型中提取地理空间知识）。
  arXiv preprint arXiv:2310.06213, 2023.
- [172]
  Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He,
  Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang.
  **Yume: An interactive world generation model** （Yume：一个交互式世界生成模型）。
  arXiv preprint arXiv:2507.17744, 2025.
- [173]
  Russell Mendonca, Shikhar Bahl, and Deepak Pathak.
  **Structured world models from human videos** （从人类视频中构建结构化世界模型）。
  arXiv preprint arXiv:2308.10901, 2023.
- [174]
  Chen Min, Dawei Zhao, Liang Xiao, Yiming Nie, and Bin Dai.
  **Uniworld: Autonomous driving pre-training via world models** （Uniworld：通过世界模型进行自动驾驶预训练）。
  arXiv preprint arXiv:2308.07234, 2023.
- [175]
  Marvin Minsky.
  **A framework for representing knowledge** （一种知识表示框架），1974.
- [176]
  Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness,
  Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg
  Ostrovski, et al.
  **Human-level control through deep reinforcement learning** （通过深度强化学习实现人类水平的控制）。
  nature, 518(7540):529–533, 2015.
- [177]
  Thomas M Moerland, Joost Broekens, Aske Plaat, and Catholijn M Jonker.
  **A0c: Alpha zero in continuous action space** （A0c：连续动作空间中的 Alpha Zero）。
  arXiv preprint arXiv:1805.09613, 2018.
- [178]
  Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos.
  **Do generative video models learn physical principles from watching videos?** （生成式视频模型是否通过观看视频学习物理原理？）。
  arXiv preprint arXiv:2501.09038, 2025.
- [179]
  Mikhail Mozikov, Nikita Severin, Valeria Bodishtianu, Maria Glushanina, Ivan
  Nasonov, Daniil Orekhov, Pekhotin Vladislav, Ivan Makovetskiy, Mikhail
  Baklashkin, Vasily Lavrentyev, et al.
  **EAI: Emotional decision-making of LLMs in strategic games and ethical dilemmas** （EAI：大型语言模型在策略游戏和伦理困境中的情感决策）。
  Advances in Neural Information Processing Systems,
  37:53969–54002, 2024.
- [180]
  Fangwen Mu, Lin Shi, Song Wang, Zhuohao Yu, Binquan Zhang, Chenxue Wang,
  Shichao Liu, and Qing Wang.
  **ClarifyGPT: Empowering LLM-based code generation with intention clarification** （ClarifyGPT：通过意图澄清增强基于大型语言模型的代码生成）。
  arXiv preprint arXiv:2310.10996, 2023.
- [181]
  Junho Myung, Nayeon Lee, Yi Zhou, Jiho Jin, Rifki Putri, Dimosthenis Antypas,
  Hsuvas Borkakoty, Eunsu Kim, Carla Perez-Almendros, Abinew Ali Ayele, et al.
  **Blend: A benchmark for LLMs on everyday knowledge in diverse cultures and languages** （Blend：针对多种文化和语言中日常知识的大型语言模型基准）。
  Advances in Neural Information Processing Systems,
  37:78104–78146, 2024.
- [182]
  Anusha Nagabandi, Gregory Kahn, Ronald S Fearing, and Sergey Levine.
  **Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning** （用于基于模型的深度强化学习并结合无模型微调的神经网络动力学）。
  In 2018 IEEE international conference on robotics and automation
  (ICRA), pages 7559–7566. IEEE, 2018.
- [183]
  Nigamaa Nayakanti, Rami Al-Rfou, Aurick Zhou, Kratarth Goel, Khaled S. Refaat,
  and Benjamin Sapp.
  **Wayformer: Motion forecasting via simple & efficient attention networks** （Wayformer：通过简单高效的注意力网络进行运动预测），2022.
- [184]
  Jiquan Ngiam, Benjamin Caine, Vijay Vasudevan, Zhengdong Zhang, Hao-Tien Lewis
  Chiang, Jeffrey Ling, Rebecca Roelofs, Alex Bewley, Chenxi Liu, Ashish
  Venugopal, et al.
  **Scene transformer: A unified multi-task model for behavior prediction and planning** （场景变换器：用于行为预测和规划的统一多任务模型）。
  arXiv preprint arXiv:2106.08417, 2(7), 2021.

- [185]
  Junhyuk Oh, Satinder Singh, and Honglak Lee.
  Value prediction network.
  Advances in neural information processing systems, 30, 2017.
- [186]
  OpenAI.
  Introducing chatgpt.
  [https://openai.com/blog/chatgpt](https://openai.com/blog/chatgpt), 2022.
  (访问于 2025年6月5日).
- [187]
  OpenAI.
  Sora: Creating video from text.
  [https://openai.com/sora](https://openai.com/sora), 2024.
  (访问于 2025年6月5日).
- [188]
  Marios Papachristou and Yuan Yuan.
  Network formation and dynamics among multi-llms.
  arXiv preprint arXiv:2402.10659, 2024.
- [189]
  Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy
  Liang, and Michael S Bernstein.
  Generative agents: Interactive simulacra of human behavior.
  In Proceedings of the 36th Annual ACM Symposium on User
  Interface Software and Technology, pages 1–22, 2023.
- [190]
  Joon Sung Park, Lindsay Popowski, Carrie Cai, Meredith Ringel Morris, Percy
  Liang, and Michael S Bernstein.
  Social simulacra: Creating populated prototypes for social computing
  systems.
  In Proceedings of the 35th Annual ACM Symposium on User
  Interface Software and Technology, pages 1–18, 2022.
- [191]
  Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian
  Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar,
  Jimmy Shi, Stephen Spencer, Jessica Yung, Michael Dennis, Sultan Kenjeyev,
  Shangbang Long, Vlad Mnih, Harris Chan, Maxime Gazeau, Bonnie Li, Fabio
  Pardo, Luyu Wang, Lei Zhang, Frederic Besse, Tim Harley, Anna Mitenkova, Jane
  Wang, Jeff Clune, Demis Hassabis, Raia Hadsell, Adrian Bolton, Satinder
  Singh, and Tim Rocktäschel.

2024.

- [192]
  Sudipta Paul, Amit Roy-Chowdhury, and Anoop Cherian.
  Avlen: Audio-visual-language embodied navigation in 3d environments.
  Advances in Neural Information Processing Systems,
  35:6236–6249, 2022.
- [193]
  Judea Pearl.
  Causal inference in statistics: An overview.

2009.

- [194]
  Tung Phan-Minh, Elena Corina Grigore, Freddy A Boulton, Oscar Beijbom, and
  Eric M Wolff.
  Covernet: Multimodal behavior prediction using trajectory sets.
  In Proceedings of the IEEE/CVF conference on computer vision and
  pattern recognition, pages 14074–14083, 2020.
- [195]
  Jinghua Piao, Yuwei Yan, Jun Zhang, Nian Li, Junbo Yan, Xiaochong Lan, Zhihong
  Lu, Zhiheng Zheng, Jing Yi Wang, Di Zhou, et al.
  Agentsociety: Large-scale simulation of llm-driven generative agents
  advances understanding of human behaviors and society.
  arXiv preprint arXiv:2502.08691, 2025.
- [196]
  Giorgio Piatti, Zhijing Jin, Max Kleiman-Weiner, Bernhard Schölkopf,
  Mrinmaya Sachan, and Rada Mihalcea.
  Cooperate or collapse: Emergence of sustainability behaviors in a
  society of llm agents.
  arXiv preprint arXiv:2404.16698, 2024.
- [197]
  David Premack and Guy Woodruff.
  Does the chimpanzee have a theory of mind?
  Behavioral and brain sciences, 1(4):515–526, 1978.
- [198]
  Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and
  Antonio Torralba.
  Virtualhome: Simulating household activities via programs.
  In Proceedings of the IEEE conference on computer vision and
  pattern recognition, pages 8494–8502, 2018.
- [199]
  Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas.
  Pointnet: Deep learning on point sets for 3d classification and
  segmentation.
  In Proceedings of the IEEE conference on computer vision and
  pattern recognition, pages 652–660, 2017.
- [200]
  Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas.
  Pointnet++: Deep hierarchical feature learning on point sets in a
  metric space, 2017.
- [201]
  Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed
  Elhoseiny, and Bernard Ghanem.
  Pointnext: Revisiting pointnet++ with improved training and scaling
  strategies.
  Advances in neural information processing systems,
  35:23192–23204, 2022.
- [202]
  Shuofei Qiao, Runnan Fang, Ningyu Zhang, Yuqi Zhu, Xiang Chen, Shumin Deng,
  Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen.
  Agent planning with world knowledge model.
  Advances in Neural Information Processing Systems,
  37:114843–114871, 2024.
- [203]
  Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei
  Yin, Xihui Liu, Lu Sheng, Jing Shao, et al.
  Worldsimbench: Towards video generation models as world simulators.
  arXiv preprint arXiv:2410.18072, 2024.
- [204]
  Julian Quevedo, Percy Liang, and Sherry Yang.
  Evaluating robot policies in a world model.
  arXiv preprint arXiv:2506.00613, 2025.
- [205]
  Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
  Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
  Gretchen Krueger, and Ilya Sutskever.
  Learning transferable visual models from natural language
  supervision, 2021.
- [206]
  Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano
  Ermon, and Chelsea Finn.
  Direct preference optimization: Your language model is secretly a
  reward model.
  Advances in Neural Information Processing Systems, 36, 2024.
- [207]
  Aravind Rajeswaran, Igor Mordatch, and Vikash Kumar.
  A game theoretic framework for model based reinforcement learning.
  In International conference on machine learning, pages
  7953–7963. PMLR, 2020.
- [208]
  Md Shohel Rana, Mohammad Nur Nobi, Beddhu Murali, and Andrew H Sung.
  Deepfake detection: A systematic literature review.
  IEEE access, 10:25494–25513, 2022.
- [209]
  Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
  Faster r-cnn: Towards real-time object detection with region proposal
  networks.
  In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett,
  editors, Advances in Neural Information Processing Systems, volume 28.
  Curran Associates, Inc., 2015.
- [210]
  Weiming Ren, Huan Yang, Ge Zhang, Cong Wei, Xinrun Du, Wenhao Huang, and Wenhu
  Chen.

- [210]
  Consisti2v: Enhancing visual consistency for image-to-video generation.
  arXiv preprint arXiv:2402.04324, 2024.
- [211]
  Jonathan Richens, David Abel, Alexis Bellot, and Tom Everitt.
  General agents need world models.
  arXiv preprint arXiv:2506.01622, 2025.
- [212]
  Marc Rigter, Tarun Gupta, Agrin Hilmkil, and Chao Ma.
  Avid: Adapting video diffusion models to world models.
  arXiv preprint arXiv:2410.12822, 2024.
- [213]
  Jonathan Roberts, Timo Lüddecke, Sowmen Das, Kai Han, and Samuel Albanie.
  Gpt4geo: How a language model sees the world’s geography.
  arXiv preprint arXiv:2306.00020, 2023.
- [214]
  Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter.
  Learning to walk in minutes using massively parallel deep reinforcement learning.
  In Conference on Robot Learning, pages 91–100. PMLR, 2022.
- [215]
  Runway.
  Runway gen-2.
  [https://runwayml.com/product](https://runwayml.com/product), 2025.
  (Accessed on 06/05/2025).
- [216]
  Mohammad Reza Samsami, Artem Zholus, Janarthanan Rajendran, and Sarath Chandar.
  Mastering memory tasks with world models.
  arXiv preprint arXiv:2403.04253, 2024.
- [217]
  Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, and Peter Battaglia.
  Learning to simulate complex physics with graph networks.
  In International conference on machine learning, pages 8459–8468. PMLR, 2020.
- [218]
  Maarten Sap, Ronan LeBras, Daniel Fried, and Yejin Choi.
  Neural theory-of-mind? on the limits of social intelligence in large lms.
  arXiv preprint arXiv:2210.13312, 2022.
- [219]
  Nedko Savov, Naser Kazemi, Mohammad Mahdi, Danda Pani Paudel, Xi Wang, and Luc Van Gool.
  Exploration-driven generative interactive environments.
  In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 27597–27607, 2025.
- [220]
  Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al.
  Habitat: A platform for embodied ai research.
  In Proceedings of the IEEE/CVF international conference on computer vision, pages 9339–9347, 2019.
- [221]
  Dhruv Mauria Saxena, Sangjae Bae, Alireza Nakhaei, Kikuo Fujimura, and Maxim Likhachev.
  Driving in dense traffic with model-free reinforcement learning.
  In 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, May 2020.
- [222]
  Thomas C Schelling.
  Dynamic models of segregation.
  Journal of mathematical sociology, 1(2):143–186, 1971.
- [223]
  Ingmar Schubert, Jingwei Zhang, Jake Bruce, Sarah Bechtle, Emilio Parisotto, Martin Riedmiller, Jost Tobias Springenberg, Arunkumar Byravan, Leonard Hasenclever, and Nicolas Heess.
  A generalist dynamics model for control.
  arXiv preprint arXiv:2305.10912, 2023.
- [224]
  John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
  Proximal policy optimization algorithms.
  arXiv preprint arXiv:1707.06347, 2017.
- [225]
  Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki Hirose, and Sergey Levine.
  Gnm: A general navigation model to drive any robot.
  In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 7226–7233. IEEE, 2023.
- [226]
  Yu Shang, Jiansheng Chen, Hangyu Fan, Jingtao Ding, Jie Feng, and Yong Li.
  Urbanworld: An urban world model for 3d city generation.
  arXiv preprint arXiv:2407.11965, 2024.
- [227]
  Yu Shang, Yu Li, Fengli Xu, and Yong Li.
  Defint: A default-interventionist framework for efficient reasoning with hybrid large language models.
  arXiv preprint arXiv:2402.02563, 2024.
- [228]
  Yu Shang, Xin Zhang, Yinzhou Tang, Lei Jin, Chen Gao, Wei Wu, and Yong Li.
  Roboscape: Physics-informed embodied world model.
  NeurIPS, 2025.
- [229]
  Chenyang Shao, Fengli Xu, Bingbing Fan, Jingtao Ding, Yuan Yuan, Meng Wang, and Yong Li.
  Beyond imitation: Generating human mobility from context-aware reasoning with large language models.

arXiv 预印本 arXiv:2402.09836，2024。

- [230]
  Bokui Shen, Fei Xia, Chengshu Li, Roberto Martín-Martín, Linxi Fan,
  Guanzhi Wang, Claudia Pérez-D’Arpino, Shyamal Buch, Sanjana Srivastava,
  Lyne Tchapmi, 等人。
  iGibson 1.0：用于大型真实场景中交互任务的仿真环境。
  收录于 2021 年 IEEE/RSJ 智能机器人与系统国际会议（International Conference on Intelligent Robots and Systems, IROS），第 7520–7527 页。IEEE，2021。
- [231]
  Haochen Shi, Huazhe Xu, Zhiao Huang, Yunzhu Li, and Jiajun Wu。
  Robocraft：利用图网络学习在三维空间中观察、仿真与塑造弹塑性物体。

2024.

- [232]
  Haojun Shi, Suyu Ye, Xinyu Fang, Chuanyang Jin, Layla Isik, Yen-Ling Kuo, and Tianmin Shu.
  **Muma-tom: 多模态多智能体心智理论（Multi-modal multi-agent theory of mind）** 。
  arXiv 预印本 arXiv:2408.12574, 2024.
- [233]
  Hongzhi Shi, Jingtao Ding, Yufan Cao, Li Liu, Yong Li, et al.
  **学习图结构物理机制的符号模型（Learning symbolic models for graph-structured physical mechanism）** 。
  载于 **第十一届国际学习表征会议（The Eleventh International Conference on Learning Representations）** , 2022.
- [234]
  Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele.
  **具有全局意图定位与局部运动细化的运动变换器（Motion transformer with global intention localization and local movement refinement）** 。
  **神经信息处理系统进展（Advances in Neural Information Processing Systems）** ,
  35:6531–6543, 2022.
- [235]
  Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht.
  **Alfworld: 为交互式学习对齐文本与具身环境（Aligning text and embodied environments for interactive learning）** 。
  arXiv 预印本 arXiv:2010.03768, 2020.
- [236]
  David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al.
  **利用深度神经网络与树搜索掌握围棋游戏（Mastering the game of go with deep neural networks and tree search）** 。
  **自然（nature）** , 529(7587):484–489, 2016.
- [237]
  David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al.
  **无需人类知识掌握围棋游戏（Mastering the game of go without human knowledge）** 。
  **自然（nature）** , 550(7676):354–359, 2017.
- [238]
  Laura Smith, Ilya Kostrikov, and Sergey Levine.
  **公园漫步：使用无模型强化学习在20分钟内学会行走（A walk in the park: Learning to walk in 20 minutes with model-free reinforcement learning）** 。
  arXiv 预印本 arXiv:2208.07860, 2022.
- [239]
  Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.
  **通过随机微分方程进行基于分数的生成建模（Score-based generative modeling through stochastic differential equations）** 。
  arXiv 预印本 arXiv:2011.13456, 2020.
- [240]
  James WA Strachan, Dalila Albergo, Giulia Borghini, Oriana Pansardi, Eugenio Scaliti, Saurabh Gupta, Krati Saxena, Alessandro Rufo, Stefano Panzeri, Guido Manzi, et al.
  **测试大型语言模型与人类的心智理论（Testing theory of mind in large language models and humans）** 。
  **自然人类行为（Nature Human Behaviour）** , 页码 1–11, 2024.
- [241]
  Winnie Street, John Oliver Siy, Geoff Keeling, Adrien Baranes, Benjamin Barnett, Michael McKibben, Tatenda Kanyere, Alison Lentz, Robin IM Dunbar, et al.
  **大型语言模型在高级心智理论任务上达到成人人类水平（Llms achieve adult human performance on higher-order theory of mind tasks）** 。
  arXiv 预印本 arXiv:2405.18870, 2024.
- [242]
  Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu.
  **T2v-compbench: 一个用于组合式文本到视频生成的综合基准（A comprehensive benchmark for compositional text-to-video generation）** 。
  载于 **IEEE/CVF计算机视觉与模式识别会议论文集（Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR）** , 页码 8406–8416, 2025年6月.
- [243]
  Richard S Sutton.
  **基于近似动态规划的集成学习、规划与反应架构（Integrated architectures for learning, planning, and reacting based on approximating dynamic programming）** 。
  载于 **1990年机器学习会议录（Machine learning proceedings 1990）** , 页码 216–224. Elsevier,

1990.

- [244]
  Damian A Tamburri.
  **《通用数据保护条例（General Data Protection Regulation, GDPR）》的设计原则：一项形式概念分析及其评估** 。
  Information Systems, 91:101469, 2020.
- [245]
  Hao Tang, Darren Key, and Kevin Ellis.
  **Worldcoder：一个基于模型的 LLM 智能体——通过编写代码并与环境交互来构建世界模型** 。
  Advances in Neural Information Processing Systems,
  37:70148–70212, 2024.
- [246]
  Jiakai Tang, Heyang Gao, Xuchen Pan, Lei Wang, Haoran Tan, Dawei Gao, Yushuo
  Chen, Xu Chen, Yankai Lin, Yaliang Li, et al.
  **Gensim：一个基于大型语言模型智能体的通用社会模拟平台** 。
- [247]
  Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou,
  Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, et al.
  **Aether：几何感知的统一世界建模** 。
  arXiv preprint arXiv:2503.18945, 2025.
- [248]
  HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang,
  Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, et al.
  **Hunyuanworld 1.0：从文字或像素生成沉浸式、可探索、可交互的 3D 世界** 。
  arXiv preprint arXiv:2507.21809, 2025.
- [249]
  ManyCore Research Team.
  **Spatiallm：用于空间理解的大型语言模型** 。
  [https://github.com/manycore-research/SpatialLM](https://github.com/manycore-research/SpatialLM), 2025.
- [250]
  Marvin Teichmann, Michael Weber, Marius Zoellner, Roberto Cipolla, and Raquel
  Urtasun.
  **Multinet：用于自动驾驶的实时联合语义推理** ，

2018.

- [251]
  田然（Ran Tian）、李博一（Boyi Li）、翁鑫硕（Xinshuo Weng）、陈宇霄（Yuxiao Chen）、爱德华·施默林（Edward Schmerling）、王悦（Yue Wang）、鲍里斯·伊万诺维奇（Boris Ivanovic）、马可·帕沃内（Marco Pavone）。
  **将世界标记化为对象级知识以应对自动驾驶中的长尾事件（Tokenize the world into object-level knowledge to address long-tail events in autonomous driving）** ，2024。
- [252]
  爱德华·C·托尔曼（Edward C Tolman）。
  **大鼠与人类的认知地图（Cognitive maps in rats and men）** 。
  《心理学评论（Psychological review）》，55(4):189，1948。
- [253]
  雨果·图夫龙（Hugo Touvron）、蒂博·拉夫里尔（Thibaut Lavril）、高蒂埃·伊扎卡尔（Gautier Izacard）、泽维尔·马蒂内（Xavier Martinet）、玛丽-安娜·拉绍（Marie-Anne Lachaux）、蒂莫泰·拉克鲁瓦（Timothée Lacroix）、巴蒂斯特·罗齐埃（Baptiste Rozière）、纳曼·戈亚尔（Naman Goyal）、埃里克·汉布罗（Eric Hambro）、费萨尔·阿扎尔（Faisal Azhar）等。
  **Llama：开放且高效的基础语言模型（Llama: Open and efficient foundation language models）** 。
  arXiv 预印本 arXiv:2302.13971，2023。
- [254]
  达尼·瓦列夫斯基（Dani Valevski）、亚尼夫·利维坦（Yaniv Leviathan）、莫阿布·阿拉尔（Moab Arar）、什洛米·弗鲁克特（Shlomi Fruchter）。
  **扩散模型是实时游戏引擎（Diffusion models are real-time game engines）** 。
  arXiv 预印本 arXiv:2408.14837，2024。
- [255]
  A·瓦斯瓦尼（A Vaswani）。
  **注意力机制就是你所需要的一切（Attention is all you need）** 。
  《神经信息处理系统进展（Advances in Neural Information Processing Systems）》，2017。
- [256]
  阿什马尔·瓦亚尼（Ashmal Vayani）、迪努拉·迪萨纳亚克（Dinura Dissanayake）、哈辛德里·瓦塔瓦纳（Hasindri Watawana）、努尔·阿赫桑（Noor Ahsan）、内瓦西尼·萨西库马尔（Nevasini Sasikumar）、奥姆卡·塔瓦卡尔（Omkar Thawakar）、赫诺克·比亚德格利格·阿德姆特乌（Henok Biadglign Ademtew）、叶海亚·赫迈提（Yahya Hmaiti）、阿曼迪普·库马尔（Amandeep Kumar）、卡尔蒂克·库克雷贾（Kartik Kukreja）等。
  **所有语言都重要：在文化多样的 100 种语言上评估 LMMs（All languages matter: Evaluating lmms on culturally diverse 100 languages）** 。
  载于《计算机视觉与模式识别会议论文集（Proceedings of the Computer Vision and Pattern Recognition Conference）》，第 19565–19575 页，2025。
- [257]
  武达（Dat Vu）、吴宝（Bao Ngo）、潘洪（Hung Phan）。
  **Hybridnets：端到端感知网络（Hybridnets: End-to-end perception network）** ，2022。
- [258]
  Wan 团队（Team Wan）、王昂（Ang Wang）、艾宝乐（Baole Ai）、文斌（Bin Wen）、毛超杰（Chaojie Mao）、谢晨玮（Chen-Wei Xie）、陈迪（Di Chen）、于飞武（Feiwu Yu）、赵海明（Haiming Zhao）、杨建晓（Jianxiao Yang）等。
  **Wan：开放且先进的大规模视频生成模型（Wan: Open and advanced large-scale video generative models）** 。
  arXiv 预印本 arXiv:2503.20314，2025。
- [259]
  王傲（Ao Wang）、陈辉（Hui Chen）、林子佳（Zijia Lin）、韩军功（Jungong Han）、丁贵广（Guiguang Ding）。
  **Repvit：从 ViT 视角重新审视移动 CNN（Repvit: Revisiting mobile cnn from vit perspective）** 。
  载于《IEEE/CVF 计算机视觉与模式识别会议论文集（Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition）》，第 15909–15920 页，2024。
- [260]
  王汉卿（Hanqing Wang）、陈嘉禾（Jiahe Chen）、黄文思（Wensi Huang）、贲庆伟（Qingwei Ben）、王泰（Tai Wang）、米博宇（Boyu Mi）、黄涛（Tao Huang）、赵思衡（Siheng Zhao）、陈逸伦（Yilun Chen）、杨思哲（Sizhe Yang）等。
  **Grutopia：规模化城市中的通用机器人梦想（Grutopia: Dream general robots in a city at scale）** 。
  arXiv 预印本 arXiv:2407.10943，2024。
- [261]
  王磊（Lei Wang）、高鹤阳（Heyang Gao）、薄晓鹤（Xiaohe Bo）、陈旭（Xu Chen）、文继荣（Ji-Rong Wen）。
  **Yulan-onesim：迈向基于大语言模型的新一代社交模拟器（Yulan-onesim: Towards the next generation of social simulator with large language models）** 。
  arXiv 预印本 arXiv:2505.07581，2025。
- [262]
  王乐宁（Lening Wang）、郑文钊（Wenzhao Zheng）、任一龙（Yilong Ren）、姜涵（Han Jiang）、崔志勇（Zhiyong Cui）、于海洋（Haiyang Yu）、卢策吾（Jiwen Lu）。
  **Occsora：作为自动驾驶世界模拟器的 4D 占据生成模型（Occsora: 4d occupancy generation models as world simulators for autonomous driving）** 。
  arXiv 预印本 arXiv:2405.20337，2024。
- [263]
  王立瑞（Lirui Wang）、凌一阳（Yiyang Ling）、袁哲成（Zhecheng Yuan）、莫希特·什里达尔（Mohit Shridhar）、鲍晨（Chen Bao）、秦宇哲（Yuzhe Qin）、王百霖（Bailin Wang）、徐华哲（Huazhe Xu）、王晓龙（Xiaolong Wang）。
  **Gensim：通过大语言模型生成机器人仿真任务（Gensim: Generating robotic simulation tasks via large language models）** 。
  arXiv 预印本 arXiv:2310.01361，2023。
- [264]
  王廷武（Tingwu Wang）、吉米·巴（Jimmy Ba）。
  **利用策略网络探索基于模型的规划（Exploring model-based planning with policy networks）** 。
  arXiv 预印本 arXiv:1906.08649，2019。
- [265]
  王晓峰（Xiaofeng Wang）、朱政（Zheng Zhu）、黄冠（Guan Huang）、陈新泽（Xinze Chen）、朱家港（Jiagang Zhu）、卢策吾（Jiwen Lu）。
  **Drivedreamer：迈向真实世界驱动的自动驾驶世界模型（Drivedreamer: Towards real-world-driven world models for autonomous driving）** 。
  arXiv 预印本 arXiv:2309.09777，2023。
- [266]
  王晓峰（Xiaofeng Wang）、朱政（Zheng Zhu）、黄冠（Guan Huang）、陈新泽（Xinze Chen）、朱家港（Jiagang Zhu）、卢策吾（Jiwen Lu）。
  **Drivedreamer：迈向真实世界驱动的自动驾驶世界模型（Drivedreamer: Towards real-world-driven world models for autonomous driving）** ，2023。
- [267]
  王晓峰（Xiaofeng Wang）、朱政（Zheng Zhu）、黄冠（Guan Huang）、王博渊（Boyuan Wang）、陈新泽（Xinze Chen）、卢策吾（Jiwen Lu）。
  **Worlddreamer：通过预测掩码词元迈向通用视频生成世界模型（Worlddreamer: Towards general world models for video generation via predicting masked tokens）** 。
  arXiv 预印本 arXiv:2401.09985，2024。
- [268]
  王一茹（Yi Ru Wang）、段佳飞（Jiafei Duan）、迪特尔·福克斯（Dieter Fox）、西达尔塔·斯里尼瓦萨（Siddhartha Srinivasa）。
  **Newton：大语言模型具备物理推理能力吗？（Newton: Are large language models capable of physical reasoning?）** 。
  arXiv 预印本 arXiv:231

Technology innovation management review, 9(11), 2019.

- [271]
  Alex Wilf, Sihyun Shawn Lee, Paul Pu Liang, and Louis-Philippe Morency.
  **三思而后行：观点采择提升大型语言模型的心智理论能力（Think twice: Perspective-taking improves large language models’ theory-of-mind capabilities）** , 2023.
- [272]
  Dong Wu, Man-Wen Liao, Wei-Tian Zhang, Xing-Gang Wang, Xiang Bai, Wen-Qing
  Cheng, and Wen-Yu Liu.
  **Yolop：一次查看实现全景驾驶感知（Yolop: You only look once for panoptic driving perception）** .
  Machine Intelligence Research, 19(6):550–562, November 2022.
- [273]
  Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and
  Mingsheng Long.
  **ivideogpt：交互式视频GPT是可扩展的世界模型（ivideogpt: Interactive videogpts are scalable world models）** .
  arXiv preprint arXiv:2405.15223, 2024.
- [274]
  Jincenzi Wu, Zhuang Chen, Jiawen Deng, Sahand Sabour, Helen Meng, and Minlie
  Huang.
  **Coke：用于机器心智理论的认知知识图谱（Coke: A cognitive knowledge graph for machine theory of mind）** .
  arXiv preprint arXiv:2305.05390, 2024.
- [275]
  Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken
  Goldberg.
  **Daydreamer：用于物理机器人学习的世界模型（Daydreamer: World models for physical robot learning）** .
  In Conference on robot learning, pages 2226–2240. PMLR, 2023.
- [276]
  Wayne Wu, Honglin He, Yiran Wang, Chenda Duan, Jack He, Zhizheng Liu, Quanyi
  Li, and Bolei Zhou.
  **Metaurban：城市空间中具身人工智能的仿真平台（Metaurban: A simulation platform for embodied ai in urban spaces）** .
  arXiv preprint arXiv:2407.08725, 2024.
- [277]
  Wayne Wu, Honglin He, Chaoyuan Zhang, Jack He, Seth Z Zhao, Ran Gong, Quanyi
  Li, and Bolei Zhou.
  **通过可扩展城市仿真实现自主微出行（Towards autonomous micromobility through scalable urban simulation）** .
  In Proceedings of the Computer Vision and Pattern Recognition
  Conference, pages 27553–27563, 2025.
- [278]
  Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua
  Liu, Hanxiao Jiang, Yifu Yuan, He Wang, et al.
  **Sapien：一个基于部件的模拟交互环境（Sapien: A simulated part-based interactive environment）** .
  In Proceedings of the IEEE/CVF conference on computer vision and
  pattern recognition, pages 11097–11107, 2020.
- [279]
  Jiannan Xiang, Guangyi Liu, Yi Gu, Qiyue Gao, Yuting Ning, Yuheng Zha, Zeyu
  Feng, Tianhua Tao, Shibo Hao, Yemin Shi, et al.
  **Pandora：迈向具有自然语言动作和视频状态的通用世界模型（Pandora: Towards general world model with natural language actions and video states）** .
  arXiv preprint arXiv:2406.09455, 2024.
- [280]
  Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, and
  Zhiting Hu.
  **语言模型遇见世界模型：具身体验增强语言模型（Language models meet world models: Embodied experiences enhance language models）** .
  Advances in neural information processing systems, 36, 2024.
- [281]
  Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and
  Xingang Pan.
  **Worldmem：具有记忆的长期一致世界仿真（Worldmem: Long-term consistent world simulation with memory）** .
  arXiv preprint arXiv:2504.12369, 2025.
- [282]
  Ziyang Xie, Zhizheng Liu, Zhenghao Peng, Wayne Wu, and Bolei Zhou.
  **Vid2sim：用于城市导航的、从视频到逼真交互仿真的方法（Vid2sim: Realistic and interactive simulation from video for urban navigation）** .
  In Proceedings of the Computer Vision and Pattern Recognition
  Conference, pages 1581–1591, 2025.
- [283]
  Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang,
  Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, et al.
  **迈向大型推理模型：基于大型语言模型的强化推理综述（Towards large reasoning models: A survey of reinforced reasoning with large language models）** .
  arXiv preprint arXiv:2501.09686, 2025.
- [284]
  Fengli Xu, Jun Zhang, Chen Gao, Jie Feng, and Yong Li.
  **城市生成智能：具身城市环境中智能体的基础平台（Urban generative intelligence (ugi): A foundational platform for agents in embodied city environment）** .
  arXiv preprint arXiv:2312.11813, 2023.
- [285]
  Wenrui Xu, Dalin Lyu, Weihang Wang, Jie Feng, Chen Gao, and Yong Li.
  **定义与评估视觉语言模型的基本空间能力：来自心理测量学的视角（Defining and evaluating visual language models’ basic spatial abilities: A perspective from psychometrics）** .
  In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher
  Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the
  Association for Computational Linguistics (Volume 1: Long Papers), pages
  11571–11590, Vienna, Austria, July 2025. Association for Computational
  Linguistics.
- [286]
  Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, and
  Yang Liu.

- [287]
  闫文森（Wilson Yan）、张云志（Yunzhi Zhang）、彼得·阿比尔（Pieter Abbeel）、阿拉温德·斯里尼瓦斯（Aravind Srinivas）。
  **探索大型语言模型在沟通游戏中的应用：一项关于狼人杀的实证研究（Exploring large language models for communication games: An empirical study on werewolf）** 。
  arXiv 预印本 arXiv:2309.04658，2023年。
- [288]
  闫旭（Xu Yan）、张海明（Haiming Zhang）、蔡英杰（Yingjie Cai）、郭敬明（Jingming Guo）、邱伟超（Weichao Qiu）、高斌（Bin Gao）、周凯强（Kaiqiang Zhou）、赵越（Yue Zhao）、金欢（Huan Jin）、高建涛（Jiantao Gao）等。
  **VideoGPT：使用 VQ-VAE 和 Transformer 进行视频生成（Videogpt: Video generation using vq-vae and transformers）** 。
  arXiv 预印本 arXiv:2104.10157，2021年。
- [289]
  闫旭（Xu Yan）、张海明（Haiming Zhang）、蔡英杰（Yingjie Cai）、郭敬明（Jingming Guo）、邱伟超（Weichao Qiu）、高斌（Bin Gao）、周凯强（Kaiqiang Zhou）、赵越（Yue Zhao）、金欢（Huan Jin）、高建涛（Jiantao Gao）等。
  **锻造自动驾驶视觉基础模型：挑战、方法与机遇（Forging vision foundation models for autonomous driving: Challenges, methodologies, and opportunities）** 。
  arXiv 预印本 arXiv:2401.08045，2024年。
- [290]
  闫宇伟（Yuwei Yan）、曾庆斌（Qingbin Zeng）、郑志恒（Zhiheng Zheng）、袁敬哲（Jingzhe Yuan）、冯杰（Jie Feng）、张军（Jun Zhang）、徐凤丽（Fengli Xu）、李勇（Yong Li）。
  **OpenCity：一个使用海量 LLM 智能体模拟城市活动的可扩展平台（Opencity: A scalable platform to simulate urban activities with massive llm agents）** 。
  arXiv 预印本 arXiv:2410.21286，2024年。
- [291]
  杨德顺（Deshun Yang）、胡露慧（Luhui Hu）、田宇（Yu Tian）、李子豪（Zihao Li）、克里斯·凯利（Chris Kelly）、杨邦（Bang Yang）、杨辛迪（Cindy Yang）、邹月娴（Yuexian Zou）。
  **WorldGPT：一个受 Sora 启发的、从文本和图像输入构建丰富世界模型的视频 AI 智能体（Worldgpt: a sora-inspired video ai agent as rich world models from text and image inputs）** 。
  arXiv 预印本 arXiv:2403.07944，2024年。
- [292]
  杨继涵（Jihan Yang）、杨树生（Shusheng Yang）、安贾利·W·古普塔（Anjali W Gupta）、韩瑞林（Rilyn Han）、李飞飞（Li Fei-Fei）、谢赛宁（Saining Xie）。
  **空间思维：多模态大语言模型如何观察、记忆和回忆空间（Thinking in space: How multimodal large language models see, remember, and recall spaces）** 。
  载于《计算机视觉与模式识别会议论文集》，第 10632–10643 页，2025年。
- [293]
  杨梦娇（Mengjiao Yang）、杜一伦（Yilun Du）、戴博（Bo Dai）、戴尔·舒尔曼斯（Dale Schuurmans）、约书亚·B·特南鲍姆（Joshua B Tenenbaum）、彼得·阿比尔（Pieter Abbeel）。
  **文本到视频模型的概率适应（Probabilistic adaptation of text-to-video models）** 。
  arXiv 预印本 arXiv:2306.01872，2023年。
- [294]
  杨梦娇（Mengjiao Yang）、杜一伦（Yilun Du）、卡米亚尔·加塞米普尔（Kamyar Ghasemipour）、乔纳森·汤普森（Jonathan Tompson）、戴尔·舒尔曼斯（Dale Schuurmans）、彼得·阿比尔（Pieter Abbeel）。
  **学习交互式真实世界模拟器（Learning interactive real-world simulators）** 。
  arXiv 预印本 arXiv:2310.06114，2023年。
- [295]
  雪莉·杨（Sherry Yang）、杜一伦（Yilun Du）、赛义德·卡米亚尔·赛义德·加塞米普尔（Seyed Kamyar Seyed Ghasemipour）、乔纳森·汤普森（Jonathan Tompson）、莱斯利·帕克·凯布林（Leslie Pack Kaelbling）、戴尔·舒尔曼斯（Dale Schuurmans）、彼得·阿比尔（Pieter Abbeel）。
  **学习交互式真实世界模拟器（Learning interactive real-world simulators）** 。
  载于《第十二届国际学习表征会议》，2024年。
- [296]
  雪莉·杨（Sherry Yang）、雅各布·沃克（Jacob Walker）、杰克·帕克-霍尔德（Jack Parker-Holder）、杜一伦（Yilun Du）、杰克·布鲁斯（Jake Bruce）、安德烈·巴雷托（Andre Barreto）、彼得·阿比尔（Pieter Abbeel）、戴尔·舒尔曼斯（Dale Schuurmans）。
  **视频作为现实世界决策的新语言（Video as the new language for real-world decision making）** 。
  arXiv 预印本 arXiv:2402.17139，2024年。
- [297]
  杨悦（Yue Yang）、孙凡云（Fan-Yun Sun）、卢卡·韦斯（Luca Weihs）、伊莱·范德比尔特（Eli VanderBilt）、阿尔瓦罗·赫拉斯蒂（Alvaro Herrasti）、韩文森（Winson Han）、吴佳俊（Jiajun Wu）、尼克·哈伯（Nick Haber）、兰杰·克里希纳（Ranjay Krishna）、刘凌杰（Lingjie Liu）等。
  **Holodeck：语言引导的 3D 具身 AI 环境生成（Holodeck: Language guided generation of 3d embodied ai environments）** 。
  载于《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 16227–16237 页，2024年。
- [298]
  杨泽元（Zeyuan Yang）、刘佳庚（Jiageng Liu）、陈培浩（Peihao Chen）、阿努普·切里安（Anoop Cherian）、蒂姆·K·马克斯（Tim K Marks）、乔纳森·勒鲁（Jonathan Le Roux）、甘闯（Chuang Gan）。
  **Rila：用于零样本语义视听导航的反思与想象语言智能体（Rila: Reflective and imaginative language agent for zero-shot semantic audio-visual navigation）** 。
  载于《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 16251–16261 页，2024年。
- [299]
  杨中奇（Zhongqi Yang）、葛文航（Wenhang Ge）、李雨琪（Yuqi Li）、陈佳琪（Jiaqi Chen）、李浩源（Haoyuan Li）、安梦吟（Mengyin An）、康飞（Fei Kang）、薛华（Hua Xue）、徐百欣（Baixin Xu）、尹宇阳（Yuyang Yin）等。
  **Matrix-3D：全向可探索的 3D 世界生成（Matrix-3d: Omnidirectional explorable 3d world generation）** 。
  arXiv 预印本 arXiv:2508.08086，2025年。
- [300]
  杨卓毅（Zhuoyi Yang）、滕佳彦（Jiayan Teng）、郑文迪（Wendi Zheng）、丁明（Ming Ding）、黄诗雨（Shiyu Huang）、徐嘉政（Jiazheng Xu）、杨远明（Yuanming Yang）、洪文怡（Wenyi Hong）、张晓涵（Xiaohan Zhang）、冯冠宇（Guanyu Feng）等。
  **CogVideoX：采用专家 Transformer 的文本到视频扩散模型（Cogvideox: Text-to-video diffusion models with an expert transformer）** 。
  arXiv 预印本 arXiv:2408.06072，2024年。
- [301]
  杨子怡（Ziyi Yang）、张再斌（Zaibin Zhang）、郑子瑞（Zirui Zheng）、蒋宇贤（Yuxian Jiang）、甘子越（Ziyue Gan）、王志宇（Zhiyu Wang）、凌子健（Zijian Ling）、马马丁（Martin Ma）、董博文（Bowen Dong）、普拉蒂克·古

Advances in Neural Information Processing Systems,
37:128734–128768, 2024.

- [303]
  Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni,
  Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al.
  **Nuwa-xl: Diffusion over diffusion for extremely long video
  generation.**
  arXiv preprint arXiv:2303.12346, 2023.
- [304]
  Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu.
  **Wonderworld: Interactive 3d scene generation from a single image.**
  In Proceedings of the Computer Vision and Pattern Recognition
  Conference, pages 5916–5926, 2025.
- [305]
  Jifan Yu, Xiaozhi Wang, Shangqing Tu, Shulin Cao, Daniel Zhang-Li, Xin Lv, Hao
  Peng, Zijun Yao, Xiaohan Zhang, Hanming Li, et al.
  **Kola: Carefully benchmarking world knowledge of large language
  models.**
  arXiv preprint arXiv:2306.09296, 2023.
- [306]
  Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu.
  **Gamefactory: Creating new games with generative interactive videos.**
  arXiv preprint arXiv:2501.08325, 2025.
- [307]
  Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang,
  Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al.
  **Magvit: Masked generative video transformer.**
  In Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition, pages 10459–10469, 2023.
- [308]
  Sheng Yu, Di-Hua Zhai, Yuanqing Xia, Haoran Wu, and Jun Liao.
  **Se-resunet: A novel robotic grasp detection method.**
  IEEE Robotics and Automation Letters, 7(2):5238–5245, 2022.
- [309]
  Yang Yu, Hong Qian, and Yi-Qi Hu.
  **Derivative-free optimization via classification.**
  In Proceedings of the AAAI Conference on Artificial
  Intelligence, volume 30, 2016.
- [310]
  Yuan Yuan, Jingtao Ding, Depeng Jin, and Yong Li.
  **Learning the complexity of urban mobility with deep generative
  network.**
  PNAS nexus, 4(5):pgaf081, 2025.
- [311]
  Hu Yue, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen,
  Maoqing Yao, and Guanghui Ren.
  **Ewmbench: Evaluating scene, motion, and semantic quality in embodied
  world models.**
  arXiv preprint arXiv:2505.09694, 2025.
- [312]
  Jintian Zhang, Xin Xu, and Shumin Deng.
  **Exploring collaboration mechanisms for llm agents: A social
  psychology view.**
  arXiv preprint arXiv:2310.02124, 2023.
- [313]
  Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun.
  **Copilot4d: Learning unsupervised world models for autonomous driving
  via discrete diffusion, 2024.**
- [314]
  Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah
  Snavely, Jiajun Wu, and William T Freeman.
  **Physdreamer: Physics-based interaction with 3d objects via video
  generation.**
  In European Conference on Computer Vision, pages 388–406.
  Springer, 2025.
- [315]
  Weichen Zhang, Chen Gao, Shiquan Yu, Ruiying Peng, Baining Zhao, Qian Zhang,
  Jinqiang Cui, Xinlei Chen, and Yong Li.
  **Citynavagent: Aerial vision-and-language navigation with hierarchical
  semantic planning and global memory.**
  ACL, 2025.
- [316]
  Wenqi Zhang, Ke Tang, Hai Wu, Mengna Wang, Yongliang Shen, Guiyang Hou, Zeqi
  Tan, Peng Li, Yueting Zhuang, and Weiming Lu.
  **Agent-pro: Learning to evolve via policy-level reflection and
  optimization.**
  arXiv preprint arXiv:2402.17574, 2024.
- [317]
  Xinnong Zhang, Jiayu Lin, Xinyi Mou, Shiyue Yang, Xiawei Liu, Libo Sun, Hanjia
  Lyu, Yihang Yang, Weihong Qi, Yue Chen, et al.
  **Socioverse: A world model for social simulation powered by llm agents
  and a pool of 10 million real-world users.**
  arXiv preprint arXiv:2504.10157, 2025.
- [318]
  Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao
  Jiang, Zedong Gao, Eric Li, Yang Liu, et al.
  **Matrix-game: Interactive world foundation model.**
  arXiv preprint arXiv:2506.18701, 2025.
- [319]
  Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu,
  Zhenhua Dong, and Ji-Rong Wen.

- [320]
  张哲俊（Zhejun Zhang）、亚历山大·林尼格（Alexander Liniger）、戴登新（Dengxin Dai）、余飞（Fisher Yu）、卢克·范·古尔（Luc Van Gool）。
  **Trafficbots: Towards world models for autonomous driving simulation and motion prediction（Trafficbots：迈向自动驾驶仿真与运动预测的世界模型）** 。
  发表于《2023年IEEE机器人与自动化国际会议（2023 IEEE International Conference on Robotics and Automation, ICRA）》，第1522–1529页。IEEE，2023年。
- [321]
  张哲俊（Zhejun Zhang）、亚历山大·林尼格（Alexander Liniger）、克里斯托斯·萨卡里迪斯（Christos Sakaridis）、余飞（Fisher Yu）、卢克·范·古尔（Luc Van Gool）。
  **Real-time motion prediction via heterogeneous polyline transformer with relative pose encoding（基于相对位姿编码的异构折线变换器的实时运动预测）** ，2023年。
- [322]
  赵柏宁（Baining Zhao）、方建杰（Jianjie Fang）、戴子超（Zichao Dai）、王子由（Ziyou Wang）、查继荣（Jirong Zha）、张伟晨（Weichen Zhang）、高晨（Chen Gao）、王悦（Yue Wang）、崔金强（Jinqiang Cui）、陈新雷（Xinlei Chen）等。
  **Urbanvideo-bench: Benchmarking vision-language models on embodied intelligence with video data in urban spaces（Urbanvideo-bench：基于城市空间视频数据对具身智能的视觉-语言模型进行基准测试）** 。
  发表于《第63届计算语言学协会年会论文集（第1卷：长论文）（Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)）》，2025年。
- [323]
  赵柏宁（Baining Zhao）、唐荣泽（Rongze Tang）、贾明远（Mingyuan Jia）、王子由（Ziyou Wang）、满凡航（Fanhang Man）、张欣（Xin Zhang）、尚宇（Yu Shang）、张伟晨（Weichen Zhang）、吴伟（Wei Wu）、高晨（Chen Gao）等。
  **Airscape: An aerial generative world model with motion controllability（Airscape：一个具有运动可控性的空中生成世界模型）** 。
  发表于《第33届ACM国际多媒体会议论文集（Proceedings of the 33rd ACM International Conference on Multimedia）》，第12519–12528页，2025年。
- [324]
  赵甘龙（Ganlong Zhao）、李冠彬（Guanbin Li）、陈伟楷（Weikai Chen）、余轶舟（Yizhou Yu）。
  **Over-nav: Elevating iterative vision-and-language navigation with open-vocabulary detection and structured representation（Over-nav：利用开放词汇检测和结构化表示提升迭代式视觉-语言导航）** 。
  发表于《IEEE/CVF计算机视觉与模式识别会议论文集（Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition）》，第16296–16206页，2024年。
- [325]
  赵国胜（Guosheng Zhao）、王晓峰（Xiaofeng Wang）、朱铮（Zheng Zhu）、陈新泽（Xinze Chen）、黄冠（Guan Huang）、鲍晓艺（Xiaoyi Bao）、王新刚（Xingang Wang）。
  **Drivedreamer-2: Llm-enhanced world models for diverse driving video generation（Drivedreamer-2：用于多样化驾驶视频生成的LLM增强世界模型）** ，2024年。
- [326]
  赵勇（Yong Zhao）、徐凯（Kai Xu）、朱正球（Zhengqiu Zhu）、胡越（Yue Hu）、郑志恒（Zhiheng Zheng）、陈迎丰（Yingfeng Chen）、季亚泰（Yatai Ji）、高晨（Chen Gao）、李勇（Yong Li）、黄金才（Jincai Huang）。
  **Cityeqa: A hierarchical llm agent on embodied question answering benchmark in city space（Cityeqa：城市空间具身问答基准上的分层LLM智能体）** 。
  发表于《2017年自然语言处理实证方法会议论文集（Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing）》，2025年。
- [327]
  赵子瑞（Zirui Zhao）、李伟孙（Wee Sun Lee）、徐大卫（David Hsu）。
  **Large language models as commonsense knowledge for large-scale task planning（作为大规模任务规划常识知识的大型语言模型）** 。
  《神经信息处理系统进展（Advances in Neural Information Processing Systems）》，第36卷，2024年。
- [328]
  甄浩宇（Haoyu Zhen）、邱晓文（Xiaowen Qiu）、陈培浩（Peihao Chen）、杨金城（Jincheng Yang）、闫鑫（Xin Yan）、杜一伦（Yilun Du）、洪一宁（Yining Hong）、甘创（Chuang Gan）。
  **3d-vla: A 3d vision-language-action generative world model（3d-vla：一个三维视觉-语言-动作生成世界模型）** 。
  arXiv预印本 arXiv:2403.09631，2024年。
- [329]
  甄浩宇（Haoyu Zhen）、孙乔（Qiao Sun）、张洪鑫（Hongxin Zhang）、李俊彦（Junyan Li）、周思远（Siyuan Zhou）、杜一伦（Yilun Du）、甘创（Chuang Gan）。
  **Tesseract: learning 4d embodied world models（Tesseract：学习四维具身世界模型）** 。
  arXiv预印本 arXiv:2504.20995，2025年。
- [330]
  郑典（Dian Zheng）、黄子骐（Ziqi Huang）、刘洪波（Hongbo Liu）、邹凯（Kai Zou）、何一楠（Yinan He）、张帆（Fan Zhang）、张元翰（Yuanhan Zhang）、何静雯（Jingwen He）、郑伟诗（Wei-Shi Zheng）、乔宇（Yu Qiao）等。
  **Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness（Vbench-2.0：推进视频生成基准套件以实现内在保真度）** 。
  arXiv预印本 arXiv:2503.21755，2025年。
- [331]
  郑斯蒂芬（Stephan Zheng）、亚历山大·特罗特（Alexander Trott）、苏尼尔·斯里尼瓦萨（Sunil Srinivasa）、大卫·C·帕克斯（David C Parkes）、理查德·索赫尔（Richard Socher）。
  **The ai economist: Taxation policy design via two-level deep multiagent reinforcement learning（AI经济学家：通过双层深度多智能体强化学习设计税收政策）** 。
  《科学进展（Science advances）》，第8卷，第18期，eabk2607，2022年。
- [332]
  郑文钊（Wenzhao Zheng）、陈伟良（Weiliang Chen）、黄元辉（Yuanhui Huang）、张博睿（Borui Zhang）、段月琪（Yueqi Duan）、卢继文（Jiwen Lu）。
  **Occworld: Learning a 3d occupancy world model for autonomous driving（Occworld：学习用于自动驾驶的三维占据世界模型）** 。
  arXiv预印本 arXiv:2311.16038，2023年。
- [333]
  郑藏伟（Zangwei Zheng）、彭翔宇（Xiangyu Peng）、杨天骥（Tianji Yang）、沈晨辉（Chenhui Shen）、李圣贵（Shenggui Li）、刘洪鑫（Hongxin Liu）、周宇坤（Yukun Zhou）、李天一（Tianyi Li）、尤洋（Yang You）。
  **Open-sora: Democratizing efficient video production for all（Open-sora：为所有人普及高效视频制作）** 。
  arXiv预印本 arXiv:2412.20404，2024年。
- [334]
  周宏宇（Hongyu Zhou）、葛政（Zheng Ge）、李泽明（Zeming Li）、张祥雨（Xiangyu Zhang）。
  **Matrixvt: Efficient multi-camera to bev transformation for 3d perception（Matrixvt：用于三维感知的高效多相机到鸟瞰图变换）** ，2022年。
- [335]
  周思远（Siyuan Zhou）、杜一伦（Yilun Du）、陈佳犇（Jiaben Chen）、李彦东（Yandong Li）、杨丁仁（Dit-Yan Yeung）、甘创（Chuang Gan）。
  **Robodreamer: Learning compositional world models for robot imagination（Robodreamer：学习用于机器人想象的组合式世界模型）** 。

arXiv 预印本 arXiv:2404.12377, 2024.

- [336]
  Zikang Zhou, Jianping Wang, Yung-Hui Li, and Yu-Kai Huang.
  **以查询为中心的轨迹预测（Query-centric trajectory prediction）** 。
  收录于《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 17863–17873 页，2023 年。
- [337]
  Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong.
  **Irasim：学习交互式真实机器人动作模拟器（Irasim: Learning interactive real-robot action simulators）** 。
  arXiv 预印本 arXiv:2406.14540, 2024.
- [338]
  Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, Chi Zhang, 等。
  **Sora 是世界模拟器吗？关于通用世界模型及超越的全面综述（Is sora a world simulator? a comprehensive survey on general world models and beyond）** 。
  arXiv 预印本 arXiv:2405.03520, 2024.
- [339]
  Alex Zyner, Stewart Worrall, and Eduardo Nebot.
  **使用循环神经网络进行自然主义驾驶员意图与路径预测（Naturalistic driver intention and path prediction using recurrent neural networks）** 。

<a id="appendix-a"></a>

**附录 A（Appendix A）**

## 附录 A 相关综述（Appendix A Related survey）

> 表 S1. 与现有综述的对比。本文侧重于对 **世界模型（World Model）** 的系统性定义及其能力的全面概述。

| Survey | Venue and Year  | Main Focus                  | Deficiency                            |
| :----- | :-------------- | :-------------------------- | :------------------------------------ |
| Survey | Venue and Year  | Main Focus                  | Deficiency                            |
| Survey | Venue and Year  | Main Focus                  | Deficiency                            |
| Survey | Venue and Year  | Main Focus                  | Deficiency                            |
| [338]  | Arxiv, 2024     | General world model         | Limited to discussion on applications |
| [338]  | Arxiv, 2024     | General world model         | Limited to discussion on applications |
| [338]  | Arxiv, 2024     | General world model         | Limited to discussion on applications |
| [338]  | Arxiv, 2024     | General world model         | Limited to discussion on applications |
| [166]  | Arxiv, 2024     | Efficient multimodal models | Limited to discussion on techniques   |
| [166]  | Arxiv, 2024     | Efficient multimodal models | Limited to discussion on techniques   |
| [166]  | Arxiv, 2024     | Efficient multimodal models | Limited to discussion on techniques   |
| [166]  | Arxiv, 2024     | Efficient multimodal models | Limited to discussion on techniques   |
| [35]   | Arxiv, 2024     | Text-to-video generation    | Limited scope                         |
| [35]   | Arxiv, 2024     | Text-to-video generation    | Limited scope                         |
| [35]   | Arxiv, 2024     | Text-to-video generation    | Limited scope                         |
| [35]   | Arxiv, 2024     | Text-to-video generation    | Limited scope                         |
| [83]   | IEEE T-IV, 2024 | Autonomous driving          | Limited scope                         |
| [83]   | IEEE T-IV, 2024 | Autonomous driving          | Limited scope                         |
| [83]   | IEEE T-IV, 2024 | Autonomous driving          | Limited scope                         |
| [83]   | IEEE T-IV, 2024 | Autonomous driving          | Limited scope                         |
| [145]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |
| [145]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |
| [145]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |
| [145]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |
| [288]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |
| [288]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |
| [288]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |
| [288]  | Arxiv, 2024     | Autonomous driving          | Limited scope                         |

<a id="appendix-b"></a>

## 附录 B 图表

![robot_wm](images/robot_wm.png)

> 图 S1. 机器人世界模型的发展。

![simulacra](images/simulacra.png)

> 图 S2. 世界模型与社会模拟。

<a id="appendix-c"></a>

## 附录 C 更新历史

- 2025.09.09：发布用于 ACM Computing Survey 的版本。
- 2025.11.10：重写了第 [2](#section-2) 节（历史与当前发展），总结了深度学习时代世界模型的发展路线图（图 [2](#figure-2)），重组了第 [5](#section-5) 节中的应用领域，并更新了近期论文。
