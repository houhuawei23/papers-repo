# 标题：边缘高效推理（Efficient Reasoning on the Edge）

- ArXiv: 2603.16867
- 作者：Yelysei Bondarenko, Thomas Hehn, Rob Hesselink, Romain Lepert, Fabio Valerio Massoli, Evgeny Mironov, Leyla Mirvakhabova, Tribhuvanesh Orekondy, Spyridon Stasis, Andrey Kuzmin, Anna Kuzina, Markus Nagel, Ankita Nayak, Corrado Rainone, Ork de Rooij, Paul N Whatmough, Arash Behboodi%, Babak Ehteshami Bejnordi
- 章节：1
- 估计词元数：27.0k

# Efficient Reasoning on the Edge（高效边缘推理）

# Efficient Reasoning on the Edge（高效边缘推理）

# 引言（Introduction）

<a id="fig:overview"></a>

![](./images/fig1_overview.png)

> 图 6： **所提出的面向边缘设备的高效推理框架概览。** (a) 模型架构利用参数高效的 **低秩自适应（Low-Rank Adaptation, LoRA）** 适配器和一个轻量级切换器来动态路由查询。这种设计使得基础模型和推理激活模式能够在预填充阶段无缝共享一个可重用的 **键值缓存（Key-Value Cache, KV Cache）** 。(b) 并行测试时扩展策略，同时生成多个推理流以提高准确性，而不会造成严重的延迟惩罚。(c) 端到端部署流程，展示了从多阶段训练（ **监督微调（Supervised Fine-Tuning, SFT）** 和预算强制 **强化学习（Reinforcement Learning, RL）** ）到量化、模型导出以及最终设备上执行的整个过程。

**大型语言模型（Large Language Models, LLMs）** 的成功与影响力持续扩大， **推理（Reasoning）** 已成为其成就的一个基本组成部分。商业级代码代理能够对代码结构、重构、调试和依赖关系解析进行推理 [@anthropic2024claude; @jaech2024openai]。在科学领域，LLMs 越来越多地被用于协助专业数学家解决研究级别的问题 [@Abouzaid2026-first-proof; @Cao2025-ai4math; @Feng2026-ai4math]。在移动设备上，具备推理能力的 LLMs 解锁了一类新型的智能个人助手，它们能够规划多步骤任务、根据上下文响应用户查询，并在应用程序和界面之间自主操作。来自 Gemini 和 OpenAI 的最新进展展示了日益强大的独立推理能力 [@OpenAI2026-first-proof; @Woodruff2026-gemini]，并且一波模型直接与设备 **用户界面（User Interfaces, UIs）** 和现实世界服务交互的 **智能体（Agentic）** 用例正在涌现 [@Yao2022-react; @Zhou2025-maiui; @Wang2025-uitars2; @Veuns-Team2026-uivenus1.5]。然而，这些成就的代价是生成了海量的 **词元（Tokens）** ，其中推理轨迹占据了整体计算量的很大一部分 [@thinking-efficiency2025]。

在移动场景中，在边缘设备上部署推理模型具有吸引力，因为它能将敏感数据保留在设备端、降低往返延迟，并且在连接受限时仍可使用。然而，在实践中，设备端推理面临着几个关键限制。首先是 **内存瓶颈（Memory bottleneck）** ：受当前 DRAM（Dynamic Random Access Memory）容量的限制，移动设备通常只能支持经过适度量化（Quantization）的较小模型，而对于较大模型，则只能采用更激进的量化方案 [@Alizadeh2024-llm-flash; @Xiao2026-llm-mobile]。第二个限制是 **词元生成（Token generation）** 在功耗、延迟和内存占用方面的成本。冗长的推理轨迹（Reasoning traces）和大的上下文长度会显著增加 KV 缓存（KV cache）的大小，迅速耗尽可用内存。最后，具有广泛能力范围的通用大型语言模型（Large Language Models, LLMs）难以在边缘设备支持的模型大小内实现。虽然专用的小型语言模型（Small Language Models, SLMs）可以在特定任务上匹配较大模型的性能，但模型切换会引入额外的内存移动开销。因此，边缘部署的模型必须在严格的内存预算内运行，同时实现可用的每秒词元数（Tokens Per Second, TPS）和可接受的响应时间。

本文提出了一种端到端（End-to-end）的流水线（Pipeline），用于在严格的词元、延迟和内存预算下，在边缘设备上部署具备推理能力的语言模型。我们的设计从一个基础的、不具备推理能力的指令（Instruct）模型出发，通过 **LoRA（Low-Rank Adaptation）适配器（Adapters）** 来启用“推理模式”，使得同一个骨干网络（Backbone）既可以运行在标准聊天模式（无适配器），也可以运行在推理模式（启用推理适配器）。一个轻量级的切换器（Switcher）将每个传入的查询路由到适当的模式，仅在推理可能有益时才启用它。

在训练阶段，我们使用 **参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 来使基础模型在不同领域专业化，同时保持部署的实用性。我们证明 LoRA [@hulora] 对于实现面向特定领域的推理和任务专业化是有效的。 **将 LoRA 秩（Rank）作为基础模型和期望性能的函数进行选择是一个核心问题** ，我们通过广泛的分析来解决这个问题。由于 LoRA 适配器可以在运行时切换，因此单个基础模型只需加载一次，然后通过启用或禁用适配器来动态适应不同的任务。为了实现基础模型与 LoRA 增强的推理模型之间的 KV 缓存共享，我们在预填充（Prefill）阶段提出了 **掩码 LoRA 训练（Masked LoRA training）** ，并证明这对准确性没有显著影响。

我们的训练方案遵循推理模型常用的两阶段结构：首先在高质量推理轨迹上进行 **监督微调（Supervised Fine-Tuning, SFT）** ，然后进行 **强化学习（Reinforcement Learning, RL）** 阶段以进一步对齐 [@deepseekai2025deepseekr1]。由于 SFT 阶段并未明确处理推理的冗长性问题，模型在初始训练后往往会变得冗长和重复，因此我们 RL 阶段的一个关键目标是通过 **预算强制（Budget forcing）** 来惩罚过长的推理轨迹 [@muennighoff2025s1; @li2025budgetguidance]。我们在 RL 阶段应用预算强制，并探索不同的训练策略、奖励设计、提示方法和硬性执行机制，从实验中提炼出最佳实践。

在推理阶段，我们将 **内存受限的解码（Memory-bound decoding）** 作为设备端的主要瓶颈。我们利用 **并行测试时缩放（Parallel test-time scaling）** 结合 **神经验证（Neural verification）** 来提高准确性，而不会产生显著的延迟开销，尤其是在典型的设备端设置中。具体来说，由于设备端推理分为计算受限（Compute-bound）的预填充阶段和内存受限的解码阶段，我们可以通过运行并行解码路径来更好地利用计算单元（例如 NPU），同时保持最小的增量开销。我们进一步证明，以结果奖励模型（Outcome reward model）风格实现的神经验证，可以通过在基础模型的潜在表示（Latent representations）上训练的轻量级验证器头部（Verifier head）来实现。

这些组件共同使我们能够从一个非推理模型出发，逐步提升推理性能，同时保持在边缘设备上的可部署性。图 [1](#fig:overview) 全面概述了我们提出的端到端框架，展示了从适配器训练到设备端部署的完整流程。我们在 Qwen2.5 系列模型上实例化了这一流水线，并使用我们提出的流水线为这些模型启用了推理能力。设备端部署还需要额外的考量。我们讨论了一系列量化方案，支持 4 到 8 位的权重量化，同时允许激活值（Activations）使用更高精度以适应其动态范围，从而在整个流水线中最小化量化损失。得到的量化模型随后被导出并编译以供设备端执行。整个工作流程使用高通（Qualcomm）开源工具实现，包括 Qualcomm FastForward [@fastforward] 和 Qualcomm GENIE SDK [@GENIE]。在本文中，我们为支撑这项工作的设计选择和实证分析提供了更深入的见解。我们将最终的系统定位为一个实用的蓝图，用于在资源受限的边缘设备上部署具备推理能力的语言模型。

# Reasoning on Edge: System Design（边缘推理：系统设计）

<a id="fig:system_design"></a>

> 图 1 | **混合推理模型架构** 。该流程始于一个紧凑的基础 **大型语言模型（Large Language Model, LLM）** ，它通过基于 **低秩适配器（Low-Rank Adapter, LoRA）** 的 **监督微调（Supervised Fine-Tuning, SFT）** 专门用于推理。为了强制生成简洁内容并防止过度冗长，这些适配器会经历带有 **预算强制（Budget Forcing）** 的 **强化学习（Reinforcement Learning, RL）** 。最后，引入一个轻量级的 **切换器（Switcher）** 模块作为推理需求分类器，从而创建一个混合模型，该模型能根据任务复杂度动态地将传入查询路由到快速的基础模型或专门的推理适配器。

开发紧凑且功能强大的推理模型需要一个训练流程，该流程能够可靠地从相对较小的基础 LLM 中引出高质量的 **思维链（Chain-of-Thought, CoT）** 行为，同时避免不必要的冗长和过度计算。尽管预训练的 LLM 有时可以作为 **零样本（Zero-shot）** 推理器 [1]，但通过诸如 **草稿纸（Scratchpad）** [2] 或 **监督思维链（Supervised CoT）** [3] 等方法进行显式推理，对于在数学和编码任务上获得强大性能仍然至关重要。最近的系统，如 OpenAI O1 [4] 和专门的小型推理模型，包括 Tina [5]、Phi-4-mini [6] 和混合推理架构 [7]，表明当结合有针对性的微调和对齐技术时，这些能力可以被提炼到相对较小的模型中。

我们的目标是微调一个适中的基础 LLM，使其在复杂推理任务上具有竞争力，同时产生简洁的输出并最小化词元生成。为了保持适配的轻量级和可部署性，我们主要通过 **低秩适配器（Low-Rank Adapters, LoRA）** [8] 进行训练，这保留了可重用的冻结主干网络，并实现了模块化的推理专业化。图 [2](#fig:system_design) 所示的整体流程以保持效率并支持在严格的内存和延迟约束下部署的方式，集成了监督微调、强化学习和轻量级路由。

**监督微调（Supervised Fine-Tuning）** 构成了我们流程的第一阶段，旨在释放预训练基础 LLM 的推理能力。我们不进行密集微调，而是采用使用 LoRA 的 **参数高效微调（Parameter-Efficient Fine-Tuning）** 。LoRA 已被证明在推理场景下可以匹配甚至超越密集微调 [9]，同时使基础模型保持冻结状态并可跨多个领域重用。此阶段使模型具备了通过多步骤问题进行推理的基本能力，但正如广泛观察到的那样，它也增加了冗长性，并可能导致不必要地冗长或重复的推理轨迹 [10]。

为了完善推理行为并控制冗长性，我们应用了 **强化学习（Reinforcement Learning）** [11, 12]（RL），使用针对两个目标定制的自定义奖励函数：准确性和效率。首先，我们使用 **预算强制（Budget Forcing）** [13]，这是一种惩罚过长响应的机制。此约束鼓励模型在不牺牲正确性的前提下产生简洁的推理轨迹。其次，我们整合了一个基于答案的奖励，它直接激励模型生成正确的最终答案。为了优化，我们采用了 **基于组的相对策略优化（Group-based Relative Policy Optimization, GRPO）** 算法 [14]，该算法更新 LoRA 参数。

最后，并非所有用户查询都需要多步推理。为了避免不必要的计算，我们引入了一个轻量级的 **切换器（Switcher）模块** ，该模块基于隐藏的提示表示来预测是否需要推理。当不需要推理时，模型绕过 LoRA 适配器，直接依赖基础模型，从而减少延迟并限制 **KV 缓存（KV Cache）** 的增长，这是边缘部署的一个重要考量。

在接下来的章节中，我们将端到端流程分解为其主要组成部分：基于 LoRA 的适配（第 [3](#sec:lora) 节）、使用切换器模块动态激活或绕过这些适配器的动态推理时路由（第 [4](#sec:switcher) 节）、用于冗长控制的预算强制 RL（第 [5](#sec:budget_forcing) 节）、并行测试时缩放（第 [6](#sec:parallel) 节），以及包括量化和设备上执行的部署路径（第 [7](#sec:quantization) 节）。在每一节中，我们将描述该组件，然后通过有针对性的实验量化其影响，突出关键见解和聚焦消融研究，以分离出哪些设计选择驱动准确性，哪些改善了设备上的效率。

# 用于模块化推理的 LoRA {#sec:lora}

在我们的推理框架中，我们采用 **参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** ，原因有二：首先，它能够以较低的训练成本进行可扩展的实验；其次，它能生成模块化的适配器，可以在运行时启用或禁用，从而允许单个基础模型在通用聊天模式和增强推理模式之间切换。为了激发推理行为，我们在由更强教师模型（如 DeepSeek-R1 [@deepseekai2025deepseekr1] 和 QwQ-32B [@QwQ-32B]）生成的推理轨迹数据集上进行 **监督微调（Supervised Fine-Tuning, SFT）** 。

我们的实验表明，3B 和 7B 模型可以在成本效益高的设置下，通过对精选的轨迹数据集进行直接的 SFT 获得强大的推理能力，达到与规模大得多的蒸馏基线（例如 DeepSeek-R1-Distill-Qwen-7B）相当的性能。这些结果表明，强大的推理能力并不需要繁重的蒸馏流程或庞大的训练预算；高质量的轨迹数据（例如 OpenThoughts3 [@guha2025openthoughts]）结合轻量级的微调就足以弥合大部分差距。总的来说，这为构建有能力的推理模型提供了一条实用且可扩展的路径，而无需昂贵的基础设施。

## 实验设置 {#sec:experimental-setup}

我们 LoRA 适配阶段的主要目标是确定小型通用指令模型是否可以在不进行全参数蒸馏的情况下获得专家级的推理能力。因此，我们采用 Qwen2.5-3B-Instruct 和 Qwen2.5-7B-Instruct 模型 [@qwen2025qwen25technicalreport] 作为我们的核心实验骨干。在本节中，我们将概述用于通过 LoRA 激发推理行为的数据集和优化策略，以及用于严格评估由此产生的性能权衡的多样化数学、科学和编码基准测试套件。此外，我们还提供了一个广泛的 LoRA 超参数研究，旨在为 3B 和 7B 骨干模型确定最稳定且计算效率最高的适配策略。

## 训练细节 {#sec:training-details}

#### 数据。

在我们的实验中，我们使用了两个 SFT 数据集。第一个是 **Mixture of Thoughts（MoT）** [@openr1]，它包含从 DeepSeek-R1 模型蒸馏出的 350k 条推理轨迹。该数据集涵盖三个核心领域：数学（93.7k 条轨迹）、代码（83.1k 条轨迹）和科学（173k 条轨迹）。第二个数据集是 **OpenThoughts3-1.2M（OT3）** [@guha2025openthoughts]，包含 850k 个数学问题、250k 个代码问题和 100k 个科学问题。OT3 的标注轨迹是使用 QwQ-32B 模型生成的。

#### 训练配置。

所有模型均使用 bfloat16 数据类型训练 5 个轮次，启用了 DeepSpeed zero2 配置和 CPU 卸载。在所有配置中，我们应用了预热比例为 0.1 的余弦学习率调度，权重衰减设置为 0。

对于 MoT 数据集上的基线密集训练，Qwen2.5-3B-Instruct 和 Qwen2.5-7B-Instruct 模型的学习率设置为 $1\mbox{e$-$}5$，全局批次大小为 128。模型权重使用 AdamW 优化器进行优化，其参数为 $(\beta_1,\beta_2) = (0.9, 0.95)$。对于 OT3 数据集上的密集训练，我们遵循 [@guha2025openthoughts] 中描述的方案。密集模型使用 $8\mbox{e$-$}5$ 的学习率和 512 的批次大小进行训练，使用 AdamW 优化器，参数为 $(\beta_1,\beta_2) = (0.9, 0.999)$。

在我们的 LoRA 训练设置中，我们遵循采用相对较大学习率的常见做法。我们还发现，减小批次大小通常会导致更稳定的优化和更好的结果。在所有 PEFT 实验中，我们使用 LoRA 秩为 $128$，将 LoRA alpha 设置为该值的两倍，采用 $2\mbox{e$-$}{4}$ 的学习率，并以 $64$ 的批次大小进行训练。

## 评估细节 {#sec:evaluation-details}

#### 基准测试。

我们使用一套涵盖数学、科学和编程领域的多样化基准测试来评估所训练模型的推理能力。为评估多步骤数学问题求解，我们使用具有挑战性的竞赛数据集，包括 AIME 24/25 [@aime]、AMC23 [@amc23] 和 MATH500 [@hendrycks2021measuring]。科学推理能力则使用博士级别的 GPQA Diamond 数据集 [@rein2024gpqa] 来衡量。最后，为评估代码生成能力，我们利用 LiveCodeBench（v2，仅代码生成场景）[@livecodebench] 处理近期的竞赛编程问题，同时使用来自 HumanEval [@humaneval] 和 MBPP [@mbpp] 的标准 Python 编程任务，包括其经过严格验证的 EvalPlus 变体（HumanEval+ 和 MBPP+）[@evalplus]。每个基准测试的详细描述，包括问题数量和具体测试场景，在附录 [10] 中提供。

#### 评估流程（Evaluation pipeline）

我们的评估流程结构如下。对于推理基准测试，包括 AIME24、AIME25、MATH500、GPQA 和 AMC，我们允许的最大生成长度为 32,768 个词元（tokens），温度（temperature）设为 0.6，$\mathrm{top}\_{\_}\mathrm{p}$ 设为 0.95。对于在 OT3 数据集上训练的模型，我们采用原作者在 [@guha2025openthoughts] 中推荐的生成参数，具体将温度设为 0.7，$\mathrm{top}\_{\_}\mathrm{p}$ 设为 1.0，并保持最大生成长度为 32,768 个词元。同样，对于 LCB、MBPP 和 HumanEval（及其 "+" 版本），我们按照模型创建者的建议设置生成参数；我们将 LCB 的最大生成长度设为 32768 个词元，HumanEval 和 MBPP 的最大生成长度设为 1024 个词元。

鉴于某些基准测试包含的问题数量有限，因此更容易受到准确率波动的影响，我们执行多次评估运行以确保鲁棒性，其协议类似于 [@guha2025openthoughts] 中概述的方法。具体来说，我们对 AIME24、AIME25 和 AMC 数据集评估 10 次并报告平均结果。对于 GPQA，我们进行 4 次评估运行，对于 MATH500，我们运行一次评估。对于编程基准测试，pass@1 分数是从 LCB 的 16 个候选解决方案池和 HumanEval 与 MBPP 的 200 个候选解决方案池中估算得出的，使用的是首次在 [@humaneval] 中提出的无偏 pass@$k$ 估计器。所有评估均使用支持 vLLM [@kwon2023efficient] 的 lighteval 框架 [@lighteval] 进行，但对于 HumanEval、MBPP 及其增强变体，我们使用了 Evalplus 包[^4]。

## 结果（Results）

如表 [$$tab:lora_main_results$$](#tab:lora_main_results) 所示，我们评估了 Qwen2.5-3B-Instruct 和 Qwen2.5-7B-Instruct 模型在密集（dense）和基于 LoRA 的微调下的表现，并与关键基线进行了比较。在 OpenThoughts3（OT3）数据集上进行微调，对两个骨干模型都带来了最大且最一致的推理性能提升，显著提高了数学和科学基准测试的准确率，同时也提升了在更依赖推理的编程基准测试（LiveCodeBench）上的性能。相比之下，思维混合（Mixture of Thoughts, MoT）相比基础模型提供了明显的改进，但其增益始终小于 OT3。此外，在 OT3 上密集训练的 Qwen2.5-3B 模型，其性能与在 MoT 上密集训练的 Qwen2.5-7B 模型相当或略高，这表明更高的数据质量可以部分弥补较小的骨干模型规模。

对于 Qwen2.5-7B，在 OT3 上进行 LoRA 微调恢复了大部分密集 OT3 微调的改进，并且将适配器（adapter）的秩（rank）从 64 增加到 128 通常缩小了在核心推理基准测试（例如 AIME24 和 LCB）上的差距。值得注意的是，使用秩为 128 的 LoRA 进行 OT3 微调，与密集微调相比仅需更新 4.24% 的参数，却在多个推理基准测试上达到了接近 R1-Distill-Qwen-7B 基线的性能，这表明轻量级的适配器训练能够以显著更低的适应成本恢复蒸馏模型的大部分能力。然而，对于 Qwen2.5-3B，秩为 128 的 OT3 LoRA 在所有推理基准测试上都大幅落后于密集 OT3 模型，这突显了适配器容量和/或优化细节在较小规模下更为重要，并促使我们在后续章节中进行消融实验。

编码结果揭示了 **推理专业化（reasoning specialization）** 与“直接答案”代码生成之间的权衡。虽然 **监督微调（Supervised Fine-Tuning, SFT）** 持续提升了在 LCB 上的性能，但它总是导致 MBPP、HumanEval 及其各自的 .+ 变体性能有所下降。这种模式与向 **显式多步推理（explicit multi-step reasoning）** 的转变是一致的：它有助于解决像 LCB 这样更困难、对推理敏感的编码任务，但对于奖励简短、直接代码输出的基准测试则可能适得其反。具体而言，我们的 SFT 阶段旨在将推理行为引导至原本不进行推理的模型中，而 LCB 正是这种显式推理可能有益的场景。相比之下，MBPP 和 HumanEval 通常不会提示模型进行推理，并且必须直接给出响应 [^5]。有趣的是，对于 Qwen2.5-7B 模型， **OT3 LoRA（rank 64/128）** 通常比 **密集 OT3（dense OT3）** 保持了更强的 HumanEval/MBPP 性能，这表明 **参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 相对于 **密集微调（dense fine-tuning）** 可以部分缓解专业化/遗忘的权衡。先前的研究也观察到 [@huan2025does]，对推理轨迹进行 SFT 可能导致通用能力的部分遗忘，并且这种现象或许可以通过采用 **强化学习微调（Reinforcement Learning fine-tuning）** 来缓解 [@lai2025reinforcement]。

我们还为 7B 模型探索了一种两阶段训练策略，以检验我们是否能结合两个数据集的独特优势。我们的动机是，虽然 OT3 数据集（由较小的 QwQ 模型生成）提供了最强的基线性能，但 MoT 数据集可能包含互补的、高度复杂的推理模式，模型可以在后续训练阶段吸收这些模式。然而，有趣的是，在 OT3 训练之后再进行 MoT 训练，导致各基准测试的结果变化极小。准确率基本保持稳定，除了 AIME24 下降了 0.12 分，GPQA 提高了 0.05 分。

### LoRA 超参数研究：秩、学习率和批量大小 {#sec:ablations}

为了研究超参数对 PEFT 模型性能的影响，我们考虑了一系列学习率、批量大小和 LoRA 秩的值。我们在由 50,000 条数据组成的 OT3 子集上，对 Qwen2.5-3B-Instruct 和 Qwen2.5-7B-Instruct 模型进行了 1 个轮次的训练。对于每个模型，我们在以下范围内改变这些值：学习率在 $\{1\mbox{e$-$}4, 2\mbox{e$-$}4, 5\mbox{e$-$}4\}$ 中，批量大小在 $\{32, 64, 128\}$ 中，LoRA 秩在 $\{32, 64, 128, 256\}$ 中。LoRA 适配器应用于所有线性层，$\alpha$ 设置为 $2\times \text{rank}$， **丢弃率（dropout）** 固定为 $0.1$。对于我们的 **消融研究（ablation study）** ，我们在数学和科学基准测试上评估训练好的模型，包括 AIME24、AIME25、MATH500、GPQA Diamond 和 AMC23。与 [@guha2025openthoughts] 中使用的设置类似，我们对较小的数据集（如 AIME24、AIME25 和 AMC23）的准确率值进行 10 次运行平均，对 GPQA 进行 4 次运行平均，对 MATH500 则运行一次。我们将 **温度（temperature）** 设置为 0.7，$top\_p$ 值设置为 1.0。此消融研究的完整评估结果可在附录 [11](#sec:lora_ablation){reference-type="ref" reference="sec:lora_ablation"} 中找到。

#### Qwen2.5-3B-Instruct 结果。

在一组考虑的超参数上训练的模型的完整性能总结在附录的 [10](#tab:ablations_qwen3b){reference-type="ref+Label" reference="tab:ablations_qwen3b"} 中。为了分离每个超参数的影响，我们通过对其他两个维度（学习率、批量大小和 LoRA 秩）的准确率进行平均来总结结果，并突出主要趋势。如 [1](#tab:lr_summary){reference-type="ref+Label" reference="tab:lr_summary"} 所示，学习率对性能有显著影响，$2\mbox{e$-$}4$ 在所有任务中提供了最高的总体平均准确率。我们还观察到不同基准测试之间存在相反的敏感性：AIME24 和 AIME25 随着学习率增加而改善，而 MATH500 则下降，这表明较大的学习率可能导致对更复杂的数学推理产生 **过适应（over-adaptation）** 。

[]{#tab:lr_summary label="tab:lr_summary"}

**LR** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

$1\mbox{e$-$}4$ 0.040 0.025 0.573 0.277 0.281 0.239
$2\mbox{e$-$}4$ **0.051** 0.033 **0.574** 0.274 **0.299** **0.246**
$5\mbox{e$-$}4$ **0.051** **0.038** 0.556 **0.280** 0.275 0.240
: 按学习率分组的 Qwen2.5-3B-Instruct 平均性能。

如表 [2](#tab:rank_summary) 所示， **低秩适应（Low-Rank Adaptation, LoRA）** 秩对整体性能表现出最一致的正向影响。提高秩通常能提升准确率，秩 $256$ 达到了最高的总体平均值。然而，秩 $128$ 在大多数基准测试上已经表现强劲，使其成为适配器内存受限时的一个实用操作点。在不同任务中，AIME25 对秩特别敏感，而 MATH500 则相对稳定。总体而言，在资源允许的情况下，更高的秩更可取，但秩 $128$ 为边缘部署提供了有利的 **准确率-效率权衡（accuracy-efficiency trade-off）** 。

[]{#tab:rank_summary label="tab:rank_summary"}

**秩（Rank）** **可训练参数百分比（%TP）** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **平均值（Avg）**

---

32 1.94% 0.047 0.020 0.568 0.270 0.284 0.238
64 3.88% 0.040 0.023 0.570 **0.283** 0.284 0.240
128 7.76% **0.054** 0.038 0.560 0.272 **0.287** 0.242
256 15.52% 0.048 **0.048** **0.573** 0.282 0.284 **0.247**
: 按 LoRA 秩分组的 Qwen2.5-3B-Instruct 平均性能。%TP 表示可训练参数的百分比。

最后，与学习率和秩相比， **批大小（Batch size）** 的影响相对较小，如 [3](#tab:bs_summary) 所示。当批大小设置为 64 时观察到最佳整体性能，尽管差异很小。MATH500 从较大的批次（如 128）中略有受益，而 AIME25 在 64 时达到峰值。这些结果表明，批大小的选择可以主要基于计算约束，而对整体性能没有显著影响。

[]{#tab:bs_summary label="tab:bs_summary"}

**批大小（Batch size）** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **平均值（Avg）**

---

32 **0.054** 0.030 0.562 0.277 **0.288** 0.242
64 0.048 **0.038** 0.566 **0.278** 0.287 **0.243**
128 0.040 0.028 **0.575** 0.277 0.281 0.240
: 按批大小分组的 Qwen2.5-3B-Instruct 平均性能。

#### Qwen2.5-7B-Instruct 结果。

附录中的 [11](#tab:qwen7_ablations) 总结了 Qwen2.5-7B-Instruct 在学习率、批大小和适配器秩上的 LoRA 消融实验。与 3B 设置（学习率影响性能，但各测试值间的平均值变化不大）相比，7B 主干模型表现出更窄的稳定学习率范围（表 [4](#tab:lr_summary_second_model_full)）：LR=$1\mbox{e$-$}4$ 和 $2\mbox{e$-$}4$ 训练可靠，而 LR=$5\mbox{e$-$}4$ 通常不稳定，并可能导致训练崩溃。总体而言，一个实用的指导原则是： **在较大的主干模型上进行稳定训练时使用较低的学习率，并在该稳定范围内调整其余超参数** 。对于下面展示的各个表格，我们明确地考虑了这种区别。报告不同学习率设置平均值的表格包含了所有运行结果，以突显较高学习率引入的不稳定性。相反，报告批大小和 LoRA 秩平均准确率的表格则排除了学习率为 $5\mbox{e$-$}4$ 的运行结果，因为这些配置包含了发散的结果。

[]{#tab:lr_summary_second_model_full label="tab:lr_summary_second_model_full"}

**LR** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

$1\times10^{-4}$ 0.158 0.132 0.775 **0.366** 0.533 0.393
$2\times10^{-4}$ **0.170** **0.148** **0.779** 0.354 **0.543** **0.399**
$5\times10^{-4}$ 0.148 0.112 0.635 0.350 0.439 0.337
: 按学习率分组的 Qwen2.5-7B-Instruct 平均性能。

如表 [5](#tab:rank_summary_second_model_full){reference-type="ref" reference="tab:rank_summary_second_model_full"} 所示， **低秩自适应（Low-Rank Adaptation, LoRA）** 的秩对 Qwen2.5-7B 性能有可测量但相对较小的影响：平均得分从 0.388（秩 32）提升至 0.402（秩 128），而秩 256 的结果为 0.397，与之相当。总体而言，秩 64 至 128 形成了一个紧密的权衡区域，其中秩 128 在平均表现上最佳，但仅略优于较低的秩。与 30 亿参数模型相比，70 亿参数模型的结果在不同秩之间分布更为集中，这表明对于更大的骨干模型，秩是一个相对较弱的调节杠杆。

[]{#tab:rank_summary_second_model_full label="tab:rank_summary_second_model_full"}

**Rank** **%TP** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

32 1.06% 0.152 0.119 0.776 0.357 0.534 0.388
64 2.12% 0.159 0.150 0.779 0.355 0.535 0.396
128 4.24% **0.178** 0.141 **0.784** **0.367** **0.541** **0.402**
256 8.48% 0.165 **0.152** 0.771 0.359 0.536 0.397
: 按 LoRA 秩分组的 Qwen2.5-7B-Instruct 平均性能。%TP 表示可训练参数的百分比。

类似地，表 [6](#tab:batch_summary_second_model_full){reference-type="ref" reference="tab:batch_summary_second_model_full"} 显示，在我们的参数扫描中， **批大小（Batch size）** 对 Qwen2.5-7B 性能的影响可以忽略不计。平均准确率仅在 0.394 到 0.398 之间变化。这与 30 亿参数模型的趋势一致，表明一旦学习率处于稳定区间，批大小主要可以根据计算资源约束来选择。

[]{#tab:batch_summary_second_model_full label="tab:batch_summary_second_model_full"}

**Batch size** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

32 0.154 **0.148** **0.781** 0.355 0.530 0.394
64 0.164 0.146 0.774 **0.366** 0.538 **0.398**
128 **0.172** 0.127 0.776 0.358 **0.541** 0.395
: 按批大小分组的 Qwen2.5-7B-Instruct 平均性能。

根据超参数研究的结果，可以得出以下结论： **学习率（Learning rate）** 为 $2\times10^{-4}$、批大小为 64、LoRA 秩设置为 128 的配置能够持续产生良好的结果，同时在效率和训练稳定性之间提供了良好的折衷。

# 通过切换器模块实现动态 LoRA 路由 {#sec:switcher}

虽然推理模型擅长解决复杂问题，但并非每个用户查询都需要详尽的多步骤 **思维链（Chain-of-Thought, CoT）** 。对于标准的对话提示或简单的知识性问题，生成长推理轨迹会产生不必要的延迟和计算开销，这可能成为边缘设备的关键瓶颈。为了解决这个问题，我们引入了一个轻量级的 **切换器（Switcher）** 模块，以实现动态适配器路由。通过分析用户提示，切换器决定是绕过还是激活特定于推理的 **LoRA（Low-Rank Adaptation）** 适配器。当切换器被禁用时，系统作为高效的原始指令模型运行，用于常规对话；当被激活时，它会无缝切换为专门的推理引擎。

在架构上，切换器作为基础 **大型语言模型（Large Language Model, LLM）** 顶部的辅助分类头，并在预填充阶段运行。它处理来自最终 Transformer 层的隐藏状态，并计算平均序列表示。基于此表示，切换器执行二元分类，以确定输入序列是否对应于面向推理的任务。如果被分类为此类任务，则推理 LoRA 适配器将在后续解码中被激活。

切换器头被实现为一个轻量级 **多层感知机（Multilayer Perceptron, MLP）** ，其单个隐藏维度为 $8$，使用 **ReLU（Rectified Linear Unit）** 激活函数， **丢弃率（Dropout rate）** 为 $p = 0.2$。这种紧凑的架构确保了高效设备端推理的开销可忽略不计，同时保持了足够的表达能力以进行序列级分类。

在边缘设备上，预填充阶段是计算密集型的。一次性处理长输入提示会产生过高的计算开销。为了缓解这个问题，实际的设备端实现通常将预填充序列划分为更小的离散块。切换器模块被明确设计为支持这种分块预填充策略。切换器不是缓冲整个提示的隐藏状态来计算全局平均值，而是动态更新其序列表示。具体来说，我们计算这些块上隐藏状态的运行指数移动平均值。在我们的设置中，我们使用块大小为 $128$ 个词元，平滑系数为 $\alpha = 0.5$。为了增强对量化伪影的鲁棒性，我们在训练期间向平均表示中注入均值为零、标准差为 $\sigma = 0.5$ 的独立高斯噪声。

#### 用于 KV 缓存重用的掩码 LoRA 训练。

在推理时动态激活 LoRA 适配器的一个主要挑战是 **KV 缓存（Key-Value cache）** 的兼容性。在标准的 LoRA 训练下，模型期望提示词元（预填充阶段）的 KV 缓存是在 LoRA 适配器完全激活的情况下生成的。如果一个查询在基础模型已经编码了提示之后被路由到推理模式，系统通常需要重新编码整个提示并激活 LoRA 适配器，以生成兼容的 KV 缓存。在边缘设备上，这种重新编码会导致严重的延迟和计算惩罚。为了消除这种低效性，我们引入了一种 **掩码 LoRA（Masked LoRA）** 训练策略。在对推理适配器进行微调期间，我们在提示词元的前向传播过程中掩码（禁用）LoRA 权重，仅在生成响应词元时激活它们。这迫使 LoRA 适配器去适应严格由基础模型生成的提示 KV 缓存。根据经验，我们观察到这种策略不会导致推理准确性下降，同时允许基础模型和推理模式无缝共享单个预填充 KV 缓存，完全消除了切换时需要重新编码提示词元的需求。

## 训练细节

为了训练切换器，我们构建了一个小型数据集，其中结合了简单的对话或基于知识的查询以及复杂的查询，使模型能够学习如何区分何时需要推理。数据集的结构如下：每个条目由一个提示问题和一个标签（0 或 1）配对组成，分别表示低复杂度或高复杂度。复杂度是根据提示所采样的数据集来源确定的。我们纳入了涵盖数学和非数学领域的数据集中的提示，以降低切换器对领域特定线索过拟合的风险（例如，假设所有数学问题本质上都更难）。

最终数据集包含约 2000 个样本。简单查询随机抽取自 **SQuAD2.0 数据集（SQuAD2.0 dataset）** （600 个问题）[@rajpurkar2018knowdontknowunanswerable]，该数据集主要由常识理解问题构成；以及 **MMLU 数学子集（MMLU math subset）** （419 个问题）[@hendrycks2021measuring]，其中包含简单的数学问题。困难提示词则来源于 **S1K 数据集（S1K dataset）** 的一个子集（500 个问题）[@muennighoff2025s1]，涵盖数学、科学和填字游戏等领域的挑战性问题，以及 **StrategyQA** （500 个问题）[@geva2021didaristotleuselaptop]，后者覆盖了非科学领域的推理问题。

## 结果（Results）

**切换器模块（Switcher module）** 的主要动机是优化标准的日常用户交互。在实际的边缘部署中，绝大多数用户查询是简单的对话、事实查询或基本指令，无需多步推理。通过为切换器设置积极的阈值，将这些简单查询路由到基础指令模型，我们可以在 **词元生成（token generation）** 、延迟和功耗方面实现巨大的总体节省，从而将 **LoRA 推理适配器（LoRA reasoning adapters）** 严格保留给复杂任务。

<a id="fig:switcher_pareto"></a>

![](./images/switcher_pareto.png)

> 图 7： **切换器模块对 MATH500 的影响。** **左图：** 组合模型准确率随路由至推理适配器的查询比例变化。 **右图：** 不同切换器阈值下的平均完成长度与总体准确率。

为了严格评估这种动态路由的影响，我们在具有挑战性的 **MATH500 基准测试（MATH500 benchmark）** 上评估了切换器的性能，因为该基准包含不同复杂程度的问题。我们遍历了不同的切换器置信度阈值，以改变路由到推理适配器与基础模型的提示词比例。图 [3](#fig:switcher_pareto){reference-type="ref" reference="fig:switcher_pareto"} 说明了随着更大比例的查询被路由到推理模型，整体模型性能如何变化。在此，我们研究了基础的 **Qwen2.5-7B-Instruct 模型（Qwen2.5-7B-Instruct model）** 及其在 **OT3** 上训练的对应模型，该模型配备了秩为 128 的 LoRA 适配器，且未施加任何预算强制约束。

随着更多答案通过推理生成，准确率从基础模型的基线水平平滑上升，趋近于纯推理模型的上限。这表明切换器有效地在更复杂、最能受益的查询上优先使用推理模型，从而实现了仅靠基础模型无法达到的准确率水平。

右图显示了相应的计算成本，以平均完成长度衡量。选择更高准确率的运行模式需要按比例增加计算成本，而当准确率要求适中时，则可以实现较低成本的运行模式。因此，切换器为权衡这一取舍提供了一个灵活的机制。

# 预算强制与推理时计算优化（Budget Forcing and Inference-Time Compute Optimization） {#sec:budget_forcing}

**思维链提示（Chain-of-Thought prompting, CoT prompting）** [@wei2022chain] 通过解码中间推理步骤来扩展 **推理时计算（Inference-Time Compute, ITC）** ，显著提升了 **大型语言模型（Large Language Models, LLMs）** 在复杂任务上的性能，但通常以高延迟和大量词元开销为代价。理论分析 [@zhang2025laws] 认为，最优的测试时计算量应与问题难度线性相关。然而，无约束的模型通常会违反这种最优性，即使在简单任务上也表现出退化的冗长和过度思考 [@muennighoff2025s1]。

为了协调推理能力与计算效率， **预算强制（Budget Forcing）** [@muennighoff2025s1] 方法旨在使生成过程与明确的词元/计算约束对齐。基于 **强化学习（Reinforcement Learning, RL）** 的方法，以及其他方法 [@xu2025cod; @renze2024ccot; @wang2025ton; @huang2025hapo]，在性能与 CoT 长度缩减之间实现了令人印象深刻的权衡。这些方法通常需要通过添加基于长度的惩罚项来增强奖励 [@aggarwal2025l1]，或者在达到目标预算时强制执行硬截断约束 [@liu2025dler]。形式上，一个标准的预算强制奖励目标可以表示为加性惩罚：

$$
\label{eq:penalty}
    R(y,x) = R_{\text{accuracy}}(y,x) - \lambda \cdot R_{\text{budget}}(L)
$$

其中 $x$ 是提示（prompt），$y$ 是生成的响应，$R_{\text{accuracy}}(y,x)$ 是准确性奖励，$L$ 代表总词元长度，$R_{\text{budget}}(L)$ 是由超参数 $\lambda$ 缩放的惩罚函数。

## 软屏障奖励公式（Soft-Barrier Reward Formulation） {#subsec:reward_design}

基于预算惩罚强化学习的基础，我们的奖励设计基于三个核心理由：

1.  **避免严格的词元匹配：** 我们不强制模型精确匹配预定义的预算。这样做假设了对特定任务所需最优计算量的完美先验知识，这与生成式探索的前提相矛盾。
2.  **轨迹探索：** 模型必须保留足够的自由度来探索多样化的推理路径，而不会过早地被截断。
3.  **遵循提示的预算合规性：** 模型必须可靠地满足提示中提供的用户定义预算约束。

为了实现这些原则，我们使用离散的生成长度预算来提示模型，具体将约束划分为 $1000$、$3000$、$4000$ 和 $6000$ 个词元。我们不是采用加性惩罚，而是引入了一个乘性的、分段线性的 **软屏障（soft barrier）** 。当生成长度超过提示的预算区间时，该屏障会使预算奖励从 $1.0$ 衰减到 $0.0$。这种线性衰减起到了缓冲作用，既阻止模型超出限制，又不会因轻微的预算违规而施加灾难性的惩罚。

这种衰减在一个以目标预算 $B$ 为中心的对称窗口内起作用，其中窗口的半宽 $m \in [0,1]$ 被视为一个可调的超参数。形式上，我们将预算奖励修正因子定义为：

$$
\label{eq:budget}
    R_{\text{budget}}(L) =
    \begin{cases}
        1 & L \le L_{\text{low}} \\
        p & L > L_{\text{high}} \\
        1 - (1-p)\frac{L - L_{\text{low}}}{L_{\text{high}} - L_{\text{low}}} & L_{\text{low}} < L \le L_{\text{high}}
    \end{cases}
$$

其中 $L$ 是生成响应的总长度，$p$ 是最大预算惩罚下限，$L_{\text{low}} = (1-m)B$，且 $L_{\text{high}} = (1+m)B$。通过经验观察，我们注意到设置负的惩罚下限没有带来优化收益；因此，我们设 $p = 0$。

最终的 **整体奖励（holistic reward）** $R$ 随后被定义为任务准确率奖励与预算遵从修正因子的乘积：

$$
\label{eqn:bf_reward}
    R(y,x) = R_{\text{accuracy}}(y,x) \times R_{\text{budget}}(L)
$$

其中 $R_{\text{accuracy}}(y,x) \in \{0, 1\}$ 代表二元的准确率奖励。

#### 挑战与奖励黑客行为（Challenges and reward hacking）。

由于 **预算强制（budget forcing）** 对优化流形施加了严格的约束，它本质上引发了推理性能与计算成本之间的权衡。我们观察到，简单地应用长度惩罚（例如，*仅*惩罚思维链（Chain-of-Thought, CoT）推理轨迹中的词元）极易受到 **奖励黑客行为（reward hacking）** 的影响。在早期迭代中，策略会迅速坍缩为一种退化的、“懒惰”的策略：模型学会通过过早地用 `</think>` 标记关闭推理块来规避惩罚，然后在最终响应输出中继续其冗长的 CoT。通过惩罚总生成长度 $L$ 而不仅仅是推理轨迹，我们的乘法公式有效地中和了这种利用。此外，虽然我们的最终奖励公式去除了显式的格式遵循奖励，但经验评估证实，模型在整个训练过程中始终保持着所需的结构化格式。

## 实验设置（Experimental Setup） {#subsec:experimental_setup}

### 训练细节（Training Details）

为了证明我们的 **软屏障奖励公式（soft-barrier reward formulation）** 在压缩 CoT 推理方面的有效性，我们在最先进的推理模型上进行了广泛的实验。我们使用 DeepScaleR 数据集 [@luo2025deepscaler] 作为主要的训练语料。为了最大化训练稳定性并防止退化的优化步骤，我们对数据集应用了严格的过滤标准：任何提示（prompt）若其组奖励标准差为零，则被移除，以确保模型在策略更新期间始终接收到有意义的比较信号。

我们使用 GRPO（Group Relative Policy Optimization，组相对策略优化）[@shao2024deepseekmath] 来优化我们的模型。GRPO 特别适合推理任务，因为它通过利用组缩放奖励绕过了对单独价值模型的需求。对于给定的提示 $x$，目标是最小化以下损失：
$$\mathcal{L}_{\mathrm{GRPO}}(\theta \mid x) = -\frac{1}{G}\sum_{i=1}^{G} \min\!\Big( \rho_i\,A_i,\; \operatorname{clip}(\rho_i,\,1-\epsilon,\,1+\epsilon)\,A_i \Big) + \beta\,D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\big\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big),$$
其中 $G$ 表示组大小，概率比 $\rho_i$ 和优势 $A_i$ 定义为：
$$\rho_i = \frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}, \qquad A_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}$$
这里，$\mu_r$ 和 $\sigma_r$ 分别代表组内奖励的均值和标准差：
$$\mu_r = \frac{1}{G}\sum_{j=1}^{G} r_j, \qquad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^{G} (r_j - \mu_r)^2}$$

我们的实现基于 `trl` 库（版本 0.26.2）[@vonwerra2020trl]。我们在一个配备 8 块 NVIDIA H100（80GB）GPU 的单一计算节点上执行训练。在 GRPO 推演（rollouts）期间，我们为每个提示采样 8 个生成结果（$G=8$）。训练超参数的完整总结详见表 [12](#tab:hyperparams){reference-type="ref" reference="tab:hyperparams"}（位于 [12](#app:budget_forcing) 部分）。

### 评估细节（Evaluation Details）

[为评估我们的预算强制技术对数学推理的影响，我们使用大规模 Math500 [@lightman2023lets] 基准作为主要测试平台。]{style="color: black"} 为确保评估的稳健性和可复现性，我们采用 `lighteval` 框架（版本 0.8.1）。所有推理过程均使用 vLLM（版本 0.10.2）进行加速。为标准化 pass@1 准确率的评估，我们在所有基准测试中应用一致的采样策略：以温度（temperature）为 0.6、$\text{top}\_\text{p}$ 为 0.95 进行生成采样，并将最大完成长度（maximum completion length）扩展至 32K 词元（tokens），以容纳基线模型可能遗留的冗长推理轨迹。

## 结果：效率-准确率权衡（Results: Efficiency-Accuracy Trade-off） {#subsec:main_results}

如前所述，我们的主要目标是在任务性能下降最小的情况下压缩生成的推理轨迹。由于我们的软屏障奖励（soft-barrier reward）公式有意省略了预算惩罚项的严格正则化权重（参见公式 [$$eq:penalty$$](#eq:penalty){reference-type="ref" reference="eq:penalty"}），我们发现 GRPO（Group Relative Policy Optimization）中的 **库尔贝克-莱布勒散度（Kullback-Leibler divergence, KL divergence）** 惩罚系数 $\beta_{\text{KL}}$ 可作为强制预算友好行为的有效控制机制。经验表明，设置 $\beta_{\text{KL}} = 10^{-3}$ 能产生最佳平衡，显著减少生成长度，同时性能下降可忽略不计。相反，较宽松的惩罚 $\beta_{\text{KL}} = 10^{-4}$ 能在极短的完成长度下提高格式遵循度，但代价是在更大、无约束的上下文上进行评估时，性能回归（performance regression）会略微增加。

图 [4](#fig:cdf_avg_len){reference-type="ref" reference="fig:cdf*avg_len"} 展示了无约束基线模型（紫色）与使用 $\beta*{\text{KL}} = 10^{-3}$ 训练的两个中间检查点（checkpoints）的平均完成长度分布。左图和右图分别描绘了最大完成长度被严格限制在 4K 和 6K 词元时的评估情况。为了在推理中强制执行此硬预算（hard budget），我们在达到词元限制时立即截断生成，随后附加一个提示（prompt），强制模型立即输出其最终答案。

![ **平均完成长度分布（Average Completion Length Distributions）。**   **左图：**  强制最大完成长度为 4K 词元的评估。 **右图：**  最大完成长度为 6K 词元的评估。请注意，分布尾部延伸至零以下或超出最大预算是 **核密度估计（Kernel Density Estimation, KDE）** 曲线平滑的标准产物。从基线（[紫色]{style="color: purple"}）到中间检查点（[蓝色]{style="color: blue"}），再到最终 **强化学习（Reinforcement Learning, RL）** 微调检查点（[绿色]{style="color: green"}）的进展，展示了稳定、渐进地学习简洁生成的过程（$\beta_{\text{KL}}=10^{-3}$）。](media/cdf_avg_len.pdf){#fig:cdf_avg_len width="\\linewidth"}

如图 [4](#fig:cdf_avg_len){reference-type="ref" reference="fig:cdf_avg_len"} 所示，我们的 RL 微调有效地将分布密度向显著更短的长度方向移动。至关重要的是，从基线（紫色）到中间检查点（蓝色），再到最终策略（绿色）的转变，突显了一个稳定、渐进的优化轨迹。模型并未经历突然、不稳定的策略崩溃（policy collapse），而是在训练过程中平滑且单调地学会了生成更简洁的推理轨迹来解决问题。

为了量化这种压缩效果，图 [5](#fig:compl_dist_comparison){reference-type="ref" reference="fig:compl_dist_comparison"} 提供了通过我们的 RL 微调实现的 **思维链（Chain-of-Thought, CoT）** 长度减少的细粒度分解。具体而言，图 [5](#fig:compl_dist_comparison){reference-type="ref" reference="fig:compl_dist_comparison"} 的右图显示，我们的方法实现了平均完成长度减少因子约为 $\sim 2.4\times$，在某些查询上最大压缩率可达 $\sim8\times$。如前所述，这种对冗余性的积极削减是在保持与基础模型（base model）相当性能的同时实现的，准确率下降极小——在许多情况下可忽略不计。实际上，这种减少直接转化为更低的总体推理延迟（inference latency）和更快的最终答案生成时间（time-to-final-answer），使得高级推理模型在资源受限环境中的部署可行性显著提高。

![ **平均完成长度比较（Average Completion Length Comparison）。**   **左图：**  基础模型（[橙色]{style="color: orange"}曲线）与 RL 微调模型（[绿色]{style="color: green"}曲线）的平均完成长度的 **累积分布函数（Cumulative Distribution Function, C.D.F.）** ，其中 $\beta_{KL}=1.e^{-3}$。我们考虑的最大完成长度为 6K 词元。 **右图：**  RL 微调模型带来的完成长度减少。我们使用了左图中的相同模型。RL 微调模型实现的平均减少长度为 $2.38 \pm 0.07$。](media/compl_dist_comparison.pdf){#fig:compl_dist_comparison width="\\linewidth"}

**模型** &\

& **预算 = 1K** & **预算 = 2K** & **预算 = 4K** & **预算 = 6K** & **平均（预算 = 32K）** \
SFT 基线（r=128） & 34 & 57 & 73 & 83 & **95** \
BF RL${}^{\beta_{KL}=1.e-3}$ & [62]{.underline} & [78]{.underline} & **85** & **90** & [92]{.underline}\
BF RL${}^{\beta_{KL}=1.e-4}$ & **72** & **80** & [84]{.underline} & [85]{.underline} & 90\

---

<a id="fig:cot_example_2"></a>

> 图 2： **代数化简的定性比较。** **中部：** 基线轨迹虽然立即正确识别了平方差策略，但陷入了过度的自我验证，通过展开、直接计算和替代因式分解等方式重新计算结果。 **底部：** 预算强制轨迹识别出嵌套的平方差结构，并线性地执行求解，没有冗余检查。

## 预算强制思维链的定性分析 {#subsec:qualitative_examples}

为了在轨迹层面考察我们预算强制目标的作用机制，我们定性地比较了无约束基线模型与我们的预算强制模型在四个不同数学领域的推理轨迹：数论、代数化简、模式识别和模运算（图 [6](#fig:cot_example_2){reference-type="ref" reference="fig:cot_example_2"}--[7](#fig:cot_example_4){reference-type="ref" reference="fig:cot_example_4"} 以及 [12](#app:budget_forcing) 中的图 [9](#fig:cot_example_1){reference-type="ref" reference="fig:cot_example_1"}--[10](#fig:cot_example_3){reference-type="ref" reference="fig:cot_example_3"}）。分析中呈现出一致的模式：无约束基线经常遭受严重的 **认知犹豫（Epistemic hesitation）** 。虽然它通常在生成过程的早期就识别出正确的逻辑策略，但它会耗费数千个词元在冗余的自我验证、测试替代（且通常效率较低）的方法以及假设存在微小错误上。例如，在图 [6](#fig:cot_example_2){reference-type="ref" reference="fig:cot_example_2"} 和图 [7](#fig:cot_example_4){reference-type="ref" reference="fig:cot_example_4"} 中，基线几乎立即得出了正确答案，但却陷入了广泛的验证循环，使用三到四种不同的方法重新计算结果。

相比之下，我们的预算强制策略学会了自信地信任其初始的、正确的逻辑推导。它成功地剪除了这些冗余的验证循环和冗长的句法解析，在严格保留核心推理主干和人类可读性的同时，大幅缩短了轨迹长度。此外，尽管为了简洁起见，我们在可视化中省略了特殊的分隔符标记（例如 `<think>`），但我们观察到预算强制模型能够稳健地生成它们，保持了所需的思维链（CoT）和最终答案的格式约束。

<a id="fig:cot_example_4"></a>

> 图 3： **模运算的定性比较。** **中部：** 基线轨迹立即正确计算了和与余数，但使用了四种不同的方法（逐步加法、数字和规则、配对和重新计算）进行了广泛而冗余的验证。 **底部：** 预算强制轨迹执行了直接计算并毫不犹豫地返回了结果。

# 并行测试时扩展与推理 {#sec:parallel}

在使用 **大型语言模型（Large Language Models, LLMs）** 进行自回归生成时，主要的运行时瓶颈之一是反复加载各层的模型权重以生成下一个词元。因此，通常存在在不显著增加运行时开销的情况下，增加每个用户查询总计算量的潜力。实现这一点的一个直接方法是并行生成多个独立样本：这种方法不是生成一条 **思维链（Chain-of-Thought, CoT）** 轨迹和一个最终答案，而是分配额外的计算资源来同时生成多条轨迹。

并行生成不仅从系统角度来看具有吸引力，还能提高准确性。在各种基准测试和模型中，生成多个候选解决方案然后进行聚合，已反复显示出持续的性能提升。一种常见且简单的聚合策略是对各个答案进行 **多数投票（Majority Voting）** ，即选择出现频率最高的答案作为最终答案。在实践中，多数投票被广泛用作一种可靠工具，用于从相同的基础模型中进一步提升最终模型性能。

早期的研究工作 [@cobbe2021training; @wang2022self] 表明，可以使同一个 LLM 生成多个多样且独立的响应，以找到更好的解决方案。在聚合之前，通过采样多个独立响应来扩展计算已被证明 [@brown2024large; @wu2025inference; @snell2025scaling] 是一种有效的范式，使得较小的模型在相同的推理计算预算下能够超越较大的模型。在这些情况下，关键是对解决方案进行评分（例如，通过频率或使用外部验证器）并预测得分最高的解决方案。并行推理设计依赖于选择采样方案、设计和采用奖励模型，以及最终的聚合方案来生成最终答案。响应之间的 **多样性（Diversity）** [@brown2024large] 对于提高采样到正确响应的概率至关重要。增加多样性通常通过以更高的 **采样温度（Sampling Temperature）** [@renze2024effect] 进行自回归生成来实现。虽然响应通常是独立采样的，但最近的研究探索了 **相互依赖的采样（Inter-dependent Sampling）** [@hsu2025group; @pan2025learning; @rodionov2025hogwild; @zheng2025parallel] 和 **引导搜索（Guided-search）** [@yao2023tree; @ning2023skeleton; @li2025enhancing] 以生成更好的响应候选池。在本工作中，我们专注于独立采样方案。 **奖励模型（Reward Models）** （也称为 **验证器（Verifiers）** ）为给定响应估计一个或多个标量值分数。关于改进奖励模型的研究由来已久，包括开发更准确的奖励模型 [@liu2024skywork; @wang2024helpsteer2]、更细粒度的分数分配（例如，使用 **过程奖励模型（Process Reward Models）** 的每步分数）[@lightman2023let; @wang2023math; @zhang2025lessons]、利用更多计算资源扩展验证 [@liu2025inference]，以及将奖励模型扩展到数学推理之外的其他推理领域 [@chae2025web; @chen2025scaling]。在本工作中，我们使用奖励模型对结果进行评分。最后， **聚合（Aggregation）** 对于从多个响应-分数对的池中草拟一个特定响应至关重要。现有的工作主要从 **排序-选择（Rank-and-select）** 的角度来看待这个问题：要么根据出现频率（“自我一致性”或“多数投票”）[@wang2022self] 对响应进行排序，要么使用外部奖励模型对响应进行评分 [@cobbe2021training; @wang2023math]。需要注意的是，这些策略会导致零和局面：排名最高的解决方案被选中，其余的被丢弃。因此，最近的一系列工作 [@khairi2025making; @li2025drafts; @qi2025learning; @zhao2025majority] 研究基于候选响应 **合成（Synthesizing）** （而非选择）最终响应。

在测试时扩展计算通常意味着额外的延迟和计算开销，因此在资源受限的 **边缘设备（Edge Devices）** 上实现尤其具有挑战性。文献 [@hao2025scaling] 讨论了使用独立验证器在边缘设备上进行并行 **测试时扩展（Test-Time Scaling, TTS）** 的一些优势。这种分离阻碍了中间计算（例如通过 **KV 缓存（KV-cache）** ）的有效重用。因此， **高效地（Efficiently）** 扩展并行计算已引起研究界的一些关注。为了实现高效推理，已有几项工作研究了通过执行更少的 **浮点运算（Floating Point Operations, FLOPs）** [@wu2025inference] 和减少生成过程中所需的 **内存占用（Memory Footprint）** [@sun2024fast; @hooper2025ets] 来生成解决方案。与本文提出的验证器设计最接近的是 **GenRM** [@zhang2025generative]，其中作者研究了微调基础生成模型，使其额外执行基于提示的验证。这种联合生成-验证范式对边缘设备很有吸引力，因为生成和验证可以在 **DRAM** 和 **闪存（Flash Memory）** 之间以最小的参数移动来完成。

## **面向边缘计算的高效验证器设计（Efficient Verifier Design for Edge Compute）**

并行推理会产生一组 $N$ 条独立的 **思维链（Chain-of-Thought, CoT）** 轨迹及其对应的最终答案。随之而来的核心问题是：给定这 $N$ 个候选答案，我们如何可靠地选择出正确的那一个？ **多数投票（Majority voting）** 提供了一个强有力的基线，但它并非总是足够有效，尤其是在答案空间很大或存在歧义的情况下。这促使我们引入一个明确的 **选择机制（selection mechanism）** ，该机制能够对候选答案进行评分并倾向于选择最可能正确的答案，同时仍能从独立采样所产生的多样性中获益。

一种自然的方法是添加一个 **验证器模型（verifier model）** 来评估每个候选解决方案。然而，在边缘设备上部署一个与生成器规模相当的独立验证器，在存储和内存占用方面可能代价高昂，并且通常在延迟方面尤其昂贵。为了避免这种情况，我们的目标是尽可能高效地复用生成器。具体来说，我们保持相同的 **基础模型（base model）** ，并添加一个轻量级的 **验证器头部（verifier head）** ：一个应用于最终 **词元嵌入（token embedding）** 的独立线性层，后接一个 **Sigmoid 激活函数（sigmoid activation）** ，以产生一个标量形式的“正确性”分数。图 [1](#fig:overview) b 展示了这种方法。这种设计将额外参数保持在最低限度，并且至关重要的是，它允许对所有生成的响应进行 **KV 缓存（KV-cache）** 复用，因为验证器头部可以基于生成过程中已经计算好的表征进行操作。

除了线性头部，我们还在每个生成的响应后附加一个简短的 **验证提示（verification prompt）** ，询问模型所提出的解决方案是否正确。经验表明，与仅依赖线性头部而没有明确验证提示相比，这种额外的查询是有益的。在操作上，这种策略仅需为每个独立生成过程的验证提示增加一个小的额外 **预填充（prefill）** 步骤，同时保留了原始生成过程的 KV 缓存复用。因此，验证器仅为每个候选答案增加了适度的开销，同时提高了选择过程的可靠性。

为了获得最佳结果，我们将多数投票与验证器评分结合为 **加权多数投票（weighted majority vote）** 。我们不再对每个候选答案进行等权重计数，而是根据其验证器评分对每个候选答案的投票进行加权。直观地说，验证器认为更可能正确的候选答案对最终决策的贡献更大，同时仍然保留了跨多个样本聚合所带来的鲁棒性优势。这种混合方法保留了投票的简单性和稳定性，同时融入了对解决方案质量的学习认知，这在候选集合中混杂着表面看似合理但实际错误的解决方案以及正确解决方案时尤其有帮助。

## 训练与评估细节 {#sec:training_and_quantization}

验证被视为一个二元分类问题，其中生成器的一个候选响应根据真实答案被标记为正确或错误。为了生成验证器的训练数据，我们使用了 MATH 训练集 [@hendrycks2021measuring] 中 7.5k 个问题的 97.5%。剩余 2.5% 的问题留作验证器的验证集。为了构建一个多样化的训练集和一个用于快速评估的验证集，我们为每个训练问题生成了 16 个候选响应，而在验证集中我们限制自己只生成 4 个响应。验证器头部使用 **Sigmoid 激活函数（Sigmoid activation）** 来产生一个类概率分数，我们使用 **二元交叉熵损失（Binary Cross-Entropy loss）** 对其进行训练。这种设置使验证器的目标直接与下游选择问题对齐：在推理时使用的相同采样程序下，区分生成器产生的正确与错误候选解。

[]{#tab:wmv_results label="tab:wmv_results"}

**并行响应数** **1** **2** **4** **6** **8**

---

Greedy (baseline) 71.0 \- \- \- \-
Majority Vote 69.9 $\pm$ 1.3 70.0 $\pm$ 1.3 75.1 $\pm$ 1.0 76.6 $\pm$ 1.0 77.5 $\pm$ 0.8
Weighted MV (ours) 69.9 $\pm$ 1.3 72.7 $\pm$ 1.0 76.1 $\pm$ 0.9 77.5 $\pm$ 0.8 78.2 $\pm$ 0.7
: 我们的轻量级验证器加权多数投票与无验证器的多数投票以及贪婪解码在 MATH500 上的准确率对比。均值和标准差是根据从 16 个独立的 4 比特权重量化 Qwen-2.5-7B-Instruct 响应中进行的 20 次随机抽取计算得出。

## 结果

我们使用一个 4 比特权重量化的 Qwen-2.5-7B-Instruct 模型在 MATH500 上评估所提出的轻量级验证器，比较了贪婪解码、标准多数投票和我们的加权多数投票。表 [7](#tab:wmv_results){reference-type="ref" reference="tab:wmv_results"} 总结了准确率作为并行响应数量的函数，采样温度设置为 0.7。

即使在并行性非常有限的情况下，并行测试时扩展也能立即带来好处。仅使用两个并行响应，加权多数投票就将准确率提高到 72.7%，优于贪婪基线（71.0%）和标准多数投票（70.0%）。这突显了引入验证的一个关键优势：当两个采样响应不一致时，多数投票无法打破平局，而验证器加权方案可以持续选择更可靠的候选。

随着并行度的增加，多数投票和加权多数投票都表现出稳定的增益，这证实了先前的观察，即并行采样提高了生成正确解的概率。然而，加权多数投票在所有并行度水平上都持续优于未加权的多数投票，并且在八个并行响应时，加权多数投票相对于基线的改进达到了 10%。重要的是，与多数投票相比，随机抽取间的方差也略有减小，这表明验证器加权提供了一种更稳定的聚合机制。

尽管其结构简单，验证器仍能带来显著的好处。在架构上，它仅产生极小的开销，实际上每个流只多一个词元。因为验证器 **复用（Reuses）** 了生成器的 **KV 缓存（KV-cache）** ，所以不需要重新处理原始提示或响应，避免了通常与独立验证器模型相关的主要内存和延迟成本。这使得性能提升在内存带宽和存储资源严格受限的边缘场景中尤其具有吸引力。

# 量化（Quantization） {#sec:quantization}

在接下来的章节中，我们将讨论我们量化方法的细节。我们首先在 [7.1](#sec:quantization_background) 节简要概述 **神经网络量化（Neural Network Quantization）** ，并总结近期量化 **大型语言模型（Large Language Models, LLMs）** 的方法。

我们在 [7.2](#sec:quantization_base) 节概述了量化基础 LLM 的策略，并以 Qwen2.5-7B-Instruct 模型为例进行演示；随后在 [7.3](#sec:qarm) 节为其赋予推理能力。最后，我们在 [7.4](#sec:quantization_on_device) 节提供了量化模型在设备上导出和部署的具体细节。

## 背景与相关工作（Background and Related Work） {#sec:quantization_background}

#### 量化（Quantization）。

神经网络量化是减少模型占用空间、数据传输和计算需求最有效的方法之一 [@krishnamoorthi2018quantizing; @nagel2021white]。通过对模型进行量化，高比特宽度的浮点权重和激活值可以用低比特数字表示。除了减小模型大小，使用低比特定点表示（例如 **INT8** ）还能显著降低延迟和能耗 [@horowitz]。

我们采用以下 **量化-反量化函数（Quantization-dequantization function）** 的定义：

$$
\widehat{{\bm{x}}} := q\left({\bm{x}};\,{\bm{s}},{\bm{z}},b\right) = {\bm{s}}\cdot
    \vphantom{\Bigg(} \Big(\,\smash{\underbrace{\mathop{\mathrm{clip}}\!\left(\ensuremath{\left\lfloor{\frac{{\bm{x}}}{{\bm{s}}}}\right\rceil}+{\bm{z}};-2^{b-1},2^{b-1}-1\right)}_{\text{\normalsize $=: {\bm{x}}_\mathbb{Z}$}}} - {\bm{z}}\Big),
    \label{eq:dequant}
$$

其中，${\bm{x}}$ 表示量化器输入（即网络权重或激活张量），${\bm{s}}$ 表示高精度（[FP32]{.sans-serif} / [FP16]{.sans-serif} / [BF16]{.sans-serif}）的 **量化尺度（quantization scale）** ，${\bm{z}}$ 表示 **整数零偏移（integer zero offset）** ，$b$ 表示 **位宽（bitwidth）** 。$\ensuremath{\left\lfloor{\cdot}\right\rceil}$ 表示 **四舍五入取整算子（round-to-nearest-integer operator）** 。${\bm{x}}_\mathbb{Z}$ 是输入 ${\bm{x}}$ 的 $b$ 位整数 **量化表示（quantized representation）** 。量化参数 ${\bm{s}}$、${\bm{z}}$ 可以在 ${\bm{x}}$ 的分量间共享（通常是 **逐通道（per-channel）** 或 **分块（block-wise）** ）。这种量化方案被称为 **均匀仿射（uniform affine）** 或 **非对称（asymmetric）** 量化 [@hubara2017quantized; @krishnamoorthi2018quantizing; @zhou2016dorefa]，它是最常用的量化方案之一，因为它允许定点算术的高效实现。在 **对称（symmetric）** 量化的情况下，我们将量化网格限制为围绕 ${\bm{z}}={\bm{0}}$ 对称。

量化方法通常可分为 **训练后量化（Post-training quantization, PTQ）** 和 **量化感知训练（Quantization-aware training, QAT）** 两大类。PTQ 算法将预训练的高精度网络直接转换为定点模型，无需原始训练流程 [@banner2018post; @cai2020zeroq; @choukroun2019low; @hubara2020improving; @meller2019same; @zhao2019improving; @Nagel_2019_ICCV; @nagel_up_2020; @li2021brecq]。这些方法快速、易用，并且通常仅依赖于一个小的校准数据集。相比之下，QAT 方法 [@gupta2015deep; @jacob2018quantization; @lsq; @nagel_oscillations_2022] 在训练过程中模拟量化以寻找更优的解决方案，但通常需要更长的训练时间、更多的内存、带标签的数据以及仔细的超参数调优。

#### **LLM 量化（LLM Quantization）** .

传统 QAT 方法过高的训练成本和内存使用量使其在量化现代 LLMs 时不太实用，尽管一些工作如 LLM-QAT [@liu2023llm] 和 BitDistiller [@du2024bitdistiller] 探索了结合知识蒸馏的 QAT。值得注意的是，[@liu_paretoq_2025; @chen2025scaling_qat] 是我们所知唯一成功将 QAT 扩展到数十亿词元规模的研究。有几篇论文探索了 QAT 与 **参数高效微调（Parameter-efficient fine-tuning, PEFT）** 的结合，包括 [@dettmers2024qlora; @xu2023qa; @li2023loftq; @guo2023lq; @kim2024memory; @bondarenko2024low]。与传统的 QAT 相比，这些方法中的大多数都提供了显著的内存减少，但通常并不专注于推理效率。例如，QLoRA [@dettmers2024qlora] 使用（非均匀的）[NF4]{.sans-serif} 格式将预训练权重量化为 4 位，但在前向传播过程中将其反量化回 [BF16]{.sans-serif}。

由于权重和激活中存在强烈的数值 **离群值（outliers）** ，LLMs 的训练后量化是一项具有挑战性的任务 [@bondarenko_understanding_2021; @kovaleva_bert_2021; @dettmers_gpt3_int8_2022; @bondarenko_quantizable_2023; @sun_massive_2024]。核心挑战在于，将离群值量化到定点网格上会迫使在 **动态范围（dynamic range）** 和 **精度（precision）** 之间进行权衡：增加动态范围可以捕获离群值，但会牺牲零附近的精度；而保持精度则需要将它们 **裁剪（clipping）** 掉——这两种情况都会严重降低模型性能。

现有的 LLM PTQ 方法可以大致分为 **仅权重（weights-only）** 量化和 **权重-激活（weight-activation）** 量化。仅权重量化专注于将权重转换为低比特值。GPTQ [@frantar_gptq_2022] 利用二阶信息迭代地对分组权重进行舍入，并校正剩余组中的量化误差。SpQR [@dettmers2023spqr]、AWQ [@lin2023awq] 和 OWQ [@lee2024owq] 强调了对应于高幅度激活的所谓"显著（salient）"权重的重要性。其他近期的仅权重方法包括 [@jeon2023frustratingly; @lee2023flexround; @luo2023long; @chee2024quip]。权重-激活量化同时压缩权重和激活。SmoothQuant [@xiao_smoothquant_2024]、`LLM.int8()` / `GPT3.int8()` [@dettmers_gpt3_int8_2022] 和 Outlier Suppression [@wei2022outlier] 通过管理激活离群值实现了 W8A8 量化。`LLM.int8()` 使用 **混合精度分解（mixed-precision decomposition）** ，而其他两种方法则采用 **逐通道缩放（channel-wise scaling）** 。其他一些近期的权重-激活 PTQ 方法包括 [@lee2023enhancing; @liu2023qllm; @wei_outlier_2023; @yuan2023rptq; @tang2024easyquant; @yao2022zeroquant; @lin2024qserve]。

#### **使用 FPTs 的 LLM 量化（LLM Quantization using FPTs）** .

**大型语言模型（Large Language Model, LLM）** 量化中一个前景广阔的方向是使用 **旋转（rotations）** 和其他 **函数保持变换（Function-Preserving Transformations, FPTs）** 。Nagel 等人 [1] 首次探索了用于 **卷积神经网络（Convolutional Neural Network, CNN）** 量化的 FPTs，证明了 **ReLU（Rectified Linear Unit）** 激活函数和逐通道缩放操作可交换，从而实现了权重的跨层重缩放。在 LLM 场景下，Xiao 等人 [2] 提出通过在线逐通道缩放（在 **线性层（linear layers）** 之前应用），将激活值中的异常值迁移到权重中。后续工作扩展了这一思想，包括：

- 在缩放中引入偏移量 [3]，
- 为查询（queries）和键（keys）使用缩放向量 [4]，
- 通道混合变换 [5]，
- 使用随机 **哈达玛变换（Hadamard transforms）** 来减少异常值 [6, 7]，
- 其他在线旋转方法 [8]，
- 缩放与旋转的组合 [9]，
- 以及 **克罗内克结构（Kronecker-structured）** 矩阵变换 [10]。

最近，FPTQuant [11] 引入了三种新颖、轻量且表达能力强的 FPTs，以促进 **Transformer** 模型的量化。通过利用现代 Transformer 固有的 **等变性（equivariances）** 和独立性，这些 FPTs 旨在保持模型功能的同时，将中间激活分布塑造成对量化更友好的形态。因此，FPTQuant 能够实现静态 **INT4** 量化，几乎不产生额外开销，也无需定制内核，速度非常快，性能与大多数先前工作相当或更优。

## **量化基础语言模型（Quantizing Base Language Model）** {#sec:quantization_base}

#### **量化设置（Quantization setup）** 。

为了在我们的硬件上高效运行模型，我们使用 **INT4** 逐通道均匀仿射量化（公式 [eq:dequant]）对所有线性层（包括最终的 **语言模型头部（LM head）** ）的权重进行量化。为了进一步提高效率并降低延迟，我们使用 **INT8** 的 **KV 缓存（KV-cache）** 、 **INT8** 的输入嵌入（input embeddings）以及 **INT16** 用于所有剩余的激活值（均为逐张量）。为简洁起见，我们将此配置称为“W4A16KV8”。我们对权重、KV 缓存和嵌入使用对称量化，对激活值使用非对称量化，这是一种常见的设置。

#### **变换（Transformations）** 。

为了最大化量化模型的精度，我们应用了来自 FPTQuant [11] 的完全可合并变换子集（图 [8]）：

- 一对 **RoPE（Rotary Position Embedding）** 前变换 $(\mathscompmodern{T}_k$, $\bar{\mathscompmodern{T}}_k)$，其中 $\mathscompmodern{T}_k$ 应用于键（keys），而 $\bar{\mathscompmodern{T}}_k$ 可解释为 $\mathscompmodern{T}_k$ 的逆，应用于查询（queries）；
- $(\mathscompmodern{T}_u,\mathscompmodern{T}^{-1}_u)$：一个逐通道缩放器，被合并到上投影（up projection）和下投影（down projection）权重中；
- 多头值变换 $(\mathscompmodern{T}_v,\bar{\mathscompmodern{T}}_v)$：由每个头（head）的可逆矩阵组成，被合并到值（value）和输出（output）权重中；
- 以及一个用于旋转残差（residuals）的旋转矩阵 $(\mathscompmodern{T}_r,\mathscompmodern{T}^{-1}_r)$（在每个 Transformer 块的开始和末尾应用，并且是共享的）。

这组变换 $\mathscompmodern{T}:= \{\mathscompmodern{T}_k, \mathscompmodern{T}_u, \mathscompmodern{T}_v, \mathscompmodern{T}_r\}$ 将有助于塑造中间激活分布，使其对量化更友好，同时保持未量化模型的输出不变。上述所有 FPTs 都是完全可合并的，因此我们可以在硬件上运行模型而无需任何额外的推理开销。

<a id="fig:quantization_transforms"></a>

![](./images/quantization_transforms.png)

> 图 8 | 量化变换示意图。

## **参考文献（References）**

1.  Nagel 等人 (2019)
2.  Xiao 等人 (2024)
3.  Wei 等人 (2023)
4.  Shao 等人 (2024)
5.  Chee 等人 (2024)
6.  Ashkboos 等人 (2024)
7.  Liu 等人 (2024)
8.  Lin 等人 (2024)
9.  Hu 等人 (2025)
10. Sun 等人 (2024)
11. van 等人 (2025)

> 图 8： **函数保持变换（Function-Preserving Transformations）** 。我们使用了来自 FPTQuant 的 4 种变换类型：合并到查询（query）和键（key）中的缩放-旋转变换 $\mathscompmodern{T}_k$，合并到上投影（up projection）和下投影（down projection）中的逐通道缩放器 $\mathscompmodern{T}_u$，由每个注意力头（head）的可逆矩阵组成并合并到值（value）和输出权重中的 $\mathscompmodern{T}_v$，以及用于旋转残差（residuals）的旋转矩阵 $\mathscompmodern{T}_r$（跨层共享）。变换训练完成后，每个“合并组（merge group）”中的变换参数会被合并到原始模型权重 ${\bm{\mathsfit{W}}}$ 中。

#### 训练与评估细节

遵循文献 [@van2025fptquant; @lee2025unifying]，我们在 DCLM-Edu [@allal2025smollm2] 数据集上训练模型，该数据集是通过应用教育质量分类器 [@penedo2024fineweb] 对 DCLM [@li2024datacomp] 进行过滤后得到的更干净的版本。我们通过最小化量化张量与未量化张量之间的 $L^p$（$p=2$）范数来初始化量化参数，然后端到端地训练变换 $\mathscompmodern{T}$ 和量化参数 $\{{\bm{s}}, {\bm{z}}\}$，严格遵循 FPTQuant 的流程。我们使用 FastForward [@fastforward] 来模拟量化。为简洁起见，我们将上述量化流程记为 $\textbf{FPTQuant}^{\boldsymbol{\circ}}$。

为了评估量化后的基础语言模型（base language model）的预测性能，我们遵循先前的工作 [@frantar_gptq_2022; @xiao_smoothquant_2024; @shao_omniquant_2024; @van2025fptquant; @sun_flatquant_2024]，报告 WikiText-2 测试集上的困惑度（perplexity）（假设序列长度为 $4096$）。我们还报告了一组常识推理（Common Sense Reasoning, CSR）任务的平均零样本（zero-shot）准确率，这些任务包括 PIQA [@bisk_piqa_2020]、WinoGrande [@sakaguchi_winogrande_2021]、HellaSwag [@zellers_hellaswag_2019]、ARC-e 和 ARC-c [@clark_think_2018] 以及 LAMBADA [@paperno_lambada_2016]。最后，我们报告了 MMLU [@hendrycks2020measuring] 上的 5 样本（5-shot）准确率。对于 CSR 和 MMLU 的评估，我们使用 LM Harness 框架 [@sutawika2025eleutherai]。

#### 结果

我们在表 [8](#tbl:quantization_results_base_model){reference-type="ref" reference="tbl:quantization_results_base_model"} 中总结了量化基础模型的结果。正如我们所见，采用最小-最大值（min-max）范围估计的最简单的训练后量化（Post-Training Quantization, PTQ）流程经历了不可接受的准确率/困惑度下降。激活值中强烈的数值异常值（即使使用 16 位！）以及 **主要是量化后 4 位权重中灾难性的精度损失** 导致了如此糟糕的性能。

采用一组 **函数保持可合并变换（function-preserving mergeable transformations）** $\mathscompmodern{T}$ 已经显著改善了权重和激活值的分布，即使在最小-最大值范围设置下也能带来更好的准确率。此外，使用更好的范围初始化（range initialization）结合端到端学习，两者共同逐步恢复了全精度（full-precision）模型性能的更大一部分。最终，我们在 CSR 上匹配了全精度准确率，在 WikiText-2 上仅有约 0.4 的困惑度下降，在 MMLU 上仅有不到 1.5% 的准确率下降，而后者是公认相当具有挑战性的基准测试。总体而言，$\text{FPTQuant}^{\circ}$ 量化的基础模型表现出强大的性能，考虑到整个过程在单个 Nvidia H100 80GB GPU 上耗时不到 24 小时。

| **方法**                                        | **位宽（Bitwidth）** | $L^p$ | $\mathscompmodern{T}$ | **训练（train）** | **WikiText-2** ($\downarrow$) | **CSR** ($\uparrow$) | **MMLU** ($\uparrow$) |
| :---------------------------------------------- | :------------------- | :---- | :-------------------- | :---------------- | :---------------------------- | :------------------- | :-------------------- |
| ---                                             | ---                  | ---   | ---                   | ---               | ---                           | ---                  | ---                   |
| 全精度（Full-precision）                        | BF16                 | -     | -                     | -                 | 6.85                          | 72.90                | 74.28                 |
| 最小-最大值量化（Min-max quantization）         | W4A16KV8             | **-** | **-**                 | **-**             | 102.4                         | 51.71                | 62.35                 |
|                                                 | W4A16KV8             | **-** | **-**                 |                   | 9.18                          | 65.83                | 67.59                 |
|                                                 | W4A16KV8             | **-** |                       | **-**             | 8.48                          | 67.85                | 69.06                 |
|                                                 | W4A16KV8             |       | **-**                 |                   | 7.53                          | 70.68                | 72.26                 |
| $\textbf{FPTQuant}^{\boldsymbol{\circ}}$ (ours) | W4A16KV8             |       |                       |                   | **7.26**                      | **72.94**            | **72.81**             |

> **表 8 | 量化 W4A16KV8 基础模型（Qwen2.5-7B-Instruct）结果** 。我们报告了 Wikitext 困惑度、平均 0-shot CSR 和 5-shot MMLU 准确率。'$L^p$' 表示使用 $L^p$ 范围初始化，$\mathscompmodern{T}$ = 使用可合并变换集，'train' = 变换和量化参数的端到端训练。

## 量化感知模块化推理（Quantization-Aware Modular Reasoning） {#sec:qarm}

量化基础模型将不可避免地影响底层的激活分布。为了获得最佳性能，在应用后续的微调（fine-tuning）时，包括我们在第 [3](#sec:lora){reference-type="ref" reference="sec:lora"} 节中描述的推理 LoRA（Reasoning LoRA）协议， **必须考虑这些变化** 。

我们的方法遵循 QLoRA [@dettmers2024qlora] 及相关技术 [@xu2023qa; @li2023loftq] 的通用范式，即在冻结的、量化的基础模型之上训练 LoRA 适配器（adapters）。为了进一步提高内存和运行时效率，我们将训练好的 LoRA 适配器权重量化为 INT8，并在推理时使用 INT16 激活值。与基础模型一样，我们对权重使用对称的逐通道量化（symmetric per-channel quantization），对激活值使用非对称的逐张量量化（asymmetric per-tensor quantization）。我们将上述技术称为 **量化感知模块化推理（Quantization-Aware Modular Reasoning, QAMR）** 。

#### **训练与评估细节（Training and evaluation details）**

我们遵循第 [3](#sec:lora) 节中描述的 **训练与评估协议（training and evaluation protocols）** 。对于训练，我们使用 **OT3（OpenThoughts 3）** 数据集 [@guha2025openthoughts]。为了评估我们量化后推理模型的能力，我们使用了一套全面的基准测试，包括 **AIME 24/25** [@aime]、 **MATH500** [@hendrycks2021measuring]、 **GPQA Diamond** [@rein2024gpqa] 和 **AMC23** [@amc23]。我们进行了一项消融研究，以评估改进的 **基础模型量化流水线（base model quantization pipeline）** 和所提出的 **QAMR（Quantization-Aware Modular Reasoning）** 方法的贡献，该研究使用了 50k 个训练样本的随机子集，同时保留完整的训练数据集用于最终结果集。

#### **结果（Results）**

我们在表 [9](#tbl:quantization_results_reasoning_model) 中观察到，一个 **朴素量化（naïvely quantized）** 的基础模型与一个 **全精度（full-precision）** 推理模块（即不使用 QARM）结合时，基本上无法工作。从定性角度看，这样的模型输出看似随机的 **词元（tokens）** ，没有任何结构或与当前任务的相关性。

相比之下，应用 **QARM** —— 即使训练时间相对较短 —— 也能在 MATH500 和 GPQA 等任务上恢复大部分性能。值得注意的是， **量化感知的模块化推理对于学习任何内容都至关重要** 。

此外，使用 $\text{FPTQuant}^{\circ}$ 为训练推理模块提供了一个更强的起点，并在除 AMC23 外的所有基准测试中持续提升性能（AMC23 需要更长的训练）。通过使底层的 **激活分布（activation distributions）** 对量化更友好，使用 $\text{FPTQuant}^{\circ}$ 量化的基础模型相比使用标准 **最小-最大范围设置（min-max range setting）** 量化的基础模型，能显著减少训练不稳定性，并实现更快的学习。

最后，当结合扩展训练时，我们的方法平均能达到与同等训练的全精度推理模型性能相差约 2% 以内的水平，同时模型显著更紧凑，推理效率更高。

| **Bitwidth** | $\textbf{FPTQuant}^{\boldsymbol{\circ}}$ | **QAMR** | **N** | **AIME24** | **AIME25** | **MATH500** | **GPQA** | **AMC23** |  **Avg**  |
| :----------- | :--------------------------------------: | :------: | :---: | :--------: | :--------: | :---------: | :------: | :-------: | :-------: |
| BF16         |                    -                     |    -     |  50k  |    21.8    |    20.3    |    82.6     |   38.6   |   65.2    |   45.70   |
| W4A16KV8     |                  **-**                   |  **-**   |  50k  |    0.0     |    0.0     |     0.0     |   0.0    |    0.0    |   0.00    |
| W4A16KV8     |                  **-**                   |    ✓     |  50k  |    17.3    |    6.3     |    75.6     |   33.0   | **64.0**  |   39.25   |
| W4A16KV8     |                    ✓                     |    ✓     |  50k  |  **23.3**  |  **15.0**  |  **79.6**   | **33.7** |   57.0    | **41.72** |
| BF16         |                    -                     |    -     | 1.2M  |    53.3    |    33.0    |    94.0     |   39.9   |   82.5    |   60.54   |
| W4A16KV8     |                    ✓                     |    ✓     | 1.2M  |  **46.6**  |  **36.6**  |  **89.6**   | **37.8** | **80.0**  | **58.12** |

> 表 9 | **量化 W4A16KV8 推理模型结果（基于 Qwen2.5-7B-Instruct 基础模型）** 。我们报告了 AIME24/25、MATH500、GPQA 和 AMC23 的准确率（越高越好）。N = 用于训练推理模块的 OT 数据样本数量。

## **验证器量化与端侧部署（Verifier Quantization and On-Device Deployment）** {#sec:quantization_on_device}

作为我们量化流水线的最后阶段，我们处理 **验证器（verifier）** 的量化，这对于在资源受限的硬件上部署所提出的验证器至关重要。为了最小化因数值精度降低而导致的性能下降，我们直接在由第 [7.2](#sec:quantization_base) 节获得的 **4 位权重量化（4-bit weight-quantized）** 的 **Qwen-2.5-7B-Instruct** 模型产生的 **嵌入（embeddings）** 上训练验证器。这一选择减少了训练和推理之间的 **分布偏移（distribution shift）** ，确保验证器学习在部署时将遇到的相似表示上进行操作。在此设置下训练完验证器头部后，我们进一步使用 **FastForward** [@fastforward] 将激活和验证器头部的权重量化为 **8 位表示（8-bit representations）** 。

一旦包括基础模型、推理适配器和验证器在内的所有组件都被量化，我们就为端侧部署准备模型。量化后的第一步是进行 **模型转换（model transformation）** ，以确保在 **PyTorch** 表示层面与 **GENIE SDK** [@GENIE] 支持的格式兼容。这涉及到 **预填充（prefill）** 和 **解码（decoding）** 阶段的 **自回归并行与顺序生成（autoregressive parallel and sequential generation）** ，以及处理 **注意力操作（attention operations）** 、 **掩码（masking）** 和 **位置嵌入（position embeddings）** 。

接下来，我们在 **ONNX（Open Neural Network Exchange）** 表示层面建立与 GENIE 的兼容性。我们在此阶段使用 **Qualcomm FastForward** [@fastforward]，并在 **线性层（linear layers）** 、 **多头注意力（multi-head attention）** 以及 **模型分区（model partitioning）** 上实现转换。我们使用结合了 FastForward 的 PyTorch 来获取 **ONNX 计算图（ONNX graph）** 和相关的 **量化编码（quantization encodings）** 。我们导出为 **深度学习容器（Deep Learning Container, DLC）** 格式，并对 FastForward 遗漏的任何剩余未量化节点（例如偏置）进行量化。我们为部署目标（例如 **aarch64-android** ）进行编译，并使用 **_adb_（Android Device Bridge）** 上传到设备。

# **讨论与挑战（Discussions and Challenges）**

在资源受限的边缘设备上部署具备推理能力的模型，需要在任务性能、延迟、内存占用和功耗之间进行复杂的权衡。在本工作中，我们提出了一个实用的端到端框架来克服这些限制。通过使用模块化的 **低秩自适应（Low-Rank Adaptation, LoRA）** 适配器将推理能力与基础权重解耦，我们证明了 **参数高效微调（Parameter-Efficient Fine-Tuning）** 能够达到与计算成本高昂的全参数蒸馏方法（如 DeepSeek-R1-Distill）相媲美的推理精度。为了优化日常用户交互，我们轻量级的动态 **切换器（Switcher）** 将标准对话路由到高效的基础模型，而将推理适配器严格保留给需要多步逻辑的复杂查询。为了进一步对抗延迟，我们的 **预算强制强化学习（Budget-Forced Reinforcement Learning, RL）对齐** 明确惩罚生成的冗余性，在不牺牲任务准确性的前提下，将平均推理词元数量减少了 $2.4\times$。在推理时，我们利用自回归解码的内存受限特性，引入了 **并行测试时缩放（Parallel Test-Time Scaling）** ，并结合一个轻量级验证器，在复杂推理基准测试上提供了高达 10% 的准确率提升。最后，我们展示了通过 **FPTQuant** 和 **量化感知模块化推理（Quantization-Aware Modular Reasoning）** 实现的 4 位权重量化能够保持这种鲁棒性能，在实现巨大内存节省的同时，其准确率与全精度推理模型相差不到 2%。下文，我们总结了从我们流程的每个组件中得出的关键见解，并概述了为未来研究铺平道路的剩余挑战。

#### **用于模块化推理的 LoRA（LoRA for modular reasoning）**

我们的实验表明，通过 LoRA 进行的参数高效微调在激发小型（3B 和 7B）基础模型的推理能力方面非常有效，其性能常常能与计算成本高昂的全参数蒸馏相媲美。一个关键见解是，LoRA 的成功在很大程度上取决于适配器容量和基础模型规模。虽然 LoRA 秩为 128 时，7B 模型在具有挑战性的基准测试上几乎能与密集微调的基线模型匹配，但 3B 模型表现出更大的性能差距，这表明较小的骨干网络对适配器容量限制更为敏感。此外，虽然推理专业化提高了复杂任务（例如 LiveCodeBench）的性能，但它也引入了权衡，偶尔会降低在需要直接答案的简单编码任务上的 **零样本（Zero-Shot）** 性能。管理这种 **专业化-遗忘权衡（Specialization-Forgetting Trade-Off）** 仍然是一个持续的挑战。

#### **通过切换器进行动态 LoRA 路由（Dynamic LoRA routing via the switcher）**

我们证明了，通过认识到并非所有查询都需要复杂的多步逻辑，推理的计算开销可以大幅减少。轻量级的切换器模块成功地充当了按需路由器，为标准查询保留了基础模型的速度，仅在必要时才激活推理 LoRA 适配器。一个重要的部署见解是在 **预填充阶段（Prefill Stage）** 进行 **掩码 LoRA 训练（Masked LoRA Training）** 的必要性。该策略确保基础模型生成的 **KV 缓存（KV-Cache）** 可以被推理适配器无缝复用，从而完全消除了在切换模式时重新编码提示词元所带来的严重延迟惩罚。

虽然当前的切换器依赖于一个监督分类器头来评估查询复杂度，但未来工作的一个前景方向是通过强化学习来学习这种路由策略。这将与我们的预算强制目标产生协同作用：当 LoRA 激活时，预算强制 RL 会明确缩短推理轨迹；而一个针对准确性和长度进行优化的 RL 驱动路由器，则会尽可能学习完全绕过适配器。因为冻结的基础模型本身就能产生直接答案，而无需冗长的 **思维链（Chain-of-Thought, CoT）** ，成功地将查询路由到非 LoRA 模式会自动产生一个大幅缩短的响应，从而有机地将推理适配器严格保留给基础模型会失败的复杂查询。

未来的工作可以将切换器扩展到二元路由之外，将其转变为动态 LoRA 选择的通用机制。系统可以不是仅在基础模型和单个推理适配器之间选择，而是将每个查询路由到一个特定任务适配器库 [@huang2024mixture; @feng2024mixture]（例如专门用于数学、编码或其他领域），允许同一骨干网络支持更丰富的能力，同时保持模块化部署。一个特别有前景的方向是包含为 **隐式推理（Latent Reasoning）** 量身定制的适配器，因为最近的研究表明，隐式推理 LoRA 适配器可以在保持强大推理性能的同时，大幅减少显式 CoT 生成的词元开销 [@hao2024training; @shen2025codi; @wu2025parallel; @kuzina2026kava; @wang2025system]。在这种情况下，切换器不仅需要决定是否需要推理，还需要决定哪种推理形式对给定查询最有效，这使得动态适配器路由成为迈向能力更强、计算效率更高的设备端系统的自然路径。

#### **预算强制（Budget forcing）**

我们提出了利用 **强化学习（Reinforcement Learning, RL）** 微调 **大型语言模型（Large Language Models, LLMs）** 以生成更短补全文本的方案。通过利用一个乘法惩罚项（公式 [$$eqn:bf_reward$$](#eqn:bf_reward)），我们成功地对齐了 LLMs，使其在回答给定问题时减少生成的 **词元（token）** 数量。我们的实证评估表明，在任务准确率仅有最小程度下降的前提下，平均补全长度减少了 $2.4\times$，最大压缩比可达 $8\times$。至关重要的是，我们的公式在结构上缓解了我们在早期实验中观察到的 **奖励黑客（reward hacking）** 问题，确保了效率提升源于真正的 **推理压缩（rationale compression）** ，而非格式上的投机取巧。虽然我们的“软屏障（soft-barrier）”方法为计算感知推理提供了一个稳健的机制，但它也为未来的研究开辟了几个令人兴奋的方向。我们概述了这些前瞻性的方向，它们自然地解决了我们方法当前的局限性。

我们的实验提供了预算强制在 **最先进（state-of-the-art）** 推理模型上的一个快照。然而，基础模型规模（例如，参数量）与推理压缩能力之间的关系仍然是一个悬而未决的问题。研究更大、能力更强的模型是否表现出更强的“认知犹豫（epistemic hesitation）”，从而为 **推理词元压缩（Inference Token Compression, ITC）** 提供更大的压缩空间，对于制定预算强制的一般性缩放定律至关重要。

当前预算强制方法（包括我们的方法）的一个根本性局限在于假设了统一的词元成本，即每个生成的词元无论其效用如何，对预算消耗的贡献是均等的。然而，一个代表关键逻辑跃迁的词元，其语义价值远高于用于句法粘合或模糊表达（例如，“让我想想这个……”）的词元。一个有前景的未来方向在于开发 **语义感知（semantic-aware）** 的预算先验，该先验能够根据信息密度或局部熵动态调整惩罚权重，正如 [@Massoli2026-eu] 中所探索的那样。通过将优化目标从纯粹的长度最小化转向 **推理密度最大化（reasoning density maximization）** ，我们可以鼓励模型优先考虑高效用词元，同时不成比例地惩罚低熵填充词，从而有效地将计算成本与推理深度解耦。

#### **并行测试时缩放与推理（Parallel test-time scaling and reasoning）**

我们已经证明，即使是一个轻量级的验证器设计，也能显著提高在边缘设备上进行推理的并行测试时缩放的有效性。通过以可忽略的额外成本，将聚合的鲁棒性与习得的正确性概念相结合， **加权多数投票（weighted majority voting）** 为在资源受限的设备上部署并行推理提供了一种实用且高效的方法。当前的设计可以扩展为使用 **过程奖励模型（process reward model）** 对推理步骤进行评分。另一个有趣的扩展是采用具有相互依赖生成过程的并行推理方案，如 [@hsu2025group; @rodionov2025hogwild; @cesa2026lanerope] 中所述。

#### **量化（Quantization）** {#quantization}

我们在 **Qwen2.5-7B-Instruct** 上的实验突显了从一个强大的量化基础模型开始的重要性。利用 **函数保持变换（function-preserving transformations）** 、改进的范围初始化以及变换与量化参数的联合微调，为量化现代 LLMs 同时保持稳健的预测性能提供了一条直接且轻量级的路径。在为这类模型扩展推理能力时，我们进一步表明，必须考虑量化引起的 **分布偏移（distribution shifts）** 。受先前 **参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 文献的启发，我们通过提出 **量化感知模块化推理（Quantization-Aware Modular Reasoning, QAMR）** 方法来应对这一挑战，该方法通过在量化后的基础模型上直接训练推理适配器来减轻量化噪声和分布偏移。最后，我们的结果表明，一个相对紧凑的 4 位权重量化 7B 模型可以实现与规模大得多的模型相媲美的推理性能。

推理本质上仍然是一个密集生成词元的任务，这使得它从根本上受限于内存而非计算。因此，效率和性能的进一步提升很可能依赖于减少模型权重的内存占用。未来工作的一个前景广阔的方向是，通过利用最先进的压缩技术（如 **Quip#** [@tseng2024quip]）将量化推至 4 位以下，或探索 2-3 位的 **量化感知训练（Quantization-Aware Training, QAT）** 方法，例如 **ParetoQ** [@liu_paretoq_2025]。

# 结论（Conclusion）

在本工作中，我们提出了一个端到端框架，使得最先进的 **大型语言模型（Large Language Model, LLM）** 推理能够在资源受限的边缘设备上变得实用。我们证明了，由 **动态路由切换器（Dynamic Routing Switcher）** 控制的 **参数高效低秩适应（Parameter-Efficient Low-Rank Adaptation, LoRA）** ，能够在不影响日常交互速度的前提下，解锁强大的推理能力。通过引入 **预算强制强化学习（Budget-Forced Reinforcement Learning）** ，我们成功地抑制了模型的冗长性，以适应严格的设备端词元限制。为了进一步最大化硬件利用率，我们利用了 **并行测试时缩放（Parallel Test-Time Scaling）** 和一个 **轻量级潜在验证器（Lightweight Latent Verifier）** ，以在内存受限的生成阶段提升准确性。在整个流程中， **硬件感知（Hardware-Awareness）** 始终是统一的原则：从最大化 **键值缓存（Key-Value Cache, KV-Cache）** 重用，到将 **量化（Quantization）** 直接交织到适配器、切换器和验证器的训练中。最终，这种协同设计的方法弥合了基于云的推理与移动硬件严格的 **内存（Memory）** 、 **延迟（Latency）** 和 **功耗（Power）** 预算之间的差距，为设备端人工智能提供了一个实用的蓝图。

# 基准测试描述（Benchmark Description） {#App:benchmark}

为了全面评估我们微调模型的推理能力，我们利用了一套涵盖数学、科学和编程领域的多样化基准测试套件。

- **AIME 24/25** [@aime] 包含来自 2024 年和 2025 年美国数学邀请赛（American Invitational Mathematics Examination）的 30 个极具挑战性的数学竞赛问题。这些问题涵盖代数、几何和数论等一系列主题，面向高中生。问题需要多步推理，答案是在 0 到 999 之间的整数。

- **MATH500** [@hendrycks2021measuring] 是一个包含 500 个数学问题的基准测试，涵盖代数、几何、数论、预微积分和概率等不同主题。问题需要多步解决方案，答案可能包含 LaTeX 格式的表达式。

- **GPQA Diamond** [@rein2024gpqa] 包含 198 个来自物理、化学和生物学领域的科学博士水平问题。所有问题均以多项选择题形式呈现。

- **AMC23** [@amc23] 包含来自 2023 年美国数学竞赛（American Mathematics Competition）的 40 个问题，答案为整数。

- **LiveCodeBench** [@livecodebench] 是一个持续更新的（因此称为“实时”）编程基准测试。在每次发布中，会从竞争性编程平台（例如 CodeForces、LeetCode）获取一定数量的编程问题，每个问题用于构建四个编程“场景”： **代码生成（Code Generation）** 、 **代码修复（Code Repair）** 、 **测试输出预测（Test Output Prediction）** 和 **代码执行（Code Execution）** 。在本工作中，我们使用 v2 版本，包含 511 个问题，并将自己限制在代码生成场景。

- **HumanEval 和 HumanEval+** [@humaneval; @evalplus] 是 HumanEval 基准测试的原始版本和改进版本，包含 164 个问题，要求模型根据 Python 函数的签名和文档字符串生成其函数体。然后使用多个单元测试来验证生成的函数。 **HumanEval+** [@evalplus] 通过将用于验证的单元测试数量增加 80 倍来改进原始基准测试。

- **MBPP 和 MBPP+** [@mbpp; @evalplus] 是最基础 Python 程序（Most Basic Python Programs）基准测试的原始版本和改进版本，包含 1000 个来自人类程序员的基本 Python 编程任务。每个任务都涉及根据自然语言需求和三个必须通过的单元测试编写一个简单的 Python 函数。 **MBPP+** 增强版选取了 378 个任务的子集，并将单元测试数量增加了 35 倍。

# **LoRA 消融研究（LoRA ablation study）** {#sec:lora_ablation}

本附录提供了第 [3.4.1](#sec:ablations) 节中介绍的 **参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 消融研究的完整详细结果，见表格 [10](#tab:ablations_qwen3b) 和 [11](#tab:qwen7_ablations)。该研究探讨了不同 **学习率（Learning Rate, LR）** 、 **批大小（Batch Size, BS）** 和 **LoRA 适配器秩（LoRA adapter rank）** 对 3B 和 7B 模型骨干推理能力的影响。模型在 OpenThoughts3（OT3）数据集的 50,000 条数据子集上训练了一个 **周期（epoch）** ，并在核心数学与科学基准测试（AIME24、AIME25、MATH500、GPQA 和 AMC23）上进行了评估。

[]{#tab:ablations_qwen3b label="tab:ablations_qwen3b"}

表 1 | Qwen2.5-3B-Instruct 的消融结果。LR 表示学习率，BS 表示批大小。在每个学习率子组中，最佳性能已用粗体标出。
| **LR** | **BS** | **Rank** | **AIME24** | **AIME25** | **MATH500** | **GPQA** | **AMC23** | **Avg** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0001 | 32 | 32 | 0.05 | 0.01 | 0.55 | 0.25 | 0.24 | 0.220 |
| 0.0001 | 32 | 64 | 0.03 | 0.02 | 0.58 | 0.29 | 0.28 | 0.240 |
| 0.0001 | 32 | 128 | **0.08** | 0.03 | 0.56 | 0.28 | 0.30 | 0.250 |
| 0.0001 | 32 | 256 | 0.03 | 0.04 | 0.57 | 0.28 | 0.28 | 0.240 |
| 0.0001 | 64 | 32 | 0.05 | 0.01 | 0.54 | 0.27 | 0.28 | 0.230 |
| 0.0001 | 64 | 64 | 0.03 | 0.03 | 0.57 | 0.29 | 0.29 | 0.242 |
| 0.0001 | 64 | 128 | 0.02 | 0.03 | 0.58 | 0.27 | 0.30 | 0.240 |
| 0.0001 | 64 | 256 | 0.04 | **0.06** | 0.59 | **0.30** | **0.34** | **0.266** |
| 0.0001 | 128 | 32 | 0.04 | 0.01 | **0.61** | 0.26 | 0.29 | 0.242 |
| 0.0001 | 128 | 64 | 0.05 | 0.00 | 0.59 | 0.29 | 0.27 | 0.240 |
| 0.0001 | 128 | 128 | 0.03 | 0.01 | 0.56 | 0.27 | 0.25 | 0.224 |
| 0.0001 | 128 | 256 | 0.03 | 0.05 | 0.58 | 0.27 | 0.25 | 0.236 |
| 0.0002 | 32 | 32 | **0.08** | 0.01 | 0.58 | 0.26 | 0.33 | 0.252 |
| 0.0002 | 32 | 64 | 0.05 | 0.03 | 0.59 | **0.30** | 0.32 | 0.258 |
| 0.0002 | 32 | 128 | 0.06 | 0.04 | 0.57 | 0.25 | 0.29 | 0.242 |
| 0.0002 | 32 | 256 | 0.07 | 0.04 | **0.60** | 0.28 | **0.34** | 0.266 |
| 0.0002 | 64 | 32 | 0.03 | 0.02 | 0.56 | 0.27 | 0.30 | 0.236 |
| 0.0002 | 64 | 64 | 0.07 | 0.03 | 0.58 | 0.26 | 0.28 | 0.244 |
| 0.0002 | 64 | 128 | 0.07 | **0.06** | 0.55 | 0.29 | 0.25 | 0.244 |
| 0.0002 | 64 | 256 | 0.06 | 0.04 | 0.58 | 0.27 | 0.30 | 0.250 |
| 0.0002 | 128 | 32 | 0.00 | 0.03 | 0.59 | 0.27 | 0.28 | 0.234 |
| 0.0002 | 128 | 64 | 0.00 | 0.02 | 0.55 | 0.28 | 0.28 | 0.226 |
| 0.0002 | 128 | 128 | 0.05 | 0.04 | 0.55 | 0.27 | 0.30 | 0.242 |
| 0.0002 | 128 | 256 | 0.07 | 0.04 | 0.59 | 0.29 | 0.32 | **0.262** |
| 0.0005 | 32 | 32 | 0.05 | 0.02 | 0.55 | **0.31** | 0.29 | 0.244 |
| 0.0005 | 32 | 64 | 0.05 | 0.03 | 0.53 | 0.28 | 0.27 | 0.232 |
| 0.0005 | 32 | 128 | **0.06** | 0.04 | 0.54 | 0.26 | 0.27 | 0.234 |
| 0.0005 | 32 | 256 | 0.04 | 0.05 | 0.53 | 0.28 | 0.24 | 0.228 |
| 0.0005 | 64 | 32 | **0.06** | 0.04 | 0.56 | 0.26 | 0.29 | 0.242 |
| 0.0005 | 64 | 64 | 0.05 | 0.05 | 0.57 | 0.28 | 0.30 | 0.250 |
| 0.0005 | 64 | 128 | **0.06** | 0.04 | 0.57 | 0.29 | 0.27 | 0.246 |
| 0.0005 | 64 | 256 | 0.03 | 0.05 | 0.54 | 0.28 | 0.24 | 0.228 |
| 0.0005 | 128 | 32 | **0.06** | 0.03 | 0.57 | 0.28 | 0.26 | 0.240 |
| 0.0005 | 128 | 64 | 0.03 | 0.00 | 0.57 | 0.28 | 0.27 | 0.230 |
| 0.0005 | 128 | 128 | **0.06** | 0.05 | 0.56 | 0.27 | **0.35** | **0.258** |
| 0.0005 | 128 | 256 | **0.06** | **0.06** | **0.58** | 0.29 | 0.25 | 0.248 |

[]{#tab:qwen7_ablations label="tab:qwen7_ablations"}

表 2 | Qwen2.5-7B-Instruct 的消融结果。LR 表示学习率，BS 表示批大小。在每个学习率

# 预算强制细节（Budget Forcing Details） {#app:budget_forcing}

表 [12](#tab:hyperparams) 列出了我们在预算强制强化学习（Budget Forcing Reinforcement Learning, RL）训练中使用的超参数（Hyperparameters）列表。

表 1 | **GRPO 训练超参数（GRPO Training Hyperparameters）** 。除非另有说明，所有实验均共享这些设置。
| **超参数（Hyperparameter）** | **值（Value）** |
| :--- | :--- |
| 优化器（Optimizer） | AdamW |
| 学习率（Learning Rate） | $2 \times 10^{-5}$ |
| 学习率调度器（LR Scheduler） | Cosine |
| 预热比例（Warmup Ratio） | 0.05 |
| 批次大小（全局）（Batch Size (Global)） | 256 |
| 每提示生成次数（Generations per Prompt, $G$） | 8 |
| 温度（Temperature） | 0.8 |
| 最大生成长度（Max Completion Length） | 6144 |
| 最大梯度范数（Max Gradient Norm） | 1.0 |
| KL 惩罚系数（KL Penalty Coefficient, $\beta_{\text{KL}}$） | $\{10^{-3}, 10^{-4}\}$ |
| 训练步数（Training Steps） | 200 |

我们在图 [9](#fig:cot_example_1) 和图 [10](#fig:cot_example_3) 中报告了无约束基线（Unconstrained Baseline）与我们预算强制模型（Budget-forced Model）的推理轨迹（Reasoning Traces）之间的额外定性比较。

<a id="fig:cot_example_1"></a>

> 图 4： **数论推理的定性比较（Qualitative comparison on number theory reasoning）** 。红色高亮文本表示冗余的验证和语言解析，而加粗文本标识了关键的推理步骤。为简洁起见，我们使用“ **[…]** ”作为占位符。 **顶部：** 提示（Prompt）。 **中部：** 基线（Baseline）轨迹很早就正确识别了属性（$p^{2}$），但陷入了大量冗余的验证循环，检查合数并重新列出质数（以红色高亮显示）。 **底部：** 预算强制（Budget Forced）轨迹直接应用质数平方属性并计算结果，没有不必要的犹豫或语法噪音。

<a id="fig:cot_example_3"></a>

> 图 5： **模式识别的定性比较（Qualitative comparison on pattern recognition）** 。 **中部：** 基线（Baseline）轨迹最初正确识别了公式 $2n - 1$，但花费了近 1000 个词元（Tokens）来验证其与替代算术公式（$a + (n - 1)d$）和假设的用户错误（混淆项数与值，测试 $n^{2}$ 等）的对比。 **底部：** 预算强制（Budget Forced）轨迹直接检索公式并计算所请求的特定项。
