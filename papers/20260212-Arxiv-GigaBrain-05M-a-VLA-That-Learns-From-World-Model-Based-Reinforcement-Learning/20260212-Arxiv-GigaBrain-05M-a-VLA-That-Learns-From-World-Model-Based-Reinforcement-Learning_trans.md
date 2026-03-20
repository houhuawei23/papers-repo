# 标题：GigaBrain-0.5M*：一种通过基于世界模型的强化学习进行学习的视觉-语言-动作模型
- ArXiv：2602.12099
- 作者：GigaBrain 团队（按字母顺序排列），：，Boyuan Wang
- 章节数：15
- 估计词元数：9.6k

## 目录
- 1 引言
- 2 相关工作
  - 2.1 视觉-语言-动作模型
  - 2.2 用于策略模型的世界模型
  - 2.3 用于视觉-语言-动作模型的强化学习
- 3 GigaBrain-0.5M*
  - 3.1 GigaBrain-0.5
  - 3.2 RAMP
    - 3.2.1 RAMP 公式化
    - 3.2.2 RAMP 实现
- 4 实验
  - 4.1 基础模型性能
  - 4.2 RAMP 性能
- 5 结论与未来工作
- 参考文献

## 摘要

 **摘要** ：直接从当前观测预测多步动作块的 **视觉-语言-动作模型（Vision-Language-Action models, VLA）** ，由于受限的场景理解和较弱的未来预测能力而面临固有的局限性。相比之下，在 **网络规模视频语料库（web-scale video corpora）** 上预训练的 **视频世界模型（video world models）** 表现出强大的 **时空推理（spatiotemporal reasoning）** 和准确的未来预测能力，使其成为增强 VLA 学习的天然基础。因此，我们提出了  **GigaBrain-0.5M** *，一种通过 **基于世界模型的强化学习（world model-based reinforcement learning）** 训练的 VLA 模型。它建立在  **GigaBrain-0.5**  之上，后者在超过 10,000 小时的机器人操作数据上进行了预训练，其中间版本目前在 **国际 RoboChallenge 基准（international RoboChallenge benchmark）** 上排名第一。GigaBrain-0.5M* 进一步通过  **RAMP（Reinforcement leArning via world Model-conditioned Policy，通过世界模型条件化策略的强化学习）**  集成了基于世界模型的强化学习，以实现鲁棒的跨任务适应。实证结果表明，与  **RECAP**  基线相比，RAMP 实现了显著的性能提升，在包括 **衣物折叠（Laundry Folding）** 、 **装箱（Box Packing）**  和 **意式浓缩咖啡制备（Espresso Preparation）**  在内的挑战性任务上带来了约 30% 的改进。至关重要的是， **GigaBrain-0.5M** * 展现出可靠的 **长时程执行（long-horizon execution）** 能力，能够持续完成复杂的操作任务而不失败，这已通过我们项目页面上的真实世界部署视频得到验证。

<a id="section-1"></a>

## 1 引言（Introduction）

近期， **视觉-语言-动作模型（Vision-Language-Action models, VLA）** （pi0; pi05; go1; gr00t; gr3; walloss; galaxea; gigabrain0）在理解指令、感知环境和执行复杂操作方面取得了令人瞩目的成果。然而，主流 VLA 架构中仍存在一个根本性局限：它们依赖 **近视观测（myopic observations）** 进行 **长时程动作规划（long-horizon action planning）** 。这一缺陷源于架构上偏向 **反应式控制（reactive control）** 而非 **前瞻性规划（prospective planning）** 。相反，在大规模视频语料库上训练的 **基础世界模型（foundation world models）** 已展现出预测合理未来状态的卓越能力，此类 **预测先验（predictive priors）** 为赋予 VLA 模型 **前瞻能力（foresight）** 提供了一条途径。

因此，我们提出了  **GigaBrain-0.5M\** *，一个通过 **基于世界模型的强化学习（world model-based reinforcement learning）** 训练的 VLA 模型。具体而言，GigaBrain-0.5M\* 通过集成  **RAMP（Reinforcement leArning via world Model-conditioned Policy）**  对  **GigaBrain-0.5M** （我们最新发布的、在超过 1 万小时真实世界机器人交互数据上预训练的 VLA 模型）进行了扩展。RAMP 框架遵循一个迭代的四阶段训练范式：
1.   **世界模型预训练** ：在大规模机器人操作数据上预训练世界模型，以预测未来状态及其关联的 **价值（value）** 。
2.   **策略微调** ：通过将世界模型预测的未来状态和价值估计作为条件，对 **策略（policy）** 的动作选择进行微调。
3.   **策略部署与数据收集** ：将条件化策略部署到物理环境中，在 **人在回路（human-in-the-loop）** 干预下收集 **轨迹数据（rollout trajectories）** 。
4.   **联合精炼** ：使用收集整理的轨迹数据集，对世界模型和策略进行联合精炼。

这一迭代训练范式实现了 **持续学习（continual learning）** 与 **自我改进（self-improvement）** 。我们提出的 RAMP 方法灵感来源于 $\pi^{*}_{0.6}$（pi06）中的  **RECAP** ，因为两者都利用额外信息作为 VLA 模型的条件。然而，RECAP 仅使用稀疏的 **优势（advantage）** （0 或 1）作为输入，提供的信息增益有限。相比之下，我们提出的 RAMP 利用了由充分预训练的世界模型预测的未来状态，从而产生了显著的信息增益。此外，我们从理论上验证了 RECAP 是 RAMP 的一个特例。

在我们的实验中，我们首先进行了全面的内部评估，以衡量  **GigaBrain-0.5**  相对于强大基线（包括 $\pi_{0.5}$（pi05）和  **GigaBrain-0** （gigabrain0））的性能。我们的方法在一系列多样化的操作任务上取得了 **最先进（state-of-the-art）** 的成功率，在具有挑战性的 **可变形物体操作（deformable object manipulation）** 和 **长时程程序性任务（long-horizon procedural tasks）** 上优势尤为明显。此外，GigaBrain-0.5 的一个中间版本在公开的  **RoboChallenge**  基准排行榜（robochallenge）上获得了首位。

我们还进行了广泛的 **消融研究（ablation studies）** ，以分析不同强化学习算法对真实机器人性能的影响。结果表明，我们提出的 RAMP 显著优于其他方法，如  **AWR** （awr）和  **RECAP** （pi06），在策略学习过程中实现了更优的 **多任务泛化（multi-task generalization）** 和显著提升的 **样本效率（sample efficiency）** 。

值得注意的是， **GigaBrain-0.5M\** * 展现出强大的长时程推理能力，能够无缝执行复杂的顺序任务，包括叠衣服、装箱和制作意式浓缩咖啡，在延长的交互过程中无需中断。

<a id="section-2"></a>

## 2 相关工作（Related Works）

<a id="section-2-1"></a>

### 2.1 视觉-语言-动作模型（Vision-Language-Action Models）

基础语言模型（Foundation Language Models）的最新进展催化了 **视觉-语言-动作模型（Vision-Language-Action Models, VLA）**  的发展 (o2024open; team2024octo; kimopenvla; pi0; pertsch2025fast; pi05; doshi2024scaling; wang2024scaling; liu2024rdt; qu2025spatialvla; li2024cogact; gr00t; gr3; gigabrain0; swiftvla)。这些模型通过联合扩展模型参数和训练语料库，旨在增强跨任务和跨具身（Cross-embodiment）的泛化能力。此类系统通常利用冻结或微调过的视觉-语言骨干网络（qwen25vl; paligemma; kosmos; llava; flamingo; paligemma2; smolvlm）来处理异构的感官输入并生成可执行的运动指令，所采用的方法要么是自回归（Autoregressive）的标记化（Tokenization）策略，要么是通过基于流的生成范式（Flow-based Generative Paradigms）(lipman2022flow; liu2022rectified) 构建的连续动作空间。

尽管当代的 VLA 模型整合了广泛的跨具身数据集 (o2024open; khazatsky2024droid; dasari2019robonet; walke2023bridgedata; ebert2021bridge) 以及大规模专有档案库以增强泛化能力，但其在 **时间扩展推理（Temporally Extended Reasoning）**  能力方面仍存在根本性局限。具体而言，在处理长视野（Long-horizon）操作任务时，这些模型倾向于主要基于即时观测输入来生成动作。

<a id="section-2-2"></a>

### 2.2 用于策略模型的世界模型（World Models for Policy Models）

世界建模（World Modeling）领域的最新突破 (hunyuanvideo; wan; cosmos; cosmospredict; vjepa2; genieEnvisioner; enerverseac; dreamgen) 加速了利用生成数据来弥合具身人工智能（Embodied AI）系统中仿真与现实差距的进程 (sorasurvey)。在自动驾驶领域，世界模型被用来生成 **极端案例（Corner Cases）**  数据 (drivedreamer; drivedreamer2; gaia; gaia2; magicdrive; vista; cosmosdrivedream) 并构建交通场景 (drivedreamer4d; recondreamer; recondreamer++; recondreamerrl)。

在具身机器人领域，诸如 (gigaworld0) 等技术利用世界模型生成的样本——涵盖纹理变化的场景 (emma; robotransfer; roboengine)、多视角渲染 (egodemogen) 和自我中心平移 (mimicdreamer)——来丰富 VLA 模型的训练数据。

一个独特的范式涉及通过世界模型预测未来的视觉轨迹（例如，DreamGen (dreamgen) 和 ViDAR (vidar)），随后通过 **逆动力学模型（Inverse Dynamics Models, IDMs）**  推断出可执行的运动指令。此类流程的有效性关键取决于生成序列的视觉保真度和物理合理性。

除了数据生成，新兴的方法正在探索世界模型与策略学习之间更紧密的集成。像 (worldvla; drivevla; drivedreamer; motus; lingbotva; mimicvideo) 这样的方法将来自预测性世界模型的潜在表示与策略网络融合，以提高样本效率和泛化能力。更具雄心的框架，如 (cosmospolicy)，则完全绕过了显式的策略网络，直接将世界模型的预测映射为动作序列。

<a id="section-2-3"></a>

### 2.3 用于视觉-语言-动作模型的强化学习（Reinforcement Learning for Vision-Language-Action Models）

模仿学习（Imitation Learning）策略会因 **分布偏移（Distribution Shift）**  ross2011dagger 而遭受复合误差，这本质上将其性能限制在示范数据的质量范围内。虽然 DAgger 及其变体 kelly2019hg; jang2022bc 通过在线专家干预缓解了这一问题，但它们仍然依赖于持续的人工监督，并且缺乏自主策略改进的机制。

为了超越模仿学习的局限，强化学习（Reinforcement Learning, RL）已被广泛用于机器人策略优化。传统方法采用 **同策略（On-policy）**  算法 schulman2017proximal 或 **异策略（Off-policy）**  方法 kalashnikov2018qt，通过环境交互来优化策略。最近的研究通过直接策略梯度优化 Tan2025InteractivePostTraining; Lu2025VLARL 或在冻结骨干网络上进行残差策略学习 Guo2025ImprovingVLA，将这些范式扩展到了 VLA 模型。然而，由于训练不稳定性和样本效率低下，将策略梯度方法扩展到大规模 VLA 模型仍然具有挑战性。

一个新兴的方向通过将动作生成 **条件化（Conditioning）**  于价值信号，绕过了显式的策略梯度计算，这包括 **奖励条件化策略（Reward-conditioned Policies）**  Kumar2019RewardConditionedPolicies 和 **优势条件化（Advantage-conditioned）**  的公式化方法 kuba2023advantage; Wu2023ElasticDecisionTransformer。最近，$\pi^{*}_{0.6}$ pi06 提出了  **RECAP**  框架，证明了优势条件化强化学习能使 VLA 模型通过机器人上的数据收集在下游任务中获得高性能。这激励我们探索基于世界模型的强化学习，其中世界模型联合预测价值和未来状态，以作为丰富的策略条件。

<a id="section-3"></a>

## 3 GigaBrain-0.5M*

基于我们的基础 **视觉语言动作模型（Vision-Language-Action model, VLA）**  GigaBrain-0.5，我们引入了 GigaBrain-0.5M*，这是一个集成了 **基于世界模型的强化学习（World Model-based Reinforcement Learning）** ： **RAMP（Reinforcement leArning via world Model-conditioned Policy）**  的增强策略模型。本节首先详细介绍 GigaBrain-0.5 的架构和预训练数据构成。然后，我们介绍 RAMP，这是一种利用世界模型预测，通过经验和纠正反馈信号迭代优化策略行为的训练方法。

<a id="section-3-1"></a>

### 3.1 GigaBrain-0.5

GigaBrain-0.5 继承了 GigaBrain-0 (gigabrain0) 的端到端 VLA 架构，旨在将视觉观测和语言指令映射为 **双臂机器人（Bi-manual robots）**  的动作序列。
它采用 **混合变换器（Mixture-of-Transformers, MoT）**  主干，利用预训练的  **PaliGemma-2（paligemma2）视觉语言模型（Vision-Language Model, VLM）**  对多模态输入进行编码，并采用带有 **流匹配（Flow Matching）**  (lipman2022flow) 的 **动作扩散变换器（Action Diffusion Transformer, DiT）**  (dit) 来预测动作块。
为了增强推理能力，GigaBrain-0.5 生成一个 **具身思维链（Embodied Chain-of-Thought, Embodied CoT）** ，该链由自回归子目标语言、离散动作标记 (pertsch2025fast) 和 2D 操作轨迹 $\mathbf{t}_{1:10}$ 组成。语言和离散标记通过 VLM 头部解码，而 2D 轨迹则通过一个轻量级的 **门控循环单元（Gated Recurrent Unit, GRU）**  解码器从可学习的标记中回归得到。在此版本中，深度信息和 2D 轨迹被视为可选状态，使模型能够适应不同的传感器模态和任务需求。
所有组件在一个统一的目标下联合优化：

$$
\mathcal{L}=\mathbb{E}_{\mathcal{D},\tau,\epsilon}\left[-\sum_{j=1}^{n-1}M_{\text{CoT},j}\log p_{\theta}\left(x_{j+1}\mid x_{1:j}\right)+\left\|\epsilon-a_{\text{chunk}}-f_{\theta}\left(a_{\text{chunk}}^{\tau,\epsilon}\right)\right\|^{2}+\lambda\left\|\text{GRU}(\hat{\mathbf{t}}_{1:10})-\mathbf{t}_{1:10}\right\|^{2}\right],(1)
$$

其中 $\mathcal{D}$ 是训练数据集，$M_{\text{CoT},j}\in\{0,1\}$ 是一个逐标记的掩码，指示位置 $j$ 是否属于 CoT 推理流（子目标语言或离散动作）。对于扩散过程，$\tau\in[0,1]$ 是流匹配时间步，$\epsilon\sim\mathcal{N}(0,\mathbf{I})$ 是高斯噪声，$a_{\text{chunk}}^{\tau,\epsilon}=\tau\cdot a_{\text{chunk}}+(1-\tau)\cdot\epsilon$ 表示加噪的动作块。项 $\hat{\mathbf{t}}_{1:10}$ 和 $\mathbf{t}_{1:10}$ 分别代表预测的和真实轨迹关键点，由超参数 $\lambda$ 平衡。
值得注意的是， **知识隔离（Knowledge Insulation, KI）**  本质上防止了语言预测项和动作预测项之间的优化干扰。

<a id="figure-2"></a>
![teaser](images/teaser.png)
> 图 2：RAMP 概览。RAMP 框架通过一个四阶段流程运行。(1)  **世界模型预训练（World Model Pre-training）**  为未来状态预测和价值估计建立统一的表示空间。(2)  **带世界模型条件的策略训练（Policy Training with World Model Condition）**  使用显式的世界模型条件初始化 GigaBrain-0.5 策略。(3)  **人在回路轨迹生成（Human-in-the-Loop Rollout, HILR）数据收集**  通过自主执行和专家纠正生成多样化和高质量的轨迹。(4)  **使用轨迹数据的持续训练（Continual Training with Rollout Data）**  利用标注的轨迹数据更新策略，同时包含成功的演示和纠正信号。这个紧密集成的闭环过程促进了策略的持续优化和自我改进。

<a id="section-3-2"></a>

### 3.2 RAMP（通过世界模型潜在状态进行奖励增强的演员-评论家策略）

在本节中，我们首先阐述所提出的  **RAMP（Reward-Augmented Actor-Critic Policy via World Model Latents）**  框架，并证明  **RECAP（Reward-Weighted Conditional Action Probabilities）**  [pi06] 是该框架内的一个特例。随后，我们将详细介绍 RAMP 的实现，该实现包含四个迭代训练阶段：
1.   **世界模型预训练（World Model Pre-training）** 
2.   **策略预训练（Policy Pre-training）** 
3.   **人在回路中推演（Human-in-the-Loop Rollout, HILR）数据收集** 
4.   **使用推演数据进行策略训练（Policy Training with Rollout Data）** 

#### 3.2.1 RAMP 公式化

为了推导一个可扩展的、能利用世界模型潜在状态的训练目标，我们将  **KL 正则化强化学习（KL-regularized reinforcement learning）**  框架扩展到我们的增广状态空间 $\mathbf{S}=(\mathbf{o},\mathbf{z},l)$，其中 $\mathbf{z}$ 代表由世界模型提取的潜在表示。我们的目标是在最大化期望回报的同时，通过  **KL 散度（Kullback-Leibler divergence）**  约束策略 $\pi$ 偏离参考策略 $\pi_{\text{ref}}(\cdot|\mathbf{S})$。借鉴正则化强化学习中的标准结论，最优策略的闭式解由 [pi06] 给出：

$$
\hat{\pi}(a|\mathbf{S})\propto\pi_{\text{ref}}(a|\mathbf{S})\exp\left(\frac{A^{\pi_{\text{ref}}}(\mathbf{S},a)}{\beta}\right).(2)
$$

为了缓解与直接估计指数优势项相关的数值不稳定性，我们引入一个二元改进指示器 $I$，并假设观察到改进事件 $p(I|a,\mathbf{S})$ 的概率与动作的指数优势成正比。通过应用 **贝叶斯定理（Bayes' theorem）** ，我们将这个难以处理的优势项重新表述为条件概率之比：$\exp(A^{\pi_{\text{ref}}}(\mathbf{S},a)/\beta)\propto\pi_{\text{ref}}(a|I,\mathbf{S})/\pi_{\text{ref}}(a|\mathbf{S})$。将此比例代回最优策略方程，将 $\hat{\pi}$ 重新表达为无条件分布和条件改进分布的复合。因此，我们参数化一个神经网络 $\pi_{\theta}$ 来同时拟合这些分布，从而得到最小化加权负对数似然的最终训练目标：

$$
\mathcal{L}(\theta)=\mathbb{E}_{D}\left[-\log\pi_{\theta}(a|\mathbf{o},\mathbf{z},l)-\alpha\log\pi_{\theta}(a|I,\mathbf{o},\mathbf{z}_{t},l)\right],(3)
$$

其中 $I=\mathds{1}[A(\mathbf{o},\mathbf{z},l,a)>\epsilon]$ 作为改进信号。

在此目标中明确包含潜在状态 $z$ 不仅仅是一个结构上的选择，更是一种理论上的必然。为了证明这一设计的合理性，我们从概率论的角度审视了我们的方法 RAMP 与现有方法（如 RECAP [pi06]）之间的关系。我们建立了这两种范式之间的内在联系。
从概率建模的角度，我们在理论上建立了 RAMP 与 RECAP 之间的内在联系，证明 RECAP 本质上是 RAMP 的一个退化特例，其中忽略了关于未来潜在状态的信息。具体来说，RECAP 的策略形式 $\pi(a|o,I)$ 在数学上等价于 RAMP 策略 $\pi(a|\mathbf{o},\mathbf{z},I)$ 在潜在未来状态 $\mathbf{z}$ 上的边际分布：

$$
\pi_{RECAP}(a|\mathbf{o},I)=\int_{z}\pi_{RAMP}(a|\mathbf{o},\mathbf{z},I)p(\mathbf{z}|\mathbf{o},I)d\mathbf{z}.(4)
$$

这意味着， **RECAP（RECAP）**  有效地学习了一个 **平均策略（average policy）** ，该策略必须在没有具体指导的情况下，隐式地整合并折衷所有可能的未来演化。相比之下， **RAMP（RAMP）**  通过显式地以世界模型（world model）的预测 $\mathbf{z}$ 为条件，消除了这种不确定性，将问题从对未来平均值的猜测，转变为针对特定物理状态的精确规划。

此外，从信息论（Information Theory）的角度来看，引入时空潜在变量 $\mathbf{z}$ 为动作生成提供了显著的 **信息增益（Information Gain）** 。虽然 RECAP 仅依赖于稀疏的二元优势信号（$I\in\{0,1\}$）进行粗略的 **信用分配（credit assignment）** ，但 RAMP 利用 $\mathbf{z}$ 注入了密集的几何结构和物理动力学先验，从而显著降低了动作生成的 **条件熵（conditional entropy）** ：$H(a|\mathbf{o},\mathbf{z},I)\leq H(a|\mathbf{o},I)$。

#### 3.2.2 RAMP 实现（The RAMP Implementation）

RAMP 通过在整个训练生命周期中融入世界模型指导，使 **视觉语言动作模型（Vision-Language-Action models, VLA models）**  能够从经验中学习。从大规模离线预训练到基于自主推演数据的多轮迭代微调，我们的方法实现了策略的渐进式改进。如图 [2](#S3.F2) 所示，该流程被结构化为四个渐进阶段：

 **阶段 1：世界模型预训练（World Model Pre-training）** 。初始阶段建立一个世界模型 $\mathcal{W}_{\phi}$，该模型能够联合预测未来的视觉状态和 **价值估计（value estimates）** 。遵循 pi06 的方法，我们从 **片段级（episode-level）**  成功标签中推导出稀疏奖励，使得 **价值函数（value function）**  对应于负的期望完成步数。具体而言，奖励函数定义为：

$$
r_{t}=\begin{cases}0&\text{如果 }t=T\text{ 且片段成功},\\ -C_{\text{fail}}&\text{如果 }t=T\text{ 且片段失败},\\ -1&\text{其他情况},\end{cases}(5)
$$

其中 $T$ 表示片段的终止时间步，$C_{\text{fail}}$ 是一个大的正常数，其选择是为了确保失败片段获得的 **累积回报（cumulative returns）**  远低于成功片段。这种稀疏奖励公式鼓励策略在优先考虑任务完成而非部分进展的同时，最小化执行时间。遵循潜在帧注入策略（cosmospolicy），我们将价值信号嵌入为一个额外的潜在帧，在输入世界模型之前与视觉潜在状态进行拼接。这种方法无需对底层的 **扩散变换器（Diffusion Transformer, DiT）**  进行架构修改。具体来说，未来的视觉观测 $\{\mathbf{o}_{t+i}\}_{i\in\{12,24,36,48\}}$ 首先使用预训练的 **变分自编码器（Variational Autoencoder, VAE）**  编码为时空视觉潜在变量 $\mathbf{z}_{t}\in\mathbb{R}^{H^{\prime}\times W^{\prime}\times C^{\prime}}$。同时，标量和低维辅助信号，包括当前价值估计 $v_{t}\in\mathbb{R}$ 和 **本体感觉状态（proprioceptive state）**  $\mathbf{p}_{t}\in\mathbb{R}^{d}$，通过一个 **空间平铺投影（spatial tiling projection）**  $\Psi(\cdot)$ 进行变换。该投影在空间维度上复制和广播低维向量，以匹配视觉潜在变量的形状。然后构建完整的潜在状态：

$$
\mathbf{s}_{t}=\big[\mathbf{z}_{t}\,;\,\Psi(v_{t})\,;\,\Psi(\mathbf{p}_{t})\big],(6)
$$

其中 $[\cdot\,;\,\cdot]$ 表示通道级拼接。这种统一表示使世界模型能够在单次前向传播中，联合推理视觉动态、任务进展（通过价值）和机器人运动学。

我们采用 Wan2.2 (wan) 作为世界模型 $\mathcal{W}_{\phi}$ 的 **骨干架构（backbone architecture）** 。该模型通过 **流匹配（flow matching）**  (lipman2022flow) 进行训练。通过将未来的视觉状态和价值估计视为时间上扩展的视频帧，DiT 骨干自然利用其时空 **自注意力机制（self-attention mechanisms）**  来建模当前观测、动作和未来任务结果之间的关系：

$$
\mathcal{L}_{\text{WM}}=\mathbb{E}_{\mathcal{D},\tau,\epsilon}\left[\left\|\mathcal{W}_{\phi}\big(\mathbf{s}_{\text{future}}^{\tau,\epsilon}\big)-(\mathbf{s}_{\text{future}}-\epsilon)\right\|^{2}\right],(7)
$$

其中 $\mathbf{s}_{\text{future}}^{\tau,\epsilon}=\tau\mathbf{s}_{\text{future}}+(1-\tau)\epsilon$ 表示源噪声 $\epsilon\sim\mathcal{N}(0,\mathbf{I})$ 与真实潜在状态序列 $\mathbf{s}_{\text{future}}$ 之间的线性插值，其中 $\tau\sim\mathcal{U}(0,1)$。目标项 $(\mathbf{s}_{\text{future}}-\epsilon)$ 对应于沿噪声与数据分布之间 **最优传输（Optimal Transport）** 路径的恒定速度向量场。我们利用 4K 小时的真实机器人操作数据来训练世界模型，数据分布的可视化结果见图 [3](#S4.F3)。

 **阶段 2：基于世界模型条件化的策略训练（Stage 2: Policy Training with World Model Conditioning）** 。第二阶段从预训练的 GigaBrain-0.5 检查点初始化策略，并利用世界模型条件化对其进行进一步微调。训练细节见第 [4.2](#S4.SS2) 节。具体而言，策略接收由世界模型 $\mathcal{W}_{\phi}$ 预测的两个辅助信号：(1) 未来状态词元 $\mathbf{z}_{\text{future}}$，以及 (2) 价值估计 $v_{t}$。未来状态词元通过一个轻量级 **多层感知机（Multilayer Perceptron, MLP）** 进行投影，以使其维度与策略的视觉编码器输出对齐。价值估计通过 $n$ 步 **时序差分（Temporal Difference）** 估计转化为动作优势：

$$
A(\mathbf{s}_{t},a_{t})=\sum_{k=0}^{n-1}\gamma^{k}r_{t+k}+\gamma^{n}v_{t+n}-v_{t},(8)
$$

其中 $v_{t}$ 和 $v_{t+n}$ 分别表示对状态 $\mathbf{s}_{t}$ 和 $\mathbf{s}_{t+n}$ 的价值预测，$\gamma$ 是折扣因子。为了在保留偏好结构的同时简化条件化，优势被离散化为一个二元指示器 $I=\mathds{1}\big(A(\mathbf{s}_{t},a_{t})>\epsilon\big)$，其中 $\epsilon$ 为阈值。然后，通过最小化公式 [3](#S3.E3) 中定义的监督微调目标，训练策略生成以元组 $(I,\mathbf{z})$ 为条件的动作。

为了防止过度依赖合成的世界模型信号并确保灵活部署，我们在训练期间采用两种互补策略。首先，世界模型在推理期间仅执行单步去噪，以最小化计算开销。其次，我们实现了 **随机注意力掩码（Stochastic Attention Masking）** ，在训练期间以概率 $p=0.2$ 随机抑制世界模型词元。这迫使策略即使在世界模型输入部分或完全不可用时也能保持鲁棒性能，从而实现一种绕过世界模型条件化的高效推理模式。

 **阶段 3：人在回路（Human-in-the-Loop, HIL）的轨迹数据收集（Stage 3: Human-in-the-Loop Rollout Data Collection）** 。在第三阶段，我们部署策略，通过人在回路的轨迹执行来收集轨迹。生成的数据集包含自主执行和专家干预的混合。与传统的遥操作相比，自主执行的轨迹表现出显著减小的动作分布差距，因为策略是在其自身分布中生成动作，而非模仿人类演示，从而为 **视觉-语言-动作（Vision-Language-Action, VLA）** 学习提供了更有效的监督信号。然而，自主执行不可避免地会遇到需要人工纠正的失败模式。为了减轻手动干预引入的时间不连续性，我们开发了一个人在回路轨迹数据收集软件，该软件能自动检测并移除干预边界处的过渡伪影。这种平滑机制确保了整个轨迹的时间连贯性，生成了一个干净、连续的数据集，便于在后续训练阶段进行稳定的策略更新，同时保留了专家纠正的教学价值。

 **阶段 4：基于轨迹数据的持续训练（Stage 4: Continual Training with Rollout Data）** 。在此阶段，我们使用精心整理的 HILR 数据集对策略进行微调，以掌握由自主执行和专家纠正的多样化混合所产生的复杂长时程行为。至关重要的是，为了防止优势值坍缩为零（$A(\mathbf{s}_{t},a_{t})\approx 0$），世界模型 $\mathcal{W}_{\phi}$ 与 HILR 数据集和基础数据一起进行联合训练。
对于策略训练，与阶段 2 保持一致，我们保持随机注意力掩码，掩码概率 $p=0.2$ 同时应用于优势指示器 $I$ 和未来潜在词元 $\mathbf{z}_{\text{future}}$。这种正则化具有双重目的：(1) 通过强制策略对缺失的条件输入具有鲁棒性，防止其过度依赖世界模型信号；(2) 确保预训练和微调阶段在架构和训练上的一致性，避免推理时的分布偏移。
轨迹执行-标注-训练循环迭代运行，建立了一个自我改进的闭环：随着策略的改进，其自主执行的轨迹覆盖了越来越复杂和成功的行为，这反过来又为后续迭代生成了更高质量的训练数据。

 **推理（Inference）** 。在部署期间，我们通过将优势指示器固定为 $I=1$ 来强制执行一种乐观的控制策略。
关于潜在条件 $\mathbf{z}$，随机掩码促成的架构解耦实现了两种灵活的执行模式：(1)  **高效模式（Efficient Mode）** ，绕过世界模型以最大化推理频率。在此设置下，注意力掩码被配置为使未来潜在词元对策略不可见，迫使其仅基于当前观察采取行动；(2)  **标准模式（Standard Mode）** ，世界模型主动生成 $\mathbf{z}$ 以提供密集的前瞻指导。在此设置下，注意力掩码允许策略完全看到预测的未来状态，使其能够利用前瞻上下文进行复杂的、长时程的规划。

<a id="section-4"></a>

## 4 实验（Experiment）

在本节中，我们首先评估我们的基础模型  **GigaBrain-0.5**  的性能。在内部机器人评估中，我们的模型在执行 **长时程（long-horizon）** 、复杂程序（如 **装箱（box packing）** 和 **咖啡制备（coffee preparation）** ）方面展现出稳健的能力。在公开基准  **RoboChallenge**  [robochallenge] 上，与 $\pi_{0.5}$ [pi05] 相比，我们的基础模型也取得了更优的性能。
接下来，我们将我们基于 **世界模型（world model）** 的 **强化学习（Reinforcement Learning, RL）** 方法  **RAMP**  与已建立的 RL 基线进行比较，包括  **AWR**  [awr] 和  **RECAP**  [pi06]。实验结果证实，RAMP 表现出显著更高的 **样本效率（sample efficiency）** 和更强的 **多任务泛化（multi-task generalization）** 能力。
最后，我们进行 **消融研究（ablation studies）** ，分析我们世界模型中 **价值预测模块（value prediction module）** 的贡献，定量验证其对于 **策略学习（policy learning）** 和任务成功的重要性。

<a id="figure-3"></a>
![RAMP](images/RAMP.png)
> 图 3 | GigaBrain-0.5 预训练阶段的数据分布。

<a id="section-4-1"></a>

### 4.1 基础模型性能（Foundation Model Performance）

 **预训练细节（Pre-training Details）** 。
GigaBrain-0.5 在一个超过 10,000 小时的多样化数据集上进行预训练，该数据集包含超过 6,000 小时的世界模型生成数据和大约 4,000 小时的真实机器人采集数据。详细的数据构成如图 [3](#S4.F3) 所示。我们使用我们的训练框架  **GigaTrain**  [^1^11](https://github.com/open-gigaai/giga-train) 训练 GigaBrain-0.5， **批量大小（batch size）** 为 3,072，共进行 100,000 步优化。为了减少单 GPU 内存消耗，我们采用了  **全分片数据并行（Fully Sharded Data Parallel, FSDP）**  v2，选择性地对所有  **SiglipEncoderLayer**  模块和  **Gemma2DecoderLayerWithExpert**  的前 16 层应用分片。

 **后训练细节（Post-training Details）** 。
为了评估 GigaBrain-0.5 在物理机器人上的性能，我们在目标机器人平台上收集特定任务的演示数据，并进行 **后训练（post-training）** 以使模型适应每个任务。我们对八个内部设计的任务进行了全面评估，并在公开基准 RoboChallenge 的 30 个任务上对模型进行了额外的后训练。RoboChallenge 任务和评估协议的详细信息在 [robochallenge] 中描述。我们的八个内部评估任务包括： **果汁制备（Juice Preparation）** 、 **箱子搬运（Box Moving）** 、 **餐桌清理（Table Bussing）** 、 **纸巾准备（Paper Towel Preparation）** 、 **衣物折叠（Laundry Folding）** 、 **衣物收集（Laundry Collection）** 、 **装箱（Box Packing）**  和  **意式浓缩咖啡制备（Espresso Preparation）** 。这些任务的演示视频在我们的项目页面上展示。对于每个任务，我们以 256 的批量大小进行了 20,000 步优化的后训练。

<a id="figure-4"></a>
![pretrain-data](images/pretrain-data.png)
> 图 4 | GigaBrain-0.5 在内部评估中的性能。

<a id="figure-5"></a>
![gigabrain05_exp](images/gigabrain05_exp.png)
> 图 5 | GigaBrain-0.5 在 PiPER 机械臂上部署进行真实世界的装箱任务。

<a id="figure-6"></a>
![Box_Packing](images/Box_Packing.png)
> 图 6 | GigaBrain-0.5 在 G1 人形机器人上部署进行真实世界的箱子搬运任务。

<a id="figure-7"></a>
![Boxes_moving](images/Boxes_moving.png)
> 图 7 | GigaBrain-0.5 在 PiPER 机械臂上部署进行真实世界的意式浓缩咖啡制备任务。

<a id="figure-8"></a>
![Espresso_Preparation](images/Espresso_Preparation.png)
> 图 8 | GigaBrain-0.5 在 G1 人形机器人上部署进行真实世界的果汁制备任务。

<a id="figure-9"></a>
![Juice_Preparation](images/Juice_Preparation.png)
> 图 9 | GigaBrain-0.5 在 G1 人形机器人上部署进行真实世界的衣物收集任务。

<a id="figure-10"></a>
![Laundry_Collection](images/Laundry_Collection.png)
> 图 10 | GigaBrain-0.5 在 PiPER 机械臂上部署进行真实世界的衣物折叠任务。

<a id="figure-11"></a>
![Laundry_Folding](images/Laundry_Folding.png)
> 图 11 | GigaBrain-0.5 在 PiPER 机械臂上部署进行真实世界的纸巾准备任务。

<a id="figure-12"></a>
![Paper_Towel_Preparation](images/Paper_Towel_Preparation.png)
> 图 12 | GigaBrain-0.5 在 G1 人形机器人上部署进行真实世界的餐桌清理任务。

 **内部评估（Internal Evaluation）** 。在我们的实验中，我们将 GigaBrain-0.5 与多个强基线模型进行了基准测试，包括 $\pi_{0}$（pi0）、$\pi_{0.5}$ 和 GigaBrain-0（gigabrain0）。结果总结于图 [4](#S4.F4)。 **GigaBrain-0.5**  在所有评估任务中均较其前代  **GigaBrain-0（gigabrain0）**  取得了一致且显著的提升，在每种情况下都达到了最高的成功率，在复杂的多步骤程序中提升尤为显著。例如，在需要顺序处理食材和混合的 **果汁制备（Juice Preparation）**  任务中，GigaBrain-0.5 实现了 100% 的成功率，超过了 GigaBrain-0 的 90%。对于 **装箱（Box Packing）**  和 **意式浓缩咖啡制备（Espresso Preparation）**  等具有挑战性的任务，GigaBrain-0.5 的成功率分别比 $\pi_{0.5}$ 提高了 10% 和 20%。同样，在高度灵巧的操作任务（ **纸巾准备（Paper Towel Preparation）** 、 **衣物折叠（Laundry Folding）**  和 **衣物收集（Laundry Collection）** ）上，GigaBrain-0.5 的成功率超过 80%，分别比 $\pi_{0.5}$ 高出 15%、5% 和 10%。此外，我们在图 [5](#S4.F5) 至图 [12](#S4.F12) 中可视化了八个任务。

 **RoboChallenge 评估（RoboChallenge Evaluation）** 。除了内部任务评估，我们还对  **RoboChallenge（robochallenge）**  基准进行了全面评估。RoboChallenge 是世界上首个支持真实机器人测试的大规模具身人工智能（Embodied AI）评估平台。该平台已建立了一套标准化的远程评估协议，覆盖了包含四大主要平台（UR5、Franka、ARX5 和 ALOHA）的 20 台物理机器人集群。它还进一步提供了包含 30 个标准化操作任务的开源数据集（736 GB）。详细的任务规范和评估方法见 (robochallenge)。截至 2026 年 2 月 9 日，一个中间迭代模型（ **GigaBrain-0.1** ）目前在排行榜上排名第一，实现了 51.67% 的平均成功率，比 $\pi_{0.5}$（42.67%）提高了 9%。

<a id="section-4-2"></a>

### 4.2 RAMP 性能（RAMP Performance）

在本节中，我们通过系统的实证评估来回答三个核心问题：(1) 与 $\pi^{*}_{0.6}$ 中采用的基于 **视觉语言模型（Vision-Language Model, VLM）**  的方法相比，基于 **世界模型（World Model）**  的价值预测是否提供了更高的准确性和效率？(2) 世界模型条件化是否能增强视觉-语言-动作策略的跨任务泛化能力？(3) 我们提出的  **RAMP 算法（RAMP algorithm）**  在真实机器人环境中与替代的 **强化学习（Reinforcement Learning, RL）**  方法相比表现如何？

 **价值预测性能（Value Prediction Performance）** 。
为了评估基于世界模型的价值预测的有效性，我们将其与一个基于 VLM 的基线（pi06）和我们基于世界模型的方法进行了比较。对于 VLM 中的价值预测，我们在视觉词元序列的末尾插入一个可学习的 [CLS] 词元以聚合全局场景表示，然后通过一个回归头（regression head）将该词元的隐藏状态投影，以在 $[0,1]$ 区间内产生一个标量价值预测。模型使用 **均方误差（Mean Squared Error, MSE）**  进行优化。
基于 VLM 和基于世界模型的价值预测器都在相同的预训练数据上进行训练，并在一个验证集上进行评估，该验证集包含图 [4](#S4.F4) 所示的八个操作任务中大约 100 万帧数据。我们使用四个互补的指标评估预测质量： **平均绝对误差（Mean Absolute Error, MAE）** 、MSE、 **均方根误差（Root Mean Square Error, RMSE）** （三者均为越低越好）以及 **肯德尔秩相关系数（Kendall's tau rank correlation coefficient）** （越高越好，1 表示完美的秩次保持）。所有任务的平均结果总结于表 [1](#S4.T1)。

我们的分析揭示了三个关键发现。首先，尽管使用了轻量级 VLM（pi06），但由于  **SigLIP（siglip）**  视觉编码器的计算开销，基于 VLM 的方法产生了最高的每帧推理延迟（在 A800 GPU 上为 0.32 秒）。其次，仅预测价值的世界模型变体实现了最快的推理速度（0.11 秒），但预测精度下降（MAE=0.0838，Kendall=0.7288），这表明仅建模价值未能充分利用世界模型固有的未来预测能力。第三，我们提出的联合预测方案，同时预测价值和未来状态，达到了最佳平衡。它实现了最高的肯德尔系数（0.8018）和最低的 MAE（0.0621），同时保持了有竞争力的推理速度（0.25 秒）。这表明，利用未来状态预测为准确的价值估计提供了关键的上下文基础。代表性任务的定性价值预测可视化见图 [13](#S4.F13)。

<a id="figure-13"></a>
![Table_Bussing](images/Table_Bussing.png)
> 图 13 | 来自世界模型的价值预测可视化。橙色边界框高亮了在衣物折叠任务中，当一件绿色衣物干扰折叠过程时出现的价值下降，在机械臂成功移除障碍物后，预测价值得以恢复。

<a id="table-1"></a>
 **表 1：不同价值预测方法的性能比较（Table 1: Performance comparison of different value prediction methods.）** 
| Model | Inference Time (s) | MAE $\downarrow$ | MSE$\downarrow$ | RMSE $\downarrow$ | Kendall$\uparrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| VLM-based | 0.32 | 0.0683 | 0.0106 | 0.1029 | 0.7972 |
| WM-based (value only) | 0.11 | 0.0838 | 0.0236 | 0.1433 | 0.7288 |
| WM-based (state+value) | 0.25 | 0.0621 | 0.0099 | 0.0989 | 0.8018 |

 **用于策略学习的** 世界模型（World Model） **条件化（World Model Conditioning for Policy Learning）** 。
为了评估世界模型条件化是否能增强多任务泛化能力，我们在单任务和多任务训练方案之间进行了对照比较。我们选择了四个具有代表性的操作任务进行评估： **餐桌清理（Table Bussing）** 、 **叠衣服（Laundry Folding）** 、 **纸巾准备（Paper Towel Preparation）**  和 **装箱（Box Packing）** 。为确保公平比较并隔离世界模型条件化的效果，我们所有策略都仅在 RAMP 的 Stage-2 数据集上进行训练，不包含任何生成的轨迹数据。
对于单任务训练，每个策略独立训练 20000 步，批大小为 256。对于多任务训练，我们将所有四个任务的数据均匀混合，并使用相同的批大小训练一个单一策略 60000 步。
我们的实验结果，如图 [14](#S4.F14) 所示，表明 **世界模型条件化方法（world model condition approach）**  在单任务和多任务训练场景中均持续优于基线。具体而言，融入世界模型在所有评估任务中都带来了显著的性能提升，从 5000 步到 20000 步的整个训练轨迹中都能持续观察到显著的改进。值得注意的是，性能提升在多任务设置中尤为明显，世界模型方法与基线之间的成功率差距在训练过程中逐渐扩大，在如装箱（Box Packing）等任务中，在第 20000 步时取得了高达 $\sim$30% 的成功率提升。这表明，世界模型条件化能有效促进跨多个任务的知识迁移，同时在单任务场景中保持稳健的性能增益。

<a id="figure-14"></a>
![reward](images/reward.png)
> 图 14：有无世界模型条件的单任务与多任务性能比较（Figure 14: Comparison of single-task and multi-task performance with and without world model conditions.）

 **与强化学习（Reinforcement Learning, RL）基线的比较（Comparison with RL Baselines）** 。
我们将 RAMP 与最先进的强化学习基线进行基准测试。

*    **GigaBrain-0.5 + AWR (awr)** ：一个 **离线强化学习（Offline RL）**  基线，它使用 **加权模仿学习（Weighted Imitation Learning）**  对 GigaBrain-0.5 策略进行微调，利用了当前策略生成的轨迹。
*    **GigaBrain-0.5 + RECAP (pi06)** ：一种 **优势条件化（Advantage-conditioned）**  的离线强化学习方法，通过优势输入扩展了 GigaBrain-0.5 主干网络，作为我们方法的一个 **消融变体（Ablated Variant）** （不包含状态预测）。
*    **GigaBrain-0.5 + RAMP (GigaBrain-0.5M\*)** ：我们提出的 **基于世界模型条件化策略的强化学习（Reinforcement learning via world Model-conditioned Policy）**  框架，该框架将 GigaBrain-0.5 策略同时条件化于预测的价值和未来状态 **潜在变量（Latents）** ，以优化长视野任务性能。

我们的 RAMP 框架在所有三个极具挑战性的操作任务上均确立了顶尖性能： **装箱（Box Packing）** 、 **意式浓缩咖啡制备（Espresso Preparation）**  和 **叠衣服（Laundry Folding）** 。如图 [15](#S4.F15) 量化所示，RAMP 在所有评估任务上都取得了接近完美的成功率，显著优于所有基线方法（GigaBrain-0.5、GigaBrain-0.5+AWR 和 GigaBrain-0.5+RECAP）。值得注意的是，RAMP 在装箱和意式浓缩咖啡制备任务上表现出特别显著的提升，分别比 RECAP 基线高出约 30 个百分点。
至关重要的是，GigaBrain-0.5M^∗ 模型（即集成了我们 RAMP 框架的 GigaBrain-0.5）展现出稳健且一致的任务执行能力，在实际部署中取得了可靠的成功率，这已通过我们项目页面上的补充执行视频得到经验验证。这种在多个复杂操作任务上前所未有的性能，凸显了 RAMP 在解决具有挑战性的现实世界机器人问题上的有效性。

<a id="figure-15"></a>
![multi_task_exp](images/multi_task_exp.png)
> 图 15：不同强化学习方法的比较（Figure 15: Comparison of different RL methods.）

<a id="section-5"></a>

## 5 结论与未来工作（Conclusion and Future Work）

在本工作中，我们提出了  **GigaBrain-0.5**  及其 **世界模型（World Model）** 增强的后继版本  **GigaBrain-0.5M** *，通过 **大规模预训练（Large-scale Pretraining）** 和 **基于模型的强化学习（Model-based Reinforcement Learning, Model-based RL）** ，推进了 **视觉-语言-动作（Vision-Language-Action, VLA）** 学习的前沿。GigaBrain-0.5 在超过 10,000 小时多样化的机器人数据上进行了预训练，在八个内部操作任务和 RoboChallenge 基准测试的 30 个标准化任务上均展示了最先进的性能，取得了 51.67% 的平均成功率，并在公开排行榜上获得了首位。基于这一坚实基础，GigaBrain-0.5M* 引入了一种新颖的 **世界模型条件化架构（World Model-conditioned Architecture）** ，该架构利用 **未来状态预测（Future State Prediction）** 来克服传统 VLA 模型固有的有限预测能力。通过  **RAMP（Reinforcement Learning with Model-based Planning）**  集成基于模型的强化学习，我们的方法实现了鲁棒的跨任务泛化能力，并能可靠地执行复杂的 **长视野任务（Long-horizon Tasks）** ，例如顺序装箱和意式浓缩咖啡制作。

展望未来，GigaBrain 系列将研究如何更高效地利用 **模型推演数据（Model Rollout Data）** ，以最大化 **合成轨迹（Synthetic Trajectories）** 的信息价值，同时最小化计算开销。此外，我们旨在探索更具可扩展性的 **自进化范式（Self-evolution Paradigms）** ，通过 **闭环交互（Closed-loop Interaction）** 实现自主的数据管理、策略精炼和世界模型更新。

## 参考文献（References）