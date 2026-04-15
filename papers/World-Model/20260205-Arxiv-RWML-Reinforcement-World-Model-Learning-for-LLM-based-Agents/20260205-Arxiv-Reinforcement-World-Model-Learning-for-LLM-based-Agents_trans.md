# 标题：基于大语言模型的智能体的强化世界模型学习（Reinforcement World Model Learning for LLM-based Agents）

- ArXiv: 2602.05842
- 作者：Xiao Yu, Baolin Peng, Ruize Xu, Yelong Shen, Pengcheng He, Suman Nath, Nikhil Singh, Jiangfeng Gao, Zhou Yu
- ∗Project Lead †Equal Advising 1Columbia University, New
York 2Microsoft Research, Redmond 3Dartmouth College,
Hanover. Correspondence to: Xiao Yu <xy2437@columbia.edu>
- 章节数：34
- 估计词元数：31.2k


## 目录

- [标题：基于大语言模型的智能体的强化世界模型学习（Reinforcement World Model Learning for LLM-based Agents）](#标题基于大语言模型的智能体的强化世界模型学习reinforcement-world-model-learning-for-llm-based-agents)
  - [目录](#目录)
  - [摘要（Abstract）](#摘要abstract)
  - [1 引言（Introduction）](#1-引言introduction)
  - [2 方法（Method）](#2-方法method)
    - [2.1 符号表示（Notation）](#21-符号表示notation)
    - [2.2 强化世界模型学习（Reinforcement World Model Learning）](#22-强化世界模型学习reinforcement-world-model-learning)
  - [3 实验（Experiments）](#3-实验experiments)
    - [3.1 实验设置（Experiment Setup）](#31-实验设置experiment-setup)
      - [基准测试（Benchmarks）](#基准测试benchmarks)
      - [基线方法（Baselines）](#基线方法baselines)
      - [模型与训练数据（Models and Training Data）](#模型与训练数据models-and-training-data)
    - [3.2 主要结果（Main Results）](#32-主要结果main-results)
    - [3.3 RWML 遗忘更少（RWML Forgets Less）](#33-rwml-遗忘更少rwml-forgets-less)
    - [3.4 消融研究（Ablation Studies）](#34-消融研究ablation-studies)
  - [4 讨论（Discussion）](#4-讨论discussion)
    - [4.1 RWML对决策的影响（Impact of RWML on Decision-Making）](#41-rwml对决策的影响impact-of-rwml-on-decision-making)
    - [4.2 权重变化分析（Weight Change Analysis）](#42-权重变化分析weight-change-analysis)
    - [4.3 基础模型能力的影响（Impact of Base Model Capability）](#43-基础模型能力的影响impact-of-base-model-capability)
  - [5 相关工作（Related Work）](#5-相关工作related-work)
      - [训练决策智能体（Training Decision-Making Agents）](#训练决策智能体training-decision-making-agents)
      - [训练世界模型（Training World Models）](#训练世界模型training-world-models)
  - [6 结论（Conclusion）](#6-结论conclusion)
  - [7 影响声明（Impact Statements）](#7-影响声明impact-statements)

## 摘要（Abstract）

**大语言模型（Large Language Models, LLMs）** 在以语言为中心的任务中取得了强大的性能。然而，在智能体（agentic）场景中，LLMs 常常难以预测行动后果并适应环境动态，这凸显了基于 LLM 的智能体对 **世界建模（world-modeling）** 能力的需求。我们提出了 **强化世界模型学习（Reinforcement World Model Learning, RWML）** ，这是一种自监督方法，它利用 **仿真到现实的差距奖励（sim-to-real gap rewards）** ，在文本状态上为基于 LLM 的智能体学习 **动作条件化的世界模型（action-conditioned world models）** 。我们的方法将模型生成的模拟下一状态与从环境中观察到的实际下一状态对齐，鼓励在预训练的嵌入空间中，内部世界模拟与实际环境动态之间保持一致。与 **下一状态词元预测（next-state token prediction）** 相比——后者优先考虑词元级别的保真度（即重现确切的措辞）而非语义等价性，并可能导致 **模型崩溃（model collapse）** ——我们的方法提供了更鲁棒的训练信号，并且经验上比 **LLM 作为评判者（LLM-as-a-judge）** 更不易受到 **奖励黑客攻击（reward hacking）** 。我们在 ALFWorld 和 $\tau^{2}$ Bench 上评估了我们的方法，观察到相对于基础模型的显著提升，尽管完全是 **自监督（self-supervised）** 的。当与 **任务成功奖励（task-success rewards）** 结合时，我们的方法在 ALFWorld 和 $\tau^{2}$ Bench 上分别比直接使用任务成功奖励的强化学习高出 6.9 和 5.7 个百分点，同时达到了专家数据训练的性能。

<a id="section-1"></a>

## 1 引言（Introduction）

**大语言模型（Large Language Models, LLMs）** 在广泛的以语言为中心的任务中取得了显著成功，包括问答、代码生成和多步推理（Brown 等人，2020；Wei 等人，2022；Lample 和 Conneau，2019；Rozière 等人，2024；DeepSeek-AI 等人，2025；OpenAI 等人，2024）。这些进展促使人们越来越关注将 LLMs 用作自主智能体，以与现实环境交互并完成 **长视野任务（long-horizon tasks）** （Yao 等人，2023；Deng 等人，2023）。尽管具备强大的语言和推理能力，基于 LLM 的智能体在许多需要预测行动后果并适应环境动态的智能体场景中仍然表现不佳（Liu 等人，2025）。这种差异凸显了预训练获得的 **语言能力（language competence）** 与基于 LLM 的智能体所需的 **智能体智能（agentic intelligence）** 之间的区别。

<a id="figure-1"></a>

![wmrl-cover_fig_cropped](images/wmrl-cover_fig_cropped.png)

> 图 1：我们提出 RWML 作为一种可扩展的自监督方法，通过学习下一状态来提升基于 LLM 的智能体的世界建模能力，这发生在从任务成功奖励中学习策略的下游强化学习之前。

造成这一限制的一个关键原因是标准预训练目标与智能体用例之间的错位。标准的预训练目标，如在静态文本语料库上进行 **下一词元预测（next-token prediction）** ，强调语言理解和生成。相比之下，现代基于 LLM 的智能体在复杂、长视野的环境中运行，成功完成任务需要推理当前状态以及环境如何响应行动而演变（LeCun，2022；Hu 和 Shu，2023；Hao 等人，2023）。对自身行动潜在未来结果进行建模的能力是生物智能的核心。神经科学和心理学研究表明，人类、动物和智能系统使用内部世界模型进行推理、规划、探索，并能从极少的尝试中高效学习（Craik，1944；Tolman，1948；Daw 等人，2005；Daw 和 Dayan，2014；Bennett，2023）。我们认为，这种世界建模能力对于基于 LLM 的智能体进行有效的推理和规划同样至关重要。

最近的研究探索了为基于 LLM 的智能体配备世界建模能力，训练 LLMs 使用下一词元预测（即 **监督微调（Supervised Fine-Tuning, SFT）** ）来预测下一状态。例如，Zhang 等人（2025a）；Yu 等人（2025c）教导 LLMs 使用专家策略提供的轨迹或由更强语言模型生成的高质量合成数据来建模环境转移。虽然在某些设置下有效，但这些方法面临可扩展性挑战：（1）它们严重依赖来自专家/强 LLMs 的高质量数据；（2）它们基于 SFT，优先考虑词元级别的保真度（即重现确切的措辞）而非语义等价性，并可能导致模型崩溃。

在本文中，我们提出了 **强化世界模型学习（Reinforcement World Model Learning, RWML）** ，一种基于 **强化学习（Reinforcement Learning, RL）** 的自监督训练方法，为基于 LLM 的智能体学习动作条件化的世界模型。RWML 不是用 SFT 优化词元级别的保真度，而是训练 LLMs 以最小化模型生成的模拟下一状态与从环境中观察到的实际下一状态之间的差异，该差异在预训练的嵌入空间中进行度量。这种 **仿真到现实的对齐（sim-to-real alignment）** 促进了智能体内部世界模型与实际环境动态之间的语义一致性，同时保留了与任务相关的转移，使其适用于下游决策。我们在两个长视野智能体基准测试（ALFWorld 和 $\tau^{2}$ Bench）上评估了我们的方法，发现 RWML 在不使用任何专家数据、强 LLMs 或任务成功奖励信号的情况下，将基础模型的性能显著提高了 19.6 和 6.9 个百分点。当与任务成功奖励结合时，使用 RWML 训练的智能体在 ALFWorld 和 $\tau^{2}$ Bench 上分别比直接使用任务成功奖励的强化学习高出 6.9 和 5.7 个百分点，同时达到了使用专家数据训练的性能。

总之，我们的贡献是：（1）我们提出 RWML 作为一种可扩展的自监督训练方法，用于基于 LLM 的智能体，它从仿真到现实的差距奖励中学习动作条件化的世界模型。（2）我们在两个长视野基准测试（ALFWorld 和 $\tau^{2}$ Bench）上评估了我们的方法，发现 RWML 显著提高了基础模型的性能。当与任务成功奖励结合时，我们的模型优于标准强化学习，并达到了使用专家数据训练的性能。（3）我们进行了全面的分析——包括消融研究、模型遗忘、定性分析等——以突出强化学习和世界模型学习对基于 LLM 的智能体的益处。

<a id="figure-2"></a>

![wmrl-main_algo_cropped](images/wmrl-main_algo_cropped.png)

> 图 2：RWML 概述。给定一个目标模型 $\pi_{\theta}$，我们首先通过使用 $\pi_{\theta}$ 与环境交互收集轨迹 $(s_{0},a_{0},s_{1},a_{1},...s_{T})$ 来为 RWML 收集训练数据，然后将这些轨迹转换为所有 $t$ 对应的 $\left\langle s_{\leq t},a_{t},s_{t+1}\right\rangle$ 三元组，并在子采样了[公式˜1](https://arxiv.org/html/2602.05842v2#S2.E1)中定义的“过于简单”的样本之后。然后，我们通过 **GRPO（Group Relative Policy Optimization）** 训练 $\pi_{\theta}$ 作为世界模型进行推理，使用轻量级奖励函数（例如，基于嵌入的余弦相似度）来比较预测的 $\hat{s}_{t+1}$ 与真实的 $s_{t+1}$。

<a id="section-2"></a>

## 2 方法（Method）

- [2.1 符号表示（Notation）](#21-notation)
- [2.2 强化世界模型学习（Reinforcement World Model Learning）](#22-reinforcement-world-model-learning)

<a id="section-2-1"></a>

### 2.1 符号表示（Notation）

在复杂、长视野的环境中完成任务通常被表述为一个 **马尔可夫决策过程（Markov Decision Process, MDP）** $\left\langle\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma\right\rangle$。在多步任务的通用设定中，一个由 **大型语言模型（Large Language Model, LLM）** 驱动的智能体 $\pi_{\theta}$ 接收一个任务指令和一个来自环境 $s_{t}\sim\mathcal{S}$ 的 **观察（observation）** （^1^11从技术上讲，环境中输入到智能体的任何信息都是一个观察（如在部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）中）。然而，为简化符号，我们使用 $s$ 来泛指从环境接收到的输入数据。），生成一个 **动作（action）** $a_{t}\sim\pi(\cdot|s_{t})$，并接收一个新的观察 $s_{t+1}\sim\mathcal{S}$。在动作生成过程中，模型通常被给予最多 $H$ 步的交互历史 $\left\langle s_{t-H},a_{t-H},...,s_{t}\right\rangle$，并被允许在生成下一个动作 $a_{t}$ 之前进行思考/推理。这个交互过程会重复进行，直到任务完成或达到最大步数，届时会根据任务是否成功完成/失败返回一个 **终止奖励（terminal reward）** $r_{T}\sim\mathcal{R}(a_{T},s_{T})$。 **折扣因子（discounting factor）** $\gamma\in(0,1]$ 用于在 **强化学习（Reinforcement Learning, RL）** 训练期间对未来奖励进行折现和传播。请注意，由于本工作将 LLMs 训练为世界模型，_我们将生成的状态表示为 $\hat{s}_{t}$_，以区别于真实环境状态 $s_{t}$。

例如，在诸如 ALFWorld（Shridhar 等人，2021）这样的环境中，一个动作 $a_{t}$ 可能是“前往边桌 1”，而结果状态 $s_{t+1}$ 描述了该交互的结果，例如智能体当前可用的对象（例如，“你到达边桌 1。在边桌上你看到一个马克杯、一个胡椒瓶和一个番茄。”）。在更复杂的环境中，例如 $\tau^{2}$ Bench（Barres 等人，2025），一个动作 $a_{t}$ 可能是一个工具调用或对用户的响应，而下一个状态 $s_{t+1}$ 则返回一个工具响应（通常为 json 格式），或由用户模拟器（由 LLM 驱动）生成的自然语言响应。关于每个环境的更多细节，请分别参见[附录 B（Appendices˜B）](#appendix-b)和[C（C）](#appendix-c)。

<a id="section-2-2"></a>

### 2.2 强化世界模型学习（Reinforcement World Model Learning）

扩展诸如 RL 等 **智能体后训练方法（agentic post-training methods）** 的一个关键挑战在于它们依赖于在 **回合（episode）** 结束时提供的准确任务成功奖励。虽然有效，但这些奖励是 **稀疏的（sparse）** ，并且需要领域专家精心设计（Barres 等人，2025；Xie 等人，2024；Rawles 等人，2025；Zhou 等人，2024）。随着任务和环境变得更加复杂，这种依赖性带来了扩展性挑战。

我们引入了 **强化世界模型学习（Reinforcement World Model Learning, RWML）** ，这是一种可扩展的、 **自监督（self-supervised）** 的训练方法，其中智能体在进一步使用任务成功奖励 RL 进行微调之前，先从环境动态 $\mathcal{T}$ 中学习准确的世界模型知识。直观上，RWML 训练一个 LLM 策略 $\pi_{\theta}$，使其*也能够推理*给定动作 $a_{t}$ 和过去交互历史 $H$ 的 **后果（consequences）** $\hat{s}_{t+1}$：

$$
(\mathrm{reason},\hat{s}_{t+1})\sim\pi_{\theta}(\cdot|s_{\leq t},a_{t});s_{\leq t}\equiv\left\langle s_{t-H},a_{t-H},...,s_{t}\right\rangle
$$

其中“$\mathrm{reason}$”表示模型在生成下一个状态 $\hat{s}_{t+1}$ 的最终预测之前生成的 **推理词元（reasoning tokens）** 。为了评估预测的质量，我们使用一个简单的 **二元（binary）** （^2^22根据经验，我们发现二元化奖励更鲁棒，更不易被攻击（参见[第 3.4 节（Section 3.4）](#section-3-4)）。）奖励函数来比较 $\hat{s}_{t+1}$ 与 **真实值（ground truth）** $s_{t+1}$ 之间的距离：

$$
r^{\mathrm{WM}}(\hat{s}_{t+1},s_{t+1})=\begin{cases}1.0,&\text{如果}d(\hat{s}_{t+1},s_{t+1})<\tau_{d},\\ 0.0,&\text{否则}.\end{cases}
$$

其中 $\tau_{d}$ 是一个 **超参数（hyperparameter）** ，而 $d$ 主要通过使用一个现成的 **嵌入模型（embedding model）** $E(\cdot)$ 和 **余弦相似度（cosine similarity）** 来实现（Karpukhin 等人，2020；Zhang 等人，2025b）：

$$
d(\hat{s}_{t+1},s_{t+1})=1-\cos(E(\hat{s}_{t+1}),E(s_{t+1})).
$$

为了优化这个奖励，我们使用标准的 **组相对策略优化（Group Relative Policy Optimization, GRPO）** （Shao 等人，2024；DeepSeek-AI 等人，2025）：

$$
\displaystyle\mathbb{E}_{\pi_{\theta_{\mathrm{old}}}}\left[\min{\left(\rho_{\theta}A,\textrm{clip}(\rho_{\theta},1\pm\epsilon)A\right)}-\beta D_{\textrm{KL}}(\pi_{\theta}||\pi_{\theta_{\mathrm{ref}}})\right],
$$

其中 $\rho_{\theta}=\pi_{\theta}(y|x)/\pi_{\theta_{\mathrm{ref}}}(y|x)$ 是 **重要性采样比率（importance sampling ratio）** ，$\beta$ 是 **KL 正则化系数（KL regularization coefficient）** ，而 $A=[r^{\mathrm{WM}}-\textrm{mean}(r^{\mathrm{WM}})]/{\textrm{std}(r^{\mathrm{WM}})}$ 是使用我们奖励函数的 **组相对优势（group-relative advantage）** 。我们注意到，整个过程不需要任何专家数据、更强的 LLMs 或任务成功奖励信号。

为了收集 RWML 的训练数据，我们直接使用目标模型 $\pi_{\theta}$ 与环境收集 **轨迹（rollouts）** $(s_{0},a_{0},s_{1},a_{1},...s_{T})$，然后将轨迹转换为所有 $t$ 对应的三元组 $\left\langle s_{\leq t},a_{t},s_{t+1}\right\rangle$。为了提高覆盖范围和多样性，我们对每个训练任务执行 $N>1$ 次轨迹。为了帮助模型在 RL 期间专注于学习非平凡的世界模型知识，我们遵循（Snell 等人，2024；Sun 等人，2025）的直觉，并对数据集中“过于简单”而难以学习的部分进行 **子采样（subsample）** 。具体来说，我们首先使用 **监督微调（Supervised Fine-Tuning, SFT）** 训练一个单独的 LLM $\pi^{\prime}_{\theta}$，使其能够使用完整数据集的 10% 来预测 $\hat{s}_{t+1}$。然后，我们使用这个 $\pi^{\prime}_{\theta}$ 在数据集的另外 90%（即训练集）上生成 $\hat{s}_{t+1}$，对通过 $K=10$ 次尝试始终获得高奖励的训练样本进行子采样：

$$
\frac{1}{K}\sum\limits_{K}r^{\mathrm{WM}}(\hat{s}_{t+1},s_{t+1})\geq\tau_{\mathrm{easy}},(1)
$$

其中 $\tau_{\mathrm{easy}}$ 是一个超参数。对于高于阈值的“简单”样本，我们仅以概率 $p=0.1$ 将它们包含在最终训练集中，优先考虑更难的样本，同时保持多样性。生成的数据集用于 RWML 中的 GRPO 训练，如前一节所述。整个过程的概述如[图 2（Figure˜2）](#figure-2)所示。

<a id="section-3"></a>

## 3 实验（Experiments）

我们在两个广泛使用的长视野环境中评估 **RWML（Reasoning World Model Learning，推理世界模型学习）** ，这些环境需要准确的世界和工具理解才能进行有效的规划和任务完成。

- [3.1 实验设置（Experiment Setup）](#31-实验设置)
- [3.2 主要结果（Main Results）](#32-主要结果)
- [3.3 RWML 遗忘更少（RWML Forgets Less）](#33-rwml-遗忘更少)
- [3.4 消融研究（Ablation Studies）](#34-消融研究)

<a id="section-3-1"></a>

### 3.1 实验设置（Experiment Setup）

- [基准测试（Benchmarks）](#基准测试)
- [基线方法（Baselines）](#基线方法)
- [模型与训练数据（Models and Training Data）](#模型与训练数据)

#### 基准测试（Benchmarks）

我们在两个流行的智能体基准测试上进行实验： **ALFWorld** （Shridhar 等人，2021）和 **$\tau^{2}$ Bench** （Barres 等人，2025）。ALFWorld 是一个基于文本的具身环境，智能体需要根据自然语言指令定位物体并与之交互以完成家庭任务。$\tau^{2}$ Bench 是一个交错式工具使用环境，模型扮演客服代理的角色，在与提出问题的模拟用户对话的同时，使用工具调用来解决问题。我们使用每个基准测试官方提供的训练和测试划分进行训练和评估。

#### 基线方法（Baselines）

我们与其他来自三个类别的策略和世界模型相关训练方法进行比较：(1) 从任务成功奖励中学习；(2) 从交互/转移函数 $\mathcal{T}$ 中学习（类似于我们的方法）；(3) 从专家标注/更强的 LLMs 中学习。

- 1.  **从任务成功奖励中学习** ：我们考虑使用拒绝采样的 **强化微调（Reinforced Finetuning, RFT）** 以及使用任务成功奖励的标准 **强化学习（Reinforcement Learning, RL）** （策略 RL）。RFT 首先使用目标模型为每个训练任务生成 $N$ 条轨迹，然后仅对正确解决了任务的轨迹进行 **监督微调（Supervised Fine-Tuning, SFT）** 训练（Touvron 等人，2023；Zelikman 等人，2022）。策略 RL 直接使用 **GRPO** 训练基础模型 $\pi_{\theta}$，通过在线生成轨迹来优化任务成功奖励（Feng 等人，2025c；Yu 等人，2025a）。
- 2.  **从交互/转移函数中学习** ：我们考虑 **世界模型监督微调（World Model SFT, WM SFT）** ，它使用与 RWML 相同的训练数据，但训练模型直接使用 SFT 预测 $s_{t+1}$。请注意，WM SFT 不涉及推理，因为只有 $s_{t+1}$ 可用。
- 3.  **从专家/强 LLMs 中学习** ：我们考虑 Zhang 等人（2025a）和 Yu 等人（2025c）提出的 **隐式世界建模（Implicit World Modeling, IWM）** 和 **自我反思（Self-Reflection, SR）** 。使用专家轨迹 $(s_{0},a^{*}_{0},s_{1},a^{*}_{1},...)$，这些方法首先用目标模型 $\pi_{\theta}$ 生成的替代性非最优动作-状态对 $(s_{t},a^{\prime}_{t})$ 来增强它们。然后，要么将这些数据转换为用于世界模型学习的下一状态预测三元组 $\left\langle s_{\leq t},a_{t},s_{t+1}\right\rangle$，要么使用一个强 LLM 来合成推理数据（对比专家动作与替代的非最优动作）用于反思学习。最后，这些数据与专家策略数据（即预测 $a^{*}_{t+1}$）结合，并使用 SFT 在组合数据集上进行训练。由于这两种方法严重依赖专家轨迹，我们还考虑了一个更简单的基线，即直接使用 SFT 学习专家策略（记为“模仿学习”）。更多实现细节，请参阅 [Sections˜B.3](https://arxiv.org/html/2602.05842v2#A2.SS3) 和 [C.4](https://arxiv.org/html/2602.05842v2#A3.SS4)。

除了这些基于训练的方法，我们还评估了闭源 LLMs（如 GPT-5（Singh 等人，2025））上的 **ReACT 风格提示** （Yao 等人，2023）作为额外参考。关于这些方法与我们的方法之间更高层次的比较，请参见 [Table˜A1](#table-1)。

<a id="table-1"></a>

> 表 1：在 ALFWorld 和 $\tau^{2}$ Bench 上的性能。所有结果是 3 次运行的平均值，最大步数为 30。我们的方法以灰色高亮显示。\*对于 ALFWorld，我们使用 Qwen3-235B-A22B-instruct；对于 $\tau^{2}$ Bench，我们使用 Qwen3-235B-A22B-thinking。

| 方法                                          | ALFWorld     | $\tau^{2}$ Bench |              |              |              |              |              |
| ------------------------------------------- | ------------ | ---------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| ID                                          | OOD          | AVG              | Retail       | Telecom      | Airline      | AVG          |              |
| ReACT(Qwen2.5-7B)                           | 16.2$\pm$1.0 | 6.8$\pm$ 2.0     | 13.0$\pm$1.3 | 15.0$\pm$2.0 | 27.5$\pm$0.0 | 18.3$\pm$2.4 | 20.7$\pm$0.9 |
| ReACT(Qwen3-8B)                             | 40.9$\pm$1.5 | 31.3$\pm$2.6     | 37.7$\pm$1.8 | 37.7$\pm$4.9 | 31.2$\pm$4.2 | 21.6$\pm$5.2 | 31.9$\pm$2.9 |
| ReACT(Qwen3-235B\*)                         | 38.0$\pm$0.4 | 32.3$\pm$2.7     | 36.1$\pm$0.7 | 50.6$\pm$2.7 | 48.8$\pm$3.8 | 51.3$\pm$5.5 | 50.0$\pm$3.4 |
| ReACT(GPT-4.1)                              | 42.5$\pm$1.0 | 47.4$\pm$0.7     | 44.1$\pm$0.5 | 55.8$\pm$2.4 | 41.7$\pm$4.3 | 48.3$\pm$2.4 | 48.7$\pm$4.5 |
| ReACT(GPT-5)                                | 51.6$\pm$1.3 | 44.8$\pm$0.7     | 49.3$\pm$0.9 | 55.8$\pm$7.2 | 65.0$\pm$5.4 | 55.0$\pm$4.1 | 59.3$\pm$0.5 |
| _从任务成功奖励中学习_                                |              |                  |              |              |              |              |              |
| RFT                                         | 34.4$\pm$3.8 | 34.4$\pm$3.4     | 34.4$\pm$2.6 | 43.3$\pm$3.1 | 33.3$\pm$3.1 | 13.3$\pm$2.4 | 33.3$\pm$1.7 |
| Policy RL                                   | 82.1$\pm$3.6 | 79.2$\pm$2.0     | 81.0$\pm$1.6 | 40.8$\pm$1.2 | 39.2$\pm$1.2 | 30.0$\pm$8.2 | 38.0$\pm$1.6 |
| _自监督_                                       |              |                  |              |              |              |              |              |
| WM SFT                                      | 3.1$\pm$0.0  | 2.1$\pm$0.7      | 2.8$\pm$0.3  | 32.3$\pm$5.3 | 24.1$\pm$6.1 | 26.9$\pm$6.6 | 27.9$\pm$3.1 |
| \rowcolorlight-gray RWML (ours)             | 34.4$\pm$0.6 | 29.2$\pm$7.5     | 32.6$\pm$2.1 | 40.8$\pm$4.0 | 40.5$\pm$4.9 | 31.3$\pm$6.7 | 38.8$\pm$2.5 |
| _自监督 + 策略 RL_                               |              |                  |              |              |              |              |              |
| WM SFT + Policy RL                          | 76.2$\pm$3.4 | 82.3$\pm$0.7     | 80.4$\pm$1.5 | 40.8$\pm$4.2 | 45.0$\pm$5.4 | 30.0$\pm$7.1 | 40.3$\pm$3.9 |
| \rowcolorlight-gray RWML + Policy RL (ours) | 86.7$\pm$2.8 | 90.1$\pm$0.7     | 87.9$\pm$1.6 | 44.2$\pm$2.1 | 45.8$\pm$2.4 | 38.3$\pm$2.4 | 43.7$\pm$2.1 |

<a id="table-2"></a>

> 表 2：将我们的方法与使用专家数据/强 LLMs 的训练方法进行比较。模仿学习、IWM 和 SR 是根据 Zhang 等人（2025a）复现的，该文献报告了在 ALFWorld 上 ID 分别为 78.1、82.8、82.0，OOD 分别为 64.1、70.3、71.1。最高分以 **粗体** 显示，次高分以*下划线*显示。我们的模型在不使用专家/强 LLM 数据的情况下显示出有竞争力的性能。

| 方法                                        | ALFWorld       | $\tau^{2}$ Bench |                |                |                |                |                |
| ------------------------------------------- | -------------- | ---------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| ID                                          | OOD            | AVG              | Retail         | Telecom        | Airline        | AVG            |                |
| _从专家/强 LLMs 中学习_                     |                |                  |                |                |                |                |                |
| Imitation Learning                          | 84.9$\pm$1.9   | 77.6$\pm$3.2     | 82.5$\pm$2.3   | 48.3$\pm$1.2   | 41.7$\pm$3.1   | 38.3$\pm$2.4   | _43.7_$\pm$1.3 |
| IWM                                         | _85.6_$\pm$1.6 | 78.1$\pm$1.6     | 83.1$\pm$1.0   | 40.8$\pm$4.3   | 44.2$\pm$5.1   | 46.7$\pm$2.4   | 43.3$\pm$2.6   |
| SR                                          | 83.9$\pm$1.0   | _82.3_$\pm$0.7   | _83.3_$\pm$0.4 | _45.0_$\pm$3.5 | 45.8$\pm$8.3   | _43.3_$\pm$2.4 | 45.0$\pm$3.6   |
| _自监督 + 策略 RL_                          |                |                  |                |                |                |                |                |
| WM SFT + Policy RL                          | 76.2$\pm$3.4   | _82.3_$\pm$0.7   | 80.4$\pm$1.5   | 40.8$\pm$4.2   | _45.0_$\pm$5.4 | 30.0$\pm$7.1   | 40.3$\pm$3.9   |
| \rowcolorlight-gray RWML + Policy RL (ours) | 86.7$\pm$2.8   | 90.1$\pm$0.7     | 87.9$\pm$1.6   | 44.2$\pm$2.1   | 45.8$\pm$2.4   | 38.3$\pm$2.4   | _43.7_$\pm$2.1 |

#### 模型与训练数据（Models and Training Data）

遵循先前工作（Feng 等人，2025c；Yu 等人，2025a；Zhang 等人，2025a），对于所有方法，我们均在 ALFWorld 上基于 Qwen2.5-7B-Instruct（Qwen 等人，2025）进行训练。在 $\tau^{2}$ Bench 上，由于该基准的难度以及 Qwen3 模型增强的工具使用能力，我们对于所有方法均基于 Qwen3-8B（Yang 等人，2025）进行训练。

对于 **世界模型强化学习（Reward-weighted World Model Learning, RWML）** ，我们使用策略 $\pi_{\theta}$ 收集交互数据，在每个训练任务上以温度 $\tau=1.0$ 进行 $N$ 次轨迹推演，其中 ALFWorld 的 $N=3$，$\tau^{2}$ Bench 的 $N=6$。然后，我们将所有轮次拆分为三元组 $\left\langle s_{\leq t},a_{t},s_{t+1}\right\rangle$（对所有 $t$），使用其中 90% 的三元组进行训练，10% 用于验证。最后，我们使用一个对应于两个基准约 30% 训练数据的 $\tau_{\mathrm{easy}}$ 来对“简单”训练样本进行子采样。与我们的基线方法相比，我们注意到整个过程 **不需要任何专家标注或更强的 LLMs，也不需要任务成功/失败信号** 。仅需要 $\left\langle s_{\leq t},a_{t},s_{t+1}\right\rangle$ 形式的三元组。

最后，对于 **策略强化学习（Policy RL）** 训练，我们使用 GRPO 让模型学习利用任务成功奖励（$\gamma=1.0$）来解决问题。对于 ALFWorld，我们遵循先前工作（Yu 等人，2025a），允许每个任务最多 30 步。对于 $\tau^{2}$ Bench，出于成本考虑，我们使用 Qwen3-235B-A22B-Instruct（Yang 等人，2025）作为用户模拟器进行训练和评估，并允许每个任务最多 30 步。关于使用官方设置（GPT-4.1 作为用户模拟器）的评估结果，请参阅 [第˜C.3 节](https://arxiv.org/html/2602.05842v2#A3.SS3)。所有训练均在 B200 GPU 上进行。更多训练和超参数细节，请分别参见 ALFWorld 和 $\tau^{2}$ Bench 的 [附录˜B](#appendix-b) 和 [附录˜C](#appendix-c)。

<a id="section-3-2"></a>

### 3.2 主要结果（Main Results）

在 [表˜1](#table-1) 中，我们展示了 RWML 作为一种 **自监督（self-supervised）** 方法的有效性，该方法仅从交互数据中训练。在不使用任何专家数据、强 LLMs 或任务成功奖励信号的情况下，RWML 相比基础模型显著提升了智能体能力，在 ALFWorld 和 $\tau^{2}$ Bench 上分别提升了 19.6 分和 7.9 分。当与任务成功奖励（即策略 RL）结合时，我们发现我们的模型在所有基于训练的基线方法中表现最佳。值得注意的是，在 [表˜2](#table-2) 中我们发现：(1) 在 ALFWorld 上，我们的模型甚至优于那些使用专家标注/强 LLMs 的方法；(2) 在 $\tau^{2}$ Bench 上，尽管未使用任何专家数据/强 LLMs，我们的模型获得了第二高的总分。这证明了 RWML 的有效性，其可扩展、自监督的设计为“中期训练（mid-training）”算法指明了一个有前景的方向，这类算法可以补充诸如策略 RL 等后训练方法，以进一步提升基于 LLM 的智能体性能。

<a id="section-3-3"></a>

### 3.3 RWML 遗忘更少（RWML Forgets Less）

在 [表˜3](#table-3) 中，我们在世界模型学习的背景下，评估了强化学习（RL）和监督微调（SFT）对 **灾难性遗忘（catastrophic forgetting）** （Kirkpatrick 等人，2017；Luo 等人，2025c）的相对敏感性。我们在以下方面评估了在 ALFWorld 和 $\tau^{2}$ Bench 上训练的模型：(1) 通用知识基准，如 MMLU-Redux（Gema 等人，2025）和 IFEval（Zhou 等人，2023）；(2) 数学与 STEM 问题，如 MATH-500（Lightman 等人，2023）、GSM8k（Cobbe 等人，2021）和 GPQA-Diamond（Rein 等人，2023）；(3) 编码任务，如 LiveCodeBench（Jain 等人，2024）。在 [表˜3](#table-3) 中，我们发现与 WM SFT 相比，RWML 在几乎所有基准上都导致了更少的模型遗忘。我们认为这与先前工作的发现（Shenfeld 等人，2025；Chen 等人，2025a）一致，即由于其 **同策略（on-policy）** 性质，在线 RL 在保留先验知识和能力方面显著优于 SFT。关于模型参数更新的更多分析，请参见 [第˜4.2 节](#section-4-2)。

<a id="section-3-4"></a>

### 3.4 消融研究（Ablation Studies）

在 [表˜4](#table-4) 中，我们展示了一项消融研究，以探究 RWML 中不同组件的贡献。具体来说，我们考虑了：(1) 用 **LLM 作为评判器（LLM-as-a-judge）** （Zheng 等人，2023）替换我们基于嵌入的奖励；(2) 移除对“过于简单”样本进行子采样的步骤，记为“w/o subsample”；(3) 完全移除 RWML 训练，记为“w/o training”。对于 LLM 作为评判器，我们考虑了两种变体：提示 LLM 比较生成的 $\hat{s}_{t+1}$ 与真实值 $s_{t+1}$，并返回一个 **实值奖励** $r\in[0,1]$，允许部分得分。我们将其记为“w/ LLM-as-a-judge”。或者，我们提示 LLM 返回一个 **二元奖励** ，即 0.0 或 1.0。我们将其记为“w/ bin(LLM-as-a-judge)”。在这两种情况下，我们都使用 Qwen-3-235B-A22B-Instruct（Yang 等人，2025）作为评判模型，因为它是一个快速、强大且可本地部署的开源 LLM。

[表˜4](#table-4) 中的结果表明，我们方法的所有组件对于提升模型性能都很重要。此外，我们发现：(1) 较弱的模型（如 Qwen2.5-7B）更容易受到数据质量/噪声奖励函数的影响；(2) LLM 作为评判器不可靠，有时在训练中可能被“攻击”（参见 [附录˜D](#appendix-d) 中的示例）；(3) 对“简单”训练样本进行子采样有利于进一步提升模型性能。

<a id="table-3"></a>

> 表 3：在 ALFWorld 和 $\tau^{2}$ Bench 上训练后测量遗忘情况。对于 LiveCodeBench，我们使用 2025-01-01 至 2025-04-30 期间的问题。评估使用温度 1.0 和最大响应长度 16k，通过 EvalScope（ModelScope，2024）进行。最大的性能下降（$\Delta$）用深红色高亮显示。建议彩色查看。

|             |               | ALFWorld   | $\tau^{2}$ Bench                          |                                           |          |                                          |                                           |
| ----------- | ------------- | ---------- | ----------------------------------------- | ----------------------------------------- | -------- | ---------------------------------------- | ----------------------------------------- |
|             |               | Qwen2.5-7B | +WM SFT                                   | +RWML                                     | Qwen3-8B | +WM SFT                                  | +RWML                                     |
| 通用知识    | MMLU-Redux    | 77.26      | \cellcolorforgetDark67.16($\Delta$-10.10) | \cellcolorforgetLight74.88($\Delta$-2.38) | 87.75    | \cellcolorforgetDark87.02($\Delta$-0.73) | \cellcolorforgetLight87.42($\Delta$-0.33) |
|             | IFEval        | 71.34      | \cellcolorforgetDark68.39($\Delta$-2.95)  | \cellcolorforgetLight69.32($\Delta$-2.02) | 84.46    | \cellcolorforgetDark82.07($\Delta$-2.39) | \cellcolorforgetLight83.36($\Delta$-1.10) |
| 数学与 STEM | MATH-500      | 75.40      | \cellcolorforgetDark71.60($\Delta$-3.80)  | 75.40($\Delta$0.00)                       | 92.80    | 92.80($\Delta$0.00)                      | 92.80($\Delta$0.00)                       |
|             | GSM8k         | 91.66      | \cellcolorforgetDark90.45($\Delta$-1.21)  | \cellcolorforgetLight91.28($\Delta$-0.38) | 96.13    | \cellcolorforgetDark95.53($\Delta$-0.60) | \cellcolorforgetLight95.68($\Delta$-0.45) |
|             | GPQA-Diamond  | 32.83      | \cellcolorforgetDark25.25($\Delta$-7.58)  | \cellcolorforgetLight28.79($\Delta$-4.05) | 59.09    | \cellcolorforgetDark57.07($\Delta$-2.02) | \cellcolorforgetLight58.08($\Delta$-1.01) |
| 编码        | LiveCodeBench | 19.23      | \cellcolorforgetDark15.38($\Delta$-3.85)  | \cellcolorforgetLight16.48($\Delta$-2.75) | 43.41    | \cellcolorforgetDark41.21($\Delta$-2.20) | 43.41($\Delta$0.00)                       |

<a id="table-4"></a>

> 表 4：RWML 的消融研究。我们在 ALFWorld 上使用 Qwen2.5-7B-Instruct，在 $\tau^{2}$ Bench 上使用 Qwen3-8B。我们发现更强的基础模型（例如 $\tau^{2}$ Bench 上的 Qwen3-8B）对数据质量/奖励攻击的敏感性较低，并且对“过于简单”的训练样本进行子采样有利于进一步提升性能。

| 方法                     | ALFWorld     | $\tau^{2}$ Bench |              |              |              |              |              |
| ------------------------ | ------------ | ---------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| ID                       | OOD          | AVG              | Retail       | Telecom      | Airline      | AVG          |              |
| RWML（我们的方法）       | 34.4$\pm$0.6 | 29.2$\pm$7.5     | 32.6$\pm$2.1 | 40.8$\pm$4.0 | 40.5$\pm$4.9 | 31.3$\pm$6.7 | 38.8$\pm$2.5 |
| - w/ bin(LLM-as-a-judge) | 21.9$\pm$2.4 | 9.9$\pm$1.9      | 14.5$\pm$1.3 | 30.0$\pm$5.4 | 34.2$\pm$1.2 | 28.3$\pm$4.7 | 31.3$\pm$2.9 |
| - w/ LLM-as-a-judge      | 3.9$\pm$1.0  | 3.0$\pm$1.2      | 3.6$\pm$1.3  | 36.1$\pm$2.5 | 34.2$\pm$4.3 | 21.7$\pm$4.7 | 33.7$\pm$3.9 |
| - w/o subsample          | 3.1$\pm$1.3  | 2.6$\pm$1.5      | 2.9$\pm$1.0  | 39.2$\pm$6.2 | 40.0$\pm$2.0 | 28.3$\pm$6.2 | 36.3$\pm$2.5 |
| - w/o training           | 16.2$\pm$1.0 | 6.8$\pm$2.0      | 13.0$\pm$1.3 | 37.7$\pm$4.9 | 31.2$\pm$4.2 | 21.6$\pm$5.2 | 31.9$\pm$2.9 |

<a id="section-4"></a>

## 4 讨论（Discussion）

![alf_layerwise_singlecol_noWMSFT_RWML](images/alf_layerwise_singlecol_noWMSFT_RWML.png)

> (a) ALFWorld

- [4.1 RWML对决策的影响](#41-impact-of-rwml-on-decision-making)
- [4.2 权重变化分析](#42-weight-change-analysis)
- [4.3 基础模型能力的影响](#43-impact-of-base-model-capability)

<a id="section-4-1"></a>

### 4.1 RWML对决策的影响（Impact of RWML on Decision-Making）

在本节中，我们对模型在 **现实世界建模学习（Real-World Modeling Learning, RWML）** 训练前后的决策行为进行了一些定性和定量分析。定性分析方面，在[图˜5](#figure-5)中，我们发现经过RWML训练的模型能利用其改进的环境知识，做出更准确、更高效的决策。例如，在ALFWorld中，我们的模型正确预测“刀”最可能在“台面”上，而不是其他位置，并在5步内完成任务。在$\tau^{2}$ Bench中，它正确地考虑了“飞行模式”可能已开启的可能性——这是基础模型忽略的情况。

定量分析方面，我们发现RWML有效地减少了在两个基准测试中生成无效/低效动作的比例，尽管并未为此进行显式训练。在ALFWorld上，无效动作（例如，格式错误）或低效动作（例如，“查看”和“检查”动作）的比例从59.30%下降到39.45%。类似地，在$\tau^{2}$ Bench上，每次工具调用中无效工具调用（例如，编造的工具名称或不正确的参数）的比例从24.90%下降到8.84%。总体而言，我们的定性和定量结果表明， **RWML显著提升了LLM在智能体环境中的决策能力** 。

<a id="figure-4"></a>

![tau2bench_layerwise_singlecol_noWMSFT_RWML](images/tau2bench_layerwise_singlecol_noWMSFT_RWML.png)

> 图 4：在 $\tau^{2}$ Bench上使用不同基础模型进行RWML训练的结果。

<a id="section-4-2"></a>

### 4.2 权重变化分析（Weight Change Analysis）

为了理解RWML的有效性，我们还分析了它在训练过程中如何重塑模型参数。遵循Zhu等人（2025）的方法，我们检查了相对于未经训练的基础模型的逐参数权重变化，采用相同的定义和阈值$\eta=10^{-3}$来识别主要的逐点更新：

$$
|\hat{w_{i}}-w_{i}|>\eta\cdot\max(|w_{i}|,|\hat{w_{i}}|),
$$

其中$w_{i},\hat{w_{i}}\in\mathbb{R}$是微调前后模型权重点的有限非零标量。

对于每一层，我们计算了经历主要更新的参数比例。Qwen3-8B在$\tau^{2}$-Bench以及Qwen2.5-7B-Instruct在ALFWorld上的结果如[图˜3](#figure-3)所示。完整结果见[附录˜E](#appendix-e)。一个一致的模式是，与 **权重匹配监督微调（Weight-Matched Supervised Fine-Tuning, WM SFT）** 相比，RWML在各层引起的参数变化显著更少，这表明它通过更小、更具针对性的更新集合来编码任务相关信息（另见[附录˜E](#appendix-e)）。这表明RWML以一种 **参数效率更高、结构上更保守** 的方式学习，避免了对预训练表示空间进行广泛的修改。

重要的是，这种紧凑的更新行为也有助于解释为什么RWML能很好地与后续的策略学习相结合，如[图˜3](#figure-3)所示。当后续进行 **策略强化学习（Policy Reinforcement Learning, Policy RL）** 时，产生的权重变化率与直接对基础模型应用Policy RL的结果保持非常接近。相比之下，用WM SFT初始化的模型在策略优化后表现出明显更高的变化率，反映了更强的参数干扰。这些观察结果表明，RWML保持了与策略学习更兼容的参数格局，减少了训练后的冲突和冗余。

总体而言，我们发现RWML的这种参数更新行为在两个基准测试中是一致的，并且对于不同的Transformer组件（包括注意力（Q/K/V/O）和MLP投影层）基本保持不变（见[附录˜E](#appendix-e)）。与WM SFT训练的模型相比，RWML训练模型的持续较低变化率也与我们在[第˜3.3节](#section-3-3)中的发现一致，该节表明RWML能更好地缓解 **灾难性遗忘（catastrophic forgetting）** 。这些结果提供了一个不同于传统“SFT-then-RL”范式的视角：在“中期训练”和训练后阶段都应用RL似乎能产生更稳定、更一致的参数更新，这可能有助于解释性能的提升。

<a id="section-4-3"></a>

### 4.3 基础模型能力的影响（Impact of Base Model Capability）

在具有挑战性的$\tau^{2}$ Bench上，我们发现从RWML中学习世界模型知识并将其迁移到决策的能力，取决于基础模型的能力。在[图˜4](#figure-4)中，我们使用三个不同的基础模型进行了RWML训练：Qwen2.5-7B、Qwen3-8B和Qwen3-30B-A3B(^3^33我们使用Qwen3-30B-A3B-Thinking-2507，这是Qwen3-30B-A3B经过额外推理和智能体数据后训练的增强版本，留给进一步改进的空间较小。)。我们发现，像Qwen2.5-7B这样的较弱模型在具有挑战性的$\tau^{2}$ Bench上难以将世界知识迁移到决策中，而更强的模型（Qwen3-8B和Qwen3-30B-A3B）则显示出显著的性能提升，接近Qwen3-235B-A22B-Thinking-2507的水平。这表明 **RWML对于（足够）强大的基础模型最为有效** 。我们将提升较弱模型的迁移能力留作未来工作。

<a id="figure-5"></a>

![model_scaling](images/model_scaling.png)

> 图 5：经过RWML后，模型通过利用其改进的环境知识，做出更准确、更高效的决策。

<a id="section-5"></a>

## 5 相关工作（Related Work）

- [训练决策智能体](#training-decision-making-agents)
- [训练世界模型](#training-world-models)

#### 训练决策智能体（Training Decision-Making Agents）

基于LLM的智能体（Yao等人，2023；Shinn等人，2023）已在多个领域得到广泛应用，例如交互式游戏（Wang等人，2023；Feng等人，2025c）；软件工程（Jimenez等人，2024；Yang等人，2024）；计算机、手机、浏览器使用（Xie等人，2024；Rawles等人，2025；Zhou等人，2024；Yu等人，2025b）等。许多关于训练语言智能体的早期工作主要依赖于 **模仿学习（imitation learning）** （即SFT），使用人类专家的演示（Deng等人，2023；Chen等人，2025b；Wang等人，2025a）或由更强LLM合成的轨迹，通常辅以一套手动设计的工作流程/启发式方法（Zeng等人，2023；Chen等人，2024；Su等人，2025；Xu等人，2025）。虽然高质量的SFT数据提供了密集的监督信号，但由于收集此类演示的成本高昂，难以扩展。或者，最近在RL方面的努力绕过了对逐步演示的需求，转而通过试错直接从终端奖励（即任务成功）中学习。近期工作包括Feng等人（2025a）；Tan等人（2025）；Luo等人（2025a）；Jin等人（2025）；Wang等人（2025b），通常由 **近端策略优化（Proximal Policy Optimization, PPO）** （Schulman等人，2017）和 **分组相对策略优化（Group Relative Policy Optimization, GRPO）** （Shao等人，2024）等算法驱动。然而，在复杂环境中设计任务成功奖励函数仍然需要大量的人类专业知识（Chowdhury等人，2024；Xie等人，2024；Gou等人，2025），限制了可扩展性。总之，这些工作激发了对更具可扩展性的训练方法的需求，以弥合下一词元预测预训练模型与其在长视野智能体环境中的下游应用之间的差距。

#### 训练世界模型（Training World Models）

除了任务成功奖励，现实世界的交互数据包含丰富的信息，可用于辅助决策。早期的例子包括 **Dyna算法（Dyna algorithms）** （Sutton，1991），它单独训练一个世界模型，将基于模型的学习与无模型学习相结合，以实现高效的策略训练。最近在LLM智能体上的应用要么训练一个 **独立的（separate）** 世界模型来支持推理时算法，如 **蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）** （Hao等人，2023；Wu等人，2025；Chae等人，2025；Gu等人，2025），要么在单个模型内联合学习世界模型和策略以提高泛化能力（FAIR CodeGen团队等人，2025；Zhang等人，2025a；Yu等人，2025a, c；Feng等人，2025b；Li等人，2025；Qian等人，2026）。然而，这些方法要么需要多个模型的昂贵训练/推理，要么在世界模型学习过程中依赖专家/更强LLM的额外标注。我们提出RWML作为一种可扩展的、 **自监督的（self-supervised）** 方法，用于提升单个模型的世界知识和决策能力。

<a id="section-6"></a>

## 6 结论（Conclusion）

我们提出了RWML，一种可扩展、自监督的方法，它在使用任务成功奖励进行下游RL之前，增强了基于LLM的智能体对环境理解和决策的能力。无需专家/更强LLM的标注或任务成功信号，RWML通过将模拟的下一个状态与预训练嵌入空间中观察到的环境状态对齐，将LLM训练为一个 **动作条件化的世界模型（action-conditioned world model）** 。我们在两个长视野智能体基准测试ALFWorld和$\tau^{2}$ Bench上评估了RWML，发现仅使用交互数据即可获得显著的性能提升。当与策略RL中的任务成功奖励结合时，我们的方法在两个基准测试上都优于直接的策略RL，并与使用专家数据进行训练的效果相当。我们相信，我们的工作为可扩展、自监督的训练方法开辟了新途径，以在智能体RL时代进一步推进基于LLM的智能体。

<a id="section-7"></a>

## 7 影响声明（Impact Statements）

本文提出了一项工作，旨在通过一种可扩展、自监督的方法来提升基于LLM的智能体的智能体能力。虽然大多数基于LLM的智能体方法并非为不道德的使用而设计，但其应用和数据收集过程仍可能带来误用的风险。在这项工作中，我们提出了RWML，它使用交互数据（无需专家标注或更强LLM）来改进基于LLM的智能体中的世界建模，并且仅在已建立的、隔离的基准测试上进行训练，对现实世界没有影响。我们认为，开发防护措施，如 **安全过滤器（safety filters）** （OpenAI，2022；Inan等人，2023；Luo等人，2025b），以及使用像 **沙箱（sandboxes）** （AgentInfra Team，2025；Pan等人，2025）这样的隔离环境，对于安全的AI智能体研究至关重要。我们不认可将RWML或其构成方法用于任何非法或不道德的目的。
