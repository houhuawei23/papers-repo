# Title: Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning （让离线强化学习在线化：用于离线视觉强化学习的协作世界模型）

- ArXiv: 2305.15260
- 作者：Qi Wang, Junming Yang, Yunbo Wang, Xin Jin, Wenjun Zeng, Xiaokang Yang, Ningbo Institute of Digital Twin, Eastern Institute of Technology, China, School of Computer Science and Engineering, Southeast University, China
- 章节数：49
- 估计词元数：22.7k

## 目录

- 1 引言
- 2 问题设定
- 3 方法
  - 3.1 离线到在线状态对齐
    - 源模型预训练。
    - 状态对齐。
  - 3.2 在线到离线奖励对齐
  - 3.3 最小-最大值约束
- 4 实验
  - 4.1 实验设置
    - 数据集。
    - 对比方法。
  - 4.2 Meta-World 上的跨任务实验
    - 主要结果。
    - 使用随机源域的结果。
    - 使用多源域的结果。
  - 4.3 跨环境：从 Meta-World 到 RoboDesk
  - 4.4 DMC 上的跨动力学实验
  - 4.5 进一步分析
    - 消融研究。
    - CoWorld 能否解决价值高估问题？
    - CoWorld 对领域相似性的依赖。
    - 与跨领域联合训练单一世界模型的比较。
    - 超参数敏感性。
- 5 相关工作
- 6 结论与局限性
- 致谢
- 参考文献
- 附录
- 附录 A 模型细节
  - A.1 CoWorld 框架
  - A.2 世界模型
  - A.3 行为学习
  - A.4 超参数
- 附录 B 额外的定量与定性结果
  - B.1 策略评估的可视化
  - B.2 DMC Medium-Expert 数据集上的定量结果
  - B.3 Meta-World 上的定量结果
  - B.4 潜在空间对齐的效果
  - B.5 现实 Sim2Real 设置上的额外结果
  - B.6 与预训练基础模型 R3M 的比较
  - B.7 训练效率
- 附录 C 多源 CoWorld
- 附录 D 源域与目标域
  - Meta-World。
  - RoboDesk。
  - DeepMind Control。
- 附录 E 对比方法
- 附录 F 更广泛的影响

## 摘要

###### 摘要

使用视觉输入训练 **离线强化学习（Offline Reinforcement Learning, Offline RL）** 模型面临两大挑战： **表示学习（Representation Learning）** 中的 **过拟合（Overfitting）** 问题，以及对期望未来奖励的 **高估偏差（Overestimation Bias）** 。近期工作试图通过鼓励 **保守行为（Conservative Behaviors）** 来缓解高估偏差。与之相反，本文尝试为价值估计构建更灵活的约束，同时不阻碍对潜在优势的探索。其核心思想是利用现成的、可以轻松以在线方式交互的 RL 模拟器，作为离线策略的“测试平台”。为了实现有效的 **在线到离线知识迁移（Online-to-Offline Knowledge Transfer）** ，我们提出了 **CoWorld** ，一种 **基于模型的强化学习（Model-based Reinforcement Learning, Model-based RL）** 方法，旨在减轻状态空间和奖励空间中的 **跨领域差异（Cross-domain Discrepancies）** 。实验结果表明了 CoWorld 的有效性，其性能大幅超越了现有的 RL 方法。

<a id="section-1"></a>

## 1 引言（Introduction）

由于与物理世界交互的成本高昂，通过视觉观测学习控制策略可能具有挑战性。 **离线强化学习（Offline Reinforcement Learning, RL）** 是应对这一挑战的一种有前景的方法 [11, 21, 37, 3, 65]。然而，在当前视觉控制任务中直接使用现有的离线 RL 算法存在两个主要困难。首先，离线视觉 RL 在表示学习过程中更容易出现过拟合问题，因为它需要从有限的高维视觉输入中提取隐藏状态。此外，与其状态空间对应物类似，离线视觉 RL 也容易受到 **价值高估（value overestimation）** 挑战的影响，正如我们从现有方法中所观察到的那样 [22, 16]。

<a id="figure-1"></a>

![intro1](images/intro1.png)

> 图 1 | 我们用于离线视觉 RL 的方法。

改进离线视觉 RL 仍然是一个探索不足的研究领域。我们的目标是在价值函数的高估与过度保守之间取得平衡，以避免对离线数据分布之外的估计值施加过度的惩罚。直观地说，_我们不应过度限制具有潜在优势的探索。_ 我们的基本思想如图 [1](#figure-1) 所示，即利用现成的、用于相关（不一定完全相同）视觉控制任务的在线模拟器作为辅助源域，从而将离线视觉 RL 构建为一个 **离线-在线-离线（offline-online-offline）** 迁移学习问题，以学习适度保守的策略。

我们提出了一种新颖的基于模型的迁移 RL 方法，称为 **协同世界模型（Collaborative World Models, CoWorld）** 。具体来说，我们为源域和目标域分别训练独立的世界模型和 RL 智能体，每个模型都有其特定于域的参数。为了减轻世界模型之间的差异，我们引入了一种新颖的表示学习方案，包含两个迭代训练阶段。如图 [1](#figure-1) 所示，这两个阶段分别促进了潜在状态分布（从离线到在线）和奖励函数（从在线到离线）的对齐。通过这样做，源域评论家可以作为评估目标离线策略的在线“测试平台”。它也更“知识渊博”，因为它可以主动与在线环境交互并收集丰富的信息。领域协同世界模型的另一个好处是能够缓解离线表示学习的过拟合问题，从而从有限的离线视觉数据中推导出更具泛化性的潜在状态。

对于离线数据集中的行为学习，我们利用源模型的知识，并在目标域评论家模型的训练目标中引入一个温和的正则化项。该正则化项鼓励源评论家重新评估目标策略。如图 [2](#figure-2) 所示，它允许对从“知识渊博”的源评论家那里获得低价值的轨迹的高估价值进行灵活的约束。相反，如果一个策略从源评论家那里获得了高价值，我们倾向于保留离线智能体的原始估计。这种方法是可行的，因为源评论家在世界模型学习阶段已经与目标域对齐。

我们在 Meta-World、RoboDesk 和 DeepMind Control 基准测试的离线视觉控制任务中展示了 CoWorld 的有效性。我们的方法被证明可以轻松扩展到具有多个源域的场景。即使存在多样化的物理动力学、动作空间、奖励尺度和视觉外观，它也能通过从辅助域迁移知识来有效解决价值高估问题。

总之，我们的工作带来了以下贡献：

- 我们创新性地将离线视觉 RL 构建为一个领域迁移问题。其基本思想是利用跨领域知识来解决离线视觉控制任务中的表示过拟合和价值高估问题。
- 我们提出了 CoWorld，这是一种遵循离线-在线-离线范式的方法，融合了世界模型对齐和灵活价值约束的特定技术。

<a id="figure-2"></a>

![intro2](images/intro2.png)

> 图 2 | 为了解决离线 RL 中的价值高估问题（a），我们可以直接惩罚离线数据分布之外的估计值，但这可能会阻碍智能体探索具有高奖励的潜在状态（b）。与现有方法不同，CoWorld 在在线辅助域中训练一个跨域评论家模型来重新评估离线策略（c），并用灵活的约束对目标价值进行正则化（d）。该方法的可行性在于世界模型学习阶段的领域对齐技术。

<a id="section-2"></a>

## 2 问题设定（Problem Setup）

我们将 **离线视觉强化学习（Offline Visual Reinforcement Learning）** 视为一个 **部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）** ，其目标是在固定的目标数据集 $\mathcal{B}^{(T)}$ 中最大化累积奖励。
我们特别关注那些可以访问辅助环境的场景，这些环境能够实现丰富的交互和高效的在线数据收集。
目标是通过从源 POMDP $\left\langle\mathcal{O}^{(S)},\mathcal{A}^{(S)},\mathcal{T}^{(S)},\mathcal{R}^{(S)},\gamma^{(S)}\right\rangle$ 进行知识迁移，来提升目标 POMDP $\left\langle\mathcal{O}^{(T)},\mathcal{A}^{(T)},\mathcal{T}^{(T)},\mathcal{R}^{(T)},\gamma^{(T)}\right\rangle$ 的离线性能。
这些符号分别表示视觉观测空间、动作空间、状态转移概率、奖励函数和折扣因子。

<a id="table-1"></a>

> 表 1 | RoboDesk（目标域）与 Meta-World（辅助源域）对比。

|          | 源：Meta-World             | 目标：RoboDesk             | 相似性 / 差异  |
| -------- | -------------------------- | -------------------------- | -------------- |
| 任务     | Window Close               | Open Slide                 | 相关的操作任务 |
| 动力学   | Simulated Sawyer robot arm | Simulated Franka robot arm | 不同           |
| 动作空间 | Box(-1, 1, (4,), float64)  | Box(-1, 1, (5,), float32)  | 不同           |
| 奖励尺度 | [0, 1]                     | [0, 10]                    | 不同           |
| 观测     | Right-view images          | Top-view images            | 不同的视角     |

例如，在我们的一个实验中，我们使用 RoboDesk 作为离线目标域，并使用来自 Meta-World 的各种任务作为源域。
如表 [1](#table-1) 所示，由于观测图像来自不同的相机视角，这两个环境在物理动力学、动作空间、奖励定义和视觉外观方面都存在显著差异。
我们的首要任务是解决 **领域差异（Domain Discrepancies）** ，以实现 **跨域行为学习（Cross-domain Behavior Learning）** 。

<a id="section-3"></a>

## 3 方法（Method）

本节将介绍 CoWorld 的技术细节。CoWorld 由一对 **世界模型（World models）** $\{\mathcal{M}_{\phi^{\prime}},\mathcal{M}_{\phi}\}$、 **行动者网络（Actor networks）** $\{\pi_{\psi^{\prime}},\pi_{\psi}\}$ 和 **评论者网络（Critic networks）** $\{v_{\xi^{\prime}},v_{\xi}\}$ 组成，其中 $\{\phi,\psi,\xi\}$ 和 $\{\phi^{\prime},\psi^{\prime},\xi^{\prime}\}$ 分别是目标域和源域的参数。
由于在 $\{\mathcal{O},\mathcal{A},\mathcal{T},\mathcal{R}\}$ 的所有元素中都可能存在潜在的跨域差异，整个训练过程被组织成三个迭代阶段，遵循一个 **离线-在线-离线（offline-online-offline）** 的迁移学习框架：

- **A) 离线到在线状态对齐（Offline-to-online state alignment）** ：通过将其状态空间与源世界模型 $\mathcal{M}_{\phi^{\prime}}$ 的状态空间对齐，来训练离线世界模型 $\mathcal{M}_{\phi}$。
- **B) 在线到离线奖励对齐（Online-to-offline reward alignment）** ：通过整合目标奖励信息，在在线环境中训练 $\mathcal{M}_{\phi^{\prime}}$ 和 $\{\pi_{\psi^{\prime}},v_{\xi^{\prime}}\}$。
- **C) 在线到离线价值约束（Online-to-offline value constraint）** ：利用源评论者 $v_{\xi^{\prime}}$ 提供的价值约束，训练目标离线域智能体 $\{\pi_{\psi},v_{\xi}\}$。

<a id="section-3-1"></a>

### 3.1 离线到在线状态对齐（Offline-to-Online State Alignment）

#### 源模型预训练（Source model pretraining）。

我们从一个源域预热阶段开始，采用一种基于模型的 **行动者-评论者方法（Actor-critic method）** ，即 DreamerV2 [16]。
为了促进跨域知识迁移，我们额外引入了一个状态对齐模块，记作 $g(\cdot)$，并使用 softmax 操作实现。
世界模型 $\mathcal{M}_{\phi^{\prime}}$ 由以下组件构成：

$$
\begin{aligned}
\text{循环转移（Recurrent transition）} &: \displaystyle h_{t}^{(S)}=f_{\phi^{\prime}}(h_{t-1}^{(S)},z_{t-1}^{(S)},a_{t-1}^{(S)}) \\
\text{图像编码（Image encoding）} &: \displaystyle{e}_{t}^{(S)}=e_{\phi^{\prime}}(o_{t}^{(S)}) \\
\text{后验状态（Posterior state）} &: \displaystyle z_{t}^{(S)}\sim q_{\phi^{\prime}}(h_{t}^{(S)},{e}_{t}^{(S)}) \\
\text{先验状态（Prior state）} &: \displaystyle\hat{z}_{t}^{(S)}\sim p_{\phi^{\prime}}(h_{t}^{(S)}) \\
\text{重建（Reconstruction）} &: \displaystyle\hat{o}_{t}^{(S)}\sim p_{\phi^{\prime}}(h_{t}^{(S)},z_{t}^{(S)}) \\
\text{奖励预测（Reward prediction）} &: \displaystyle\hat{r}_{t}^{(S)}\sim r_{\phi^{\prime}}(h_{t}^{(S)},z_{t}^{(S)}) \\
\text{折扣因子（Discount factor）} &: \displaystyle\hat{\gamma}^{(S)}_{t}\sim p_{\phi^{\prime}}(h_{t}^{(S)},z_{t}^{(S)}) \\
\text{状态对齐目标（State alignment target）} &: \displaystyle s_{t}^{(S)}=g({e}_{t}^{(S)}),
\end{aligned}
$$

其中 $\phi^{\prime}$ 代表世界模型的组合参数。
我们在动态扩展的源域 **经验回放缓冲区（Experience replay buffer）** $\mathcal{B}^{(S)}$ 上训练 $\mathcal{M}_{\phi^{\prime}}$，通过最小化

$$
\displaystyle\mathcal{L}(\phi^{\prime})= \displaystyle\mathbb{E}_{q_{\phi^{\prime}}}\Big{[}\sum_{t=1}^{N}\underbrace{- \ln p_{\phi^{\prime}}(o_{t}^{(S)}\mid h_{t}^{(S)},z_{t}^{(S)})}_{\text{图像重建}}\underbrace{-\ln r_{\phi^{\prime}}(r_{t}^{(S)}\mid h_{t}^{(S)} ,z_{t}^{(S)})}_{\text{奖励预测}}\underbrace{-\ln p_{\phi^{\prime}}( \gamma_{t}^{(S)}\mid h_{t}^{(S)},z_{t}^{(S)})}_{\text{折扣预测}} (2) \displaystyle\underbrace{+\ \mathrm{KL}\left[q_{\phi^{\prime}}(z_{t}^{(S)}\mid h _{t}^{(S)},o_{t}^{(S)})\ \|\ p_{\phi^{\prime}}(\hat{z}_{t}^{(S)}\mid h_{t}^{(S )})\right]}_{\text{KL 散度}}\Big{]}.
$$

我们训练源智能体的 **行动者（actor）** $\pi_{\psi^{\prime}}(\hat{z}_{t})$ 和 **评论者（critic）** $v_{\xi^{\prime}}(\hat{z}_{t})$，其目标分别是最大化与估计由 $\mathcal{M}_{\phi^{\prime}}$ 生成的期望未来奖励 $\mathbb{E}_{p_{\phi^{\prime}},p_{\psi^{\prime}}}[\sum_{\tau\geq t}\hat{\gamma}_{\tau-t}\hat{r}_{\tau}]$。更多细节请参阅附录 [A.3](https://arxiv.org/html/2305.15260v4#A1.SS3)。我们部署 $\pi_{\psi^{\prime}}$ 与辅助环境交互，并收集新数据用于进一步的世界模型训练。

#### **状态对齐（State alignment）**

一种直接的 **迁移学习（transfer learning）** 解决方案是在源智能体的检查点（checkpoints）基础上，使用离线数据集训练目标智能体。然而，由于不同领域在任务、视觉观测、物理动力学和动作空间上的差异，这种方法可能会遭受潜在的 **不匹配（mismatch）** 问题。当在线数据收集自与离线数据集不同的环境时（例如，Meta-World $\rightarrow$ RoboDesk），这个问题会更加严重。

我们通过分离源智能体与目标智能体的参数，同时显式地对齐它们的 **潜在状态空间（latent state spaces）** 来解决这个问题。具体而言，目标世界模型 $\mathcal{M}_{\phi}$ 与源模型 $\mathcal{M}_{\phi^{\prime}}$ 具有相同的网络架构。我们将从 $\mathcal{B}^{(T)}$ 中采样的相同目标域观测输入到这两个模型中，并拉近 $e_{\phi^{\prime}}(o_{t}^{(T)})$ 和 $e_{\phi}(o_{t}^{(T)})$ 的距离。我们通过最小化以下损失来优化 $\mathcal{M}_{\phi}$：

$$
\displaystyle\mathcal{L}(\phi) \displaystyle=\mathbb{E}_{q_{\phi}}\Big{[}\sum_{t=1}^{N}\underbrace{-\ln p_{\phi}(o_{t}^{(T)}\mid h_{t}^{(T)},z_{t}^{(T)})}_{\text{图像重建}} \underbrace{-\ln r_{\phi}(r_{t}^{(T)}\mid h_{t}^{(T)},z_{t}^{(T)})}_{\text{奖励预测}}\underbrace{-\ln p_{\phi}(\gamma_{t}^{(T)}\mid h_{t}^{(T)}, z_{t}^{(T)})}_{\text{折扣预测}} (3) \displaystyle\underbrace{+\ \beta_{1}\mathrm{KL}\left[q_{\phi}(z_{t}^{(T)}\mid h _{t}^{(T)},o_{t}^{(T)})\|\ p_{\phi}(\hat{z}_{t}^{(T)}\mid h_{t}^{(T)})\right]} _{\text{KL 散度}}\underbrace{+\ \beta_{2}\mathrm{KL}\left[\texttt{sg}(g( e_{\phi^{\prime}}(o_{t}^{(T)})))\ \|\ g(e_{\phi}(o_{t}^{(T)}))\right]}_{\text{域对齐损失}}\Big{]},
$$

其中 $\texttt{sg}(\cdot)$ 表示 **梯度停止（gradient stopping）** ，我们使用源模型的编码作为状态对齐的目标。由于源世界模型可以主动与在线环境交互并收集丰富信息，这能防止目标世界模型对离线数据 **过拟合（overfitting）** 。该损失项的重要性由 $\beta_{2}$ 控制。我们将在实验中检验其敏感性。

<a id="algorithm-1"></a>

**算法 1 CoWorld 的训练方案（Algorithm 1 The training scheme of CoWorld）**

- 1: **要求（Require）** ：离线数据集 $\mathcal{B}^{(T)}$。
- 2: **初始化（Initialize）** ：源模型参数 $\{\phi^{\prime},\psi^{\prime},\xi^{\prime}\}$ 与目标模型参数 $\{\phi,\psi,\xi\}$。
- 3: **预训练（Pretrain）** 源智能体并收集一个 **回放缓冲区（replay buffer）** $\mathcal{B}^{(S)}$。
- 4: **当（while）** 未收敛 **执行（do）**
- 5: **对（for）** 在 $\{1:K_{1}\}$ 中的每一步 **执行（do）** $\triangleright$ 在离线域中
- 6: 从 $\mathcal{B}^{(T)}$ 中采样 $\{(o_{t}^{(T)},a_{t}^{(T)},r_{t}^{(T)})\}_{t=1}^{N}$。
- 7: 使用公式 (3) 训练目标世界模型 $\mathcal{M}_{\phi}$。 $\triangleright$ 离线到在线状态对齐
- 8: 使用 $\pi_{\psi}$ 和 $\mathcal{M}_{\phi}$ 生成 $\{(z_{i}^{(T)},a_{i}^{(T)})\}_{i=t}^{t+H}$。 $\triangleright$ 带约束的行为学习
- 9: 在 $\{(z_{i}^{(T)},a_{i}^{(T)})\}_{i=t}^{t+H}$ 上使用公式 (6) 训练评论者 $v_{\xi}$。
- 10: 在 $\{(z_{i}^{(T)},a_{i}^{(T)})\}_{i=t}^{t+H}$ 上使用公式 (7) 训练行动者 $\pi_{\psi}$。
- 11: **结束循环（end for）**
- 12: **对（for）** 在 $\{1:K_{2}\}$ 中的每一步 **执行（do）** $\triangleright$ 在线域中
- 13: 从 $\mathcal{B}^{(S)}$ 中采样 $\{(o_{t}^{(S)},a_{t}^{(S)},r_{t}^{(S)})\}_{t=1}^{N}$。
- 14: 从 $\mathcal{B}^{(T)}$ 中采样 $\{(o_{t}^{(T)},a_{t}^{(T)},r_{t}^{(T)})\}_{t=1}^{N}$。 $\triangleright$ 在线到离线奖励对齐
- 15: 使用公式 (LABEL:eq:corrected*reward) 重新标注源奖励 $\{\tilde{r}*{t}^{(S)}\}\_{t=1}^{N}$。
- 16: 使用公式 (2) 结合公式 (5) 训练 $\mathcal{M}_{\phi^{\prime}}$。
- 17: 使用 $\pi_{\psi^{\prime}}$ 和 $\mathcal{M}_{\phi^{\prime}}$ 生成 $\{(z_{i}^{(S)},a_{i}^{(S)})\}_{i=t}^{t+H}$。 $\triangleright$ 源域行为学习
- 18: 在想象的 $\{(z_{i}^{(S)},a_{i}^{(S)})\}_{i=t}^{t+H}$ 上训练 $\pi_{\psi^{\prime}}$ 和 $v_{\xi^{\prime}}$。
- 19: 使用 $\pi_{\psi^{\prime}}$ 收集新的源数据并附加到 $\mathcal{B}^{(S)}$。
- 20: **结束循环（end for）**
- 21: **结束循环（end while）**

<a id="section-3-2"></a>

### 3.2 在线到离线奖励对齐（Online-to-Offline Reward Alignment）

为使源智能体（source agent）能够评估目标策略（target policy），必须为其提供关于离线任务（offline task）的先验知识。为此，我们使用来自两个回放缓冲区（replay buffers）$\mathcal{B}^{(S)}$ 和 $\mathcal{B}^{(T)}$ 的混合数据来训练源奖励预测器（source reward predictor）$r_{\phi^{\prime}}(\cdot)$。通过在源域想象（source domain imaginations）上进行行为学习（behavior learning），这个 **目标信息化的奖励预测器（target-informed reward predictor）** 使得源强化学习（Reinforcement Learning, RL）智能体能够评估目标模型（target model）产生的想象状态，并为目标价值估计（target value estimation）提供一个灵活的约束（我们将在第 [3.3](#section-3-3) 节讨论）。

具体而言，我们首先从 $\mathcal{B}^{(T)}$ 中采样一条目标域数据轨迹 $\{(o_{t}^{(T)},a_{t}^{(T)},r_{t}^{(T)})\}_{t=1}^{T}$（算法 [1](#algorithm-1) 第 14 行）。然后，我们使用由 $\phi^{\prime}$ 参数化的源世界模型（source world model）来提取对应的潜在状态，并重新标注目标信息化的源奖励（算法 [1](#algorithm-1) 第 15 行）：

$$
\displaystyle\tilde{h}_{t}=f_{\phi^{\prime}}(\tilde{h}_{t-1},\tilde{z}_{t-1},a _{t-1}^{(T)}) \displaystyle\tilde{e}_{t}=e_{\phi^{\prime}}(o_{t}^{(T)}) (4) \displaystyle\tilde{z}_{t}\sim q_{\phi^{\prime}}(\tilde{h}_{t},\tilde{e}_{t}) \displaystyle\tilde{r}_{t}^{(S)}=(1-k)\cdot r_{\phi^{\prime}}(\tilde{h}_{t}, \tilde{z}_{t})+k\cdot r^{(T)}_{t},
$$

其中 $k$ 是 **目标信息化奖励因子（target-informed reward factor）** ，它在真实目标奖励 $r^{(T)}_{t}$ 与源奖励预测器 ${r}_{\phi^{\prime}}(\cdot)$ 在给定目标状态下的输出之间起到平衡作用。 **必须强调的是** ，由于目标状态空间（target state space）与源状态空间（source state space）是对齐的，因此使用目标数据作为输入来计算 ${r}_{\phi^{\prime}}(\cdot)$ 是可行的。

我们联合使用重新标注的奖励 $\tilde{r}_{t}^{(S)}$ 和从 $\mathcal{B}^{(S)}$ 采样的原始源域奖励 $r_{t}^{(S)}$ 来训练源奖励预测器。此训练通过最小化一个 **最大似然估计（Maximum Likelihood Estimation, MLE）** 损失来实现：

$$
\displaystyle\mathcal{L}_{r}(\phi^{\prime})=\ \eta\cdot\mathbb{E}_{\mathcal{B} ^{(S)}}\Big{[}\sum_{t=1}^{N}-\ln r_{\phi^{\prime}}(r_{t}^{(S)}|h_{t}^{(S)},z_{t}^{(S)})\Big{]}+(1-\eta)\mathbb{E}_{\mathcal{B}^{(T)}}\Big{[}\sum_{t=1}^{N}- \ln r_{\phi^{\prime}}(\tilde{r}_{t}^{(S)}|h_{t}^{(T)},z_{t}^{(T)})\Big{]},(5)
$$

其中第二项衡量了观测到重新标注的源奖励 $\tilde{r}_{t}^{(S)}$ 的负对数似然（negative log-likelihood）。$\eta$ 是一个超参数（hyperparameter），在此训练阶段从 $1$ 逐渐减小到 $0.1$。直观上，$\eta$ 控制着训练良好的源奖励预测器在有限的目标奖励监督下，逐步适应目标域的过程。

我们将公式 ([5](https://arxiv.org/html/2305.15260v4#S3.E5)) 整合到公式 ([2](https://arxiv.org/html/2305.15260v4#S3.E2)) 中，为源域智能体训练完整的世界模型 $\mathcal{M}_{\phi^{\prime}}$（算法 [1](#algorithm-1) 第 16 行），随后执行行为学习，使源评论家（source critic）能够评估目标策略（算法 [1](#algorithm-1) 第 17-19 行）。

<a id="section-3-3"></a>

### 3.3 最小-最大值约束（Min-Max Value Constraint）

在目标智能体（Target Agent）的行为学习阶段（算法 [1](#algorithm-1) 的第 8-10 行），我们通过向目标评论家模型（Target Critic Model）$v_{\xi}$ 的目标函数引入一个最小-最大正则化项，来缓解离线数据集中的价值高估问题。
首先，我们使用辅助的源评论家（Auxiliary Source Critic）$v_{\xi^{\prime}}$ 来估计想象出的目标状态的价值函数。
随后，我们通过额外 **最小化源评论家和目标评论家提供的估计值中的最大值** 来训练 $v_{\xi}$：

$$
\displaystyle\mathcal{L}(\xi)=\ \mathbb{E}_{p_{\phi},p_{\psi}}\Big{[}\sum^{H-1}_{t=1}\underbrace{\frac{1}{2}\left(v_{\xi}(\hat{z}_{t}^{(T)})-\texttt{sg}\big {(}V_{t}^{(T)}\big{)}\right)^{2}}_{\text{价值回归（value regression）}}+\ \underbrace{\alpha\max\left(v_{\xi}(\hat{z}_{t}^{(T)}),\ \texttt{sg}\big{(}v_{\xi^{\prime}}(\hat{z}_{t}^{(T)})\big{)}\right)}_{\text{价值约束（value constraint）}}\Big{]},(6)
$$

其中 $V_{t}^{(T)}$ 包含了未来 $n$ 步视野内奖励信息的加权平均值。
所提供的损失函数中的第一项拟合累积价值估计（其具体公式可在附录 [A.3](https://arxiv.org/html/2305.15260v4#A1.SS3) 中找到），而第二项则以一种温和保守的方式对分布外数据（Out-of-distribution Data）的高估价值进行正则化。
超参数 $\alpha$ 代表了价值约束的重要性。
运算符 sg($\cdot$) 表示我们停止梯度（Stop Gradient），以防止源评论家受到正则化项的影响。

这种方法提供了灵活保守的价值估计，在缓解高估和避免价值函数过度保守之间找到了平衡。
当目标评论家高估价值函数时，源评论家由于使用丰富的交互数据进行训练，不易受到价值高估问题的影响。因此，有可能观察到 $v_{\xi}(\hat{z}_{t}^{(T)})>v_{\xi^{\prime}}(\hat{z}_{t}^{(T)})$，而我们的方法旨在将 $v_{\xi}$ 的输出降低到 $v_{\xi^{\prime}}$ 的输出水平。这可以防止目标评论家高估真实价值。
相反，当源评论家在 $v_{\xi^{\prime}}(\hat{z}_{t}^{(T)})$ 中产生更大的价值时，最小-最大正则化项不会对目标评论家 $v_{\xi}$ 的训练产生影响。这鼓励了在目标世界模型（Target World Model）的想象中对潜在有利状态的探索。
与 DreamerV2 [16] 一致，
我们通过最大化一个带有熵正则化（Entropy Regularization）的 REINFORCE 目标函数来训练目标行动者（Target Actor）$\pi_{\psi}$，允许梯度直接通过已学习的动态模型进行反向传播：

$$
\displaystyle\mathcal{L}(\psi)=\mathbb{E}_{p_{\phi},p_{\psi}}\sum_{t=1}^{H-1}( \underbrace{\beta\mathrm{H}[a_{t}^{(T)}\mid\hat{z}_{t}^{(T)}]}_{\text{熵正则化（entropy regularization）}}+\underbrace{\rho V_{t}^{(T)}}_{\text{动态反向传播（dynamics backprop）}}+ \underbrace{(1-\rho)\ln\pi_{\psi}(\hat{a}_{t}^{(T)}\mid\hat{z}_{t}^{(T)}) \texttt{sg}(V_{t}^{(T)}-v_{\xi}(\hat{z}_{t}^{(T)})}_{\text{REINFORCE}}).(7)
$$

如前所述，$V_{t}^{(T)}$ 涉及未来 $n$ 步视野内奖励信息的加权平均值，详细公式见附录 [A.3](https://arxiv.org/html/2305.15260v4#A1.SS3)。

此外，必须指出的是，CoWorld 可以通过自适应地选择一个有用的任务作为辅助域，轻松扩展到具有多个源域（Multiple Source Domains）的场景。这种扩展可以通过测量目标域与每个源域之间的潜在状态距离来轻松实现。
关于自适应源域选择（Adaptive Source Domain Selection）的技术细节，请参阅附录 [C](#appendix-c)。

<a id="section-4"></a>

## 4 实验（Experiments）

<a id="table-2"></a>

> 表 2 | Meta-World 上 $3$ 个随机种子下 $10$ 个回合的平均回合回报及其标准差。
> Table 2: Mean episode returns and standard deviations of $10$ episodes over $3$ seeds on Meta-World.

| Model                  | BP$\rightarrow\text{DC}^{*}$ | DC $\rightarrow$ BP | BT$\rightarrow$ WC | BP$\rightarrow$ HP | WC$\rightarrow$ DC | HP$\rightarrow$ BT | Avg. |
| :--------------------- | :--------------------------- | :------------------ | :----------------- | :----------------- | :----------------- | :----------------- | :--- |
| Offline DV2            | 2143$\pm$579                 | 3142$\pm$533        | 3921$\pm$752       | 278$\pm$128        | 3899$\pm$679       | 3002$\pm$346       | 2730 |
| DrQ + BC               | 567$\pm$19                   | 587$\pm$68          | 623$\pm$85         | 1203$\pm$234       | 134$\pm$64         | 642$\pm$99         | 626  |
| CQL                    | 1984$\pm$13                  | 867$\pm$330         | 683$\pm$268        | 988$\pm$39         | 577$\pm$121        | 462$\pm$67         | 927  |
| CURL                   | 1972$\pm$11                  | 51$\pm$17           | 281$\pm$73         | 986$\pm$47         | 366$\pm$52         | 189$\pm$10         | 641  |
| LOMPO                  | 2883$\pm$183                 | 446$\pm$458         | 2983$\pm$569       | 2230$\pm$223       | 2756$\pm$331       | 1961$\pm$287       | 1712 |
| DV2 Finetune           | 3500$\pm$414                 | 2456$\pm$661        | 3467$\pm$1031      | 3702$\pm$451       | 4273$\pm$1327      | 3499$\pm$713       | 3781 |
| DV2 Finetune + EWC     | 1566$\pm$723                 | 167$\pm$86          | 978$\pm$772        | 528$\pm$334        | 2048$\pm$1034      | 224$\pm$147        | 918  |
| LOMPO Finetune         | 259$\pm$191                  | 95$\pm$53           | 142$\pm$70         | 332$\pm$452        | 3698$\pm$1615      | 224$\pm$88         | 792  |
| CoWorld (Best-Source)  | 3967$\pm$312                 | 3623$\pm$543        | 4521$\pm$367       | 4570$\pm$677       | 4845$\pm$14        | 3889$\pm$159       | 4241 |
| CoWorld (Multi-Source) | 3864$\pm$352                 | 3573$\pm$541        | 4507$\pm$59        | 4460$\pm$783       | 4678$\pm$137       | 3626$\pm$275       | 4094 |

<a id="section-4-1"></a>

### 4.1 实验设置（Experimental Setups）

#### 数据集（Datasets）。

我们在三个视觉控制环境中评估 CoWorld，即 Meta-World [54]、RoboDesk [18] 和 DeepMind Control Suite（DMC）[47]，包括跨任务和跨环境设置（Meta-World $\rightarrow$ RoboDesk）。受 D4RL [9] 启发，我们使用 DreamerV2 [16] 构建了中等回放质量（medium-replay）的离线数据集。这些数据集包含训练过程中收集的回放缓冲区（replay buffer）中的所有样本，直到策略达到中等性能水平，其定义为达到 DreamerV2 智能体所能达到的最大分数的 $1/3$。关于使用中等专家（medium-expert）离线数据训练的 CoWorld 的更多结果，请参阅附录 [B.2](https://arxiv.org/html/2305.15260v4#A2.SS2)。

#### 对比方法（Compared methods）。

我们将 CoWorld 与基于模型（model-based）和无模型（model-free）的强化学习（Reinforcement Learning, RL）方法进行比较，包括 Offline DV2 [25]、DrQ+BC [25]、CQL [25]、CURL [22] 和 LOMPO [39]。此外，我们引入了 DV2 Finetune 方法，该方法涉及将在在线源域（online source domain）中预训练的 DreamerV2 [16] 模型随后在离线目标数据集（offline target dataset）中进行微调（finetuning）。此外，DV2 Finetune 可以与持续学习（continual learning）方法——弹性权重巩固（Elastic Weight Consolidation, EWC）[19] 相结合，以正则化模型以保留源域知识，即 Finetune+EWC。更多细节请参阅附录 [E](#appendix-e)。

<a id="section-4-2"></a>

### 4.2 Meta-World 上的跨任务实验（Cross-Task Experiments on Meta-World）

Meta-World 是一个开源的模拟基准测试平台，旨在解决广泛的机器人操作任务。我们选择 $6$ 个任务作为离线数据集或在线辅助域（online auxiliary domain）的潜在候选。这些任务包括：关门（Door Close, $\textbf{DC}^{*}$）、按钮按压（Button Press, BP）、关窗（Window Close, WC）、手柄按压（Handle Press, HP）、抽屉关闭（Drawer Close, DC）、按钮下压（Button Topdown, BT）。

#### 主要结果（Main results）。

如表 [2](#table-2) 所示，我们比较了 CoWorld 与其他模型在 Meta-World 上的结果。CoWorld 在所有 $6$ 个任务中都取得了最佳性能。值得注意的是，它超越了 Offline DV2 [25]——一种同样基于 DreamerV2 并专门为离线视觉强化学习（offline visual RL）设计的方法。对于在线到离线（online-to-offline）微调模型，DV2 Finetune 通过利用来自辅助源域（auxiliary source domain）的迁移知识（transferred knowledge）取得了第二好的结果。然而，我们观察到，在涉及源域和目标域之间在视觉观察、物理动力学、奖励定义甚至机器人动作空间上存在显著数据分布偏移（data distribution shifts）的场景中（例如，Meta-World $\rightarrow$ RoboDesk），其性能会出现明显下降。另一个重要的基线模型是 DV2 Finetune+EWC，它侧重于缓解源域预训练中获得的知识的灾难性遗忘（catastrophic forgetting）。然而，如果没有额外的领域自适应（domain adaptation）模型设计，保留源域知识最终可能导致目标域性能下降。LOMPO 模型在引入源预训练阶段时会遭受负迁移（negative transfer）效应。当在离线域中从头开始训练时，其平均回报为 $1{,}712$，而在进行在线到离线微调时，其平均回报仅为 $792$。这表明，简单的迁移学习方法（naïve transfer learning method）可能会因意外的偏差（unexpected bias）而导致目标性能退化。

<a id="figure-3"></a>

![heatmap_and_comparison](images/heatmap_and_comparison.png)

> 图 3 | 左图：每个网格中的数值表示 CoWorld 与 Offline DV2 相比所获回报（Returns）的比率。高亮显示的网格代表性能最佳的源域（Source Domain）。右图：不同源域下在 Drawer Close (DC\*) 任务上的回报，其中多源（Multi-source）CoWorld（黄线）被证明能自动发现（即 Door Close）作为源域，并取得与性能最佳的单源（Single-source）CoWorld（红线）相当的结果。

<a id="figure-4"></a>

![4in1](images/4in1.png)

> 图 4 | Meta-World $\rightarrow$ RoboDesk 领域迁移场景下的定量结果。

#### 随机源域的结果

鉴于我们在表 [2](#table-2) 中展示了最佳源域的结果，即我们手动从 Meta-World 中选择一个源任务，人们可能会质疑辅助环境与目标离线数据集之间领域差异（Domain Discrepancies）的影响。在图 [3](#figure-3)（左）中，展示了 CoWorld 在 Meta-World 的 $6$ 个任务间的迁移矩阵（Transfer Matrix），其中大于 $1$ 的值表示积极的领域迁移效果。值得注意的是，存在源任务和目标任务关联性较弱的挑战性案例。在大多数情况下（$30$ 例中的 $26$ 例），CoWorld 的表现优于 Offline DV2，如热力图所示。

#### 多源域的结果

**关键** 在于，CoWorld 可以通过自适应地选择一个有用的任务作为辅助域，轻松扩展到多源域（Multiple Source Domains）场景。从表 [2](#table-2) 可以看出，多源 CoWorld 取得了与使用手动指定的在线模拟器（Online Simulators）训练的模型相当的结果。在图 [3](#figure-3)（左）中，多源 CoWorld 在所有情况下都相对于 Offline DV2 取得了正向改进，接近使用每个源任务作为辅助域的模型的最佳结果。在图 [3](#figure-3)（右）中，它也始终优于 DV2 Finetune 基线模型。这些结果表明，我们的方法能够在 **不严格假设领域相似性** 的情况下执行，并且能够从一组相关和不太相关的源域中 **自动识别出有用的在线模拟器** 。

<a id="section-4-3"></a>

### 4.3 跨环境：从 Meta-World 到 RoboDesk

为了探索具有更显著领域差距（Domain Gaps）的跨环境迁移，我们使用 RoboDesk 中的四个任务来构建独立的离线数据集，即：Push Button、Open Slide、Drawer Open、Upright Block off Table。这些任务需要处理具有随机位置物体的图像输入。表 [1](#table-1) 展示了两者在物理动力学（Physical Dynamics）、动作空间（Action Space）、奖励定义（Reward Definitions）和视觉外观（Visual Appearances）上的差异。

图 [4](#figure-4) 展示了定量比较结果，其中 CoWorld 以较大优势超越了 Offline DV2 和 DV2 Finetune。对于最佳源域实验，我们手动从 Meta-World 中选择一个源域。对于多源实验，我们联合使用所有 Meta-World 任务作为源域。与先前的发现相反，在这种跨环境设置中，由于存在更明显的领域差异，直接对源世界模型（Source World Model）进行微调（Finetuning）并不会带来最终性能的显著提升。相比之下，CoWorld 通过利用领域特定的世界模型（Domain-specific World Models）和强化学习（Reinforcement Learning, RL）智能体（Agents），并显式地对齐跨域的状态（State）和奖励（Reward）空间，更成功地应对了这些挑战。我们还展示了多源 CoWorld 的性能，它取得了与专门使用我们指定源域的最佳源域模型相当的结果。

<a id="section-4-4"></a>

### 4.4 在 DMC 上的跨动力学实验

**DMC（DeepMind Control Suite）** 是一个被广泛探索的连续控制基准。我们使用 Walker（行走者）和 Cheetah（猎豹）作为基础智能体，并对环境进行修改，创建了一组 $8$ 个不同的任务，即：Walker Walk（WW，行走者行走）、Walker Downhill（WD，行走者下坡）、Walker Uphill（WU，行走者上坡）、Walker Nofoot（WN，行走者无右足）、Cheetah Run（CR，猎豹奔跑）、Cheetah Downhill（CD，猎豹下坡）、Cheetah Uphill（CU，猎豹上坡）、Cheetah Nopaw（CN，猎豹无前爪）。
具体来说，Walker Nofoot 是一个我们无法控制 Walker 智能体右足的任务。Cheetah Nopaw 是一个我们无法控制 Cheetah 智能体前爪的任务。

![robo_ablation](images/robo_ablation.png)

> (a) Meta-World $\rightarrow$ RoboDesk（按下绿色按钮）

<a id="table-3"></a>

> 表 3：离线 DMC 中 $3$ 个随机种子下 $10$ 幕的平均奖励和标准差。

| 模型           | WW $\rightarrow$ WD | WW $\rightarrow$ WU | WW $\rightarrow$ WN | CR $\rightarrow$ CD | CR $\rightarrow$ CU | CR $\rightarrow$ CN | 平均 |
| -------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ---- |
| Offline DV2    | 435$\pm$22          | 139$\pm$4           | 214$\pm$4           | 243$\pm$7           | 3$\pm$1             | 51$\pm$4            | 181  |
| DrQ+BC         | 291$\pm$ 10         | 299$\pm$15          | 318$\pm$40          | 663$\pm$15          | 202$\pm$12          | 132$\pm$33          | 355  |
| CQL            | $46\pm$19           | 64$\pm$32           | 29$\pm$2            | 2$\pm$1             | 52 $\pm$57          | 111$\pm$157         | 51   |
| CURL           | 43$\pm$5            | 21$\pm$3            | 23$\pm$3            | 26$\pm$7            | 4$\pm$2             | 11$\pm$4            | 21   |
| LOMPO          | 462$\pm$87          | 260$\pm$21          | 460$\pm$9           | 395$\pm$52          | 46$\pm$19           | 120$\pm$4           | 291  |
| DV2 Finetune   | 379$\pm$23          | 354$\pm$29          | 407$\pm$37          | 702$\pm$41          | 208$\pm$22          | 454$\pm$82          | 417  |
| LOMPO Finetune | 209$\pm$21          | 141$\pm$27          | 212$\pm$9           | 142$\pm$29          | 17$\pm$11           | 105$\pm$12          | 137  |
| CoWorld        | 629$\pm$9           | 407$\pm$141         | 426$\pm$32          | 745$\pm$28          | 225$\pm$20          | 493$\pm$10          | 488  |

我们应用所提出的多源域选择方法来构建表 [3](#table-3) 中所示的域迁移设置。值得注意的是，CoWorld 在 $6$ 个 DMC 离线数据集中的 $5$ 个上优于其他对比模型，并在剩余任务中取得了第二佳的性能。平均而言，它比 Offline DV2 高出 $169.6\%$，比 DrQ+BC 高出 $37.5\%$。相应的定性比较可在附录 [B.1](https://arxiv.org/html/2305.15260v4#A2.SS1) 中找到。

<a id="section-4-5"></a>

### 4.5 进一步分析

#### 消融研究

我们进行了一系列消融研究，以验证 **状态空间对齐（State Space Alignment）** （阶段 A）、 **奖励对齐（Reward Alignment）** （阶段 B）和 **最小-最大值约束（Min-Max Value Constraint）** （阶段 C）的有效性。
我们在图 [5(a)](https://arxiv.org/html/2305.15260v4#S4.F5.sf1) 中展示了来自 RoboDesk 的离线 Push Green Button 数据集上的相应结果。当我们在 CoWorld 中放弃每个训练阶段时，性能都会出现显著下降。

#### CoWorld 能否解决价值高估问题？

我们在训练过程完成后，评估了 CoWorld 的 **评论家网络（Critic Network）** 在离线 Meta-World 数据集上估计的价值。
在图 [5(b)](https://arxiv.org/html/2305.15260v4#S4.F5.sf2) 中，我们计算了贯穿 $500$ 步的累积价值预测。
真实价值是通过计算 **行动者（Actor）** 在相同的 $500$ 步周期内获得的实际奖励的折现和来确定的。
我们观察到，包括 Offline DV2 和 CQL 在内的现有方法，在离线设置中经常高估价值函数。
基线模型“CoWorld w/o Max”是 CoWorld 的一个变体，它对评论家损失施加了一种 **暴力约束（Brute-force Constraint）** 。它将方程 ([6](https://arxiv.org/html/2305.15260v4#S3.E6)) 重新表述为 $\sum^{H-1}_{t=1}\frac{1}{2}(v_{\xi}(\hat{z}_{t})-\texttt{sg}(V_{t}))^{2}+
\alpha v_{\xi}(\hat{z}_{t})$。正如观察到的，该模型倾向于低估真实价值函数，这可能导致策略过于保守。
相比之下，CoWorld 估计的价值明显更准确，更接近真实值。

#### CoWorld 对域相似性的依赖

我们从不同 **观测空间（Observation Space）** 和 **奖励空间（Reward Space）** 的角度，进一步研究了 CoWorld 对域相似性的依赖。
我们首先探讨了当我们只有与目标域观测空间显著不同的源域时，CoWorld 的表现如何。
如表 [4](#table-4) 所示，智能体在源域（Meta-World）接收低维状态输入，在目标域（RoboDesk）接收高维图像。
我们可以看到，由于能够有效利用低维源数据，CoWorld 的表现分别比 Offline DV2 高出 $13.3\%$ 和 $34.0\%$。
值得注意的是， **微调方法（Finetuning Method）** （DV2 Finetune）在此场景下不适用。
在表 [5](#table-5) 中，我们还观察到，即使奖励信号显著不同，CoWorld 也能从源域中受益。
与之前的实验不同，我们对源 Meta-World 任务使用了 **稀疏奖励函数（Sparse Reward Function）** 。它仅在任务完成时设置为 $500$，在此之前保持为 $0$。
实验结果表明，尽管过度稀疏的奖励会阻碍训练过程，但在相同设置下，CoWorld 相比 DV2 Finetune 仍能实现平均 $6.6\%$ 的性能提升。

#### 与跨域联合训练单一世界模型的比较

值得注意的是， **CoWorld** 是为源域和目标域分别实现独立的世界模型（World Model）的。
另一种方案是，我们可以跨不同域使用一个联合训练的世界模型，以实现更高效的内存使用。
在表 [6](#table-6) 中，我们比较了原始 CoWorld 与“ **多任务 DV2（Multi-Task DV2）** ”的结果。
多任务 DV2 涉及使用一个联合世界模型和独立的行动者-评论者（Actor-Critic）模型，在离线和在线数据上共同训练 **DreamerV2** 。
CoWorld 始终表现更优。
直观地说，使用独立的世界模型允许源域和目标域具有不同的物理动力学、观测空间或奖励形成方式，如表 [4](#table-4) 和表 [5](#table-5) 所示的场景。

<a id="table-4"></a>

> 表 4：跨域观测空间显著不同的实验。我们在源域中使用低维状态数据作为强化学习（Reinforcement Learning, RL）智能体的输入，在目标域中使用高维图像观测。MW 代表 Meta-World，RD 代表 RoboDesk。

| 方法        | MW: Button Press $\rightarrow$ RD: Push Button | MW: Window Close $\rightarrow$ RD: Open Slide |
| :---------- | :--------------------------------------------- | :-------------------------------------------- |
| Offline DV2 | 347 $\pm$ 24                                   | 156 $\pm$ 46                                  |
| CoWorld     | 393 $\pm$ 64                                   | 209 $\pm$ 43                                  |

<a id="table-5"></a>

> 表 5：跨域奖励形成方式显著不同的实验。我们在源域中使用稀疏奖励（Sparse Reward），同时在目标域中保持密集奖励（Dense Reward）。

| 方法         | MW: Button Press $\rightarrow$ RD: Push Button | MW: Window Close $\rightarrow$ RD: Open Slide |
| :----------- | :--------------------------------------------- | :-------------------------------------------- |
| DV2 Finetune | 314 $\pm$ 51                                   | 173 $\pm$ 39                                  |
| CoWorld      | 335 $\pm$ 28                                   | 184 $\pm$ 32                                  |

<a id="table-6"></a>

> 表 6：与跨域联合训练一个世界模型（多任务 DV2）的比较。

| 方法           | MW: Button Press $\rightarrow$ RD: Push Button | MW: Window Close $\rightarrow$ RD: Open Slide |
| :------------- | :--------------------------------------------- | :-------------------------------------------- |
| Multi-Task DV2 | 342 $\pm$ 29                                   | 173 $\pm$ 22                                  |
| CoWorld        | 428 $\pm$ 42                                   | 202 $\pm$ 19                                  |

#### **超参数敏感性（Hyperparameter sensitivity）**

我们在 **Meta-World** （DC $\rightarrow$ BP）上进行了敏感性分析。
从图 [6](#figure-6) 中，我们观察到，当用于 **域 KL 散度损失（Domain KL Loss）** 的 $\beta_{2}$ 过小时，源编码器和目标编码器之间的状态对齐变得不足，阻碍了迁移学习（Transfer Learning）过程。相反，如果 $\beta_{2}$ 过大，目标编码器会受到源编码器的过度影响，导致性能下降。
我们还发现， **目标信息奖励因子（Target-informed Reward Factor）** $k$ 在平衡源数据和目标奖励信息的影响方面起着关键作用，在 $[0.1,0.7]$ 的范围内，它相对于 DV2 Finetune（$2456\pm 661$）取得了一致的改进。
此外，我们发现用于 **目标价值约束（Target Value Constraint）** 的超参数 $\alpha$ 在 $[1,3]$ 范围内表现良好，而过大的 $\alpha$ 可能导致目标评论者中的价值过度保守。

<a id="figure-6"></a>

![Q-value](images/Q-value.png)

> 图 6：在 Meta-World（DC $\rightarrow$ BP）上进行的超参数敏感性分析。

<a id="section-5"></a>

## 5 相关工作（Related Work）

从图像中学习控制策略在现实应用中至关重要。现有方法可根据其使用的算法分为 **无模型（Model-free）** [22, 41, 44, 48, 36] 或 **基于模型（Model-based）** [15, 14, 16, 43, 35, 13, 28, 29, 61, 51] 的 **强化学习（Reinforcement Learning, RL）** 。

在 **离线强化学习（Offline RL）** 中，智能体利用预先收集的离线数据来优化策略，并面临与 **价值高估（Value overestimation）** 相关的挑战 [23]。先前的方法主要建议采取离线数据集中先前存在的动作，或学习保守的价值估计 [11, 21, 4, 55, 53, 40]。最近的方法引入了特定技术来应对离线视觉强化学习中的挑战 [27, 7, 23, 2, 39, 52, 43, 57, 5, 25]。Rafailov 等人 [39] 提出使用 **潜在动力学模型（Latent dynamics models）** 和 **不确定性量化（Uncertainty quantification）** 来处理高维观测。Cho 等人 [5] 提出合成原始观测数据以扩充训练缓冲区，旨在缓解 **过拟合（Overfitting）** 问题。在一项相关研究中，Lu 等人 [25] 基于 DreamerV2 [16] 建立了一个有竞争力的离线视觉强化学习模型，因此我们将其作为我们方法的一个重要基线。

我们的工作也与 **迁移强化学习（Transfer RL）** 相关，其旨在利用过去任务中学到的知识来促进在未见任务中的学习 [64, 42, 58, 45, 59, 8, 49, 46, 12, 20, 38, 24, 33]。大多数与“离线数据集 + 模拟器”相关的现有方法都侧重于 **离线到在线（Offline-to-online）** 的设置，即策略首先在离线数据集上进行预训练，然后在交互环境中进行微调和部署 [33, 60, 56, 63]。这些方法旨在弥合离线学习和在线学习之间的差距，并促进模型快速适应在线环境。相比之下，我们探索了 **在线到离线（Online-to-offline）** 的设置，这为价值高估问题提供了一种新的解决方案。

此外，Niu 等人 [34] 引入了一个 **动态感知（Dynamics-aware）** 的混合离线与在线框架，以整合离线数据集和在线模拟器进行策略优化。与 CoWorld 不同，该方法主要关注低维 **马尔可夫决策过程（Markov Decision Processes, MDPs）** ，无法直接用于视觉控制任务。

在视觉强化学习的背景下，CtrlFormer [31] 通过一种 **样本高效（Sample-efficient）** 的 **视觉 Transformer（Vision Transformer）** 学习可迁移的状态表示。APV [43] 在源域视频上执行 **无动作（Action-free）** 的世界模型预训练，并在下游任务上对模型进行微调。Choreographer [28] 构建了一个基于模型的智能体，利用其世界模型在想象中学习和适应技能，并使用 **元控制器（Meta-controller）** 将学到的技能适应到新领域。VIP [26] 提出了一种 **自监督（Self-supervised）** 、 **目标条件化（Goal-conditioned）** 的价值函数目标，这使得能够使用未标记的视频数据进行模型预训练。

与先前方法不同，我们使用 **辅助模拟器（Auxiliary simulators）** 来处理离线视觉强化学习，通过 **协同训练（Co-trained）** 的世界模型来缓解价值高估问题。

<a id="section-6"></a>

## 6 结论与局限性（Conclusions and Limitations）

在本文中，我们提出了一种名为 **CoWorld** 的迁移强化学习方法，主要解决离线视觉强化学习中表征学习和价值估计的困难。其核心思想是利用可访问的在线环境来训练一个辅助强化学习智能体，以提供额外的价值评估。

为了解决领域差异并改进离线策略，我们提出了具体的技术贡献： **跨域状态对齐（Cross-domain state alignment）** 、 **奖励对齐（Reward alignment）** 和 **最小-最大价值约束（Min-max value constraint）** 。CoWorld 在三个强化学习基准测试中展示了有竞争力的结果。

CoWorld 一个尚未解决的问题是与辅助域训练阶段相关的计算复杂度增加（参见附录 [B.7](https://arxiv.org/html/2305.15260v4#A2.SS7)）。在未来的研究中提高训练效率是很有价值的。

## Acknowledgments

本研究得到了以下项目的资助：国家自然科学基金（项目批准号：62250062、62106144、62302246）、上海市科技重大专项（项目编号：2021SHZDZX0102）、中央高校基本科研业务费专项资金，以及 CCF-腾讯犀牛鸟基金。

本研究还得到了以下项目的资助：浙江省自然科学基金（项目编号：LQ23F010008）、东方理工大学（宁波）高性能计算中心，以及宁波数字孪生研究院。

## 参考文献

## 参考文献

- [1]
  Brian M Adams, Harvey T Banks, Hee-Dae Kwon, and Hien T Tran.
  Dynamic multidrug therapies for hiv: Optimal and sti control approaches.
  Mathematical Biosciences & Engineering, 1(2):223–241, 2004.
- [2]
  Rishabh Agarwal, Dale Schuurmans, and Mohammad Norouzi.
  An optimistic perspective on offline reinforcement learning.
  In ICML, pages 104–114, 2020.
- [3]
  Huayu Chen, Cheng Lu, Chengyang Ying, Hang Su, and Jun Zhu.
  Offline reinforcement learning via high-fidelity generative behavior modeling.
  In ICLR, 2023.
- [4]
  Xi Chen, Ali Ghadirzadeh, Tianhe Yu, Jianhao Wang, Alex Yuan Gao, Wenzhe Li, Liang Bin, Chelsea Finn, and Chongjie Zhang.
  Lapo: Latent-variable advantage-weighted policy optimization for offline reinforcement learning.
  In NeurIPS, volume 35, pages 36902–36913, 2022.
- [5]
  Daesol Cho, Dongseok Shim, and H Jin Kim.
  S2p: State-conditioned image synthesis for data augmentation in offline reinforcement learning.
  In NeurIPS, 2022.
- [6]
  Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter.
  Fast and accurate deep network learning by exponential linear units (elus).
  arXiv preprint arXiv:1511.07289, 2015.
- [7]
  Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn.
  Robonet: Large-scale multi-robot learning.
  In CoRL, 2019.
- [8]
  Benjamin Eysenbach, Swapnil Asawa, Shreyas Chaudhari, Sergey Levine, and Ruslan Salakhutdinov.
  Off-dynamics reinforcement learning: Training for transfer with domain classifiers.
  In ICLR, 2021.
- [9]
  Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine.
  D4rl: Datasets for deep data-driven reinforcement learning.
  arXiv preprint arXiv:2004.07219, 2020.
- [10]
  Scott Fujimoto and Shixiang Shane Gu.
  A minimalist approach to offline reinforcement learning.
  In NeurIPS, 2021.
- [11]
  Scott Fujimoto, David Meger, and Doina Precup.
  Off-policy deep reinforcement learning without exploration.
  In ICML, pages 2052–2062, 2019.
- [12]
  Dibya Ghosh, Chethan Bhateja, and Sergey Levine.
  Reinforcement learning from passive data via latent intentions.
  arXiv preprint arXiv:2304.04782, 2023.
- [13]
  Danijar Hafner, Kuang-Huei Lee, Ian Fischer, and Pieter Abbeel.
  Deep hierarchical planning from pixels.
  arXiv preprint arXiv:2206.04114, 2022.
- [14]
  Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi.
  Dream to control: Learning behaviors by latent imagination.
  In ICLR, 2020.
- [15]
  Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson.
  Learning latent dynamics for planning from pixels.
  In ICML, pages 2555–2565, 2019.
- [16]
  Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba.
  Mastering atari with discrete world models.
  In ICLR, 2021.
- [17]
  William Hua, Hongyuan Mei, Sarah Zohar, Magali Giral, and Yanxun Xu.
  Personalized dynamic treatment regimes in continuous time: a bayesian approach for optimizing clinical decisions with timing.
  Bayesian Analysis, 17(3):849–878, 2022.
- [18]
  Harini Kannan, Danijar Hafner, Chelsea Finn, and Dumitru Erhan.
  Robodesk: A multi-task reinforcement learning benchmark.
  [https://github.com/google-research/robodesk](https://github.com/google-research/robodesk), 2021.
- [19]
  James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al.
  Overcoming catastrophic forgetting in neural networks.
  Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.
- [20]
  Aviral Kumar, Rishabh Agarwal, Xinyang Geng, George Tucker, and Sergey Levine.
  Offline q-learning on diverse multi-task data both scales and generalizes.
  In ICLR, 2023.
- [21]
  Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine.
  Conservative q-learning for offline reinforcement learning.
  In NeurIPS, volume 33, pages 1179–1191, 2020.
- [22]
  Michael Laskin, Aravind Srinivas, and Pieter Abbeel.
  Curl: Contrastive unsupervised representations for reinforcement learning.
  In ICML, pages 5639–5650, 2020.
- [23]
  Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu.
  Offline reinforcement learning: Tutorial, review, and perspectives on open problems.
  arXiv preprint arXiv:2005.01643, 2020.
- [24]
  Xin Liu, Yaran Chen, Haoran Li, Boyu Li, and Dongbin Zhao.
  Cross-domain random pre-training with prototypes for reinforcement learning.
  arXiv preprint arXiv:2302.05614, 2023.
- [25]
  Cong Lu, Philip J Ball, Tim GJ Rudner, Jack Parker-Holder, Michael A Osborne, and Yee Whye Teh.
  Challenges and opportunities in offline reinforcement learning from visual observations.
  Transactions on Machine Learning Research, 2023.
- [26]
  Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang.
  Vip: Towards universal visual reward and representation via value-implicit pre-training.
  In ICLR, 2023.
- [27]
  Ajay Mandlekar, Jonathan Booher, Max Spero, Albert Tung, Anchit Gupta, Yuke Zhu, Animesh Garg, Silvio Savarese, and Li Fei-Fei.
  Scaling robot supervision to hundreds of hours with roboturk: Robotic manipulation dataset through human reasoning and dexterity.
  In IROS, pages 1048–1055, 2019.
- [28]
  Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Alexandre Lacoste, and Sai Rajeswar.
  Choreographer: Learning and adapting skills in imagination.
  In ICLR, 2023.
- [29]
  Vincent Micheli, Eloi Alonso, and François Fleuret.
  Transformers are sample efficient world models.
  In ICLR, 2023.
- [30]
  Zhiyu Mou, Yusen Huo, Rongquan Bai, Mingzhou Xie, Chuan Yu, Jian Xu, and Bo Zheng.
  Sustainable online reinforcement learning for auto-bidding.
  In NeurIPS, volume 35, pages 2651–2663, 2022.
- [31]
  Yao Mark Mu, Shoufa Chen, Mingyu Ding, Jianyu Chen, Runjian Chen, and Ping Luo.

- [31]
  Wendong Zhang, Geng Chen, Xiangming Zhu, Siyu Gao, Yunbo Wang, and Xiaokang Yang.
  **Ctrlformer: 通过 Transformer 学习可迁移的视觉控制状态表示（Ctrlformer: Learning transferable state representation for visual control via transformer）** 。
  发表于 **国际机器学习会议（International Conference on Machine Learning, ICML）** ，第 16043–16061 页，2022 年。
- [32]
  Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta.
  **R3M: 用于机器人操作的通用视觉表示（R3m: A universal visual representation for robot manipulation）** 。
  arXiv 预印本 arXiv:2203.12601，2022 年。
- [33]
  Mitsuhiko Nakamoto, Yuexiang Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine.
  **Cal-QL: 用于高效在线微调的校准离线强化学习预训练（Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning）** 。
  arXiv 预印本 arXiv:2303.05479，2023 年。
- [34]
  Haoyi Niu, Yiwen Qiu, Ming Li, Guyue Zhou, Jianming HU, Xianyuan Zhan, et al.
  **何时信任你的模拟器：动态感知的混合离线与在线强化学习（When to trust your simulator: Dynamics-aware hybrid offline-and-online reinforcement learning）** 。
  发表于 **神经信息处理系统大会（Conference on Neural Information Processing Systems, NeurIPS）** ，第 35 卷，第 36599–36612 页，2022 年。
- [35]
  Minting Pan, Xiangming Zhu, Yunbo Wang, and Xiaokang Yang.
  **Iso-dream: 在世界模型中隔离并利用不可控视觉动态（Iso-dream: Isolating and leveraging noncontrollable visual dynamics in world models）** 。
  发表于 **神经信息处理系统大会（NeurIPS）** ，第 35 卷，第 23178–23191 页，2022 年。
- [36]
  Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, and Abhinav Gupta.
  **预训练视觉模型用于控制的意料之中的有效性（The unsurprising effectiveness of pre-trained vision models for control）** 。
  发表于 **国际机器学习会议（ICML）** ，第 17359–17371 页，2022 年。
- [37]
  Han Qi, Yi Su, Aviral Kumar, and Sergey Levine.
  **通过不变表示学习进行数据驱动的离线决策（Data-driven offline decision-making via invariant representation learning）** 。
  发表于 **神经信息处理系统大会（NeurIPS）** ，2022 年。
- [38]
  Rafael Rafailov, Kyle Beltran Hatch, Victor Kolev, John D Martin, Mariano Phielipp, and Chelsea Finn.
  **MOTO: 基于模型的机器人学习从离线预训练到在线微调（MOTO: Offline pre-training to online fine-tuning for model-based robot learning）** 。
  发表于 **第七届机器人学习年度会议（7th Annual Conference on Robot Learning）** ，2023 年。
- [39]
  Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, and Chelsea Finn.
  **使用潜空间模型从图像进行离线强化学习（Offline reinforcement learning from images with latent space models）** 。
  发表于 **机器学习研究论文集（Proceedings of Machine Learning Research）** ，第 1154–1168 页，2021 年。
- [40]
  Marc Rigter, Bruno Lacerda, and Nick Hawes.
  **Rambo-rl: 鲁棒的对抗性基于模型的离线强化学习（Rambo-rl: Robust adversarial model-based offline reinforcement learning）** 。
  发表于 **神经信息处理系统大会（NeurIPS）** ，2022 年。
- [41]
  Max Schwarzer, Nitarshan Rajkumar, Michael Noukhovitch, Ankesh Anand, Laurent Charlin, R Devon Hjelm, Philip Bachman, and Aaron C Courville.
  **为数据高效强化学习预训练表示（Pretraining representations for data-efficient reinforcement learning）** 。
  发表于 **神经信息处理系统大会（NeurIPS）** ，第 34 卷，第 12686–12699 页，2021 年。
- [42]
  Ramanan Sekar, Oleh Rybkin, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, and Deepak Pathak.
  **通过自监督世界模型规划探索（Planning to explore via self-supervised world models）** 。
  发表于 **国际机器学习会议（ICML）** ，第 8583–8592 页，2020 年。
- [43]
  Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel.
  **利用无动作预训练视频进行强化学习（Reinforcement learning with action-free pre-training from videos）** 。
  发表于 **国际机器学习会议（ICML）** ，第 19561–19579 页，2022 年。
- [44]
  Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin.
  **将表示学习与强化学习解耦（Decoupling representation learning from reinforcement learning）** 。
  发表于 **国际机器学习会议（ICML）** ，第 9870–9879 页，2021 年。
- [45]
  Yanchao Sun, Xiangyu Yin, and Furong Huang.
  **Temple: 为样本高效多任务强化学习学习状态转移模板（Temple: Learning template of transitions for sample efficient multi-task rl）** 。
  发表于 **人工智能促进协会会议（AAAI Conference on Artificial Intelligence, AAAI）** ，第 35 卷，第 9765–9773 页，2021 年。
- [46]
  Yanchao Sun, Ruijie Zheng, Xiyao Wang, Andrew Cohen, and Furong Huang.
  **通过基于模型的正则化跨观测特征空间迁移强化学习（Transfer rl across observation feature spaces via model-based regularization）** 。
  发表于 **国际学习表征会议（International Conference on Learning Representations, ICLR）** ，2022 年。
- [47]
  Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al.
  **DeepMind 控制套件（Deepmind control suite）** 。
  arXiv 预印本 arXiv:1801.00690，2018 年。
- [48]
  Tete Xiao, Ilija Radosavovic, Trevor Darrell, and Jitendra Malik.
  **用于运动控制的掩码视觉预训练（Masked visual pre-training for motor control）** 。
  arXiv 预印本 arXiv:2203.06173，2022 年。
- [49]
  Mengjiao Yang and Ofir Nachum.
  **表示至关重要：用于序列决策的离线预训练（Representation matters: offline pretraining for sequential decision making）** 。
  发表于 **国际机器学习会议（ICML）** ，第 11784–11794 页，2021 年。
- [50]
  Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto.
  **掌握视觉连续控制：改进的数据增强强化学习（Mastering visual continuous control: Improved data-augmented reinforcement learning）** 。
  arXiv 预印本 arXiv:2107.09645，2021 年。
- [51]
  Chengyang Ying, Zhongkai Hao, Xinning Zhou, Hang Su, Songming Liu, Jialian Li, Dong Yan, and Jun Zhu.
  **奖励信息驱动的 Dreamer 用于强化学习中的任务泛化（Reward informed dreamer for task generalization in reinforcement learning）** 。
  arXiv 预印本 arXiv:2303.05092，2023 年。
- [52]
  Tianhe Yu, Aviral Kumar, Yevgen Chebotar, Karol Hausman, Chelsea Finn, and Sergey Levine.
  **如何在离线强化学习中利用未标记数据（How to leverage unlabeled data in offline reinforcement learning）** 。
  发表于 **国际机器学习会议（ICML）** ，第 25611–25635 页，2022 年。
- [53]
  Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn.
  **Combo: 保守的基于模型的离线策略优化（Combo: Conservative offline model-based policy optimization）** 。
  发表于 **神经信息处理系统大会（NeurIPS）** ，第 34 卷，第 28954–28967 页，2021 年。
- [54]
  Tianhe

用于持续视觉控制与预测的预测经验回放。
arXiv 预印本 arXiv:2303.06572，2023年。

- [62]
  张致远（Zhiyue Zhang）、梅宏远（Hongyuan Mei）和徐彦勋（Yanxun Xu）。
  用于医疗健康应用的连续时间决策变换器。
  载于《人工智能与统计国际会议》，第 6245–6262 页。PMLR，2023年。
- [63]
  郑涵（Han Zheng）、罗旭芳（Xufang Luo）、魏鹏飞（Pengfei Wei）、宋璇（Xuan Song）、李东升（Dongsheng Li）和蒋晶（Jing Jiang）。
  用于离线到在线强化学习的自适应策略学习。
  载于《AAAI》，第 37 卷，第 11372–11380 页，2023年。
- [64]
  庄壮迪（Zhuangdi Zhu）、林凯翔（Kaixiang Lin）、安尼尔·K·贾因（Anil K Jain）和周家宇（Jiayu Zhou）。
  深度强化学习中的迁移学习：综述。
  arXiv 预印本 arXiv:2009.07888，2020年。
- [65]
  庄梓峰（Zifeng Zhuang）、雷坤（Kun Lei）、刘金鑫（Jinxin Liu）、王东林（Donglin Wang）和郭一郎（Yilang Guo）。
  行为近端策略优化。
  载于《ICLR》，2023年。

## 附录（Appendix）

在本附录中，我们提供以下补充材料：
([A](#appendix-a)) 所提出模型的详细信息，包括对学习方案、符号、世界模型架构、行为学习目标函数和超参数的进一步描述。
([B](#appendix-b)) 额外的实验结果，包括学习策略的可视化、混合数据质量离线数据集的定量结果、与使用预训练基础模型（如 R3M）的比较，以及计算效率分析。
([C](#appendix-c)) 多源 CoWorld 模型的实现细节，以及对所选源域的进一步实证分析。
([D](#appendix-d)) 源域和目标域的详细设置。
([E](#appendix-e)) 对比方法的详细信息。
([F](#appendix-f)) 所提出方法的潜在社会影响。

<a id="appendix-a"></a>

## 附录 A 模型详情

### A.1 CoWorld 框架

如图 [7](https://arxiv.org/html/2305.15260v4#A1.F7) 所示，CoWorld 的整个训练过程包含三个迭代阶段： **离线到在线状态对齐（Stage A）** 、 **在线到离线奖励对齐（Stage B）** 和 **在线到离线价值约束（Stage C）** 。

首先，在阶段 A，我们将从 $\mathcal{B}^{(T)}$ 中采样的相同目标域观测输入编码器，并缩小 $e_{\phi^{\prime}}(o_{t}^{(T)})$ 和 $e_{\phi}(o_{t}^{(T)})$ 之间的距离。

其次，在阶段 B，源奖励预测器 $r_{\phi^{\prime}}(\cdot)$ 使用来自两个经验回放缓冲区 $\mathcal{B}^{(S)}$ 和 $\mathcal{B}^{(T)}$ 的混合数据进行训练。值得注意的是，当我们从 $\mathcal{B}^{(T)}$ 采样数据时，奖励将被重新标记为 **目标信息化的源奖励** 。

最后，在阶段 C，我们引入一个使用源评论家对目标评论家进行 **最小-最大价值约束** 。

<a id="figure-7"></a>

![sen_pic](images/sen_pic.png)

> 图 7 | CoWorld 使用一个辅助的在线环境来构建一个能感知离线域信息的策略“测试平台”。这反过来可以指导离线域中的视觉强化学习（Visual RL）智能体学习一个 **适度保守的策略** ，在价值高估和过度保守之间取得平衡。

关于符号表示，我们使用上标 $S$ 和 $T$ 来表示来自源域和目标域的数据。此外，下标 $(\phi^{\prime},\psi^{\prime},\xi^{\prime})$ 和 $(\phi,\psi,\xi)$ 用于区分不同域的模型参数。源域和目标域的符号表示总结在表 [7](#table-7) 中。

<a id="table-7"></a>

> 表 7 | 源域和目标域的符号表示。

| 域                | 模型参数                                                                | 数据                                                                                               |
| :---------------- | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| 源/在线域 ($S$)   | 世界模型 $\phi^{\prime}$, 行动者 $\psi^{\prime}$, 评论家 $\xi^{\prime}$ | 原始数据 $(o_{t}^{(S)},a_{t}^{(S)},r_{t}^{(S)})$, 结合两域信息重新标记的奖励 $\tilde{r}_{t}^{(S)}$ |
| 目标/离线域 ($T$) | 世界模型 $\phi$, 行动者 $\psi$, 评论家 $\xi$                            | 原始数据 $(o_{t}^{(T)},a_{t}^{(T)},r_{t}^{(T)})$                                                   |

### A.2 世界模型

我们采用了文献 [16] 中使用的世界模型框架。
图像编码器是一个 **卷积神经网络（Convolutional Neural Network, CNN）** 。
图像预测器是一个转置 CNN，而状态转移、奖励和折扣因子预测器是 **多层感知机（Multi-Layer Perceptrons, MLPs）** 。
折扣因子预测器用于估计在基于模型预测学习行为时，一个回合将结束的概率。
编码器和解码器以 $64\times 64$ 的图像作为输入。

### A.3 行为学习

对于 CoWorld 的行为学习，我们采用了来自 DreamerV2 [16] 的 **行动者-评论家方法（Actor-critic method）** 。公式 ([6](https://arxiv.org/html/2305.15260v4#S3.E6)) 中的 $\lambda$-目标 $V_{t}^{(T)}$ 定义如下：

$$
V_{t}^{(T)}\doteq\hat{r}^{(T)}_{t}+\hat{\gamma}_{t}^{(T)}\begin{cases}(1- \lambda)v_{\xi}\left(\hat{z}_{t+1}^{(T)}\right)+\lambda V_{t+1}^{(T)}&\text{if}t<H\\ v_{\xi}\left(\hat{z}_{H}^{(T)}\right)&\text{if}t=H\end{cases},(8)
$$

其中 $\lambda$ 设为 $0.95$，以更多地考虑长期目标。行动者和评论家网络均为使用 **ELU 激活函数（Exponential Linear Unit activation）** [6] 的 **多层感知机（Multilayer Perceptron, MLP）** 。目标行动者和评论家网络在源评论家网络的指导下进行训练，并使用平方损失来回归 $\lambda$-回报。在行为学习期间，世界模型是固定的。源行动者和评论家网络定义为：

$$
\text{源行动者：} \displaystyle\hat{a}_{t}^{(S)}\sim\pi_{\psi^{\prime}}(\hat{a}_{t}^{(S)}|\hat{z}_{t}^{(S)}) \quad (9) \\
\text{源评论家：} \displaystyle v_{\xi^{\prime}}(\hat{z}_{t}^{(S)})\approx\mathbb{E}_{p_{\phi^{\prime}},p_{\psi^{\prime}}}\Big{[}\textstyle\sum_{\tau\geq t}\hat{\gamma}_{\tau-t}^{(S)}\hat{r}_{\tau}^{(S)}\Big{]}.
$$

我们通过最大化以下目标函数来训练源行动者 $\pi_{\psi^{\prime}}$：

$$
\displaystyle\mathcal{L}(\psi^{\prime})=\ \mathbb{E}_{p_{\phi^{\prime}},p_{\psi^{\prime}}}\Big{[}\sum_{t=1}^{H-1}(\underbrace{\beta\mathrm{H}\left[a_{t}^ {(S)}\mid\hat{z}_{t}^{(S)}\right]}_{\text{熵正则化}}+\underbrace {\rho V_{t}^{(S)}}_{\text{动力学反向传播}} (10) \displaystyle+\underbrace{(1-\rho)\ln\pi_{\psi^{\prime}}(\hat{a}_{t}^{(S)}\mid \hat{z}_{t}^{(S)})\texttt{sg}(V_{t}^{(S)}-v_{\xi^{\prime}}(\hat{z}_{t}^{(S)}))}_{\text{REINFORCE}}\Big{]}.
$$

源评论家 $v_{\xi^{\prime}}$ 则通过最小化以下目标函数进行优化：

$$
\displaystyle\mathcal{L}(\xi^{\prime})=\mathbb{E}_{p_{\phi^{\prime}},p_{\psi^{\prime}}}\Big{[}\sum_{t=1}^{H-1}\frac{1}{2}\left(v_{\xi^{\prime}}\left(\hat{z} _{t}^{(S)}\right)-\texttt{sg}\left(V_{t}^{(S)}\right)\right)^{2}\Big{]}.(11)
$$

### A.4 超参数（Hyperparameters）

CoWorld 的超参数如表 [8](#table-8) 所示。

<a id="table-8"></a>

> 表 8: CoWorld 的超参数。

| 名称                       | 符号        | 值                    |                 |
| -------------------------- | ----------- | --------------------- | --------------- |
| 协同训练                   |             | Meta-World / RoboDesk | DMC             |
| 域 KL 损失缩放系数         | $\beta_{2}$ | $1$                   | $1.5$           |
| 目标信息奖励因子           | $k$         | $0.3$                 | $0.9$           |
| 目标评论家价值损失缩放系数 | $\alpha$    | $2$                   | 1               |
| 源域更新迭代次数           | $K_{1}$     | $2\cdot 10^{4}$       | $2\cdot 10^{4}$ |
| 目标域更新迭代次数         | $K_{2}$     | $5\cdot 10^{4}$       | $2\cdot 10^{4}$ |
| 世界模型                   |             |                       |                 |
| 数据集大小                 | —           | $2\cdot 10^{6}$       |                 |
| 批大小                     | $B$         | 50                    |                 |
| 序列长度                   | $L$         | 50                    |                 |
| KL 损失缩放系数            | $\beta_{1}$ | 1                     |                 |
| 离散潜在维度               | —           | 32                    |                 |
| 离散潜在类别数             | —           | 32                    |                 |
| RSSM 单元数                | —           | 600                   |                 |
| 世界模型学习率             | —           | $2\cdot 10^{-4}$      |                 |
| 行为学习                   |             |                       |                 |
| 想象时域                   | $H$         | 15                    |                 |
| 折扣因子                   | $\gamma$    | 0.995                 |                 |
| $\lambda$-目标             | $\lambda$   | 0.95                  |                 |
| 行动者学习率               | —           | $4\cdot 10^{-5}$      |                 |
| 评论家学习率               | —           | $1\cdot 10^{-4}$      |                 |

<a id="appendix-b"></a>

## 附录 B 额外的定量与定性结果

### B.1 策略评估可视化

我们在 Meta-World 和 DMC 任务上评估不同模型的训练智能体，并选取前 $45$ 帧进行比较。图 [8](https://arxiv.org/html/2305.15260v4#A2.F8) 和图 [9](https://arxiv.org/html/2305.15260v4#A2.F9) 分别展示了在 DMC 和 Meta-World 上执行不同模型所学策略的示例。

![framework](images/framework.png)

> (a) 在 DMC Walker Downhill 任务上的策略评估

<a id="figure-9"></a>

![walk_do_cmp](images/walk_do_cmp.png)

> 图 9: 在 Meta-World Button Topdown 任务上的策略评估。无模型方法 CURL 无法完成任务（绿色框）。与 Offline DV2（蓝色框）相比，CoWorld 取得了更好的性能，并以更少的步数完成了任务（红色框）。

### B.2 DMC Medium-Expert 数据集上的定量结果

与 medium-replay 数据集的数据收集策略类似，我们使用 DreamerV2 智能体构建了具有 medium-expert 质量的离线数据集。该 medium-expert 数据集包含训练过程中，直到策略达到专家级性能（定义为达到 DreamerV2 智能体所能获得的最大分数）之前，回放缓冲区中的所有样本。如表 [9](#table-9) 所示，在大多数任务上，CoWorld 在 DMC medium-expert 数据集上的表现优于其他基线方法。

<a id="table-9"></a>

> 表 9: 在 DMC medium-expert 数据集上的性能。我们报告了基于 $3$ 个随机种子、每个种子 $10$ 轮的平均奖励和标准差。

| 模型        | WW $\rightarrow$ WD | WW $\rightarrow$ WU | WW $\rightarrow$ WN | CR $\rightarrow$ CD | CR $\rightarrow$ CU | CR $\rightarrow$ CN | 平均 |
| ----------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------------------- | ---- |
| Offline DV2 | $450\pm 24$         | $141\pm 1$          | $214\pm 8$          | $248\pm 9$          | 3$\pm 0$            | $48\pm 3$           | 184  |
| DrQ+BC      | 808$\pm$47          | 762$\pm$61          | 808$\pm$45          | 862$\pm$13          | 454$\pm$12          | 730$\pm$17          | 737  |
| LOMPO       | 548$\pm$245         | 449$\pm$117         | 688$\pm$97          | 174$\pm$29          | 19$\pm$10           | 113$\pm$35          | 332  |
| Finetune    | 784$\pm$46          | 671$\pm$65          | 851$\pm$91          | 858$\pm$9           | 428$\pm$49          | 833$\pm$7           | 738  |
| CoWorld     | 848$\pm$9           | 774$\pm$29          | 919$\pm$7           | 871$\pm$13          | 475$\pm$16          | 844$\pm$1           | 789  |

### B.3 Meta-World 上的定量结果

图 [10(a)](https://arxiv.org/html/2305.15260v4#A2.F10.sf1) 比较了不同模型在 Meta-World 上的性能。得益于其对源环境的直接访问，DV2 Finetune 在初始训练阶段表现出更好的性能。相反，CoWorld 引入了辅助的源价值指导来协助目标智能体的训练。在训练的最终阶段，源价值指导更为有效，因此 CoWorld 的表现超过了 DV2 Finetune。图 [10(b)](https://arxiv.org/html/2305.15260v4#A2.F10.sf2) 展示了在 Meta-World 上对 CoWorld 进行的消融研究，突出了每个训练阶段的有效性和必要性。

![walk_up_cmp](images/walk_up_cmp.png)

> (a) 与其他基线的比较

### B.4 潜在空间对齐的效果

我们将相同的观测输入到 CoWorld 的源编码器和目标编码器中，然后使用 t-分布随机邻域嵌入（t-distributed stochastic neighbor embedding, t-SNE）方法对潜在状态进行可视化。如图 [11](https://arxiv.org/html/2305.15260v4#A2.F11) 所示，表示学习对齐弥合了源编码器和目标编码器隐藏状态分布之间的差距。

![walk_no_cmp](images/walk_no_cmp.png)

> (a) 对齐前的潜在空间

### B.5 关于现实仿真到真实（Sim2Real）设置的补充结果

由于实验资源有限，我们无法使用真实机器人进行实验。我们努力构建了一个更贴近现实的仿真到真实（Simulation-to-Real, Sim2Real）设置。该实验在源域和目标域中使用相同的机器人控制任务。我们手动在目标域的视觉观测和动作空间中引入了两种类型的噪声，以模拟复杂且嘈杂的真实世界场景。

- **视觉噪声（Visual noise）** ：我们修改了原始的 DeepMind Control（DMC）环境，将静态背景替换为随机真实世界视频的动态背景。
- **动作噪声（Action noise）** ：我们在 Meta-World 和 RoboDesk 环境中，对原本取值范围在 (-1,1) 的动作的每个维度上，添加从 $\mathcal{N}(0,1)$ 中采样的高斯噪声 $n_{t}$。这模拟了使用低成本（精度较低）的真实机器人收集离线数据集的场景。

如表 [10](#table-10) 所示，我们在这种新设置下比较了 CoWorld 与微调后的 DreamerV2（DV2）模型。我们在 Meta-World 的 Button Topdown 任务中应用了三种幅度的噪声，$w\sim\{0.1,1,5\}$，从而得到带噪声的动作 $a_{t}^{\text{real}}=a_{t}+w\cdot n_{t}$。

<a id="table-10"></a>

> 表 10 | 具有更显著域差距的结果。

| 目标域                              | CoWorld | DV2 Finetune |
| :---------------------------------- | :------ | :----------- |
| DMC Walker Walk                     | 544     | 457          |
| DMC Cheetah Run                     | 296     | 220          |
| RoboDesk Push Green ($w=1$)         | 406     | 358          |
| Meta-World Button Topdown ($w=0.1$) | 3752    | 2693         |
| Meta-World Button Topdown ($w=1$)   | 3567    | 3104         |
| Meta-World Button Topdown ($w=5$)   | 951     | 670          |

从以上结果可以明显看出，在这种“仿真到真实（Sim2Real）”设置中，CoWorld 始终优于朴素的微调方法。重要的是，我们在更具挑战性的设置下评估了模型，这些设置具有更显著的域差距，如表 [1](#table-1) 所示。

### B.6 与预训练基础模型 R3M 的比较

R3M [32] 在 Ego4D 人类视频数据集上进行了预训练，有助于高效学习下游机器人任务。R3M 被证明是一个具有竞争力的模型，尤其是在跨具有不同视觉输入的领域迁移表示的能力方面。为了进行模型比较，我们利用官方仓库中的 R3M 预训练权重来初始化表示模型。然后，我们基于此在 Meta-World 环境中为下游任务执行策略优化。我们分别使用了来自官方仓库的专家数据，以及我们自己收集的、数据质量参差不齐的数据。而 DV2 Finetune 也在相关任务上进行了预训练，并在离线数据集上进行了微调。如表 [11](#table-11) 所示，我们的方法优于 R3M / DV2 微调模型。

<a id="table-11"></a>

> 表 11 | CoWorld 与使用预训练基础模型 R3M 的比较。

|                      | CoWorld | R3M（专家数据） | R3M（我们的数据） | DV2 Finetune |
| :------------------- | :------ | :-------------- | :---------------- | :----------- |
| Button Press Topdown | 3889    | 1609            | 311               | 3499         |
| Drawer Close         | 4845    | N/A             | 4616              | 4273         |
| Handle Press         | 4570    | N/A             | 1603              | 3702         |

需要指出的是：

- 尽管 R3M 具有可泛化的表示能力，但它 **并非** 专门为解决 **价值高估问题（value overestimation problem）** 而设计的，而该问题是离线强化学习（Offline Reinforcement Learning, Offline RL）的基础。相比之下，我们的方法不仅跨域对齐了状态表示，还有效解决了价值高估问题，因此获得了更好的性能。
- R3M 的微调过程需要专家演示来进行高质量的模仿学习（Imitation Learning）。然而，经验表明，当应用于中等回放（medium-replay）数据的离线数据集时，其性能会下降。
- R3M 的预训练过程在 V100 GPU 上通常需要大约 $5$ 天，而我们方法的整个训练过程在 3090 GPU 上仅需约 $2$ 天。

### B.7 训练效率

如表 [12](#table-12) 所示，我们使用单块 RTX 3090 GPU 评估了在 Meta-World（Handle Press $\rightarrow$ Button Topdown）上的训练/推理时间。经验上，CoWorld 大约在 $14$ 小时内达到收敛（达到最高回报的 $90\%$）；而 DV2 Finetune 需要大约 $13$ 小时。这些结果表明，CoWorld 所需的训练挂钟时间与 DV2 Finetune 相当，同时在模型收敛后，在回报方面始终保持更好的性能。

<a id="table-12"></a>

> 表 12 | 在 Meta-World（HP $\rightarrow$ BT）上评估的运行时间比较。

| 模型             | 训练迭代次数 | 训练时间  | 每回合推理时间 |
| :--------------- | :----------- | :-------- | :------------- |
| Offline DV2      | 300k         | 2054 分钟 | 2.95 秒        |
| DrQ+BC           | 300k         | 200 分钟  | 2.28 秒        |
| CQL              | 300k         | 405 分钟  | 1.88 秒        |
| CURL             | 300k         | 434 分钟  | 2.99 秒        |
| LOMPO            | 100k         | 1626 分钟 | 4.98 秒        |
| DV2 Finetune     | 460k         | 1933 分钟 | 6.63 秒        |
| DV2 Finetune+EWC | 460k         | 1533 分钟 | 5.58 秒        |
| CoWorld          | 460k         | 3346 分钟 | 4.47 秒        |

<a id="appendix-c"></a>

## **附录 C：多源协同世界模型（Multi-Source CoWorld）**

多源协同世界模型（Multi-Source CoWorld）的核心思想是，通过计算候选源域与目标域在潜在状态空间中的 **KL散度（Kullback–Leibler divergence）** ，为其分配一组独热权重 $\omega_{t}^{i=1:M}$，其中 $i\in[1,M]$ 是每个源域的索引。该过程包括以下步骤：

- 1.  **世界模型预训练（World models pretraining）** ：
      我们分别为每个源域和目标域预训练一个世界模型。
- 2.  **域距离度量（Domain distance measurement）** ：
      在目标域的每个训练步骤中，我们度量目标域的潜在状态（由 $e_{\phi}(o_{t}^{(T)})$ 生成）与每个源域中的对应状态（由 $e_{\phi^{\prime}_{i}}(o_{t}^{(T)})$ 生成）之间的 KL 散度。其中，$e^{(T)}_{\phi}$ 是目标世界模型的编码器，$e_{\phi^{\prime}_{i}}$ 是源域 $i$ 的世界模型的编码器。
- 3.  **辅助域识别（Auxiliary domain identification）** ：我们动态识别具有最小 KL 散度的最接近的源域。我们将 $\omega_{t}^{i=1:M}$ 设为一个独热向量，其中 $\omega_{t}^{i}=1$ 表示选定的辅助域。
- 4.  **剩余训练（Rest of training）** ：利用独热权重，我们继续执行所提出的在线到离线（online-to-offline）训练方法的剩余部分。在表示学习期间，我们通过将公式 ([3](https://arxiv.org/html/2305.15260v4#S3.E3)) 中的域对齐损失项重写为以下形式，来自适应地将目标状态空间与选定的在线模拟器对齐：

$$
\mathcal{L}_{\text{M-S}}=\beta_{2}\sum_{i=1}^{M}\omega_{i}\mathrm{KL}\left[ \texttt{sg}(g(e_{\phi^{\prime}}(o_{t}^{(T)})))\ \|\ g(e_{\phi}(o_{t}^{(T)})) \right].(12)
$$

为了评估多源自适应选择算法的有效性，我们在 **元世界（Meta-World）** 和 **机器人桌面基准测试（RoboDesk Benchmark）** 上进行了实验。对于每个目标任务，使用了两个源任务，包括 CoWorld 表现最佳的任务和 CoWorld 表现最差的任务。此外，对于一些目标任务，还添加了次优的源任务。

如表 [13](#table-13) 所示，多源 CoWorld 能够为大多数多源问题自适应地选择最佳源任务，以确保充分的知识迁移。
多源 CoWorld 的性能报告在表 [2](#table-2) 中。
CoWorld 灵活地适应了具有多个源域的迁移学习场景，取得了与专门使用我们手动指定的辅助模拟器作为源域（最佳源）的模型相当的结果。这项研究显著提高了 CoWorld 在更广泛场景中的适用性。

<a id="table-13"></a>

> 表 13 | 多源协同世界模型自动选择的源域。MW 代表元世界（Meta-World），RD 代表机器人桌面（RoboDesk）。
>
> | 目标域（Target domain）     | 选定的源域（Selected source domain） |
> | :-------------------------- | :----------------------------------- |
> | MW: Door Close              | MW: Drawer Close                     |
> | MW: Button Press            | MW: Handle Press                     |
> | MW: Window Close            | MW: Button Topdown                   |
> | MW: Handle Press            | MW: Button Press                     |
> | MW: Button Topdown          | MW: Handle Press                     |
> | MW: Drawer Close            | MW: Door Close                       |
> | RD: Push Button             | MW: Button Press                     |
> | RD: Open Slide              | MW: Window Close                     |
> | RD: Drawer Open             | MW: Drawer Close                     |
> | RD: Upright Block off Table | MW:<br>Handle Press                  |

<a id="appendix-d"></a>

## 附录 D 源域与目标域

#### Meta-World

对于 **Meta-World** 环境，我们采用具有复杂视觉动态的机器人控制任务。
例如， **关门（Door Close）** 任务要求智能体在随机化门位置的同时关闭一扇带有旋转关节的门，而 **按压把手（Handle Press）** 任务则涉及在随机化把手位置的同时将把手按下。
为了评估 **CoWorld** 在这些任务上的性能，我们在六个视觉强化学习迁移任务中将其与多个基线方法进行了比较。

#### RoboDesk

我们选择 **Meta-World** 作为源域， **RoboDesk** 作为目标域。值得注意的是，这两个环境之间存在显著的 **领域差距（domain gap）** 。
两个环境的视觉观测、物理动态和动作空间均不相同。
首先，Meta-World 采用侧视视角，而 RoboDesk 使用俯视视角。
此外，Meta-World 的动作空间是 $4$ 维的，而 RoboDesk 中的动作空间是 $5$ 维的。
考虑到这些差异， **Meta-World $\rightarrow$ RoboDesk** 基准测试提出了一个具有挑战性的迁移学习问题。

#### DeepMind Control

我们在标准的 **DeepMind Control（DMC）** 环境中训练源智能体，并在修改后的 DMC 环境中训练目标智能体。 **Walker Uphill** 和 **Cheetah Uphill** 代表地面具有 $15^{\circ}$ 上坡坡度的任务。 **Walker Downhill** 和 **Cheetah Downhill** 代表平面具有 $15^{\circ}$ 下坡坡度的任务。
我们在六个具有不同源域和目标域的任务中评估了模型。

我们假设源域和目标域之间存在显著差异（见表 [1](#table-1)）。
我们提出的方法可以弱化这一假设，因为它能缓解不同源域和目标域 **马尔可夫决策过程（Markov Decision Processes, MDPs）** 之间的领域差异。
我们的实验表明， **CoWorld** 方法对跨域差异（包括视觉观测、物理动态、奖励定义，甚至机器人的动作空间）表现出显著的 **容忍度（tolerance）** 。这一特性使得根据机器人类型选择辅助模拟器变得更加方便。例如：

- 当目标域涉及机械臂（例如 RoboDesk）时，可以利用现有的机械臂仿真环境（例如 Meta-World）作为源域。
- 在涉及足式机器人的场景中，像包含类人任务（Humanoid tasks）的 DeepMind Control 这样的环境可以作为合适的辅助模拟器。
- 对于与自动驾驶相关的目标域，可以使用像 **CARLA** 这样的模拟器。

<a id="appendix-e"></a>

## **附录 E 对比方法（Appendix E Compared Methods）**

我们将 CoWorld 与几种广泛使用的基于模型和无模型的离线方法进行比较。

- **离线 DV2（Offline DV2）** [25]：
  一种基于模型的强化学习方法，它将 DreamerV2 [16] 修改为离线设置，并添加了一个与动态集成平均分歧相对应的奖励惩罚项。

- **DrQ+BC** [25]：
  它修改了 DrQ-v2 [50] 中的策略损失项，以匹配 [10] 中给出的损失。

- **CQL** [25]：
  这是一个离线强化学习框架，它学习一个 Q 函数，该函数保证期望策略值的下界低于实际策略值。我们将 CQL 正则化器添加到 DrQ-v2 [21] 的 Q 函数更新中。

- **CURL** [22]：
  一种无模型的强化学习方法，它利用对比学习从原始像素中提取高级特征。

- **LOMPO** [39]：一种基于模型的离线强化学习算法，使用潜在动态模型和不确定性量化来处理高维观测。

- **LOMPO 微调（LOMPO Finetune）** ：它使用源域数据预训练一个 LOMPO 智能体 [39]，随后在离线目标域中对预训练的智能体进行微调。

- **DV2 微调（DV2 Finetune）** ：它在在线源域中预训练一个 DreamerV2 智能体 [16]，随后在离线目标域中对预训练的智能体进行微调。值得注意的是，Meta-World $\rightarrow$ RoboDesk 任务的动作空间不一致，我们无法直接微调。因此，我们使用两个环境的最大动作空间作为共享的策略输出维度。对于 Meta-World 和 Meta-World $\rightarrow$ RoboDesk 迁移任务，我们预训练智能体 $160$k 步，并微调 $300$k 步。对于 DMC 迁移任务，我们预训练智能体 $600$k 步，并微调 $600$k 步。

- **DV2 微调+EWC（DV2 Finetune+EWC）** ：它使用 EWC [19] 修改 DV2 微调方法，以正则化模型，从而保留来自在线源域的知识。预训练和微调的步骤与 DV2 微调保持一致。

<a id="appendix-f"></a>

## **附录 F 更广泛的影响（Appendix F Broader Impacts）**

CoWorld 是一种迁移学习方法，可能有益于未来在离线强化学习（Offline RL）、基于模型的强化学习（Model-based RL）和视觉强化学习（Visual RL）领域的研究。
除了强化学习领域，这种方法在机器人学和自动驾驶等多个领域也具有巨大的贡献潜力。

在医疗保健应用的实际场景中，Zhang 等人 [62] 使用离线强化学习算法，利用大量历史数据集训练策略，以确定肾移植和 HIV 治疗中的随访计划和 **他克莫司（Tacrolimus）** 剂量。此外，还有由医学领域专家设计的相应模拟器 [17, 1]，其参数是从真实世界数据中学习得到的。

所提出设置的另一个实际用途是广告竞价，其中直接与真实的在线广告系统交互进行训练具有挑战性。最近的一种解决方案是基于历史竞价日志构建模拟竞价环境进行交互式训练，例如 [30]，并减轻虚拟广告环境与真实广告系统之间的固有差异。因此，在许多现实场景中，可以利用模拟器来优化从离线数据集中学习到的策略。

我们的方法一个潜在的负面社会影响是引入了来自附加域的现有偏见。如果用于开发我们算法的训练数据包含偏见，模型可能会学习这些偏见，从而导致决策过程中的不公平结果。 **至关重要的是，必须仔细处理数据和算法设计中的偏见，以减轻这些负面的社会影响。**
