# 标题：鲁棒智能体学习因果世界模型（Robust agents learn causal world models）

- ArXiv: 2402.10877
- 作者：Jonathan Richens, Google DeepMind, Tom Everitt
- 章节：65
- 估计词元数：45.9k

## 目录

- 1 引言
  - 论文概要。
- 2 预备知识
  - 2.1 因果模型
    - 定义 1（贝叶斯网络， Bayesian networks）。
    - 定义 2（局部干预， Local interventions）。
    - 定义 3（干预的混合， Mixtures of interventions）。
  - 2.2 决策任务
    - 定义 4（因果影响图， Causal influence diagram）。
    - 假设 1（无中介决策任务， Unmediated decision task）。
  - 2.3 分布偏移
    - 假设 2（领域依赖性， Domain dependence）。
- 3 因果模型是鲁棒适应的必要条件
  - 定理 1。
  - 3.1 放宽最优性假设
    - 定理 2。
    - 定理 3。
  - 3.2 解释
    - 智能体
      - 推论 1。
    - 迁移学习。
    - 因果推断。
- 4 讨论
  - 因果表示学习。
  - 迁移学习的因果界限。
  - 良好调节器定理。
  - 涌现能力。
  - 因果发现。
  - 没有理解就没有能力。
  - 因果方法的适用性。
  - 局限性。
- 5 相关工作
- 6 结论
  - 致谢。
- 参考文献
- 附录 A 预备知识
  - A.1 设置与假设
    - 引理 1。
    - 证明。
  - A.2 因果影响图的参数化
    - 引理 2（Okamoto, 1973）。
  - A.3 分布偏移与策略预言机
    - 定义 5（策略预言机， Policy oracle）。
- 附录 B 附录：简化证明
- 附录 C 定理 1 的证明
  - 引理 3。
  - 证明。
  - 引理 4。
  - 证明。
  - 证明。
- 附录 D 定理 2 的证明
  - 低遗憾分析。
    - 引理 5。
    - 证明。
    - 引理 6。
    - 证明。
    - 证明。
  - 估计 $P$ 的参数。
  - 学习图结构。
- 附录 E 附录：定理 3 的证明
  - 证明。
- 附录 F 实验
- 附录 G 附录：可迁移性与珀尔的因果层级
  - 可迁移性。
  - 因果层级定理。

## 摘要

###### 摘要

长期以来，人们一直假设因果推理在鲁棒和通用智能中起着基础性作用。
然而，尚不清楚智能体是否必须学习因果模型才能泛化到新领域，或者其他归纳偏置是否足够。
我们回答了这个问题，证明了任何能够满足一大类分布偏移的遗憾界限的智能体，必定已经学习了数据生成过程的近似因果模型，并且对于最优智能体，该模型会收敛到真实的因果模型。
我们讨论了这一结果对包括迁移学习和因果推断在内的若干研究领域的影响。

<a id="section-1"></a>

## 1 引言（Introduction）

**通用智能（General Intelligence）** 需要哪些能力（Legg & Hutter, 2007）？一个候选答案是 **因果推理（Causal Reasoning）** ，它在人类认知中扮演着基础性角色（Gopnik et al., 2007; Sloman & Lagnado, 2015）。甚至有人认为，没有因果推理就不可能实现 **人类水平的人工智能（Human-level AI）** （Pearl, 2018）。然而，近年来出现了许多 **智能体（Agents）** ，它们并不显式地学习或基于 **因果模型（Causal Models）** 进行推理，却仍然能够适应广泛的环境和任务（Reed et al., 2022; Team et al., 2023; Brown et al., 2020）。

这就引出了一个问题：智能体为了适应新领域，是否必须学习因果模型，还是其他 **归纳偏置（Inductive Biases）** 就足够了？要回答这个问题，我们必须小心，不能先验地假设智能体使用了因果假设。例如， **可迁移性理论（Transportability Theory）** 确定了当所有关于 **数据生成过程（Data Generating Process）** （即归纳偏置）的假设都可以表达为对 **因果结构（Causal Structure）** 的约束时，迁移学习需要哪些因果知识（Bareinboim & Pearl, 2016）。然而， **深度学习（Deep Learning）** 算法可以利用更广泛的归纳偏置集合（Neyshabur et al., 2014; Battaglia et al., 2018; Rahaman et al., 2019; Goyal & Bengio, 2022），在许多现实场景中，这些偏置可能足以识别出 **低遗憾策略（Low Regret Policies）** ，而无需因果知识。

本文的主要结果就是通过证明以下命题来回答这个问题：

> _任何能够适应足够大规模 **分布偏移（Distributional Shifts）** 集合的智能体，必然已经学习了数据生成过程的因果模型。_

在此，适应分布偏移意味着学习一个策略，该策略在数据生成过程受到干预（例如，改变特征或潜在变量的分布）后，满足一个 **遗憾界（regret bound）** 。已知，数据生成过程的 **因果模型（causal model）** 可用于识别分布偏移后满足遗憾界的策略（充分性），且模型越精确，越能找到遗憾更低的策略。我们证明了其逆命题（必要性）——给定针对大量分布偏移的遗憾界策略，我们可以学习数据生成过程的一个近似因果模型，当策略最优时，该近似会变得精确。因此，学习数据生成过程的因果模型是实现稳健适应的必要条件。

这对许多领域和问题都有影响。首先，它意味着因果识别法则同样约束着 **领域适应（domain adaptation）** 。例如，我们证明，只有当特征与标签之间的因果关系能够从训练数据中被识别时（一个非平凡的 **因果发现（causal discovery）** 问题），适应 **协变量偏移（covariate shift）** 和 **标签偏移（label shift）** 才成为可能。这为 **因果表示学习（causal representation learning）** (Schölkopf et al., 2021) 提供了进一步的理论依据，表明学习因果表示是实现强鲁棒性保证的必要条件。我们的结果还意味着，我们可以从自适应智能体中学习因果模型。我们通过在合成数据上观察一个在分布偏移下具有遗憾界的智能体的策略，来解决一个因果发现任务，从而证明了这一点。更具推测性地，我们的结果表明因果模型可能在 **涌现能力（emergent capabilities）** 中发挥作用。被训练以最小化跨多个领域损失函数的智能体，有动力去学习一个因果世界模型，这反过来可能使它们能够解决一系列更广泛的、未经明确训练的决策任务。

#### 论文大纲（Outline of paper）。

在 [第 2 节](#section-2) 中，我们介绍了用于推导结果的因果性和决策理论概念。我们在 [第 3 节](#section-3) 中呈现主要理论结果，并从自适应智能体、迁移学习和因果推断的角度讨论其解释。在 [第 4 节](#section-4) 中，我们讨论局限性，以及对若干领域和开放问题的影响。在 [第 5 节](#section-5) 中，我们讨论相关工作，包括 **可迁移性（transportability）** (Bareinboim & Pearl, 2016) 和 **因果层次定理（causal hierarchy theorem）** (Bareinboim et al., 2022)，以及关于涌现世界模型的最新实证研究。在 [附录 B](#appendix-b) 中，我们描述了将理论结果应用于因果发现问题的实验。

<a id="section-2"></a>

## 2 预备知识

<a id="section-2-1"></a>

### 2.1 因果模型

我们使用大写字母表示随机变量 $V$，小写字母表示其取值 $v\in\textit{dom}(V)$。
为简化起见，我们假设每个变量都有有限个可能取值，即 $|\textit{dom}(V)|<\infty$。
粗体表示变量集合 $\bm{V}=\{V_{1},\dots,V_{n}\}$ 及其取值 $\bm{v}\in\textit{dom}(\bm{V})=\times_{i}\textit{dom}(V_{i})$。
一个 **概率模型（Probabilistic model）** 规定了变量集合 $\bm{V}$ 上的联合分布 $P(\bm{V})$。
这些模型支持关联性查询，例如对于 $\bm{X},\bm{Y}\subseteq\bm{V}$，查询 $P(\bm{Y}=\bm{y}\mid\bm{X}=\bm{x})$。
**干预（Interventions）** 描述了对数据生成过程的外部改变（从而改变了联合分布），例如一个 **硬干预（Hard intervention）** $\textrm{do}(\bm{X}=\bm{x})$ 描述了强制变量集合 $\bm{X}\subseteq\bm{V}$ 取值为 $\bm{x}$。
这产生了一个新的分布 $P(\bm{V}\mid\textrm{do}(X=x))=P(\bm{V}_{x})$，其中 $\bm{V}_{\bm{x}}$ 指的是在此干预之后的变量 $\bm{V}$。
**因果模型（Causal models）** 的强大之处在于，它们不仅规定了 $P(\bm{V})$，还规定了在所有干预下 $\bm{V}$ 的分布，因此这些模型可用于评估关联性和干预性查询，例如 $P(\bm{Y}=\bm{y}\mid\textrm{do}(\bm{X}=\bm{x}))$。

为了推导我们的结果，我们专注于一类特定的因果模型—— **因果贝叶斯网络（Causal Bayesian Networks, CBNs）** 。
文献中还研究了其他几种替代模型和形式体系，包括 **结构方程模型（Structural Equation Models）** （Pearl, 2009）和 **内曼-鲁宾因果模型（Neyman-Rubin Causal Models）** （Rubin, 2005），我们的结果可以很容易地适用于这些模型。

###### 定义 1（贝叶斯网络）.

在一组变量 $\bm{V}=\{V_{1},\dots,V_{n}\}$ 上的一个 **贝叶斯网络（Bayesian network）** $M=(G,P)$ 是一个联合概率分布 $P(\bm{V})$，它根据一个有向无环图（Directed Acyclic Graph, DAG）$G$ 进行分解，即 $P(V_{1},\dots,V_{n})=\prod_{i=1}^{n}P(V_{i}\mid\textbf{Pa}_{V_{i}})$，其中 $\textbf{Pa}_{V_{i}}$ 是 $V_{i}$ 在 $G$ 中的父节点。

一个贝叶斯网络是 **因果的（Causal）** ，如果图 $G$ 捕捉了变量之间的因果关系，或者更正式地说，如果对于 $\bm{X}\subseteq\bm{V}$ 的任何干预 $\textrm{do}(\bm{X}=\bm{x})$ 的结果都可以通过截断分解公式计算：

$$
P(\bm{v}\mid\textrm{do}(\bm{x}))=\begin{cases}\prod_{i:v_{i}\not\in\bm{x}}P(v_ {i}\mid\textbf{pa}_{v_{i}})&\text{如果 \bm{v} 与 \bm{x} 一致}\\ 0&\text{否则。}\end{cases}
$$

更一般地，一个 **软干预（Soft intervention）** $\sigma_{v_{i}}=P^{\prime}(V_{i}\mid\textbf{Pa}^{*}_{i})$ 将 $V_{i}$ 的条件概率分布替换为一个新的分布 $P^{\prime}(V_{i}\mid\textbf{Pa}^{*}_{i})$，只要图中不引入环，可能会导致一个新的父节点集合 $\textbf{Pa}^{*}_{i}\neq\textbf{Pa}_{i}$。我们将 $\sigma_{v_{i}}$ 称为一个 **领域指示符（Domain indicator）** （Correa & Bareinboim, 2020）（它也被称为环境索引，Arjovsky et al., 2019）。
更新后的分布表示为
$P(\bm{v};\sigma_{\bm{v}^{\prime}})=\prod_{i:v_{i}\in\bm{v}^{\prime}}P^{\prime}
(v_{i}\mid\textbf{pa}^{*}_{v_{i}})\prod_{i:v_{i}\not\in\bm{v}^{\prime}}P(v_{i}
\mid\textbf{pa}_{v_{i}})$。

一般来说，如果没有 $G$ 的知识，就无法定义软干预。
例如，软干预 $\sigma_{Y}=P^{\prime}(y\mid x)$ 与因果结构 $Y\rightarrow X$ 不兼容，因为它会引发因果循环。
由于我们的结果关注于学习因果模型（从而学习因果结构），我们将理论分析集中在软干预的一个子集—— **局部干预（Local interventions）** 上，它们与所有因果结构兼容，因此可以在不默认假设已知 $G$ 的情况下使用。

###### 定义 2（局部干预）.

对 $V_{i} \in \bm{V}$ 的 **局部干预（Local intervention）** $\sigma$ 涉及对 $V_{i}$ 的状态应用一个不依赖于任何其他内生变量（Endogenous variables）的映射，即 $v_{i} \mapsto f(v_{i})$。
我们使用符号 $\sigma = \textrm{do}(V_{i} = f(v_{i}))$（变量 $V_{i}$ 被赋予状态 $f(v_{i})$）。
形式上，这是对 $V_{i}$ 的 **软干预（Soft intervention）** ，它将条件概率分布变换为：

$$
P(v_{i}\,|\,\textbf{pa}_{i};\sigma)=\sum_{v_{i}\textquoteright:f(v_{i}^{\prime})=v_{i}}P(v_{i}^{\prime}\,|\,\textbf{pa}_{i})(1)
$$

**示例** ： **硬干预（Hard interventions）** $\textrm{do}(V_{i}=v^{\prime}_{i})$ 是 $f(v_{i})$ 为常数函数的局部干预。

**示例** ：平移是局部干预，形式为 $\textrm{do}(V_{i}=v_{i}+k)=\textrm{do}(V_{i}=f(v_{i}))$，其中 $f(v_{i})=v_{i}+k$。例子包括改变强化学习（Reinforcement Learning, RL）环境（Shah et al., 2022）和图像（Engstrom et al., 2019）中物体的位置。

**示例** ：对于布尔变量 $X$，逻辑非（Logical NOT）操作 $X\mapsto\neg X$ 是一种局部干预。

我们也考虑 **随机干预（Stochastic interventions）** ，并指出 **局部干预的混合（Mixtures of local interventions）** 也可以在不知道图 $G$ 的情况下定义。
例如，向变量 $X$ 添加噪声 $X = X + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,1)$，就是对 $X$ 的一种软干预，它由局部干预（平移）的混合来描述。

###### 定义 3（干预的混合）.

一个 **混合干预（Mixed intervention）** $\sigma^{*} = \sum_{i} p_{i} \sigma_{i}$，其中 $\sum p_{i} = 1$，表示以概率 $p_{i}$ 执行干预 $\sigma_{i}$。
形式上，
$P(\bm{v} \mid \sigma^{*}) = \sum_{i} p_{i} P(\bm{v} \mid \sigma_{i})$。

###### 定义 1（贝叶斯网络）.

###### 定义 2（局部干预）.

###### 定义 3（干预的混合）.

<a id="section-2-2"></a>

### 2.2 决策任务（Decision tasks）

对 $V_{i} \in \bm{V}$ 的 **局部干预（Local intervention）** $\sigma$ 涉及对 $V_{i}$ 的状态应用一个不依赖于任何其他内生变量（Endogenous variables）的映射，即 $v_{i} \mapsto f(v_{i})$。
我们使用符号 $\sigma = \textrm{do}(V_{i} = f(v_{i}))$（变量 $V_{i}$ 被赋予状态 $f(v_{i})$）。
形式上，这是对 $V_{i}$ 的 **软干预（Soft intervention）** ，它将条件概率分布变换为：

$$
P(v_{i}\,|\,\textbf{pa}_{i};\sigma)=\sum_{v_{i}\textquoteright:f(v_{i}^{\prime})=v_{i}}P(v_{i}^{\prime}\,|\,\textbf{pa}_{i})(1)
$$

**示例** ： **硬干预（Hard interventions）** $\textrm{do}(V_{i}=v^{\prime}_{i})$ 是 $f(v_{i})$ 为常数函数的局部干预。

**示例** ：平移是局部干预，形式为 $\textrm{do}(V_{i}=v_{i}+k)=\textrm{do}(V_{i}=f(v_{i}))$，其中 $f(v_{i})=v_{i}+k$。例子包括改变强化学习（Reinforcement Learning, RL）环境（Shah et al., 2022）和图像（Engstrom et al., 2019）中物体的位置。

**示例** ：对于布尔变量 $X$，逻辑非（Logical NOT）操作 $X\mapsto\neg X$ 是一种局部干预。

我们也考虑 **随机干预（Stochastic interventions）** ，并指出 **局部干预的混合（Mixtures of local interventions）** 也可以在不知道图 $G$ 的情况下定义。
例如，向变量 $X$ 添加噪声 $X = X + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,1)$，就是对 $X$ 的一种软干预，它由局部干预（平移）的混合来描述。

###### 定义 3（干预的混合）.

一个 **混合干预（Mixed intervention）** $\sigma^{*} = \sum_{i} p_{i} \sigma_{i}$，其中 $\sum p_{i} = 1$，表示以概率 $p_{i}$ 执行干预 $\sigma_{i}$。
形式上，
$P(\bm{v} \mid \sigma^{*}) = \sum_{i} p_{i} P(\bm{v} \mid \sigma_{i})$。

###### 定义 1（贝叶斯网络）.

###### 定义 2（局部干预）.

###### 定义 3（干预的混合）.

**决策任务（Decision tasks）** 涉及一个 **决策者（decision maker， 智能体/agent）** 选择一项 **策略（policy）** ，以优化一个 **目标函数（objective function， 效用/utility）** 。为了对决策任务进行因果描述，我们使用 **因果影响图（Causal Influence Diagram, CID）** 形式体系（Howard & Matheson, 2005; Everitt et al., 2021）[1, 2]，它通过引入决策节点和效用节点，扩展了环境（机会）变量的 **因果贝叶斯网络（Causal Bayesian Network, CBN）** （示例见[图 1](#figure-1)）。为简化起见，我们专注于涉及 **单一决策（single decision）** 和 **单一效用函数（single utility function）** 的任务。

###### 定义 4：因果影响图（Causal influence diagram）

一个（单决策、单效用的） **因果影响图（Causal Influence Diagram, CID）** 是一个 CBN $M=(G,P)$，其中变量 $\bm{V}$ 被划分为决策变量、效用变量和机会变量，即 $\bm{V}=(\{D\},\{U\},\bm{C})$。效用变量是其父节点的实值函数，即 $U(\textbf{pa}_{U})$。

单决策单效用 CID 可以表示大多数决策任务，例如分类和回归，因为它们规定了应基于何种信息（$\textbf{pa}_{D}$）做出何种决策（$d\in D$），以及目标（$\mathbb{E}[U]$）。它们也可以描述一些多决策任务，例如 **马尔可夫决策过程（Markov Decision Processes, MDPs）** （注1：马尔可夫决策过程可以通过将策略选择建模为单一决策、将累积折扣奖励建模为单一效用变量，从而表述为单决策单效用 CID。）。效用可以是任何实值函数，包括标准的损失和奖励函数。

我们假设环境由一组随机变量 $\bm{C}$ 描述，这些变量通过 **因果机制（causal mechanisms）** 相互作用（注2：此假设源自 Reichenbach (1956) [3]，我们将在[附录 A.3 节](https://arxiv.org/html/2402.10877v7#A1.SS3)进一步讨论），并且 $\bm{C}$ 满足 **因果充分性（causal sufficiency）** （Pearl, 2009）[4]（包含所有共同原因），并注意到这样的 $\bm{C}$ 选择总是存在的。我们将 $\bm{C}$ 上的 CBN 称为“真实”或“底层” CBN。请注意，我们 **不** 假设智能体对底层 CBN 有任何了解，也 **不** 假设 $\bm{C}$ 中的哪些变量被智能体观测到或未观测到，仅假设智能体可以观测到 $\textbf{Pa}_{D}\subseteq\bm{C}$。我们还假设已知效用函数 $U(\textbf{Pa}_{U})$。

决策节点的条件概率分布 $\pi(d\mid\textbf{pa}_{D})$（即策略）不是模型的固定参数，而是由智能体设定以最大化其期望效用。对于一个策略 $\pi$，其期望效用为 $\mathbb{E}^{\pi}[U]=\mathbb{E}[U\mid\textrm{do}(D=\pi(\textbf{pa}_{D}))]$。如果一个策略 $\pi^{*}$ 能最大化 $\mathbb{E}^{\pi^{*}}[U]$，则称其为 **最优（optimal）** 策略。通常，智能体的行为并非最优，并会招致一些 **遗憾（regret）** $\delta$，即与最优策略相比期望效用的减少量：$\delta:=\mathbb{E}^{\pi^{*}}[U]-\mathbb{E}^{\pi}[U]$。

为了简化我们的理论分析，我们专注于一类被广泛研究的决策任务，其中智能体的决策不会因果性地影响环境（例如[图 1](#figure-1)）。

###### 假设 1：无中介决策任务（Unmediated decision task）

$\textbf{Desc}_{D}\cap\textbf{Anc}_{U}=\emptyset$。

在 **无中介决策任务（unmediated decision tasks）** 中，智能体获得环境的某些（部分）观测值，并选择一项策略，然后使用效用函数（该函数是环境状态和智能体决策的函数）对该策略进行评估。无中介决策任务的例子包括分类和回归等预测任务，而我们的定理未涵盖的 **有中介决策任务（mediated decision tasks）** 的例子包括马尔可夫决策过程，其中智能体的决策（行动）通过环境状态影响效用。

![S2.F1](images/S2.F1.svg)

> (a) 训练

**参考文献（References）**

1.  Howard, R. A., & Matheson, J. E. (2005). Influence diagrams. _Decision Analysis_, 2(3), 127–143.
2.  Everitt, T., Hutter, M., Kumar, R., & Krakovna, V. (2021). Reward tampering problems and solutions in reinforcement learning: A causal influence diagram perspective. _Synthese_, 198(Suppl 27), 6435–6467.
3.  Reichenbach, H. (1956). _The direction of time_. University of California Press.
4.  Pearl, J. (2009). _Causality: Models, reasoning, and inference_ (2nd ed.). Cambridge University Press.

###### 定义 4（因果影响图， Causal influence diagram）.

###### 假设 1（无中介决策任务， Unmediated decision task）.

<a id="section-2-3"></a>

### 2.3 分布偏移（Distributional shifts）

我们关注的是超越 **独立同分布（independent and identically distributed, iid）** 假设的泛化，即智能体在与训练环境存在分布偏移的领域中进行评估。
分布偏移可以是环境的变化（ **领域偏移， domain shifts** ），例如在领域适应（domain adaptation）和领域泛化（domain generalisation）中（Farahani 等人， 2021； Wilson & Cook， 2020）；也可以是目标的变化（ **任务偏移， task shifts** ），例如在零样本学习（zero shot learning）（Xian 等人， 2018）、上下文学习（in-context learning）（Brown 等人， 2020）和多任务强化学习（multi-task reinforcement learning）（Reed 等人， 2022）中。
我们的分析聚焦于涉及因果数据生成过程变化的领域偏移，因此可以建模为 **干预（interventions）** （Schölkopf 等人， 2021）。
这并非假设智能体将遇到的所有偏移都能建模为干预，而是要求智能体至少能够适应这些偏移。

由干预产生的偏移示例包括：图像中物体的平移（Engstrom 等人， 2019）、输入加噪和对抗鲁棒性（Hendrycks & Dietterich， 2019），以及马尔可夫决策过程（Markov Decision Processes, MDPs）中初始条件或转移函数的变化（Peng 等人， 2018）。
不易自然表示为干预的偏移示例包括：改变环境变量集合 $\bm{C}$，以及引入选择偏差（selection biases）（Shen 等人， 2018）。讨论详见 [A.3 节](https://arxiv.org/html/2402.10877v7#A1.SS3)。

我们的主要结果仅限于 **局部领域偏移（local domain shifts）** ，这对应于对机会变量 $\bm{C}$ 的 **局部干预（local interventions）** 。
我们不考虑改变智能体决策 $D$ 的偏移，但我们将那些丢弃策略输入 $\textbf{Pa}_{D}\rightarrow\textbf{Pa}_{D}^{\prime}\subseteq\textbf{Pa}_{D}$（例如掩码）的偏移视为局部干预。
我们不考虑任务偏移，即改变效用函数。

由于我们关注的是确定领域适应所需的能力，因此我们将注意力限制在领域适应非平凡（即最优策略依赖于环境分布 $P(\bm{C}=\bm{c})$）的决策任务上。

###### 假设 2（领域依赖性， Domain dependence）.

存在与 $M$ 兼容的分布 $P(\bm{C}=\bm{c})$ 和 $P^{\prime}(\bm{C}=\bm{c})$，使得 $\pi^{*}=\operatorname*{arg\,max}_{\pi}\mathbb{E}_{P}^{\pi}[U]$ 意味着 $\pi^{*}\neq\operatorname*{arg\,max}_{\pi}\mathbb{E}_{P^{\prime}}^{\pi}[U]$。

假设 2 意味着存在会改变最优策略的领域偏移。

###### 假设 2（领域依赖性， Domain dependence）.

<a id="section-3"></a>

## 3 因果模型是实现稳健适应的必要条件（Causal models are necessary for robust adaptation）

我们现在以最一般的形式呈现我们的结果——学习底层 **因果贝叶斯网络（Causal Bayesian Network, CBN）** 与学习针对局部领域转移的 **遗憾有界（regret bounded）** 策略是等价的。随后，在[第 3.2 节](#section-3-2)中，我们将我们的定理应用于三种场景： **自适应智能体（adaptive agents）** 、 **迁移学习（transfer learning）** 和 **因果推断（causal inference）** ，并证明 **稳健的智能体（robust agents）** 必须学习 **因果世界模型（causal world models）** 。

首先，我们关注理想化的情况，即我们假设策略是最优的。我们证明，对于几乎所有决策任务，给定针对大量领域转移的最优策略，可以重建底层的 CBN。

###### 定理 1（Theorem 1）。

对于满足假设 1 和 2 的几乎所有 **因果影响图（Causal Influence Diagram, CID）** $M=(G,P)$，给定策略集合 $\{\pi^{*}_{\sigma}(d\mid\textbf{pa}_{D})\}_{\sigma\in\Sigma}$，我们可以识别出效用 $U$ 的所有祖先变量 $\textbf{Anc}_{U}$ 上的 **有向无环图（Directed Acyclic Graph, DAG）** $G$ 和联合分布 $P$。其中，$\pi^{*}_{\sigma}(d\mid\textbf{pa}_{D})$ 是领域 $\sigma$ 中的最优策略，$\Sigma$ 是所有局部干预混合的集合。证明见[附录 C](#appendix-c)。

底层 CBN 的参数 $P(v_{i}\mid\textbf{pa}_{i})$ 和 $U(\textbf{pa}_{U})$ 定义了一个参数空间，而“对于几乎所有 CID”这一条件意味着，在该参数空间中，[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 不成立的子集其 **勒贝格测度（Lebesgue measure）** 为零（讨论见[第 A.2 节](https://arxiv.org/html/2402.10877v7#A1.SS2)）。这个条件是必要的，因为存在一些经过精细调整的环境，其中由于某些变量 $X\in\textbf{Anc}_{U}$ 不影响期望效用，导致仅凭智能体的策略无法识别出 CBN。例如，考虑 $X\rightarrow Y\rightarrow U$，$Y=\mathcal{N}(0,x)$ 且 $U=D+Y$，那么改变 $X$ 只能改变 $U$ 的方差，而其期望值（以及因此的最优策略）保持不变。然而，这种情况仅发生在参数 $P$ 和 $U$ 的非常特定的选择下。

在[附录 B](#appendix-b) 中，我们通过一个实例给出了证明的简化概述。我们假设存在一个 **预言机（oracle）** ，能够为 $\bm{C}$ 上的任何给定局部干预提供最优策略 $\pi^{*}_{\sigma}$。请注意，这假设智能体对一组 **因果充分（causally sufficient）** 的变量 $\bm{C}$ 上的分布偏移具有稳健性，而非假设智能体观察到的变量集是因果充分的。我们设计了一种算法，用不同的局部干预混合来查询此预言机，并识别出最优策略发生变化的那些混合。然后我们证明，这些关键的混合识别了 CBN 的参数，即确定了图 $G(\textbf{Anc}_{U})$ 和联合分布 $P(\textbf{Anc}_{U})$。

<a id="section-3-1"></a>

### 3.1 放宽最优性假设（Relaxing the assumption of optimality）

我们现在放宽最优性假设，考虑策略 $\pi_{\sigma}$ 满足一个遗憾界 $\mathbb{E}^{\pi_{\sigma}}[U]\geq\mathbb{E}^{\pi^{*}_{\sigma}}[U]-\delta$ 的情况。我们证明，对于 $\delta>0$，我们可以恢复环境 CBN 的一个近似，其误差在 $\delta\ll\mathbb{E}^{\pi^{*}}[U]$ 时随 $\delta$ 线性增长。

###### **定理 2（Theorem 2）** 。

对于几乎所有满足 **假设 1（Assumption 1）** 和 **假设 2（Assumption 2）** 的 **因果影响图（Causal Influence Diagram, CID）** $M=(G,P)$，给定一组 **局部干预（local interventions）** 的混合集合 $\Sigma$ 下的策略集 $\{\pi_{\sigma}(d\mid\textbf{pa}_{D})\}_{\sigma\in\Sigma}$，其中 $\mathbb{E}^{\pi_{\sigma}}[U]\geq\mathbb{E}^{\pi^{*}_{\sigma}}[U]-\delta$，我们可以识别出一个近似的因果模型 $M^{\prime}=(P^{\prime},G^{\prime})$。
$M^{\prime}$ 的参数满足 $\left|P^{\prime}(v_{i}\mid\textbf{pa}_{i})-P(v_{i}\mid\textbf{pa}_{i})\right|
\leq\gamma(\delta)$，对于所有 $V_{i}\in\bm{V}$ 成立，其中 $\gamma(0)=0$，并且对于远小于 $\mathbb{E}^{\pi^{*}}[U]$ 的小遗憾值 $\delta\ll\mathbb{E}^{\pi^{*}}[U]$，$\gamma(\delta)$ 随 $\delta$ 线性增长。证明见[附录 D](#appendix-d)。

参数误差的最坏情况误差界 $\gamma(\delta)$ 在[附录 D](#appendix-d) 中有详细说明。
对于 $\delta>0$，可能无法完美地识别 $G$，因为由于这些误差界，一些弱的因果关系无法被解析。
我们在[附录 D](#appendix-d) 中描述了如何学习一个可能排除对应于弱因果关系的有向边的子图 $G^{\prime}\subseteq G$。

**定理 2** 表明，我们可以从领域偏移下的 **遗憾有界策略（regret bounded policies）** 中学习到数据生成过程的（稀疏）近似因果模型，且当 $\delta\rightarrow 0$ 时，近似变得精确。
在[附录 F](#appendix-f) 中，我们使用模拟数据，针对类似于 **图 1** 的随机生成的 CID，演示了如何从遗憾有界策略中学习底层的 **因果贝叶斯网络（Causal Bayesian Network, CBN）** ，并探讨了近似 CBN 的准确性如何随遗憾界变化（ **图 2** ）。

![G_graph](images/G_graph.png)

> (a) 学习到的有向无环图（DAG）的错误率与遗憾界的关系

最后，我们证明了充分性，即拥有数据生成过程的（近似）因果模型足以识别遗憾有界策略。
对于非近似情况，这一结果是众所周知的（Bareinboim & Pearl, 2016）。

###### **定理 3（Theorem 3）** 。

给定一个因果充分的 CBN $M=(P,G)$，对于任意给定的效用 $U$（其中 $\textbf{Pa}_{U}\subseteq\bm{C}$）以及所有 **软干预（soft interventions）** $\sigma$，我们可以识别出最优策略 $\pi^{*}_{\sigma}(d\mid\textbf{pa}_{D})$。
给定一个近似因果模型 $M^{\prime}=(P^{\prime},G^{\prime})$，其满足 $\left|P^{\prime}(v_{i}\mid\textbf{pa}_{i})-P(v_{i}\mid\textbf{pa}_{i})\right|
\leq\epsilon\ll 1$，我们可以识别出遗憾有界策略，其中遗憾 $\delta$ 随 $\epsilon$ 线性增长。证明见[附录 E](#appendix-e)。

**定理 2** 和 **定理 3** 共同表明，学习数据生成过程的近似因果模型，对于在局部干预下学习遗憾有界策略是 **必要且充分** 的。

###### 定理 3（Theorem 3）。

<a id="section-3-2"></a>

### 3.2 解释（Interpretation）

在本节中，我们将通过三个视角来解释[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1)、[定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)和[定理 3](https://arxiv.org/html/2402.10877v7#Thmtheorem3)： **智能体（Agents）** 、 **迁移学习（Transfer Learning）** 和 **因果推断（Causal Inference）** 。

首先，我们推导出我们的结果： **任何能够适应局部领域偏移的智能体，必定已经学习了数据生成过程的因果模型。**

#### 智能体（Agents）

它们是 **自适应目标导向系统（adaptive goal-directed systems）** ，这意味着它们选择行动以实现某种期望的结果，并且如果它们知道自身行动的后果已经改变，就会改变其行为（Dennett, 1989），即，在经历 **领域偏移（domain shift）** （Kenton 等人, 2023）之后。例如，一家公司设定价格以最大化利润，并调整其定价以适应供需变化。

我们将 **适应（adaptation）** 定义为在分布偏移的环境中有效追求目标（最小化遗憾）的能力。适应是通过 **泛化（generalisation）** ——将在训练期间学到的知识应用于偏移环境——与在偏移环境中 **重新训练（re-training）** 相结合来实现的。

我们关注 **零样本设定（zero-shot setting）** （Kirk 等人, 2023），它描述了无需重新训练即能适应的强大智能体，即，仅利用对环境及其所发生变化（偏移）的知识。我们的目标是精确确定支持这种能力需要哪些环境知识。

因此，我们专注于适应已知分布偏移这一简化任务，即，智能体将其策略 $\pi_{\sigma}$ 以 $\sigma$ 为条件，$\pi_{\sigma}=\pi(d\mid\textbf{pa}_{D},\sigma)$（注3：$\sigma$ 等价于（上下文）不变风险最小化（Gupta 等人, 2023; Arjovsky 等人, 2019）的环境索引，或零样本强化学习（Kirk 等人, 2023）的上下文变量）。请注意，这严格比适应未知分布偏移更容易，因此，适应已知分布偏移所需的任何知识，也是适应未知偏移所必需的（注4：任何能够在没有输入 $\sigma$ 的情况下返回遗憾有界策略的智能体，在给定 $\sigma$ 时，只需丢弃 $\sigma$ 即可做到。因此，我们可以平凡地将该智能体的策略扩展为包含输入 $\sigma$，该策略仍将满足遗憾界，从而允许应用[定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)。同样地，如果智能体确实以其策略为条件，但条件是推断出的 $\sigma^{\prime}$（关于实现的综述，请参见 Kirk 等人 (2023)），并且这个以 $\sigma^{\prime}$ 为条件的策略满足遗憾界，我们也可以应用[定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)。）。

###### **推论 1（Corollary 1）**

设 $\pi(d\mid\textbf{pa}_{D},\sigma)$ 为一个智能体（Agent）的策略（Policy），该策略在由满足 [假设 1](https://arxiv.org/html/2402.10877v7#Thmassumption1) 和 [假设 2](https://arxiv.org/html/2402.10877v7#Thmassumption2) 的 **因果影响图（Causal Influence Diagram, CID）** $M=(G,P)$ 所描述的环境中，对于所有局部领域偏移（Local Domain Shifts）$\sigma$，均满足遗憾界（Regret Bound）$\mathbb{E}^{\pi(\sigma)}[U]\geq\mathbb{E}^{\pi^{*}}[U]-\delta$。
根据 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 和 [定理 3](https://arxiv.org/html/2402.10877v7#Thmtheorem3)，$\pi$ 在信息上等价于 $M$ 的一个近似 $M^{\prime}$，且当 $\delta\rightarrow 0$ 时，$M^{\prime}\rightarrow M$ 平滑地收敛。

[推论 1](https://arxiv.org/html/2402.10877v7#Thmcorollary1) 通过将 $\pi_{\sigma}(d\mid\textbf{pa}_{D})=\pi(d\mid\textbf{pa}_{D},\sigma)$ 代入 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 即可直接得出。
如果 $\pi_{\sigma}$ 对所有局部偏移 $\sigma$ 都满足一个紧致的遗憾界，那么我们仅从智能体的策略出发（遵循 [附录 C](#appendix-c) 中的步骤）就能重建底层的 **因果贝叶斯网络（Causal Bayesian Network, CBN）** 。
因此， **任何能够在已知局部领域偏移下进行泛化的智能体，必然已经学习了底层的 CBN** 。
具体而言，智能体学习了策略 $\pi(d\mid\textbf{pa}_{D},\sigma)$，该策略在信息上等价于环境的一个因果模型（Causal Model），因为任何能够使用环境的真实因果模型识别的关联性（Associative）或因果性（Causal）查询，同样可以使用 $\pi(d\mid\textbf{pa}_{D},\sigma)$ 来识别。

**示例** ：医生（Doctors）是被期望能在广泛的已知分布偏移（Known Distributional Shifts）下做出低遗憾决策的智能体，而无需在偏移后的环境中重新训练。
例如，考虑基于患者的体征和病史进行风险分层（Risk-stratifying）的任务。
医生可能被调到一个新的病房，那里的患者接受过一种治疗（已知分布偏移），该治疗对潜在变量（Latent Variables）具有随机效应（混合干预，Mixed Intervention），例如治愈疾病和引起副作用。
医生无法在这个新领域重新训练，例如，采取随机决策并观察结果。
为了具备这种适应性，[定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 意味着医生必须知道相关潜在变量之间因果关系的良好近似——治疗如何影响疾病、这些疾病与其症状之间如何因果关联，等等。
同样地，任何希望复制这种能力的医疗人工智能（Medical AI）也必须学习一个类似精确的因果模型，并且智能体的性能越好，其因果模型就必须越精确。

###### 推论 1（Corollary 1）。

#### 迁移学习（Transfer learning）。

在迁移学习（Zhuang et al., 2020）中，模型在一组源域（source domains）上进行训练，并在保留的目标域（target domains）上进行评估，其中 i) 数据分布与源域不同，且 ii) 可用于训练的数据与源域相比受到限制（Wang et al., 2022）。
例如，在无监督域适应（unsupervised domain adaptation）中，学习器仅限于来自目标域的输入特征样本 $\textbf{pa}_{D}\sim P(\textbf{Pa}_{D};\sigma)$，而在域泛化（domain generalisation）中，训练期间通常无法获得来自目标域的数据（Farahani et al., 2021）。

令 $\mathcal{D}_{S}$ 表示来自源域的训练数据，$\mathcal{D}_{\sigma}$ 表示来自给定目标域 $\sigma$ 的可用训练数据。
假设存在一个迁移学习算法，在给定此训练数据的情况下，能返回一个满足给定目标域 $\sigma$ 遗憾界（regret bound）的策略 $\pi_{\sigma}$。
由于 $\pi_{\sigma}$ 是训练数据的函数，那么根据 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)，该算法的存在意味着我们可以从 $\mathcal{D}_{S}\cup\{\mathcal{D}_{\sigma}\}_{\sigma\in\Sigma}$ 中识别出底层的因果贝叶斯网络（Causal Bayesian Network, CBN）。
为了理解这对迁移学习算法的存在施加了非平凡的约束，我们可以考虑以下简单示例。

示例：考虑 [图 1](#figure-1) 中描绘的监督学习任务对应的因果影响图（Causal Influence Diagram, CID）。
令 $\mathcal{D}_{S}=\{(x^{i},y^{i})\sim P(X,Y)\}_{i=1}^{n}$，因此对于足够大的 $n$，智能体可以从 $\mathcal{D}_{S}$ 中学习到 $P(X,Y)$。
然而，$Y\rightarrow X$ 也必须能够从训练数据 $\mathcal{D}_{S}\cup\{\mathcal{D}_{\sigma}\}_{\sigma\in\Sigma}$ 中被识别出来。
换句话说，迁移学习问题包含一个隐藏的因果发现（causal discovery）问题。
如果 $\mathcal{D}_{\sigma}=\emptyset$，那么 $Y\rightarrow X$ 必须仅从 $P(x,y)$ 中识别，这通常是不可能的，除非因果数据生成过程服从额外的假设（例如，参见 Hoyer et al., 2008）。
如果来自目标域的未标记特征被包含在训练数据 $\mathcal{D}_{\sigma}=\{x^{i}\sim P(X;\sigma)\}_{i=1}^{n_{\sigma}}$ 中，则 $Y\rightarrow X$ 原则上可以被识别，因为 $P(X;\sigma_{Y})\neq P(X)$。

#### 因果推断（Causal inference）。

**定理 1（Theorem 1）** 也可以纯粹从 **因果推断（Causal Inference）** 的角度进行解释。
我们可以将其与 **因果层次定理（Causal Hierarchy Theorem, CHT）** （Bareinboim 等人，2022）进行比较，该定理指出，一个用于 L1 查询（观测性）的 **神谕（Oracle）** 几乎总是不足以评估所有 L2 查询（干预性）。
我们的 **定理 1（Theorem 1）** 可以用类似的方式表述：一个用于局部干预混合下最优策略的神谕 $\Pi^{*}_{\Sigma}:\sigma\mapsto\pi^{*}(\sigma)$，可以评估所有 L2 查询，这源于该神谕能识别底层的 **因果贝叶斯网络（Causal Bayesian Network, CBN）** ，而 CBN 又能识别所有 L2 查询。
注意 $\Pi^{*}_{\Sigma}$ 是 L2 的一个 **严格子集（strict subset）** ，并且我们将 L2 的一个子集描述为 **_L2-完备的（L2-complete）_** ，如果评估这些查询足以评估所有 L2 查询。
因此， **定理 1（Theorem 1）** 可以概括为 $\Pi^{*}_{\Sigma}$ 是 L2-完备的。
在未来的工作中，确定 L2 的其他哪些严格子集是 L2-完备的将非常有趣，因为识别这些查询就足以识别所有干预性查询。

为什么这令人惊讶？
首先，我们可能预期最优策略只编码相对少量的因果关系，因为它们可以从 $\mathbb{E}[u\mid d,\textbf{pa}_{D};\sigma]$ 计算得出，该式描述的是单个变量 $U$ 对干预 $\sigma$ 的响应。
然而， **定理 1（Theorem 1）** 表明，最优策略编码了 $\textbf{Anc}_{U}$ 中所有的因果和关联关系，包括潜在变量之间的因果关系，例如对于任何 $\bm{X},\bm{Y}\subseteq\textbf{Anc}_{U}$ 的 $P(\bm{Y}_{x})$。
其次， **定理 2（Theorem 2）** 和 **定理 3（Theorem 3）** 结合起来意味着，学习在 **领域偏移（domain shifts）** 下泛化等价于学习数据生成过程的因果模型——这两个问题在表面上是概念上截然不同的。

<a id="figure-3"></a>

![newfig](images/newfig.png)

> **图 3（Figure 3）** ：左图将 **定理 1（Theorem 1）** 置于 **珀尔的因果层次（Pearl’s causal hierarchy）** （Bareinboim 等人，2022）中。L2 包含所有干预性查询。L2 包含了领域偏移 $\bm{\Pi}^{*}_{C}$ 和任务偏移 $\bm{\Pi}^{*}_{U}$ 下的最优策略查询集合，因为最优策略总是可以从有限数量的干预性查询中找到。 **定理 1（Theorem 1）** 令人惊讶地表明 $\bm{\Pi}^{*}_{C}$ 包含了 L2，因此 $\bm{\Pi}^{*}_{C}=\text{L2}$（右图）。也就是说，学习单个效用 $U$ 在所有偏移下的最优策略足以识别 L2。这也意味着 $\bm{\Pi}^{*}_{U}\subseteq\bm{\Pi}^{*}_{C}$，这表明学习领域偏移下的最优策略足以识别任务偏移下的最优策略。

<a id="section-4"></a>

## 4 讨论（Discussion）

在此，我们讨论本研究对若干领域的影响、遗留的开放性问题以及局限性。

#### 因果表征学习（Causal representation learning）。

**因果表征学习（Causal Representation Learning, CRL）** 旨在学习能够捕捉未知因果结构的数据表征（Schölkopf et al., 2021），其目标是利用因果不变性（causal invariances）来实现跨领域的更好泛化（generalisation）。 **定理 1（Theorem 1）** 和 **定理 2（Theorem 2）** 表明，任何能够实现跨多个领域泛化的方法，都必然涉及学习数据生成过程的一个（近似）因果模型——即一个因果表征。因此，我们的结果为 CRL 提供了理论依据，证明了它是获得强鲁棒性保证的必要条件。

#### 迁移学习的因果界限（Causal bounds on transfer learning）。

如 **第 3.2 节（Section 3.2）** 所述， **定理 1（Theorem 1）** 和 **定理 2（Theorem 2）** 意味着对某些迁移学习（transfer learning）任务存在根本性的因果约束。例如，在 **图 1（Figure 1）** 所描绘的监督学习（supervised learning）任务中，在协变量偏移（covariate shift）和标签偏移（label shift）下识别具有遗憾界（regret-bounded）的策略，需要学习特征与标签之间的因果关系。此类因果发现问题（causal discovery problems）在许多情境下已得到充分理解（Vowels et al., 2022），但一般而言，若没有干预数据（interventional data）和/或额外假设，识别这种因果结构（例如 **图 4（Figure 4）** a) 中的 $Y\rightarrow X$）是不可能的。这种关联使我们能够将因果发现的（不）可能性结果转化为迁移学习的（不）可能性结果。未来的工作可以针对更小范围的分布偏移（distributional shifts）集探索这一点，并推导出更一般的迁移学习因果界限。

#### 良好调节器定理（Good regulator theorem）。

**良好调节器定理（Good Regulator Theorem）** 通常被解释为：任何系统的良好控制器都必须拥有该系统的一个模型（Conant & Ross Ashby, 1970）。然而，要从该定理的实际表述中得出这一启示需要一些想象力，因为从技术上讲，它只陈述了存在一个最优调节器，它是系统状态的确定性函数（这个函数可能是平凡的，Wentworth (2021)）。我们的定理则更明确地指出，任何鲁棒的智能体（agent）都必须已经学习到环境的一个（近似）因果模型，如 **第 3.2 节（Section 3.2）** 所述。因此，它可以被解释为一个更精确的、因果意义上的良好调节器定理。

#### 涌现能力（Emergent capabilities）。

**因果模型（Causal models）** 赋予了一种 **通用能力（general competency）** ——智能体（agent）可以利用其环境的因果模型来优化任意给定的目标函数 $U(\textbf{Pa}_{U}\subseteq\bm{V})$，而无需额外数据（[定理 3](https://arxiv.org/html/2402.10877v7#Thmtheorem3)）。这可以解释 **通用能力（general competence）** 如何从 **狭隘的训练目标（narrow training objectives）** 中产生（Brown 等人，2020；Silver 等人，2021）。

根据 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)，在许多环境中被训练以最大化奖励的智能体，有动机去学习一个 **因果世界模型（causal world model）** （因为没有它就无法泛化），而该模型反过来又可用于解决同一环境中的任何其他决策任务（[定理 3](https://arxiv.org/html/2402.10877v7#Thmtheorem3)）。然而，这种动机并不意味着仅用简单的奖励信号训练智能体就足以学习因果世界模型。例如，如果模型无法从其训练数据中被识别，那么智能体仍然不可能学习到因果模型（因此也无法泛化）。问题在于，当前的方法和训练方案是否足以学习因果世界模型。

早期结果表明， **Transformer 模型（Transformer models）** 能够学习到具备 **分布外预测（out-of-distribution prediction）** 能力的世界模型（Li 等人，2022，讨论见 [第 6 节](#section-6)）。虽然 **基础模型（Foundation models）** 能够在因果推理基准测试中达到最先进的准确率（Kıcıman 等人，2023），但它们如何实现这一点（以及这是否构成真正的因果推理）仍存在争议（Zečević 等人，2023）。

#### **因果发现（Causal discovery）** 。

因果模型赋予了一种通用能力——智能体可以利用其环境的因果模型来优化任意给定的目标函数 $U(\textbf{Pa}_{U}\subseteq\bm{V})$，而无需额外数据（[定理 3](https://arxiv.org/html/2402.10877v7#Thmtheorem3)）。这可以解释通用能力如何从狭隘的训练目标中产生（Brown 等人，2020；Silver 等人，2021）。

根据 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)，在许多环境中被训练以最大化奖励的智能体，有动机去学习一个因果世界模型（因为没有它就无法泛化），而该模型反过来又可用于解决同一环境中的任何其他决策任务（[定理 3](https://arxiv.org/html/2402.10877v7#Thmtheorem3)）。然而，这种动机并不意味着仅用简单的奖励信号训练智能体就足以学习因果世界模型。例如，如果模型无法从其训练数据中被识别，那么智能体仍然不可能学习到因果模型（因此也无法泛化）。问题在于，当前的方法和训练方案是否足以学习因果世界模型。

早期结果表明，Transformer 模型能够学习到具备分布外预测能力的世界模型（Li 等人，2022，讨论见 [第 6 节](#section-6)）。虽然基础模型能够在因果推理基准测试中达到最先进的准确率（Kıcıman 等人，2023），但它们如何实现这一点（以及这是否构成真正的因果推理）仍存在争议（Zečević 等人，2023）。

**定理 1（Theorem 1）** 和 **定理 2（Theorem 2）** 涉及通过观察智能体在干预下的策略来学习环境的因果结构。
或许令人惊讶的是， **这个单一变量对干预的响应足以识别 $\textbf{Anc}_{U}$ 中的所有关联和因果关系** 。
通常，因果发现算法需要测量多个变量对干预的响应 [1]。
此外，许多因果发现算法假设了 **独立因果机制（Independent Causal Mechanisms）** [2]，这等价于假设数据生成过程中不存在智能体 [3]。
然而，我们的结果表明，智能体可能成为因果发现的强大资源。
在 **[附录 B（Appendix B）](#appendix-b)** 中，我们利用 **定理 2（Theorem 2）** 的证明推导出一种用于学习潜在变量因果结构的因果发现算法，并在合成数据上进行了测试。

#### **没有理解，就没有能力（No competence without understanding）**

因果模型是人类理解和解释世界的基础 [4, 5]。
深度学习模型越来越多地被用于预测和控制我们尚未完全理解的复杂系统，例如 **惯性约束等离子体（inertially confined plasmas）** [6] 和 **生物分子系统（biomolecular systems）** [7]。可以说，这些模型提供了一条 **无需理解即可获得能力的捷径** ——即无需为底层系统开发更丰富的模型，也无需理解解决方案如何以及为何有效，就能解决问题。
例如，AlphaFold 能准确预测蛋白质折叠，但人们发现它并未增进对底层生化过程的理解 [8]。
有观点认为，对 **黑盒模型（black-box models）** 的依赖可能会极大地扩大 **能力（capabilities）** 与 **理解（understanding）** 之间的差距 [9]。
然而，我们的结果指出了一个潜在的解决方案。
如果一个控制器足够强大和鲁棒，我们总是可以提取出它所控制系统的一个 **可解释的因果模型（interpretable causal model）** 。
**不存在没有“理解”的鲁棒控制** ，这里的“理解”指的是学习底层系统的因果模型。
未来的工作可以探索扩展我们的结果，以开发从鲁棒智能体中 **引出（eliciting）** 因果模型的高效算法。

#### 因果方法的适用性（Applicability of causal methods）

**因果模型（Causal models）** 已被用于形式化定义诸如意图（Halpern & Kleiman-Weiner, 2018; Ward et al., 2024）、伤害（Richens et al., 2022）、欺骗（Ward et al., 2023b）、操纵（Ward et al., 2023a）和激励（Everitt et al., 2021）等概念，并且是 **可解释性（Explainability）** （Wachter et al., 2017）和 **公平性（Fairness）** （Kusner et al., 2017）方法所必需的。基于这些定义设计安全且合乎伦理的 **人工智能（Artificial Intelligence, AI）** 系统的方法，需要数据生成过程的因果模型，而这些模型通常难以学习，这导致一些人怀疑其实用性（Fawkes et al., 2022; Rahmattalabi & Xiang, 2022）。然而，我们的结果表明，能力足够强的 **智能体（Agents）** 必须已经学习到了一个能够支持这些方法的 **因果世界模型（Causal world model）** ，并且我们证明了可以从智能体中 **引出（Elicit）** 这些世界模型。

#### 局限性（Limitations）

[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 要求智能体对大量的 **领域偏移（Domain shifts）** （即对所有环境变量的局部干预）具有鲁棒性。[定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 表明，放宽 **遗憾界（Regret bounds）** 会导致一些因果关系无法从智能体的 **策略（Policy）** 中识别出来。因此，我们预期，即使从对较小范围领域偏移具有鲁棒性的智能体中，仍然有可能学习到关于环境的一些因果知识，尽管不如完整的底层 **因果贝叶斯网络（Causal Bayesian Network, CBN）** 那么完备。最后，我们的结果仅适用于 **非中介决策任务（Unmediated decision tasks）** （假设 1）。我们预期 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 可以扩展到 **主动决策任务（Active decision tasks）** ，因为假设 1 除了简化证明外并未起到主要作用。

<a id="section-5"></a>

## 5 相关工作（Related work）

最近的几项实证研究探讨了深度学习模型是学习“表面统计量”（例如输入与输出之间的相关性），还是学习世界的内部表征（McGrath 等人，2022；Abdou 等人，2021；Li 等人，2022；Gurnee & Tegmark，2023）。我们的结果为这一讨论提供了一些理论上的澄清，将智能体的性能与其世界模型（world model）的保真度联系起来，并表明超越“表面统计量”去学习因果关系对于鲁棒性而言是根本必要的。其中一项研究（Li 等人，2022）特别发现，一个为预测棋盘游戏奥赛罗（Othello）中合法下一步棋而训练的 GPT 模型，学习到了棋盘状态的线性表征（Nanda，2023）。此外，这种对棋盘状态的内部表征可以通过干预中间激活来改变，模型会根据干预更新其预测，包括那些使棋盘状态超出训练分布的干预。这表明该网络正在学习和利用一种数据生成过程（data generating process）的表征，这种表征能够支持干预下的分布外泛化（out-of-distribution generalisation）——这很像一个因果模型（causal model）。

在分布变化下评估策略（policies）的问题已在因果可迁移性（Causal Transportability, CT）理论中得到广泛研究（Bareinboim & Pearl，2016；Bellot & Bareinboim，2022）。CT 旨在为已知领域偏移下的策略评估提供必要和充分条件，前提是数据生成过程的所有假设（即归纳偏置，inductive biases）都可以表达为对因果结构的约束（Bareinboim & Pearl，2016）。然而，深度学习算法可以利用更大范围的归纳偏置（Neyshabur 等人，2014；Battaglia 等人，2018；Rahaman 等人，2019；Goyal & Bengio，2022），在许多现实世界任务中，这可能足以识别出低遗憾（low regret）策略，而无需因果知识。因此，CT 并不意味着智能体必须学习因果模型才能泛化，除非我们假设智能体一开始就只使用因果假设，而这将是基于假设的证明。进一步讨论请参见[附录 G](#appendix-g)。

与[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1)和[定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)类似的一个结果是因果层次定理（Causal Hierarchy Theorem, CHT）（Bareinboim 等人，2022；Ibeling & Icard，2021），该定理表明观测数据几乎总是不足以识别环境变量间的所有因果关系，而我们的结果则表明，最优策略集几乎总是足以识别所有因果关系。在[第 3.2 节](#section-3-2)中，我们讨论了这些定理之间的相似之处，并在[附录 G](#appendix-g)中讨论了它们的差异。

<a id="section-6"></a>

## 6 结论（Conclusion）

**因果推理（Causal reasoning）** 是人类智能的基础，并且被推测为是实现人类水平 **人工智能（Artificial Intelligence, AI）** 所必需的（Pearl, 2019）[1]。近年来，随着能够在无需显式学习或基于 **因果模型（causal models）** 进行推理的情况下，就能泛化到新任务和新领域的人工智能体的发展，这一推测受到了挑战。尽管解决 **因果推断（causal inference）** 任务需要因果模型的必要性已经得到确立（Bareinboim et al., 2022）[2]，但它们在诸如 **分类（classification）** 和 **强化学习（reinforcement learning）** 等决策任务中的作用尚不明确。

我们以一种 **模型无关（model-independent）** 的方式解决了这一推测，证明了任何能够稳健解决决策任务的智能体，无论其如何训练或其架构细节如何，都必然已经学习了数据生成过程的因果模型。这暗示了 **因果性（causality）** 与 **通用智能（general intelligence）** 之间存在更深层次的联系，因为该因果模型可用于寻找在环境变量上优化任意给定 **目标函数（objective function）** 的策略。通过建立因果性与泛化之间的形式化联系，我们的结果表明， **因果世界模型（causal world models）** 是构建稳健且通用的人工智能的必要组成部分。

#### 致谢（Acknowledgements）

我们要感谢 Alexis Bellot、Damiano Fornasiere、Pietro Greiner、James Fox、Matt MacDermott、David Reber、David Watson 和 Philip Bachman 对本文稿提出的有益讨论和评论。

## 参考文献

- Abdou 等人 (2021)
  Mostafa Abdou, Artur Kulmizev, Daniel Hershcovich, Stella Frank, Ellie Pavlick,
  and Anders Søgaard.
  语言模型能否在不接地的情况下编码感知结构？一项关于颜色的案例研究。
  _arXiv 预印本 arXiv:2109.06129_, 2021.
- Abramson 等人 (2024)
  Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander
  Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua
  Bambrick, 等。
  使用 AlphaFold 3 对生物分子相互作用进行精确结构预测。
  《自然》, 第 1–3 页, 2024.
- Arjovsky 等人 (2019)
  Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz.
  不变风险最小化。
  _arXiv 预印本 arXiv:1907.02893_, 2019.
- Bareinboim & Pearl (2012a)
  Elias Bareinboim and Judea Pearl.
  控制因果推断中的选择偏倚。
  载于《人工智能与统计学》, 第 100–108 页。
  PMLR, 2012a.
- Bareinboim & Pearl (2012b)
  Elias Bareinboim and Judea Pearl.
  因果效应的可迁移性：完备性结果。
  载于《AAAI 人工智能会议论文集》, 第 26 卷, 第 698–704 页, 2012b.
- Bareinboim & Pearl (2016)
  Elias Bareinboim and Judea Pearl.
  因果推断与数据融合问题。
  《美国国家科学院院刊》, 113(27):7345–7352, 2016.
- Bareinboim 等人 (2022)
  Elias Bareinboim, Juan D Correa, Duligur Ibeling, and Thomas Icard.
  论 Pearl 的因果层级与因果推断的基础。
  载于《概率与因果推断：Judea Pearl 的著作》, 第 507–556 页. 2022.
- Battaglia 等人 (2018)
  Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez,
  Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam
  Santoro, Ryan Faulkner, 等。
  关系归纳偏置、深度学习与图网络。
  _arXiv 预印本 arXiv:1806.01261_, 2018.
- Bellot & Bareinboim (2022)
  Alexis Bellot and Elias Bareinboim.
  领域泛化的部分可迁移性。2022.

- Brown et al. (2020)
  汤姆·布朗（Tom Brown）、本杰明·曼（Benjamin Mann）、尼克·莱德（Nick Ryder）、梅拉妮·苏比亚（Melanie Subbiah）、贾里德·D·卡普兰（Jared D Kaplan）、普拉富拉·达里瓦尔（Prafulla Dhariwal）、阿文德·尼拉坎坦（Arvind Neelakantan）、普拉纳夫·夏姆（Pranav Shyam）、吉里什·萨斯特里（Girish Sastry）、阿曼达·阿斯克尔（Amanda Askell）等。
  **语言模型是小样本学习者（Language models are few-shot learners）** 。
  《神经信息处理系统进展（Advances in neural information processing systems）》，33:1877–1901，2020。
- Castro et al. (2020)
  丹尼尔·C·卡斯特罗（Daniel C Castro）、伊恩·沃克（Ian Walker）、本·格洛克（Ben Glocker）。
  **因果关系在医学成像中至关重要（Causality matters in medical imaging）** 。
  《自然通讯（Nature Communications）》，11(1):3673，2020。
- Cohen & Welling (2016)
  塔科·科恩（Taco Cohen）、马克斯·韦林（Max Welling）。
  **群等变卷积网络（Group equivariant convolutional networks）** 。
  载于《国际机器学习会议（International conference on machine learning）》，第 2990–2999 页。PMLR，2016。
- Conant & Ross Ashby (1970)
  罗杰·C·科南特（Roger C Conant）、W·罗斯·阿什比（W Ross Ashby）。
  **任何系统的良好调节器都必须是该系统的模型（Every good regulator of a system must be a model of that system）** 。
  《国际系统科学杂志（International journal of systems science）》，1(2):89–97，1970。
- Correa & Bareinboim (2020)
  胡安·科雷亚（Juan Correa）、埃利亚斯·巴雷因博伊姆（Elias Bareinboim）。
  **一种用于随机干预的演算：因果效应识别与替代实验（A calculus for stochastic interventions: Causal effect identification and surrogate experiments）** 。
  载于《美国人工智能协会人工智能会议论文集（Proceedings of the AAAI conference on artificial intelligence）》，第 34 卷，第 10093–10100 页，2020。
- Dawid (2002)
  A·P·达维德（A P Dawid）。
  **用于因果建模与推理的影响图（Influence diagrams for causal modelling and inference）** 。
  《国际统计评论（International Statistical Review / Revue Internationale de Statistique）》，70:161–189，2002。
- Degrave et al. (2022)
  乔纳斯·德格拉夫（Jonas Degrave）、费德里科·费利奇（Federico Felici）、乔纳斯·布赫利（Jonas Buchli）、迈克尔·诺伊纳特（Michael Neunert）、布伦丹·特雷西（Brendan Tracey）、弗朗切斯科·卡尔帕内塞（Francesco Carpanese）、蒂莫·埃瓦尔德斯（Timo Ewalds）、罗兰·哈夫纳（Roland Hafner）、阿巴斯·阿卜杜勒马利基（Abbas Abdolmaleki）、迭戈·德拉斯卡萨斯（Diego de Las Casas）等。
  **通过深度强化学习实现托卡马克等离子体的磁控制（Magnetic control of tokamak plasmas through deep reinforcement learning）** 。
  《自然（Nature）》，602(7897):414–419，2022。
- Dennett (1989)
  丹尼尔·C·丹尼特（Daniel C Dennett）。
  **《意向立场（The intentional stance）》** 。
  MIT 出版社，1989。
- Engstrom et al. (2019)
  洛根·恩斯特罗姆（Logan Engstrom）、布兰登·陈（Brandon Tran）、迪米特里斯·齐普拉斯（Dimitris Tsipras）、路德维希·施密特（Ludwig Schmidt）、亚历山大·马德里（Aleksander Madry）。
  **探索空间鲁棒性的图景（Exploring the landscape of spatial robustness）** 。
  载于《国际机器学习会议（International conference on machine learning）》，第 1802–1811 页。PMLR，2019。
- Everitt et al. (2021)
  汤姆·埃弗里特（Tom Everitt）、瑞安·凯里（Ryan Carey）、埃里克·D·朗格卢瓦（Eric D Langlois）、佩德罗·A·奥尔特加（Pedro A Ortega）、肖恩·莱格（Shane Legg）。
  **智能体激励：一个因果视角（Agent incentives: A causal perspective）** 。
  载于《美国人工智能协会人工智能会议论文集（Proceedings of the AAAI Conference on Artificial Intelligence）》，第 35 卷，第 11487–11495 页，2021。
- Farahani et al. (2021)
  阿博法兹勒·法拉哈尼（Abolfazl Farahani）、萨哈尔·沃格霍伊（Sahar Voghoei）、哈立德·拉希德（Khaled Rasheed）、哈米德·R·阿拉布尼亚（Hamid R Arabnia）。
  **领域自适应简要综述（A brief review of domain adaptation）** 。
  《数据科学与信息工程进展：ICDATA 2020 和 IKE 2020 会议论文集（Advances in Data Science and Information Engineering: Proceedings from ICDATA 2020 and IKE 2020）》，第 877–894 页，2021。
- Fawkes et al. (2022)
  杰克·福克斯（Jake Fawkes）、罗宾·埃文斯（Robin Evans）、迪诺·塞伊迪诺维奇（Dino Sejdinovic）。
  **因果公平中的选择、可忽略性与挑战（Selection, ignorability and challenges with causal fairness）** 。
  载于《因果学习与推理会议（Conference on Causal Learning and Reasoning）》，第 275–289 页。PMLR，2022。
- Gopnik et al. (2007)
  艾莉森·戈普尼克（Alison Gopnik）、劳拉·舒尔茨（Laura Schulz）、劳拉·伊丽莎白·舒尔茨（Laura Elizabeth Schulz）。
  **《因果学习：心理学、哲学与计算（Causal learning: Psychology, philosophy, and computation）》** 。
  牛津大学出版社，2007。
- Goyal & Bengio (2022)
  阿尼鲁德·戈亚尔（Anirudh Goyal）、约书亚·本吉奥（Yoshua Bengio）。
  **高阶认知深度学习的归纳偏置（Inductive biases for deep learning of higher-level cognition）** 。
  《英国皇家学会会刊 A 辑（Proceedings of the Royal Society A）》，478(2266):20210068，2022。
- Gupta et al. (2023)
  沙鲁特·古普塔（Sharut Gupta）、斯特凡妮·耶格尔卡（Stefanie Jegelka）、大卫·洛佩斯-帕斯（David Lopez-Paz）、卡尔蒂克·阿胡贾（Kartik Ahuja）。
  **上下文即环境（Context is environment）** ，2023。
- Gurnee & Tegmark (2023)
  韦斯·格尼（Wes Gurnee）、马克斯·泰格马克（Max Tegmark）。
  **语言模型表征空间与时间（Language models represent space and time）** 。
  《arXiv 预印本 arXiv:2310.02207》，2023。
- Halpern & Kleiman-Weiner (2018)
  约瑟夫·哈尔彭（Joseph Halpern）、马克斯·克莱曼-韦纳（Max Kleiman-Weiner）。
  **走向可责性、意图与道德责任的形式化定义（Towards formal definitions of blameworthiness, intention, and moral responsibility）** 。
  载于《美国人工智能协会人工智能会议论文集（Proceedings of the AAAI Conference on Artificial Intelligence）》，第 32 卷，2018。
- Hendrycks & Dietterich (2019)
  丹·亨德里克斯（Dan Hendrycks）、托马斯·迪特里希（Thomas Dietterich）。
  **神经网络对常见损坏与扰动的鲁棒性基准测试（Benchmarking neural network robustness to common corruptions and perturbations）** 。
  《arXiv 预印本 arXiv:1903.12261》，2019。
- Howard & Matheson (2005)
  罗纳德·A·霍华德（Ronald A Howard）、詹姆斯·E·马西森（James E Matheson）。
  **影响图（Influence diagrams）** 。
  《决策分析（Decision Analysis）》，2(3):127–143，2005。
- Hoyer et al. (2008)
  帕特

- **Hoyer 等人（2008）**
  Patrik O. Hoyer, Dominik Janzing, Joris M. Mooij, Jonas Peters 和 Bernhard Schölkopf。
  **使用加性噪声模型进行非线性因果发现（Nonlinear causal discovery with additive noise models）** 。
  《神经信息处理系统进展（Advances in neural information processing systems）》，21，2008年。
- **Ibeling & Icard（2021）**
  Duligur Ibeling 和 Thomas Icard。
  **因果推断的拓扑学视角（A topological perspective on causal inference）** 。
  《神经信息处理系统进展（Advances in Neural Information Processing Systems）》，34:5608–5619，2021年。
- **Kenton 等人（2023）**
  Zachary Kenton, Ramana Kumar, Sebastian Farquhar, Jonathan Richens, Matt MacDermott 和 Tom Everitt。
  **智能体发现（Discovering agents）** 。
  《人工智能（Artificial Intelligence）》，第 103963 页，2023年。
- **Kıcıman 等人（2023）**
  Emre Kıcıman, Robert Ness, Amit Sharma 和 Chenhao Tan。
  **因果推理与大型语言模型：开启因果性的新前沿（Causal reasoning and large language models: Opening a new frontier for causality）** 。
  《arXiv 预印本 arXiv:2305.00050》，2023年。
- **Kirk 等人（2023）**
  Robert Kirk, Amy Zhang, Edward Grefenstette 和 Tim Rocktäschel。
  **深度强化学习中零样本泛化的综述（A survey of zero-shot generalisation in deep reinforcement learning）** 。
  《人工智能研究杂志（Journal of Artificial Intelligence Research）》，76:201–264，2023年。
- **Kusner 等人（2017）**
  Matt J Kusner, Joshua Loftus, Chris Russell 和 Ricardo Silva。
  **反事实公平性（Counterfactual fairness）** 。
  《神经信息处理系统进展（Advances in neural information processing systems）》，30，2017年。
- **Legg & Hutter（2007）**
  Shane Legg 和 Marcus Hutter。
  **通用智能：机器智能的定义（Universal intelligence: A definition of machine intelligence）** 。
  《心智与机器（Minds and machines）》，17:391–444，2007年。
- **Li 等人（2022）**
  Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister 和 Martin Wattenberg。
  **涌现的世界表征：探索在合成任务上训练的序列模型（Emergent world representations: Exploring a sequence model trained on a synthetic task）** 。
  《arXiv 预印本 arXiv:2210.13382》，2022年。
- **McGrath 等人（2022）**
  Thomas McGrath, Andrei Kapishnikov, Nenad Tomašev, Adam Pearce, Martin Wattenberg, Demis Hassabis, Been Kim, Ulrich Paquet 和 Vladimir Kramnik。
  **AlphaZero 中象棋知识的获取（Acquisition of chess knowledge in AlphaZero）** 。
  《美国国家科学院院刊（Proceedings of the National Academy of Sciences）》，119(47):e2206625119，2022年。
- **Meek（2013）**
  Christopher Meek。
  **贝叶斯网络中的强完备性与忠实性（Strong completeness and faithfulness in bayesian networks）** 。
  《arXiv 预印本 arXiv:1302.4973》，2013年。
- **Meinshausen（2018）**
  Nicolai Meinshausen。
  **从分布鲁棒性角度看因果性（Causality from a distributional robustness point of view）** 。
  载于《2018年 IEEE 数据科学研讨会（2018 IEEE Data Science Workshop (DSW)）》，第 6–10 页。

2018.

- **米特罗维奇等人（Mitrovic et al.） (2018)**
  约瓦娜·米特罗维奇（Jovana Mitrovic）、迪诺·塞季诺维奇（Dino Sejdinovic）和郑宇怀（Yee Whye Teh）。
  通过核偏差度量进行因果推断。
  《神经信息处理系统进展》，第 31 卷，2018 年。
- **穆伊等人（Mooij et al.） (2016)**
  约里斯·M·穆伊（Joris M Mooij）、乔纳斯·彼得斯（Jonas Peters）、多米尼克·扬青（Dominik Janzing）、雅各布·蔡舍尔（Jakob Zscheischler）和伯恩哈德·舍尔科普夫（Bernhard Schölkopf）。
  利用观测数据区分因果：方法与基准。
  《机器学习研究杂志》，第 17 卷，第 1 期，第 1103–1204 页，2016 年。
- **南达（Nanda） (2023)**
  尼尔·南达（Neel Nanda）。
  实际上，奥赛罗-GPT 拥有一个线性涌现世界模型，2023 年 3 月。
  网址：[https://neelnanda.io/mechanistic-interpretability/othello](https://neelnanda.io/mechanistic-interpretability/othello)。
- **内沙布尔等人（Neyshabur et al.） (2014)**
  贝赫纳姆·内沙布尔（Behnam Neyshabur）、富田亮太（Ryota Tomioka）和内森·斯雷布罗（Nathan Srebro）。
  探寻真实的归纳偏置：论隐式正则化在深度学习中的作用。
  《arXiv 预印本 arXiv:1412.6614》，2014 年。
- **冈本（Okamoto） (1973)**
  冈本正（Masashi Okamoto）。
  多元样本中二次型特征值的相异性。
  《统计学年鉴》，第 763–765 页，1973 年。
- **奥特拉尔等人（Outeiral et al.） (2022)**
  卡洛斯·奥特拉尔（Carlos Outeiral）、丹尼尔·A·尼斯利（Daniel A Nissley）和夏洛特·M·迪恩（Charlotte M Deane）。
  当前的结构预测器并未学习蛋白质折叠的物理学。
  《生物信息学》，第 38 卷，第 7 期，第 1881–1887 页，2022 年。
- **帕斯奎尔（Pasquale） (2015)**
  弗兰克·帕斯奎尔（Frank Pasquale）。
  《黑箱社会：控制金钱与信息的秘密算法》。
  哈佛大学出版社，2015 年。
- **珀尔（Pearl） (2009)**
  朱迪亚·珀尔（Judea Pearl）。
  《因果论：模型、推理与推断》。
  剑桥大学出版社，第 2 版，2009 年。
  ISBN 9780521895606。
- **珀尔（Pearl） (2018)**
  朱迪亚·珀尔（Judea Pearl）。
  机器学习的理论障碍与因果革命的七点火花。
  《arXiv 预印本 arXiv:1801.04016》，2018 年。
- **珀尔（Pearl） (2019)**
  朱迪亚·珀尔（Judea Pearl）。
  因果推断的七种工具，兼论机器学习。
  《ACM 通讯》，第 62 卷，第 3 期，第 54–60 页，2019 年。

- Pearl & Bareinboim (2011)
  朱迪亚·珀尔（Judea Pearl）和埃利亚斯·巴雷因博伊姆（Elias Bareinboim）。
  因果与统计关系的可迁移性：一种形式化方法。
  载于《人工智能 AAAI 会议论文集》，第 25 卷，第 247–254 页，2011 年。
- Pearl & Mackenzie (2018)
  朱迪亚·珀尔（Judea Pearl）和达纳·麦肯齐（Dana Mackenzie）。
  《为什么之书：因果的新科学》。
  基础图书出版社，2018 年。
- Peng et al. (2018)
  彭学斌（Xue Bin Peng）、马尔钦·安德里乔维茨（Marcin Andrychowicz）、沃伊切赫·扎伦巴（Wojciech Zaremba）和彼得·阿比尔（Pieter Abbeel）。
  利用动力学随机化的机器人控制从仿真到现实的迁移。
  载于《2018 年 IEEE 机器人与自动化国际会议（ICRA）》，第 3803–3810 页。IEEE，2018 年。
- Rahaman et al. (2019)
  纳西姆·拉赫曼（Nasim Rahaman）、阿里斯蒂德·巴拉坦（Aristide Baratin）、德万什·阿尔皮特（Devansh Arpit）、费利克斯·德拉克斯勒（Felix Draxler）、林敏（Min Lin）、弗雷德·汉普雷希特（Fred Hamprecht）、约书亚·本吉奥（Yoshua Bengio）和亚伦·库维尔（Aaron Courville）。
  论神经网络的谱偏置。
  载于《国际机器学习会议》，第 5301–5310 页。PMLR，2019 年。
- Rahmattalabi & Xiang (2022)
  艾达·拉赫马塔拉比（Aida Rahmattalabi）和项安琪（Alice Xiang）。
  因果性对伦理机器学习的承诺与挑战。
  《arXiv 预印本 arXiv:2201.10683》，2022 年。
- Reed et al. (2022)
  斯科特·里德（Scott Reed）、康拉德·佐尔纳（Konrad Zolna）、埃米利奥·帕里索托（Emilio Parisotto）、塞尔吉奥·戈麦斯·科尔梅纳雷霍（Sergio Gomez Colmenarejo）、亚历山大·诺维科夫（Alexander Novikov）、加布里埃尔·巴特-马龙（Gabriel Barth-Maron）、麦·希门尼斯（Mai Gimenez）、尤里·苏尔斯基（Yury Sulsky）、杰基·凯（Jackie Kay）、约斯特·托比亚斯·施普林根贝格（Jost Tobias Springenberg）等。
  一种通才智能体。
  《arXiv 预印本 arXiv:2205.06175》，2022 年。
- Reichenbach (1956)
  汉斯·赖欣巴哈（Hans Reichenbach）。
  《时间的方向》，第 65 卷。
  加州大学出版社，1956 年。
- Richens et al. (2022)
  乔纳森·里琴斯（Jonathan Richens）、罗里·比尔德（Rory Beard）和丹尼尔·H·汤普森（Daniel H Thompson）。
  反事实伤害。
  《神经信息处理系统进展》，35:36350–36365，2022 年。
- Rubin (2005)
  唐纳德·B·鲁宾（Donald B Rubin）。
  使用潜在结果进行因果推断：设计、建模、决策。
  《美国统计协会杂志》，100(469):322–331，2005 年。
- Schölkopf et al. (2012)
  伯恩哈德·舍尔科普夫（Bernhard Schölkopf）、多米尼克·扬青（Dominik Janzing）、约纳斯·彼得斯（Jonas Peters）、埃莱妮·斯古里察（Eleni Sgouritsa）、张坤（Kun Zhang）和约里斯·穆伊（Joris Mooij）。
  论因果与反因果学习。
  《arXiv 预印本 arXiv:1206.6471》，2012 年。
- Schölkopf et al. (2021)
  伯恩哈德·舍尔科普夫（Bernhard Schölkopf）、弗朗切斯科·洛卡泰洛（Francesco Locatello）、斯特凡·鲍尔（Stefan Bauer）、柯南·罗斯玛丽（Nan Rosemary Ke）、纳尔·卡尔赫布伦纳（Nal Kalchbrenner）、阿尼鲁德·戈亚尔（Anirudh Goyal）和约书亚·本吉奥（Yoshua Bengio）。
  迈向因果表征学习。
  《IEEE 会刊》，109(5):612–634，2021 年。

2021.

- Shah 等人 (2022)
  Rohin Shah, Vikrant Varma, Ramana Kumar, Mary Phuong, Victoria Krakovna,
  Jonathan Uesato, 以及 Zac Kenton.
  目标错误泛化：为什么正确的规范不足以实现正确的目标。
  _arXiv 预印本 arXiv:2210.01790_, 2022.
- Shen 等人 (2018)
  Zheyan Shen, Peng Cui, Kun Kuang, Bo Li, 以及 Peixuan Chen.
  具有不可知数据选择偏差的因果正则化学习。
  载于 _第 26 届 ACM 国际多媒体会议论文集_，第 411–419 页，2018.
- Silver 等人 (2021)
  David Silver, Satinder Singh, Doina Precup, 以及 Richard S Sutton.
  奖励即足够。
  《人工智能》，第 299 卷：103535，2021.
- Sloman 与 Lagnado (2015)
  Steven A Sloman 与 David Lagnado.
  思维中的因果关系。
  《心理学年度评论》，第 66 卷：223–247，2015.
- Spirtes 等人 (2000)
  Peter Spirtes, Clark N Glymour, Richard Scheines, 以及 David Heckerman.
  《因果关系、预测与搜索》。
  MIT 出版社，2000.
- Team 等人 (2023)
  Gemini 团队, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac,
  Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, 等.
  Gemini：一个能力强大的多模态模型家族。
  _arXiv 预印本 arXiv:2312.11805_, 2023.
- Vowels 等人 (2022)
  Matthew J Vowels, Necati Cihan Camgoz, 以及 Richard Bowden.
  你喜欢有向无环图吗？关于结构学习与因果发现的综述。
  《ACM 计算概览》，第 55 卷，第 4 期：1–36，2022.
- Wachter 等人 (2017)
  Sandra Wachter, Brent Mittelstadt, 以及 Chris Russell.
  无需打开黑箱的反事实解释：自动化决策与 GDPR。
  《哈佛法律与技术杂志》，第 31 卷：841，2017.
- Wang 等人 (2022)
  Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang
  Chen, Wenjun Zeng, 以及 Philip Yu.
  泛化到未见域：领域泛化综述。
  《IEEE 知识与数据工程汇刊》，2022.
- Ward 等人 (2023a)
  Francis Rhys Ward, Tom Everitt, Francesco Belardinelli, 以及 Francesca Toni.
  诚实为上策：定义与缓解人工智能欺骗。
  载于 \_第 37 届神经信息处理系统大会》，2023a.
- Ward 等人 (2023b)
  Francis Rhys Ward, Francesca Toni, 以及 Francesco Belardinelli.
  定义结构因果博弈中的欺骗。
  载于 \_2023 年自主代理与多代理系统国际会议论文集》，第 2902–2904 页，2023b.
- Ward 等人 (2024)
  Francis Rhys Ward, Matt MacDermott, Francesco Belardinelli, Francesca Toni, 以及
  Tom Everitt.
  智能体行动的原因：意图与工具性目标。
  _arXiv 预印本 arXiv:2402.07221_, 2024.
- Wentworth (2021)
  John Wentworth.
  修正良好调节器定理。
  [https://www.alignmentforum.org/posts/Dx9LoqsEh3gHNJMDk/fixing-the-good-regulator-theorem](https://www.alignmentforum.org/posts/Dx9LoqsEh3gHNJMDk/fixing-the-good-regulator-theorem),

2021. 访问日期：2023-10-17.

- Wilson & Cook (2020)
  Garrett Wilson 和 Diane J Cook.
  无监督深度领域自适应综述.
  _ACM Transactions on Intelligent Systems and Technology (TIST)_,
  11(5):1–46, 2020.
- Xian et al. (2018)
  Yongqin Xian, Christoph H Lampert, Bernt Schiele, 和 Zeynep Akata.
  零样本学习——对好的、坏的和丑陋的全面评估.
  _IEEE transactions on pattern analysis and machine intelligence_,
  41(9):2251–2265, 2018.
- Zečević et al. (2023)
  Matej Zečević, Moritz Willig, Devendra Singh Dhami, 和 Kristian Kersting.
  因果鹦鹉：大型语言模型可能谈论因果关系但并非因果性.
  _arXiv preprint arXiv:2308.13067_, 2023.
- Zhuang et al. (2020)
  Fuzhen Zhuang, Zhiyuan Qi, Keyu Duan, Dongbo Xi, Yongchun Zhu, Hengshu Zhu, Hui Xiong, 和 Qing He.
  迁移学习全面综述.
  _Proceedings of the IEEE_, 109(1):43–76, 2020.

<a id="appendix-a"></a>

## 附录 A 预备知识

### A.1 设定与假设

环境由一组随机变量 $\bm{C}=\{C_{1},C_{2},\ldots,C_{N}\}$ 描述，这些变量与决策变量 $D$ 和效用节点 $U$ 共同定义了因果影响图（Causal Influence Diagram, CID）的状态空间 $\bm{V}=\bm{C}\cup\{D,U\}$。
在我们的记法中，单个变量 $C_{i}\in\bm{C}$ 被赋予索引，而变量集合则不加索引且使用粗体表示。我们使用 $\bm{V}=\bm{v}$ 作为集合 $\bm{C}$ 中变量联合状态的简写。
联合概率分布 $P(\bm{C}=\bm{c},D=d,U=u)$ 描述了环境变量之间的统计关系。
**贝叶斯网络（Bayesian networks）** 根据图 $G$ 对联合概率分布进行分解（Pearl, 2009）。

参见 [1](https://arxiv.org/html/2402.10877v7#Thmdfn1)

变量之间的分布和统计关系可能会因施加于系统的外部 **干预（interventions）** 而改变。
**硬干预（Hard interventions）** 将变量的一个子集 $\bm{C}^{\prime}\subseteq\bm{C}$ 设置为特定值 $\bm{c}^{\prime}$，记作 $\textrm{do}(\bm{C}^{\prime}=\bm{c}^{\prime})$ 或 $\textrm{do}(\bm{c}^{\prime})$。
简单来说，对于每个可能的干预 $\textrm{do}(\bm{c}^{\prime})$，都需要一个联合概率分布 $P_{\textrm{do}(\bm{c}^{\prime})}$ 来描述更新后的关系。
幸运的是，如果 $G$ 与环境因果结构相匹配（即，每当对 $V_{i}$ 的干预直接影响另一个变量 $Y$ 的值时，图中就有一条边 $V_{i}\to V_{j}$，并且没有未建模的 **混淆变量（confounders）** ；Spirtes et al., 2000; Pearl, 2009），那么所有干预分布都可以从一个单一的贝叶斯网络推导出来。
当此条件成立时，我们称该贝叶斯网络为 **因果贝叶斯网络（causal Bayesian network）** ，并称 $G$ 为 **因果图（causal graph）** 。
关于因果图 $G$，我们将 $V_{i}$ 的直接原因（父节点）记为 $\textbf{Pa}_{i}$，所有原因（祖先节点）的集合记为 $\textbf{Anc}_{i}$，$V_{i}$ 直接导致（子节点）的变量记为 $\textbf{Ch}_{i}$，其后代节点 $\textbf{Desc}_{i}$ 是所有下游变量的集合。
特别要注意，$\textbf{Anc}_{i}$ 和 $\textbf{Desc}_{i}$ 指的是 **真** 祖先和后代，即 $V_{i}\not\in\textbf{Anc}_{i}$ 且 $V_{i}\not\in\textbf{Desc}_{i}$。
我们将一个因果贝叶斯网络（Causal Bayesian Network, CBN）记为 $M=(P,G)$，其中 $P$ 是联合分布，$G$ 是描述环境因果结构的 **有向无环图（Directed Acyclic Graph, DAG）** 。
此外，干预分布 $P_{\textrm{do}(v^{\prime})}$ 由截断分解给出：

$$
P_{\textrm{do}(v^{\prime})}(\bm{v})=\begin{cases}\prod_{i:v_{i}\not\in\bm{v}^{\prime}}P(v_{i}\mid\textbf{pa}_{v_{i}})&\text{如果 \bm{v} 与 \bm {v}^{\prime} 一致}\\ 0&\text{否则。}\end{cases}
$$

等价地，干预的效果可以通过为每个节点 $V_{i} \in V$ 添加一个额外节点 $\hat{X}$ 和边 $\hat{V}_{i} \to V_{i}$ 来计算（Correa & Bareinboim, 2020; Dawid, 2002）。在 $V_{i}$ 上进行干预，则对应于在扩展图中以 $\hat{V}_{i}$ 为条件。

更一般地， **软干预（Soft interventions）** $\sigma = P^{\prime}(V_{i} \mid \textbf{Pa}^{*}_{i})$ 将 $V_{i}$ 的条件概率分布替换为一个新的分布，可能使用一个新的父节点集 $\textbf{Pa}^{*}_{o}$，只要图中不引入循环即可（Correa & Bareinboim, 2020）。修改后的环境记为 $M(\sigma)$。

如果没有因果图 $G$ 的先验知识，就无法定义一般的软干预。例如，软干预 $\sigma_{Y} = P^{\prime}(y \mid x)$ 与因果结构 $Y \rightarrow X$ 不兼容，因为它会引入一个因果循环，因此智能体（Agent）的策略相对于此干预可能无法被良好定义。因此，我们将理论分析集中在软干预的一个子集—— **局部干预（Local interventions）** 上，这类干预可以在不假定已知 $G$ 的情况下实施。

参见 [2](https://arxiv.org/html/2402.10877v7#Thmdfn2)

- **示例** ：固定一个变量的值（ **硬干预（Hard intervention）** ）是一种局部干预，因为 $\textrm{do}(V_{i}=v^{\prime}_{i}) = \textrm{do}(V_{i}=f(v_{i}))$，其中 $f(v_{i}) = v^{\prime}_{i}$。
- **示例** ：平移是一种局部干预，因为 $\textrm{do}(V_{i}=v_{i}+k) = \textrm{do}(V_{i}=f(v_{i}))$，其中 $f(v_{i}) = v_{i}+k$。这包括在强化学习（Reinforcement Learning, RL）环境中改变物体的位置（Shah et al., 2022）。
- **示例** ：针对布尔变量 $X$ 的逻辑非（Logical NOT）操作 $X \rightarrow \neg X$。

我们也考虑 **干预的混合（Mixtures of interventions）** ，这同样可以在不了解 $G$ 的情况下描述。

**示例（Example）** ：添加高斯噪声是局部操作（平移）的混合 $\sigma_{\epsilon}=\textrm{do}(X=X+\epsilon)$，其中 $\epsilon\sim\mathcal{N}(0,1)$。

大多数决策任务，如预测、分类和强化学习，其共同点在于： **决策（decision）** 应基于某些 **信息（information）** 输出，以优化某个 **目标（objective）** 。具体术语有所不同：决策有时被称为输出、行动、预测或分类；信息有时被称为特征、上下文或状态；目标有时被称为效用函数或损失函数。然而，所有这些设置都可以在 **因果影响图（Causal Influence Diagram, CID）** 框架内描述 [3]。CID 是一种 **因果贝叶斯网络（Causal Bayesian Network）** ，其中变量被划分为 **决策变量（decision variable）** $D$、 **效用变量（utility variable）** $U$ 和 **机会变量（chance variable）** $V$，并且不为决策变量指定条件概率分布。智能体的任务是选择分布 $\pi=P(D=d\mid\textbf{Pa}_{D}=\textbf{pa}_{D})$，也称为 **策略（policy）** 或 **决策规则（decision rule）** 。 **最优策略（optimal policy）** $\pi^{*}$ 被定义为使效用期望值 $\mathbb{E}_{\pi^{*}}[U]$ 最大化的策略 $\pi^{*}$。

**参见 [4]**

按照惯例，决策节点绘制为方形，效用节点为菱形，机会节点为圆形。$D$ 的父节点 $\textbf{Pa}_{D}$ 可以解释为决策允许依赖的信息，并用虚线描绘。示例参见 [图 4]。在下文中，我们将注意力限制在一类我们称之为“ **无中介决策任务（unmediated decision task）** ”的 CID 上，其中智能体的决策不会因果性地影响任何后续会影响效用的机会变量。这简化了我们的理论分析，尽管我们的结果很可能可以推广到一般情况。

**参见 [1]**

无中介决策任务的示例包括所有标准的分类和回归任务，以及输出不包含在训练集中的生成式 AI 任务。例如，在分类中，标签的选择通常不会影响数据生成过程。属于 **中介决策任务（mediated decision task）** 而非无中介决策任务的问题包括大多数控制和强化学习任务，其中智能体的决策是影响环境状态的行为。此外，我们将专注于 **非平凡的无中介决策任务（non-trivial unmediated decision task）** ，即 $U\in\textbf{Ch}_{D}$ 的情况，因为 $\textbf{Ch}_{D}=\emptyset$ 的情况描述的是 **平凡的决策任务（trivial decision task）** （智能体的行为不影响效用）。[图 4] 是一个非平凡无中介决策任务的示例。

在 **迁移学习（Transfer Learning）** 中，我们通常关注的是从源域泛化到目标域并非平凡的问题；在平凡情况下，我们不能期望智能体为了泛化而必须学习其环境的任何信息。如果情况并非如此，那么在分布偏移下进行泛化就是平凡的。因此，我们将注意力限制在那些环境分布与智能体确定其 **策略（Policy）** 相关的决策任务上。

具体而言，如果存在一个单一的 **策略（Policy）** ，对于环境分布 $P(\bm{C}=c)$ 的所有选择都是最优的，那么我们称该决策任务是 **域无关的（Domain Independent）** 。

参见 [2](https://arxiv.org/html/2402.10877v7#Thmassumption2)

###### **引理 1（Lemma 1）** 。

**域依赖性（Domain Dependence）** 意味着：

- i)
  不存在 $d\in\textit{dom}(D)$，使得 $d\in\operatorname*{arg\,max}_{d}U(d,c)$ 对所有 $\bm{c}\in\textit{dom}(C)$ 都成立。
- ii)
  $\textbf{Pa}_{D}\subsetneq\textbf{Anc}_{U}$
- iii)
  $D\in\textbf{Pa}_{U}$

###### **证明（Proof）** 。

i) 对于任意 $P^{\prime}$，我们有 $\mathbb{E}_{P^{\prime}}[U\mid\textrm{do}(D=d),\textbf{pa}_{D}]=\sum_{\bm{c}}P^{\prime}(\bm{C}_{d}=\bm{c}\mid\textbf{pa}_{D})U(d,\bm{c})$ $=\sum_{\bm{c}}P^{\prime}(\bm{C}=\bm{c}\mid\textbf{pa}_{D})U(d,c)$，其中我们使用了 $\textbf{Desc}_{D}\cap\textbf{Anc}_{U}=\emptyset$。

因此，如果存在 $d^{*}$ 使得 $d^{*}=\operatorname*{arg\,max}_{d}U(d,\bm{c})$ 对所有 $\bm{C}=\bm{c}$ 成立，那么对于所有 $d\neq d^{*}$，有 $\mathbb{E}_{P^{\prime}}[U\mid\textrm{do}(D=d^{*}),\textbf{pa}_{D}]\geq\sum_{\bm{c}}P^{\prime}(\bm{C}=\bm{c}\mid\textbf{pa}_{D})U(d^{\prime},\bm{c})\mathbb{E}_{P^{\prime}}[U\mid\textrm{do}(D=d),\textbf{pa}_{D}]$，因此 $D=d^{*}$ 对所有 $P^{\prime}(\bm{C}=\bm{c})$ 都是最优的，这就违反了域依赖性。

ii) 由于 $D \in \textbf{Pa}_{U}$（由 iii 可知），则 $\textbf{Pa}_{U} \subseteq \textbf{Anc}_{U}$。
如果 $\textbf{Anc}_{U} = \textbf{Pa}_{D}$，那么 $\mathbb{E}_{P}[u \mid d, \textbf{pa}_{D}] = U(d, \textbf{pa}_{D})$，它与 $P(\bm{C}=\bm{c})$ 无关，因此存在一个对所有 $P$ 都最优的单一策略，这就违反了 **领域依赖性（Domain Dependence）** 。

iii) 如果 $D \notin \textbf{Anc}_{U}$，那么该 **因果影响图（Causal Influence Diagram, CID）** 是平凡的，即 $\mathbb{E}[U \mid \textrm{do}(D=d)] = \mathbb{E}[U]$，因此所有决策对于所有分布 $P(\bm{C})$ 都是最优的，这违反了领域依赖性（[假设 2](https://arxiv.org/html/2402.10877v7#Thmassumption2)）。
因此 $D \in \textbf{Anc}_{U}$，结合 $\textbf{Desc}_{D} \cap \textbf{Anc}_{U} = \emptyset$，意味着 $D \in \textbf{Pa}_{U}$。
∎

<a id="figure-4"></a>

![figure_4](images/figure_4.svg)

> **图 4（Figure 4）** ：一个 **无中介决策任务（Unmediated Decision Task）** 的因果影响图，其中 $D$ 对环境状态 $\bm{C}$ 没有因果影响。我们的主要定理意味着，一个对 $\bm{C}$ 上的 **分布偏移（Distributional Shifts）** 具有鲁棒性的智能体，必须学习覆盖 $\textbf{Anc}_{U}=\{C_{1},C_{2},C_{3},C_{4},C_{6}\}$ 的 **因果贝叶斯网络（Causal Bayesian Network, CBN）** ，注意 $C_{5} \notin \textbf{Anc}_{U}$。$C_{4}$ 是一个变量的例子，它仅通过 $D$ 成为 $U$ 的祖先，因此对 **效用（Utility）** 没有直接的因果效应，但它作为 $U$ 的一个原因 $C_{1}$ 的 **代理变量（Proxy）** ，仍然与决策任务相关。$C_{6}$ 是 $U$ 的一个原因，但不是 $D$ 的原因，人们可能会天真地认为 $C_{6}$ 上的分布偏移不会影响智能体的决策。然而，最优策略在 $C_{6}$ 上的分布偏移下可能改变，因为这些偏移会影响效用，因此，如果智能体要对 $C_{6}$ 上的偏移具有鲁棒性，就必须学习一个包含 $C_{6}$ 的因果贝叶斯网络。

###### 引理 1（Lemma 1）.

ii) 由于 $D \in \textbf{Pa}_{U}$（由 iii 可知），则 $\textbf{Pa}_{U} \subseteq \textbf{Anc}_{U}$。
如果 $\textbf{Anc}_{U} = \textbf{Pa}_{D}$，那么 $\mathbb{E}_{P}[u \mid d, \textbf{pa}_{D}] = U(d, \textbf{pa}_{D})$，它与 $P(\bm{C}=\bm{c})$ 无关，因此存在一个对所有 $P$ 都最优的单一策略，这就违反了 **领域依赖性（Domain Dependence）** 。

iii) 如果 $D \notin \textbf{Anc}_{U}$，那么该 **因果影响图（Causal Influence Diagram, CID）** 是平凡的，即 $\mathbb{E}[U \mid \textrm{do}(D=d)] = \mathbb{E}[U]$，因此所有决策对于所有分布 $P(\bm{C})$ 都是最优的，这违反了领域依赖性（[假设 2](https://arxiv.org/html/2402.10877v7#Thmassumption2)）。
因此 $D \in \textbf{Anc}_{U}$，结合 $\textbf{Desc}_{D} \cap \textbf{Anc}_{U} = \emptyset$，意味着 $D \in \textbf{Pa}_{U}$。
∎

<a id="figure-4"></a>

![figure_4](images/figure_4.svg)

> **图 4（Figure 4）** ：一个 **无中介决策任务（Unmediated Decision Task）** 的因果影响图，其中 $D$ 对环境状态 $\bm{C}$ 没有因果影响。我们的主要定理意味着，一个对 $\bm{C}$ 上的 **分布偏移（Distributional Shifts）** 具有鲁棒性的智能体，必须学习覆盖 $\textbf{Anc}_{U}=\{C_{1},C_{2},C_{3},C_{4},C_{6}\}$ 的 **因果贝叶斯网络（Causal Bayesian Network, CBN）** ，注意 $C_{5} \notin \textbf{Anc}_{U}$。$C_{4}$ 是一个变量的例子，它仅通过 $D$ 成为 $U$ 的祖先，因此对 **效用（Utility）** 没有直接的因果效应，但它作为 $U$ 的一个原因 $C_{1}$ 的 **代理变量（Proxy）** ，仍然与决策任务相关。$C_{6}$ 是 $U$ 的一个原因，但不是 $D$ 的原因，人们可能会天真地认为 $C_{6}$ 上的分布偏移不会影响智能体的决策。然而，最优策略在 $C_{6}$ 上的分布偏移下可能改变，因为这些偏移会影响效用，因此，如果智能体要对 $C_{6}$ 上的偏移具有鲁棒性，就必须学习一个包含 $C_{6}$ 的因果贝叶斯网络。

###### 证明（Proof）。

### A.2 CID 的参数化（Parameterisation of CIDs）

联合分布 $P$ 是为所有环境变量 $\bm{C}$ 定义的，而 **因果影响图（Causal Influence Diagram, CID）** 则由 $P(\bm{C})$ 和 $U(\textbf{Pa}_{U})$ 的参数定义。
我们将注意力限制在分类变量 $\bm{C}$ 上，并且不失一般性地将状态标记为 $c_{i}=0,1,\ldots,\text{dim}_{i}-1$，其中 $\text{dim}_{i}$ 是变量 $C_{i}$ 的维度。
首先，联合分布 $P(\bm{C})$ 由关于图 $G$ 的 **马尔可夫分解（Markov factorization）** 中的 **条件概率分布（Conditional Probability Distributions, CPDs）** 参数化，即 $\theta_{P}=\{P(c_{i}\mid\textbf{pa}_{i})\,\forall\,c_{i}\in\{0,\ldots,\text{dim}_{i}-2\},\textbf{pa}_{i}\in\textbf{Pa}_{i},\,C_{i}\in\bm{C}$。
请注意，CPDs $p(C_{i}=\text{dim}_{i}-1\mid\textbf{pa}_{i})$ 不包含在 $\theta_{P}$ 中，因为它们完全由归一化条件 $P(C_{i}=\text{dim}_{i}-1\mid\textbf{pa}_{i})=1-\sum_{j=0}^{\text{dim}_{i}-1}P(c_{i}\mid\textbf{pa}_{i})$ 约束。
其次， **效用函数（Utility function）** $U$ 由其父节点状态给定的值简单参数化：$\theta_{U}=\{U(\textbf{pa}_{U})\,\forall\,\textbf{Pa}_{U}=\textbf{pa}_{U}\}$。
为简化起见，我们使用归一化的效用函数：

$$
U(\textbf{pa}_{U})\rightarrow\frac{U(\textbf{pa}_{U})-\min_{\textbf{pa}_{U}^{\prime}}U(\textbf{Pa}_{U}=\textbf{pa}^{\prime}_{U})}{\max_{\textbf{pa}_{U}^{\prime}}U(\textbf{Pa}_{U}=\textbf{pa}^{\prime}_{U})-\min_{\textbf{pa}_{U}^{\prime}}U(\textbf{Pa}_{U}=\textbf{pa}^{\prime}_{U})} \tag{2}
$$

其值在 0 和 1 之间。注意到由于这是效用函数的一个正仿射变换，最优策略集保持不变，并且我们可以相应地重新缩放 **遗憾界（regret bounds）** 。
令 $\theta_{M}$ 表示 CID 的所有参数集合，$\theta_{M}=\theta_{P}\cup\theta_{U}$，并注意 $\theta_{M}$ 中的元素在 $[0,1]$ 区间内且在逻辑上是独立的，即我们可以为每个参数独立选择任何 $[0,1]$ 值，这定义了基线环境下 CID 的一个有效参数化。
在下文中，当我们提到“参数 $P,U$”时，我们指的是 $\theta_{M}$。

我们遵循 (Meek, 2013) 中概述的方法来证明对 $P,U$ 的某些约束“对几乎所有 $P,U$ 成立”，从而对几乎所有决策任务成立。
这涉及将给定的约束转换为关于 $\theta_{M}$ 的多项式方程，并应用以下引理：

###### 引理 2（Lemma 2）( Okamoto, 1973 ) .

（非平凡）多项式的解集在其参数空间上的 **勒贝格测度（Lebesgue measure）** 为零。

一个 $n$ 元多项式是 **非平凡的（non-trivial）** （即不是恒等式），当且仅当并非所有 $n$ 个变量的实例化都是该多项式的解。
例如，方程 $\text{poly}(\theta_{M})=0$ 是平凡的，当且仅当多项式表达式 $\text{poly}(\theta_{M})$ 的所有系数均为零。
因此，任何可以转换为关于 $\theta_{M}$ 的多项式方程的、对 $P, U$ 的约束，要么对所有 $\theta_{M}$ 都成立，要么仅对 $\theta_{M}$ 实例化的一个勒贝格测度为零的子集成立。

从操作上讲，这意味着如果我们在参数空间上存在任何平滑分布（例如，描述我们预期遇到的环境的分布），那么从该分布中抽取一个不满足该条件的环境的概率为 0。

###### 引理 2（Okamoto, 1973）。

### A.3 分布偏移与策略预言机

在推导我们的结果时，我们将注意力限制在那些可以建模为对数据生成过程的（软）干预的分布偏移上。
我们注意到，根据 **赖兴巴赫原理（Reichenbach's principle）** （Reichenbach, 1956），该原理指出所有统计关联都源于潜在的因果结构，我们可以假设存在一个可以用 **因果贝叶斯网络（Causal Bayesian Network, CBN）** $M=(P,G)$ 描述的因果数据生成过程。
因此，联合分布存在一个因果分解：$P(\bm{C}=\bm{c})=\prod_{i}P(c_{i}\mid\textbf{Pa}_{i})$。
通过允许干预的混合，我们可以达到 $\bm{C}$ 上的任意分布。这一点可以简单地看出：我们可以执行一个软干预来实现任意确定性分布 $P(\bm{C}=\bm{c})=\delta(\bm{C}=\bm{c}^{\prime})$，然后对这些确定性分布进行混合，从而得到 $\bm{C}$ 上的任意分布。
无法通过干预生成的分布集包括那些改变变量集 $\bm{V}$（包括决策变量和效用变量）的分布，以及引入选择偏差的分布（这在因果上通过引入额外的、被条件化的节点来表示，参见 Bareinboim & Pearl, 2012a）。
关于分布偏移与干预之间关系的进一步讨论，请参见 Schölkopf 等人（2021）和 Meinshausen（2018）。

在接下来的证明中，我们使用 **策略预言机（policy oracles）** 来形式化在分布偏移下具有有界遗憾行为的知识。

###### 定义 5（策略预言机， Policy oracle）.

对于一组干预措施 $\Sigma$，一个 **策略预言机（Policy oracle）** 是一个映射 $\Pi^{\delta}_{\Sigma}:\sigma\mapsto\pi_{\sigma}(d\mid\textbf{pa}_{D})$，$\forall$ $\sigma\in\Sigma$，其中 $\Sigma$ 是一组领域。
如果策略 $\pi_{\sigma}(d\mid\textbf{pa}_{D})$ 在因果影响图（Causal Influence Diagram, CID） $M(\sigma)$ 中达到期望效用 $\mathbb{E}^{\pi_{\sigma}}[U]\geq\mathbb{E}^{\pi^{*}}[U]-\delta$，其中 $\delta\geq 0$，则该预言机是 $\delta$-最优的。

这里的 $\delta$ 是遗憾（Regret）上界，它在所有分布偏移 $\sigma\in\Sigma$ 下均被满足。
我们将 $\delta=0$ 时的 $\delta$-最优策略预言机称为 **最优策略预言机（Optimal policy oracles）** 。
为了证明我们的主要结果，我们将注意力限制在那些 $\Sigma$ 包含所有局部干预（[定义 2](https://arxiv.org/html/2402.10877v7#Thmdfn2)）的混合的策略预言机上。

请注意，策略预言机仅指定智能体在分布偏移环境 $M(\sigma)$ 中返回什么策略。
它并不指定该策略是如何生成的，这将取决于具体的设置。
例如，在 **领域泛化（Domain generalisation）** 中，智能体通常不会从目标领域获得额外数据，但期望它能产生一个在所有目标领域都能实现低遗憾的策略（决策边界）。
另一方面，在 **领域适应（Domain adaptation）** 和 **小样本学习（Few-shot learning）** 中，智能体会获得来自每个目标领域的一些新数据，用以调整其策略。
由于我们希望容纳所有这些视角，因此我们只指定智能体的策略，而不指定用于生成它的数据。
这一点在 [第 3.2 节](#section-3-2) 中有进一步讨论。

我们考虑哪些分布偏移？在我们的证明中，我们假设智能体对任何可以描述为对环境变量 $\bm{C}$ 进行局部干预的混合的领域偏移都具有鲁棒性。我们不考虑改变效用 $U$ 或智能体决策 $D$ 的干预，尽管我们确实将策略输入的丢弃（掩码） $\textbf{Pa}_{D}\rightarrow\textbf{Pa}_{D}^{\prime}\subseteq\textbf{Pa}_{D}$ 视为局部干预。

###### 定义 5（策略预言机， Policy oracle）.

<a id="appendix-b"></a>

## 附录 B（Appendix B）：简化证明

本节将概述针对一个具有二元潜在变量的简单二元决策任务的 **[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1)** 的证明。如 **[第 4 节](#section-4)** 所述， **[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1)** 中用于识别因果贝叶斯网络（Causal Bayesian Network, CBN）的方法，可被视为一种 **算法（algorithm）** ：通过观察一个 **后悔有界（regret-bounded）** 的智能体在不同 **分布偏移（distributional shifts）** 下的 **策略（policy）** ，来学习潜在变量上的 CBN。为了证明这一点，在 **[附录 F](#appendix-f)** 中，我们在随机生成的因果影响图（Causal Influence Diagram, CID）上实现了该算法，通过实验证明我们可以通过这种方式学习底层的 CBN，并探讨智能体的后悔界如何影响所学 CBN 的准确性。

考虑 **[图 5](https://arxiv.org/html/2402.10877v7#A2.F5)** 中的 CID，它描述了一个二元决策任务 $D\in\{0,1\}$，其中包含两个二元潜在变量 $X,Y\in\textbf{Pa}_{U}$。

<a id="figure-5"></a>

![figure_5](images/figure_5.svg)

> 图 5：描述一个具有二元潜在变量 $X,Y$ 的 **上下文无关多臂老虎机（context-free multi-armed bandit）** 的示例 CID。

考虑一个选择策略 $\pi_{D}$ 的智能体，其目标是最大化期望效用。
也就是说，该 CID 描述了一个上下文无关的老虎机问题，其中 $X,Y$ 是影响臂值 $\mathbb{E}[u\mid d]=\sum_{x,y}P(x,y)U(x,y,d)$ 的潜在变量。

我们的目标是，在仅已知智能体在分布偏移下的策略及其满足一个遗憾界（regret bound）的前提下，学习这个 **因果影响图（Causal Influence Diagram, CID）** 。
我们假设已知：i) 机会变量（chance variables）集合 $\bm{C}=\{X,Y\}$，ii) 效用函数 $U(d,x,y)$，以及 iii) 在分布偏移 $\sigma$ 下的策略 $\pi_{D}(\sigma)$（其他变量 $U, X, Y$ 未被观测）。
因此，为了学习 CID，目标就是学习潜在变量联合分布 $P(x,y)$ 的参数以及未知的因果结构。
由于我们知道效用函数，我们知道 $D, X, Y \in \textbf{Pa}_{U}$，并且通过假设 CID 是无中介的（[假设 1](https://arxiv.org/html/2402.10877v7#Thmassumption1)），我们知道 $X, Y \not\in \textbf{Desc}_{D}$。
同样地，决策任务是无上下文的，因此 $D \not\in \textbf{Desc}_{X} \cup \textbf{Desc}_{Y}$。
因此，唯一的未知因果结构是潜在变量 $\bm{C}=\{X,Y\}$ 上的 **有向无环图（Directed Acyclic Graph, DAG）** 。

在对 $X$ 进行 **硬干预（hard intervention）** 后，$D=0$ 与 $D=1$ 之间的期望效用差由下式给出：

$$
\displaystyle\mathbb{E}[u\mid D=0;\textrm{do}(X=0)]-\mathbb{E}[u\mid D=1; \textrm{do}(X=0)]=\sum\limits_{y}P(Y_{X=0}=y)[U(0,0,y)-U(1,0,y)] (3) \displaystyle=P(Y_{X=0}=0)[U(0,0,Y=0)-U(1,0,0)]+(1-P(Y_{X=0}=0))[U(0,0,1)-U(1, 0,1)](4)
$$

由于我们知道 $U(d,x,y)$，因此如果我们能识别出这个期望效用差，就能识别出 $P(Y_{X=0}=0)$。
我们利用智能体在分布偏移下的策略来实现这一点，在这个简单情况下，我们可以将注意力限制在硬干预上。
遵循 [引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4) 概述的步骤，域依赖性（domain dependence）确保我们可以识别出一个硬干预 $\sigma_{2}=\textrm{do}(X=x^{\prime},Y=y^{\prime})$，该干预导致的最优策略与 $\sigma_{1}=\textrm{do}(X=0)$ 下的最优策略不同。
对于这两种干预的混合 $\sigma_{3}=q\sigma_{1}+(1-q)\sigma_{2}$，期望效用为 $\mathbb{E}[u\mid d,\sigma_{3}]=q\mathbb{E}[u\mid d,\sigma_{1}]+(1-q)\mathbb{E}
[u\mid d,\sigma_{2}]$。
这是关于 $q$ 的线性函数，并且当 $q=1$ 时，最优决策 $d_{1}$ 与 $q=0$ 时的最优决策 $d_{2}$ 不同（$d_{2}\neq d_{1}$）。
因此，存在一个唯一的 **无差异点（indifference point）** $q_{\text{crit}}$，在该点两种决策都是最优的。
可以简单地证明，这个无差异点由下式给出：

$$
q_{\text{crit}}=\left(1-\frac{\mathbb{E}[u\mid D=d_{1};\textrm{do}(X=0)]- \mathbb{E}[u\mid D=d_{2};\textrm{do}(X=0)]}{U(d_{1},x^{\prime},y^{\prime})-U(d _{2},x^{\prime},y^{\prime})}\right)^{-1}(5)
$$

当 $q \leq q_{\text{crit}}$ 时，$D=d_{1}$ 是最优的；当 $q \geq q_{\text{crit}}$ 时，$D=d_{2}$ 是最优的。
我们可以通过在 $[0,1]$ 上均匀地随机采样 $q$ 值，并观察在由此产生的混合干预下的最优决策来估计 $q_{\text{crit}}$（[算法 1](#algorithm-1)）。
也就是说，$q_{\text{crit}}$ 是策略预言机（policy oracle）对于一个随机采样的 $q$ 返回 $D=d_{1}$ 的概率。通过这种方式，我们学习到 $q_{\text{crit}}$，并且由于我们知道 $U(d,x,y)$，我们可以识别出 [公式 5](https://arxiv.org/html/2402.10877v7#A2.E5) 分子中 $\textrm{do}(X=0)$ 下的期望效用差，从而识别出 $P(Y_{X=0}=0)$。

类似地，我们识别 $P(Y_{X=1}=0), P(X_{Y=0}=0)$ 和 $P(X_{Y=1}=0)$。这些概率编码了 $X$ 和 $Y$ 之间的因果关系（例如，当且仅当对于几乎所有 **因果贝叶斯网络（Causal Bayesian Networks, CBNs）** 有 $P(Y_{X=0})\neq P(Y_{X=1})$ 时，存在一条从 $X$ 到 $Y$ 的有向路径），并将 CBN 的参数确定为 $P(C_{i}=c_{i}\mid\textrm{do}(\bm{C}\setminus C_{i}))=P(C_{i}=c_{i}\mid\textbf{
Pa}_{i}=\textbf{pa}_{i})$。

<a id="appendix-c"></a>

## 附录 C 定理 1 的证明（Proof of Theorem 1）

在本附录中，我们证明[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1)。关于证明的非正式概述，请参见[附录 B](#appendix-b)。

首先，我们证明对于给定的分布偏移 $\sigma$，对于 **几乎所有（almost all）** $P, U$，存在唯一的最优决策。虽然这对我们的证明并非必需，但它简化了我们的分析。并且由于我们的主要定理对几乎所有 $P, U$ 都成立，我们可以纳入任何有限数量的、对几乎所有 $P, U$ 都成立的独立条件，而不会加强此条件，因为 **勒贝格测度（Lebesgue measure）** 为零的集合的并集仍然是勒贝格测度为零。

###### 引理 3（Lemma 3）。

对于任何给定的 **局部干预（local intervention）** $\sigma$，对于几乎所有 $P, U$，存在唯一的 **确定性最优策略（deterministic optimal policy）** 。

###### 证明。

在干预 $\sigma$ 之后，两个决策 $d, d^{\prime}$ 在上下文 $\textbf{pa}_{D}$ 中同时最优的条件是：

$$
\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d);\sigma]=\mathbb{E}[u\mid \textbf{pa}_{D},\textrm{do}(D=d^{\prime});\sigma] \tag{6}
$$

令 $\bm{Z}=[\textbf{Anc}_{U}\setminus\textbf{Pa}_{D}]$，$\bm{X}=\textbf{Pa}_{U}\setminus\{D\}$。注意到：

$$
\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d);\sigma]=\sum_{\bm{z}}U(d,\bm{x})P(\bm{z},\textbf{pa}_{D}\mid\textrm{do}(D=d);\sigma)/P(\textbf{pa}_{D}\mid \textrm{do}(D=d);\sigma) \tag{7}
$$

并且由 $\textbf{Desc}_{D}\cap\textbf{Anc}_{U}=\emptyset$ 可得 $P(\textbf{pa}_{D}\mid\textrm{do}(D=d);\sigma)=P(\textbf{pa}_{D};\sigma)$ 和 $P(\bm{z},\textbf{pa}_{D}\mid\textrm{do}(D=d);\sigma)=P(\bm{z},\textbf{pa}_{D};\sigma)$。我们可以将方程 [6](https://arxiv.org/html/2402.10877v7#A3.E6) 两边同时乘以 $P(\textbf{pa}_{D};\sigma)$，得到：

$$
\sum_{\bm{z}}U(d,\bm{x})P(\bm{z},\textbf{pa}_{D};\sigma)=\sum_{\bm{z}}U(d^{\prime},\bm{x})P(\bm{z},\textbf{pa}_{D};\sigma) \tag{8}
$$

以及：

$$
\sum_{\bm{z}}[U(d,\bm{x})-U(d^{\prime},\bm{x})]P(\bm{z},\textbf{pa}_{D};\sigma )=0 \tag{9}
$$

令 $\sigma=\textrm{do}(v_{1}=f_{1}(v_{1}),\ldots,v_{N}=f_{N}(v_{N}))$。联合分布 $P(\bm{z},\textbf{pa}_{D};\sigma)=\prod_{i}P(c_{i}\mid\textbf{pa}_{i};\sigma)$ 是多项式，且局部干预 $P(c_{i}\mid\textbf{pa}_{i};\sigma)=\sum_{c^{\prime}_{i}:f_{i}(c^{\prime}_{i})=c_{i}}P(c^{\prime}_{i}\mid\textbf{pa}_{i})$ 保持了其多项式形式。因此，方程 [9](https://arxiv.org/html/2402.10877v7#A3.E9) 是一个关于模型参数的多项式方程，并且由于 $d\neq d^{\prime}$，它必定是非平凡的。因此，根据[引理 2](https://arxiv.org/html/2402.10877v7#Thmlemma2)，对于几乎所有 $P, U$，方程 [9](https://arxiv.org/html/2402.10877v7#A3.E9) 不成立。由于决策数量有限，这意味着对于几乎所有 $P, U$，对于给定的 $\sigma$ 和 $\textbf{pa}_{D}$，存在唯一的最优决策，从而存在唯一的最优策略。
∎

接下来，我们将详细说明如何利用 **策略预言机（policy oracle）** 来识别偏移环境 $M(\sigma)$ 中的一个特定因果查询，我们稍后将使用它来识别模型参数。

###### 引理 4（Lemma 4）。

使用一个最优策略预言机（Optimal Policy Oracle）$\Pi^{*}_{\Sigma}$，其中 $\Sigma$ 包含对 $\bm{C}$ 的所有局部干预（Local Interventions）的混合，包括掩码输入（Masking Inputs） $\textbf{Pa}_{D}^{\prime}\subseteq\textbf{Pa}_{D}$，那么对于任何给定的 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 且满足 $\textbf{Pa}_{D}^{\prime}\cap\textbf{Pa}_{U}=\emptyset$，我们可以识别出 $\sum_{z}P(\bm{C}=\bm{c};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm{c})]$，其中 $d$ 和 $d^{\prime}$ 满足 $d\neq d^{\prime}$，且 $\bm{Z}=\bm{C}\setminus\textbf{Pa}_{D}^{\prime}$。

###### 证明。

根据 [引理 3](https://arxiv.org/html/2402.10877v7#Thmlemma3)，对于几乎所有的 $P,U$，在偏移 $\sigma$ 之后存在一个单一的最优决策。
令 $d_{1}=\operatorname*{arg\,max}_{d}\mathbb{E}[u\mid\textrm{do}(D=d),\textbf{pa}_{D}^{\prime};\sigma]$，其中 $d_{1}=\pi^{*}(\sigma)$。
我们可以通过用 $\sigma$ 查询策略预言机来识别 $d_{1}$。

考虑对所有 $C_{i}\in\bm{C}$ 的硬干预（Hard Intervention），$\sigma^{\prime}:=\textrm{do}(c^{\prime}_{1},c^{\prime}_{2},\ldots,c^{\prime}_{N})$，其中对于所有 $C_{i}\in\textbf{Pa}_{D}^{\prime}$，我们设置 $C_{i}=c_{i}$ 为与观测 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 中相同的状态。
在此干预下的期望效用（Expected Utility）为 $\mathbb{E}[u\mid\textrm{do}(D=d),\textbf{pa}_{D}^{\prime};\sigma^{\prime}]=U(d,\bm{x}^{\prime})$，其中 $\bm{X}=\textbf{Pa}_{U}\setminus\{D\}$（并且根据 [引理 1](https://arxiv.org/html/2402.10877v7#Thmlemma1) iii)，我们有 $D\in\textbf{Pa}_{U}$）。

接下来我们证明，存在一种硬干预 $\sigma^{\prime}$ 的选择，使得策略预言机在上下文 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 中，对于 $\sigma^{\prime}$ 和 $\sigma$ 必须返回不同的最优决策。
由于 $\textbf{Pa}_{D}^{\prime}\cap\textbf{Pa}_{U}=\emptyset$，我们可以自由选择任何 $X=x^{\prime}$，并且由此产生的 $\sigma^{\prime}$ 将与证据 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 兼容。
注意，根据 [引理 1](https://arxiv.org/html/2402.10877v7#Thmlemma1) i)，存在 $\bm{X}=\bm{x}^{\prime}$ 使得 $d_{1}\neq\operatorname*{arg\,max}_{d}U(d,x^{\prime})$，否则 $D=d_{1}$ 对于所有 $\bm{X}=\bm{x}$ 都是最优的，这将违反领域依赖性（Domain Dependence）。
给定效用函数（Utility Function）和 $d_{1}$，我们可以确定这个 $\bm{X}=\bm{x}^{\prime}$。
令 $d_{2}=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})$，且 $\sigma^{\prime}=\textrm{do}(c^{\prime}_{1},c^{\prime}_{2},\ldots,c^{\prime}_{N})$ 是使得 $\bm{X}=\bm{x}^{\prime}$ 和 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 成立的硬干预。

考虑在混合局部干预 $\tilde{\sigma}(q)=q\sigma+(1-q)\sigma^{\prime}$ 下 $\bm{C}$ 的联合分布（Joint Distribution），

$$
\displaystyle P(\bm{C}=\bm{c}\mid\textrm{do}(D=d);\tilde{\sigma}(q)) \displaystyle=P(\bm{C}=\bm{c};\tilde{\sigma}(q)) \tag{10} \displaystyle=qP(\bm{C}=\bm{c};\sigma)+(1-q)P(\bm{C}=\bm{c};\sigma^{\prime}) \tag{11}
$$

其中在第一行我们使用了 $\textbf{Ch}_{D}=\{U\}$ 来移除干预。
注意，根据 [引理 1](https://arxiv.org/html/2402.10877v7#Thmlemma1) i)，有 $\bm{Z}=\bm{C}\setminus\textbf{Pa}_{D}\neq\emptyset$。
期望效用由下式给出，

$$
\displaystyle\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d);\tilde{\sigma}(q )]=\sum\limits_{\bm{z}}P(\bm{Z}=\bm{z}\mid\textbf{pa}_{D},\textrm{do}(D=d); \tilde{\sigma}(q))U(d,\bm{x}) (12) \displaystyle=\sum\limits_{\bm{z}}\frac{P(\bm{C}=\bm{c}\mid\textrm{do}(D=d); \tilde{\sigma}(q))}{P(\textbf{pa}_{D}\mid\textrm{do}(D=d);\tilde{\sigma}(q))}U (d,\bm{x}) (13) \displaystyle=\frac{1}{P(\textbf{pa}_{D};\tilde{\sigma}(q))}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\tilde{\sigma}(q))U(d,\bm{x}) (14) \displaystyle=\frac{1}{P(\textbf{pa}_{D};\tilde{\sigma}(q))}\sum\limits_{\bm{z}}qP(\bm{C}=\bm{c};\sigma)U(d,\bm{x})+(1-q)P(\bm{C}=\bm{c};\sigma^{\prime})U(d ,\bm{x}^{\prime})(15)
$$

注意，当 $q=1$ 时，最优决策为 $d_{1}$；当 $q=0$ 时，策略预言机（Policy Oracle）返回的最优决策属于集合 $\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$，该集合不包含 $d_{1}$。
此外，方程 [15] 关于 $d$ 的 $\operatorname*{arg\,max}$ 是一个定义域为 $q\in[0,1]$ 的分段线性函数。
因此，必然存在某个 $q=q_{\text{crit}}$，它是 $q$ 的最小值，使得当 $q<q_{\text{crit}}$ 时，策略预言机返回的最优决策在集合 $\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$ 中，而当 $q\geq q_{\text{crit}}$ 时，最优决策不在此集合中。
$q_{\text{crit}}$ 的值由 $\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d);\tilde{\sigma}(q_{\text{crit}
})]=0$ 给出，根据方程 [15]，即：

$$
q_{\text{crit}}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm{x})-U(d _{3},\bm{x})]+(1-q_{\text{crit}})[U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})]=0(16)
$$

其中 $d_{2}\in\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$ 且 $d_{3}\not\in\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$。
由此得到 $q_{\text{crit}}$ 的以下表达式：

$$
q_{\text{crit}}=\left(1-\frac{\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_ {2},\bm{x})-U(d_{3},\bm{x})]}{U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime} )}\right)^{-1}(17)
$$

这里我们使用了 $\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma^{\prime})[U(d_{2},\bm{c})-U(d_{3},
\bm{c})]=U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})$。
给定 $q_{\text{crit}}$ 和效用函数（Utility Function） $U(d,\bm{x})$，我们可以确定 $\sum_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm{x})-U(d_{3},\bm{x})]$。

最后，我们描述一个 [算法 1](#algorithm-1)（如下），它使用策略预言机对 $q_{\text{crit}}$ 进行蒙特卡洛估计（Monte Carlo Estimation），该算法可用于在渐近极限 $N\rightarrow\infty$ 下确定 $\sum_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm{c})-U(d_{3},\bm{c})]$) 并识别 $d_{2},d_{3}$。

<a id="algorithm-1"></a>

**算法 1 使用策略预言机识别 $q_{\text{crit}},d_{2},d_{3}$。输入：$(U,\Pi^{*}_{\Sigma},N,\sigma)$**

- $d_{1}\leftarrow\Pi^{*}_{\Sigma}(\sigma)$
- $\sigma^{\prime},d_{2}\leftarrow\text{对}\bm{C}\text{进行任意硬干预，使得}d_{2}=\operatorname*{arg\,max}_{d}U(d,\bm{x})\neq d_{1}$
- $D(q=1)\leftarrow\{d\text{使得}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$
- $\theta=0$
- for $i\leftarrow 1$ to $N$ do
- $q\sim\text{Uniform}(0,1)$
- $\pi^{*}(d\mid\textbf{pa}_{D})\leftarrow\Pi^{*}_{\Sigma}(\sigma_{3}(q))$
- if $d\in D(q=1)$ $\forall$ $\pi^{*}(d\mid\textbf{pa}_{D})>0$ then
- $\theta\leftarrow\theta+1$
- end if
- end for
- $q_{\text{crit}}=\theta/N$
- $D(q_{\text{crit}})\leftarrow\Pi^{*}_{\Sigma}(q_{\text{crit}}\sigma+(1-q_{\text {crit}})\sigma^{\prime})$
- $d_{3}\in D(q_{\text{crit}})$ , $d_{3}\neq d_{2}$
- return $q_{\text{crit}},d_{2},d_{3}$

∎

我们现在准备推导 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1)。

见 [1](https://arxiv.org/html/2402.10877v7#Thmtheorem1)

###### 证明。

我们通过学习“留一法”干预分布 $P(c_{i}\mid\textrm{do}(c_{1},\ldots,c_{i-1},c_{i+1},\ldots,c_{N}))$ 来学习图 $G$ 和参数 $P(c_{i}\mid\textbf{pa}_{i})$。
请注意，在此干预下，$C_{i}$ 仅依赖于其父节点集，因此 $P(c_{i}\mid\textrm{do}(c_{1},\ldots,c_{i-1},c_{i+1},\ldots,c_{N})=P(c_{i}\mid
\textbf{pa}_{i})$，其中 $\textbf{Pa}_{i}=\textbf{pa}_{i}$ 表示在留一法干预下 $\textbf{Pa}_{i}$ 的状态。
几乎所有的 $P$ 都是因果忠实的（Meek, 2013）。
因此，对于几乎所有的 $P$，这些干预分布可以用来确定 $\textbf{Pa}_{i}$，即在干预分布中 $C_{i}\not\perp C_{j}$ 当且仅当 $C_{j}\in\textbf{Pa}_{i}$。
明确地说，对于几乎所有的环境，$C_{j}\in\textbf{Pa}_{i}$ 当且仅当存在两个仅在 $C_{j}$ 上不同的留一法干预，其中 $C_{j}=c_{j}$ 和 $C_{j}=c_{j}^{\prime}$，使得 $P(c_{i}\mid\textrm{do}(c_{1},\ldots,c_{j},\ldots,c_{i-1},c_{i+1},\ldots,c_{N})
)\neq P(c_{i}\mid\textrm{do}(c_{1},\ldots,c^{\prime}_{j},\ldots,c_{i-1},c_{i+1
},\ldots,c_{N}))$。
为简化符号，我们将交替使用 $P(c_{i}\mid\textbf{pa}_{i})$ 和 $P(c_{i}\mid\textrm{do}(c_{1},\ldots,c_{j},\ldots,c_{i-1},c_{i+1},\ldots,c_{N}))$。

首先，我们学习那些具有指向 $U$ 但不包含 $D$ 的有向路径的 **机会变量（Chance variables）** 的参数，即在我们对 $D$ 进行干预的图 $G_{\hat{D}}$ 中，$U$ 的祖先节点。

**情况 1：** 学习 $C_{i}\in\textbf{Anc}_{U}(G_{\hat{D}})$ 的参数。

考虑一条有向路径 $C_{k}\rightarrow\ldots\rightarrow C_{1}$，其中 $C_{1}\in\textbf{Pa}_{U}$ 且所有变量均为 **机会节点（Chance nodes）** （该路径不包含 $D$）。
假设我们已知 $\textbf{Pa}_{k-1},\ldots,\textbf{Pa}_{1}$ 以及参数 $P(C_{i}\mid\textbf{pa}_{i})$，其中 $i=k-1,\ldots,1$。
我们将证明，给定这些参数，我们可以识别出未知参数 $P(c_{k}\mid\textbf{pa}_{k})$（从而识别出 $\textbf{Pa}_{k}$）。
定义 $\bm{Y}=\bm{C}\setminus\{C_{k},\ldots,C_{1}\}$，并考虑局部干预 $\sigma=\textrm{do}(y_{1},\ldots,y_{N-k},c_{k}=f(c_{k}))$，其中 $\textrm{do}(c_{k}=f(c_{k}))$ 是对 $C_{k}$ 的一个局部干预，使得：

$$
f(C_{k})=\begin{cases}c_{k}^{\prime}\,,\,C_{k}=c_{k}^{\prime}\\ c_{k}^{\prime\prime}\,\text{otherwise}\end{cases}(18)
$$

即，$f(C_{k})$ 将 $C_{k}$ 映射到一个二维子空间，其中 $C_{k}=c_{k}^{\prime}$ 的像为 $C_{k}=c_{k}^{\prime}$，而所有其他状态被映射到 $C_{k}=c_{k}^{\prime\prime}$，这里 $c_{k}^{\prime},c_{k}^{\prime\prime}\neq c_{k}^{\prime}$ 是 $C_{k}$ 的任意状态。
在下文中，我们将策略的所有输入屏蔽，即 $\textbf{Pa}_{D}^{\prime}=\emptyset$。

根据 [引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4)，我们可以识别：

$$
\displaystyle\sum_{\bm{c}}P(\bm{C}=\bm{c};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm {c})] \displaystyle=\sum_{c_{k}}\ldots\sum_{c_{1}}P(c_{k}\mid\textbf{pa}_{k};\sigma) \ldots P(c_{1}\mid\textbf{pa}_{1};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm{c})] \displaystyle=\sum_{c_{k}}P(c_{k}\mid\textbf{pa}_{k};\sigma)\beta(c_{k})(19)
$$

其中

$$
\beta(c_{k}):=\sum_{c_{k-1}}\ldots\sum_{c_{1}}P(c_{k-1}\mid\textbf{pa}_{k-1}; \sigma)\ldots P(c_{1}\mid\textbf{pa}_{1};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm{c})](20)
$$

且 $\beta(c_{1}):=[U(d,\bm{c})-U(d^{\prime},\bm{c})]$。
注意，在方程 [19](https://arxiv.org/html/2402.10877v7#A3.Ex5) 中，$\beta(c_{k})$ 由已知参数 $P(c_{k-1}\mid\textbf{pa}_{k-1}),\dots,P(c_{1}\mid\textbf{pa}_{1})$ 和 $U(\textbf{pa}_{U})$ 决定，并且对于几乎所有的 $P,U$，$\beta(c_{k})$ 非零，因为 $\beta(c_{k})=0$ 是这些参数的一个多项式方程，对于几乎所有的 $P,U$ 都不成立。

利用方程 [18](https://arxiv.org/html/2402.10877v7#A3.E18) 中局部干预的定义，我们有 $P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k};\sigma)=P(C_{k}=c_{k}^{\prime}\mid
\textbf{pa}_{k})$，以及 $P(C_{k}=c_{k}^{\prime\prime}\mid\textbf{pa}_{k};\sigma)=1-P(C_{i}=c_{k}^{
\prime}\mid\textbf{pa}_{k};\sigma)=1-P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k})$。
因此，方程 [19](https://arxiv.org/html/2402.10877v7#A3.Ex5) 的右侧只有一个待定参数 $P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k})$，而左侧可以通过策略预言机（[引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4)）确定，从而我们可以解出 $P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k})$。
通过使用不同的干预重复此过程，改变硬干预 $\textrm{do}(\bm{Y}=\bm{y})$ 以及 $c_{k}^{\prime},c_{k}^{\prime\prime}$ 的选择，我们可以识别出所有 $c_{k},\textbf{pa}_{k}$ 对应的 $P(c_{k}\mid\textbf{pa}_{k})$，从而识别出 $\textbf{Pa}_{k}$。

我们现在学习所有 $C_{i} \in \textbf{Anc}_{U}(G_{\hat{D}})$ 的参数。
我们知道集合 $\textbf{Pa}_{U}$，因为这是效用函数 $U(\textbf{Pa}_{U})$ 的定义域，而根据假设该函数是已知的。
然后我们可以迭代进行，首先学习 $P, G$ 中对应于某个 $C_{1} \in \textbf{Pa}_{U}$ 的参数 $P(c_{1} \mid \textbf{pa}_{1})$ 和 $\textbf{Pa}_{1}$。
我们可以这样做：$\beta(c_{1}) = U(d, \bm{x}) - U(d^{\prime}, \bm{x})$，其中 $d, d^{\prime}, \bm{x}$ 由 [引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4) 中的 [算法 1](#algorithm-1) 返回，并且 $U(\textbf{Pa}_{U})$ 是已知的。然后我们可以确定所有 $C_{j} \in \textbf{Pa}_{1}$ 的参数，依此类推，直到遍历完 $\textbf{Anc}_{1}$。
我们对所有 $C_{i} \in \textbf{Pa}_{U}$ 重复此过程，直到覆盖所有 $\textbf{Anc}_{U}(G_{\hat{D}})$。

**情况 2：学习 $C_{i} \in \textbf{Anc}_{D}$，$C_{i} \not\in \textbf{Anc}_{U}(G_{\hat{D}})$ 的参数。**

考虑 $C_{k} \in \textbf{Anc}_{U}$，其所有指向 $U$ 的有向路径都经过 $D$，即 $C_{k} \rightarrow C_{k-1} \rightarrow \ldots \rightarrow C_{1}$，其中 $C_{1} \in \tilde{P}a_{D}$。
如前所述，假设我们知道 $\textbf{Pa}_{k-1}, \ldots, \textbf{Pa}_{1}$ 以及参数 $P(C_{i} \mid \textbf{pa}_{i})$，其中 $i = k-1, \ldots, 1$。
我们现在证明，给定这些参数，我们可以识别未知参数 $P(c_{k} \mid \textbf{pa}_{k})$（从而识别 $\textbf{Pa}_{k}$）。
定义 $\bm{Y} = \bm{C} \setminus \{C_{k}, \ldots, C_{1}\}$，并令 $\sigma = \textrm{do}(y_{1}, \ldots, y_{N-k}, c_{k} = f(c_{k}))$，其中 $\textrm{do}(c_{k} = f(c_{k}))$ 是方程 [18](https://arxiv.org/html/2402.10877v7#A3.E18) 中定义的局部干预。
我们现在屏蔽除 $C_{1}$ 之外的所有证据，即 $\textbf{Pa}_{D}^{\prime} = \{C_{1}\}$。
注意，由于 $C_{1} \not\in \textbf{Pa}_{U}$，我们可以应用 [引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4)，得到（对于 $k \geq 2$）

$$
\displaystyle\sum_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm {c})] \displaystyle=\sum_{c_{k}}\ldots\sum_{c_{2}}P(c_{k}\mid\textbf{pa}_{k};\sigma) \ldots P(c_{1}\mid\textbf{pa}_{1})[U(d,\bm{c})-U(d^{\prime},\bm{c})] (21) \displaystyle=\sum_{c_{k}}P(c_{k}\mid\textbf{pa}_{k})\alpha(c_{k})(22)
$$

其中 $\bm{Z} = \bm{C} \setminus \{C_{1}\}$，并且

$$
\alpha(c_{k}):=\sum_{c_{k-1}}\ldots\sum_{c_{2}}P(c_{k-1}\mid\textbf{pa}_{k-1}) \ldots P(c_{1}\mid\textbf{pa}_{1})[U(d,\bm{c})-U(d^{\prime},\bm{c})](23)
$$

而对于 $k=1$，我们有

$$
\sum_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm{c})]=P(C_{1} =c_{1}\mid\textbf{pa}_{1};\sigma)\alpha(1)(24)
$$

其中 $\alpha(1):=[U(d,\bm{x})-U(d^{\prime},\bm{x})]$。
我们可以确定 $\alpha(c_{k})$，因为根据假设我们知道 $C_{k-1},\ldots,C_{1}$ 的参数，并且对于几乎所有的 $P,U$，有 $\alpha(c_{k})\neq 0$，因为方程 $\alpha(c_{k})=0$ 是模型参数的一个多项式，根据 [引理 2](https://arxiv.org/html/2402.10877v7#Thmlemma2)，对于几乎所有的 $P,U$ 该等式不成立。
利用局部干预方程 [18](https://arxiv.org/html/2402.10877v7#A3.E18) 的定义，我们有 $P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k};\sigma)=P(C_{k}=c_{k}^{\prime}\mid
\textbf{pa}_{k})$，以及 $P(C_{k}=c_{k}^{\prime\prime}\mid\textbf{pa}_{k};\sigma)=1-P(C_{i}=c_{k}^{
\prime}\mid\textbf{pa}_{k};\sigma)=1-P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k})$。
因此，方程 [19](https://arxiv.org/html/2402.10877v7#A3.Ex5) 的右侧只有一个未确定的参数 $P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k})$，而左侧可以使用 **策略预言机（Policy Oracle）** 确定（使用 [引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4)，注意 $\textbf{Pa}^{\prime}_{D}=\{C_{1}\}$ 且 $\{C_{1}\}\cap\textbf{Pa}_{U}=\emptyset$），从而我们可以解出 $P(C_{k}=c_{k}^{\prime}\mid\textbf{pa}_{k})$。
通过使用不同的干预重复此过程，改变 **硬干预（Hard Intervention）** $\textrm{do}(\bm{Y}=\bm{y})$ 以及 $c_{k}^{\prime},c_{k}^{\prime\prime}$ 的选择，我们可以识别出所有 $c_{k},\textbf{pa}_{k}$ 对应的 $P(c_{k}\mid\textbf{pa}_{k})$，进而确定 $\textbf{Pa}_{k}$。

现在，我们学习所有 $C_{i}\in\textbf{Anc}_{D}\setminus\textbf{Anc}_{U}(G_{\hat{D}})$ 的参数。
我们从策略预言机返回的策略的定义域中知道 $\textbf{Pa}_{D}$。
如果 $\textbf{Pa}_{D}$ 中所有变量的参数已在上一集合中学习完毕，则任务完成。
否则，存在一些变量属于 $\textbf{Anc}_{U}$，但它们所有指向 $U$ 的有向路径都经过 $D$。
令此变量集合为 $\tilde{\textbf{Pa}}_{D}\subseteq\textbf{Pa}_{D}$。
对于任意 $C_{1}\in\tilde{\textbf{Pa}}_{D}$，我们可以确定 $\alpha(c_{1})=U(d,\bm{x})-U(d^{\prime},\bm{x})$，其中 $d,d^{\prime},\bm{x}$ 由 [引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4) 中的 [算法 1](#algorithm-1) 返回，注意 $C_{1}\not\in\textbf{Pa}_{U}$。
然后，我们可以确定所有 $C_{j}\in\textbf{Pa}_{1}$ 的参数，依此类推，直到遍历完 $\textbf{Anc}_{1}$，并重复此过程，直到学习完所有 $C_{i}\in\textbf{Anc}_{D}\setminus\textbf{Anc}_{U}(G_{\hat{D}})$ 的参数。

∎

###### 引理 3（Lemma 3）.

###### 证明（Proof）.

###### 引理 4（Lemma 4）.

###### 证明（Proof）.

###### 证明（Proof）.

<a id="appendix-d"></a>

## 附录 D 定理 2 的证明（Appendix D Proof of Theorem 2）

在本节中，我们使用一个针对 $\delta>0$ 的 **$\delta$-最优策略预言机（$\delta$-optimal policy oracle）** 来推导 [引理 4](https://arxiv.org/html/2402.10877v7#Thmlemma4) 的一个版本。我们考虑这种情况的原因是，[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 假设了最优性，这是一个很强的假设，现实系统无法满足。因此，确定我们的主要结果是否依赖于这一假设至关重要。例如，可能我们只能从智能体在 $\delta=0$ 时的策略中识别出因果模型，而对于 $\delta>0$，则无法学习到任何因果模型。然而，我们发现，对于 $\delta>0$ 的现实智能体，它们必须学习 **近似因果模型（approximate causal models）** ，并且这些近似的保真度会随着 $\delta\rightarrow 0$ 以一种合理的方式提高。

#### 低遗憾分析（Low-regret analysis）

近似误差随 $\delta$ 变化的合理方式是什么？显然，如果一个智能体的遗憾界任意大，我们无法期望从其策略中学习到任何关于环境的信息。例如，一个完全随机的策略可以满足足够大的遗憾界，而智能体无需学习任何关于环境的知识就能学会这个策略。因此，在我们的分析中，我们仍然必须限制遗憾很小，标准的方法是通过阶次分析来实现。

我们将“小遗憾”定义为 $\delta\ll\mathbb{E}^{\pi^{*}}[U]$。由于我们使用归一化的效用函数（见 [第 A.1 节](https://arxiv.org/html/2402.10877v7#A1.SS1)），我们有 $\mathbb{E}^{\pi^{*}}[U]\leq 1$，因此我们可以将小遗憾区域定义为 $\delta\ll 1$。我们发现，对于小的 $\delta$，我们模型参数估计中的误差阶次，与仅产生小遗憾的智能体的遗憾增加阶次呈线性增长关系。因此，对于小的 $\delta$，我们在遗憾和准确性之间得到了 **线性权衡（linear trade-off）** 。

首先我们证明，[算法 1](#algorithm-1) 允许我们用近似值 $\tilde{Q}$ 来估计 $Q=\sum_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm{c})]$ 的值，并估计边界 $\tilde{Q}^{\pm}$，使得 $Q$ 的真实值保证满足 $\tilde{Q}^{-}\leq Q\leq\tilde{Q}^{+}$。

###### 引理 5 （Lemma 5）

使用一个 $\delta$-最优策略预言机 $\Pi^{\delta}_{\Sigma}$，其中 $\Sigma$ 包含所有局部干预的混合，包括掩码输入 $\textbf{Pa}_{D}^{\prime}\subseteq\textbf{Pa}_{D}$，那么对于任何给定的 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 且满足 $\textbf{Pa}_{D}^{\prime}\cap\textbf{Pa}_{U}=\emptyset$，我们可以确定 $d,d^{\prime},\bm{x}^{\prime}$（其中 $d\neq d^{\prime}$）以及 $Q(\textbf{pa}_{D},d,d^{\prime}):=\sum_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d,\bm{x})-U(d^{\prime},\bm{x})]<0$ 的一个点估计 $\tilde{Q}$ 和边界 $Q\in[\tilde{Q}^{-},\tilde{Q}^{+}]$，其中 $\bm{Z}=\bm{C}\setminus\textbf{Pa}_{D}$，$\bm{X}=\textbf{Pa}_{U}\setminus\{D\}$，并且，

$$
\frac{1}{1-\xi}(Q-\delta)\leq\tilde{Q}\leq\frac{1}{1+\xi}(Q+\delta) \tag{25}
$$

其中

$$
\xi:=\delta/(U(d,\bm{x}^{\prime})-U(d^{\prime},\bm{x}^{\prime}))>0 \tag{26}
$$

在最坏情况下，这些界随 $\delta$ 的变化关系为

$$
\displaystyle\tilde{Q}^{+} \displaystyle\leq\left(\frac{1-\xi}{1+\xi}\right)Q+\frac{2\delta}{1+\xi} \tag{27}
$$

$$
\displaystyle\tilde{Q}^{-} \displaystyle\geq\left(\frac{1+\xi}{1-\xi}\right)Q-\frac{2\delta}{1-\xi} \tag{28}
$$

###### 证明（Proof）.

根据 [引理 3（Lemma 3）](https://arxiv.org/html/2402.10877v7#Thmlemma3)，对于几乎所有的 $P, U$，在偏移 $\sigma$ 之后存在唯一的最优决策。
令 $d_{1}$ 为策略预言机（policy oracle）在上下文 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 中返回的策略所返回的决策，它必须满足界 $\mathbb{E}[u\mid d,\textbf{pa}_{D};\sigma]\leq\max_{d}\mathbb{E}[u\mid d,
\textbf{pa}_{D};\sigma]-\delta$。

考虑对所有 $C_{i}\in\bm{C}$ 进行硬干预（hard intervention），$\sigma^{\prime}:=\textrm{do}(c^{\prime}_{1},c^{\prime}_{2},\ldots,c^{\prime}_{
N})$，其中对于所有 $C_{i}\in\textbf{Pa}_{D}^{\prime}$，我们设定 $C_{i}=c_{i}$ 与观测 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 中的状态相同。
在此干预下的期望效用（expected utility）为 $\mathbb{E}[u\mid\textrm{do}(D=d),\textbf{pa}_{D}^{\prime};\sigma^{\prime}]=U(d
,\bm{x}^{\prime})$，其中 $\bm{X}=\textbf{Pa}_{U}\setminus\{D\}$（并且根据 [引理 1（Lemma 1）](https://arxiv.org/html/2402.10877v7#Thmlemma1) iii)，我们有 $D\in\textbf{Pa}_{U}$）。

接下来我们证明，存在一种 **硬干预（hard intervention）** $\sigma^{\prime}$ 的选择，使得 **策略预言机（policy oracle）** 在上下文 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 中，对于 $\sigma^{\prime}$ 和 $\sigma$ 必须返回不同的最优决策。
由于 $\textbf{Pa}_{D}^{\prime}\cap\textbf{Pa}_{U}=\emptyset$，我们可以自由选择任意的 $X=x^{\prime}$，由此得到的 $\sigma^{\prime}$ 将与证据 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 相容。
注意，根据 [引理 1](https://arxiv.org/html/2402.10877v7#Thmlemma1) i)，存在 $\bm{X}=\bm{x}^{\prime}$ 使得 $d_{1}\neq\operatorname*{arg\,max}_{d}U(d,x^{\prime})$，否则 $D=d_{1}$ 将对所有 $\bm{X}=\bm{x}$ 都是最优的，这将违反 **领域依赖性（domain dependence）** 。
给定效用函数和 $d_{1}$，我们可以确定这个 $\bm{X}=\bm{x}^{\prime}$。
令 $d_{2}=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})$，且 $\sigma^{\prime}=\textrm{do}(c^{\prime}_{1},c^{\prime}_{2},\ldots,c^{\prime}_{N})$ 是使得 $\bm{X}=\bm{x}^{\prime}$ 和 $\textbf{Pa}_{D}^{\prime}=\textbf{pa}_{D}^{\prime}$ 的硬干预。
注意，我们并未使用策略预言机来确定 $d_{2}$，$d_{2}$ 可以仅从 $U(\textbf{Pa}_{U})$ 确定。因此，对于任何 **遗憾界（regret bound）** ，$d_{2}$ 在 $\sigma^{\prime}$ 下是否确实最优，以及 $d_{1}$ 在 $\sigma^{\prime}$ 下是否非最优，都不存在不确定性。

考虑在 **混合局部干预（mixed local intervention）** $\tilde{\sigma}(q)=q\sigma+(1-q)\sigma^{\prime}$ 下 $\bm{C}$ 的联合分布，

$$
\displaystyle P(\bm{C}=\bm{c}\mid\textrm{do}(D=d);\tilde{\sigma}(q)) \displaystyle=P(\bm{C}=\bm{c};\tilde{\sigma}(q)) (29) \displaystyle=qP(\bm{C}=\bm{c};\sigma)+(1-q)P(\bm{C}=\bm{c};\sigma^{\prime})(30)
$$

其中第一行我们利用了 $\textbf{Ch}_{D}=\{U\}$ 来消去干预项。
注意，根据 [引理 1](https://arxiv.org/html/2402.10877v7#Thmlemma1) i)，有 $\bm{Z}=\bm{C}\setminus\textbf{Pa}_{D}\neq\emptyset$。
期望效用由下式给出，

$$
\displaystyle\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d);\tilde{\sigma}(q )]=\sum\limits_{\bm{z}}P(\bm{Z}=\bm{z}\mid\textbf{pa}_{D},\textrm{do}(D=d); \tilde{\sigma}(q))U(d,\bm{x}) (31) \displaystyle=\sum\limits_{\bm{z}}\frac{P(\bm{C}=\bm{c}\mid\textrm{do}(D=d); \tilde{\sigma}(q))}{P(\textbf{pa}_{D}\mid\textrm{do}(D=d);\tilde{\sigma}(q))}U (d,\bm{x}) (32) \displaystyle=\frac{1}{P(\textbf{pa}_{D};\tilde{\sigma}(q))}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\tilde{\sigma}(q))U(d,\bm{x}) (33) \displaystyle=\frac{1}{P(\textbf{pa}_{D};\tilde{\sigma}(q))}\sum\limits_{\bm{z}}qP(\bm{C}=\bm{c};\sigma)U(d,\bm{x})+(1-q)P(\bm{C}=\bm{c};\sigma^{\prime})U(d ,\bm{x}^{\prime})(34)
$$

注意，当 $q=1$ 时，最优决策是 $d_{1}$；当 $q=0$ 时，策略预言机返回的最优决策属于集合 $\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$，该集合不包含 $d_{1}$。
此外，方程 [34](https://arxiv.org/html/2402.10877v7#A4.E34) 关于 $d$ 的 argmax 是一个定义域为 $q\in[0,1]$ 的分段线性函数。
因此，必然存在某个 $q=q_{\text{crit}}$，它是 $q$ 的最小值，使得当 $q<q_{\text{crit}}$ 时，策略预言机返回的最优决策在集合 $\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$ 中；而当 $q\geq q_{\text{crit}}$ 时，最优决策不在此集合中。
$q_{\text{crit}}$ 的值由 $\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d);\tilde{\sigma}(q_{\text{crit}
})]=0$ 给出，根据方程 [34](https://arxiv.org/html/2402.10877v7#A4.E34)，该式为，

$$
q_{\text{crit}}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm{x})-U(d _{3},\bm{x})]+(1-q_{\text{crit}})[U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})]=0 \tag{35}
$$

其中 $d_{2}\in\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$ 且 $d_{3}\not\in\{d\text{s.t.}d=\operatorname*{arg\,max}_{d}U(d,\bm{x}^{\prime})\}$。
由此得到 $q_{\text{crit}}$ 的如下表达式：

$$
q_{\text{crit}}=\left(1-\frac{\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_ {2},\bm{x})-U(d_{3},\bm{x})]}{U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime} )}\right)^{-1} \tag{36}
$$

其中我们已使用 $\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma^{\prime})[U(d_{2},\bm{c})-U(d_{3},
\bm{c})]=U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})$。

虽然 **[算法 1（Algorithm 1）](#algorithm-1)** 能识别出使得最优策略发生改变的 $q$ 的最小值，但由于我们不再拥有最优策略预言机（Optimal Policy Oracle）， **算法 1** 返回的概率 $\tilde{q}$ 不再必然等于 $q_{\text{crit}}$。
相反， **算法 1** 可以返回的 $\tilde{q}$ 存在最小值和最大值，这由 **遗憾界（Regret Bound）** 决定（参见 [图 6（Figure 6）](https://arxiv.org/html/2402.10877v7#A4.F6)）。

我们的首要目标是利用策略预言机（Policy Oracle）返回的 $\tilde{q}$ 来界定 $q_{\text{crit}}$。
在满足遗憾界的前提下，$\tilde{q}$ 可以取到的最大（最小）值即为 $q^{\pm}$，它们是以下方程的解：

$$
\begin{aligned}
\displaystyle\delta &\displaystyle\geq\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{2});\tilde{\sigma}(q^{+})]-\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{3});\tilde{\sigma}(q^{+})] \tag{37} \\
\displaystyle-\delta &\displaystyle\leq\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{2});\tilde{\sigma}(q^{-})]-\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{3});\tilde{\sigma}(q^{-})] \tag{38}
\end{aligned}
$$

利用公式 [34](https://arxiv.org/html/2402.10877v7#A4.E34)，这些方程可简化为：

$$
\begin{aligned}
\displaystyle\delta P(\textbf{pa}_{D};\sigma(q^{+})) &\displaystyle\geq q^{+}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm {x})-U(d_{3},\bm{x})]+(1-q^{+})[U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})] \tag{39} \\
\displaystyle\delta P(\textbf{pa}_{D};\sigma(q^{-})) &\displaystyle\leq q^{-}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm {x})-U(d_{3},\bm{x})]+(1-q^{-})[U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})] \tag{40}
\end{aligned}
$$

我们可以通过取未知量 $P(\textbf{pa}_{D};\sigma(\tilde{q}))$ 的最大可能值（即令 $P(\textbf{pa}_{D};\sigma(\tilde{q}))\rightarrow 1$）来放宽并简化这些界，从而得到：

$$
\begin{aligned}
\displaystyle\delta &\displaystyle\geq q^{+}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm {x})-U(d_{3},\bm{x})]+(1-q^{+})[U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})] \tag{41} \\
\displaystyle\delta &\displaystyle\leq q^{-}\sum\limits_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm {x})-U(d_{3},\bm{x})]+(1-q^{-})[U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})] \tag{42}
\end{aligned}
$$

令 $\Delta_{1}:=\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{2});\sigma]-
\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{3});\sigma]$ 且 $\Delta_{0}:=U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})$。
注意 $\Delta_{0}>0$，因为 $d_{2}$ 在 $\sigma^{\prime}$ 下是最优的；而 $\Delta_{1}<0$，因为根据线性性，对于 $q>q_{\text{crit}}$ 且 $q_{\text{crit}}<1$，我们有 $\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{3});\tilde{\sigma}(q)]>
\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{2});\tilde{\sigma}(q)]$，因此 $\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{3});\tilde{\sigma}(1)]>
\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{2});\tilde{\sigma}(1)]$。
我们现在根据（松弛的）边界方程 [41] 和方程 [42] 定义 $q^{\pm}$，并利用方程 [36] 简化这些不等式，得到

$$
\displaystyle\tilde{q} \displaystyle\leq q_{+}=\min\{1,q_{\text{crit}}(1+\xi)\} \quad (43) \qquad \displaystyle\tilde{q} \displaystyle\geq q_{-}=\max\{0,q_{\text{crit}}(1-\xi)\} \quad (44)
$$

其中

$$
\xi:=\delta/\Delta_{0}>0 \quad (45)
$$

因此，我们利用方程 [44] 和方程 [43] 生成 $q_{\text{crit}}$ 的边界，即

$$
\displaystyle q_{\text{crit}} \displaystyle\leq\tilde{q}/(1-\xi) \quad (46) \qquad \displaystyle q_{\text{crit}} \displaystyle\geq\tilde{q}/(1+\xi) \quad (47)
$$

我们在方程 [36] 中用 $\tilde{q}$ 代替 $q_{\text{crit}}$，从而得到 $Q=\sum_{\bm{z}}P(\bm{C}=\bm{c};\sigma)[U(d_{2},\bm{x})-U(d_{3},\bm{x})]$ 的一个估计值 $\tilde{Q}$，得到

$$
\tilde{Q}=\Delta_{0}\left(1-1/\tilde{q}\right) \quad (48)
$$

最后，应用边界方程 [46] 和方程 [47] 得到

$$
\frac{1}{1-\xi}\left(\Delta_{0}-\delta\right)\leq\tilde{Q}\leq\frac{1}{1+\xi} \left(\Delta_{0}+\delta\right) \quad (49)
$$

接下来，我们利用 $Q=\Delta_{0}(1-1/q_{\text{crit}})$ 以及方程 [46] 和方程 [47] 确定上下界 $Q^{\pm}(\tilde{q})$，得到

$$
\displaystyle Q \displaystyle\leq\tilde{Q}^{+}(\tilde{q})=\Delta_{0}\left(1-\frac{1-\xi}{\tilde{q}}\right) \quad (50) \qquad \displaystyle Q \displaystyle\geq\tilde{Q}^{-}(\tilde{q})=\Delta_{0}\left(1-\frac{1+\xi}{\tilde{q}}\right) \quad (51)
$$

注意到由于公式 [48] 对 $q$ 是单调的，因此可以保证 $Q$ 的真值落在这些界限之间。
最后，我们推导出关于 $Q$ 真值的最坏情况界限表达式，这些表达式通过确定 $\tilde{Q}^{\pm}$ 得到，其中 $\tilde{q}$ 的最大值和最小值由公式 [47] 和公式 [46] 给出，

$$
\displaystyle Q^{+} \displaystyle=\max_{\tilde{q}}\tilde{Q}^{+}(\tilde{q})=\Delta_{0}\left(1-\frac {1-\xi}{1+\xi}\frac{1}{q_{\text{crit}}}\right) (52) \displaystyle=\left(\frac{1-\xi}{1+\xi}\right)Q+\frac{2\delta}{1+\xi} (53) \displaystyle Q^{-} \displaystyle=\min_{\tilde{q}}\tilde{Q}^{-}(\tilde{q})=\Delta_{0}\left(1-\frac {1+\xi}{1-\xi}\frac{1}{q_{\text{crit}}}\right) (54) \displaystyle=\left(\frac{1+\xi}{1-\xi}\right)Q-\frac{2\delta}{1-\xi}(55)
$$

<a id="figure-6"></a>

![diagram4](images/diagram4.png)

> 图 6: [引理 5（Lemma 5）](https://arxiv.org/html/2402.10877v7#Thmlemma5) 概述。$\Delta_{0}=U(d_{2},\bm{x}^{\prime})-U(d_{3},\bm{x}^{\prime})$ 且 $\Delta_{1}=\mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{2});\sigma]- \mathbb{E}[u\mid\textbf{pa}_{D},\textrm{do}(D=d_{3});\sigma]$。使用最优策略预言机（optimal policy oracle），我们可以精确地确定 $q_{\text{crit}}$，如 [引理 4（Lemma 4）](https://arxiv.org/html/2402.10877v7#Thmlemma4) 所述。对于 $\delta>0$，[算法 1（Algorithm 1）](#algorithm-1) 返回的是 $\tilde{q}$ 而非 $q_{\text{crit}}$，因为智能体（agent）可能产生遗憾（regret），因此策略（policy）改变的 $q$ 值不再被约束为 $q_{\text{crit}}$。我们使用 $\tilde{q}$ 代替 $q_{\text{crit}}$ 来计算目标查询（target query）的近似值，方法与 [引理 4（Lemma 4）](https://arxiv.org/html/2402.10877v7#Thmlemma4) 相同。$\tilde{q}$ 可以取的最大值和最小值是 $q^{\pm}$，这会导致最大遗憾 $\delta$，即 $\tilde{q}\geq q^{-}=q_{\text{crit}}(1-\delta/\Delta_{0})$ 且 $\tilde{q}\leq q^{+}=q_{\text{crit}}(1+\delta/\Delta_{0})$。因此，我们可以界定 $\tilde{Q}$ 偏离目标查询 $Q$ 的值的程度。

∎

###### 引理 6（Lemma 6）.

对于 $\delta\ll\mathbb{E}^{\pi^{*}}[U]$，$\tilde{Q}$ 和 $\tilde{Q}^{\pm}$（定义见 [引理 5（Lemma 5）](https://arxiv.org/html/2402.10877v7#Thmlemma5)）满足以下界限，

$$
\left|\tilde{Q}-Q\right|\leq\delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^ {2})(56)
$$

以及

$$
\displaystyle\tilde{Q}^{+}-Q \displaystyle\leq 2\delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^{2}) (57) \displaystyle Q-\tilde{Q}^{-} \displaystyle\geq-2\delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^{2})(58)
$$

###### 证明。

由于我们使用的是归一化效用函数（参见[第 A.1 节](https://arxiv.org/html/2402.10877v7#A1.SS1)），我们有 $\mathbb{E}^{\pi^{*}}[U]\leq 1$，因此我们可以将小遗憾（small regret）区间定义为 $\delta\ll 1$。
我们可以对 $\tilde{Q},\tilde{Q}^{\pm}$ 的界在 $\delta=0$ 处进行泰勒展开（Taylor expand），得到：

$$
Q-\delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^{2})\leq\tilde{Q}\leq Q+ \delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^{2})(59)
$$

因此，

$$
\left|\tilde{Q}-Q\right|\leq\delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^ {2})(60)
$$

并且

$$
\displaystyle\tilde{Q}^{+}-Q \displaystyle\leq 2\delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^{2}) (61) \displaystyle Q-\tilde{Q}^{-} \displaystyle\geq-2\delta(1-\frac{Q}{\Delta_{0}})+\mathcal{O}(\delta^{2})(62)
$$

因此，对于小的 $\delta$，我们估计值 $\tilde{Q}$ 的最坏情况误差（worst case error）随 $\delta$ 线性增长，并且 $\tilde{Q}$ 的上界和下界也线性增长。
∎

参见 [2](https://arxiv.org/html/2402.10877v7#Thmtheorem2)

###### 证明。

我们使用 $\delta-$最优策略预言机（$\delta-$optimal policy oracle）来估计模型参数，步骤与[附录 C](#appendix-c) 中[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 的证明相同。
然而，由于策略预言机不再是最优的，参数估计将存在误差。
在此，我们证明对于 $P$ 的参数，当 $\delta\ll 1$ 时，这些误差随 $\delta$ 线性增长，并且我们学习到 $G$ 的一个稀疏子图（sparse sub-graph）。

###### 引理 5（Lemma 5）.

###### 证明（Proof）.

###### 引理 6（Lemma 6）.

###### 证明（Proof）.

###### 证明（Proof）.

#### 估计 $P$ 的参数（Estimating parameters of $P$）

在 [定理 1（Theorem 1）](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 的证明中，我们在两种情况下估计参数 $P(c_{i}\mid\textbf{pa}_{i})$。

**情况 1（Case 1）.**

$$
Q_{k}=\sum_{\bm{c}}P(\bm{C}=\bm{c};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm{c})]= \sum_{c_{k}}P(c_{k}\mid\textbf{pa}_{k};\sigma)\beta(c_{k}) \tag{63}
$$

其中

$$
\beta(c_{k}):=\sum_{c_{k-1}}\ldots\sum_{c_{1}}P(c_{k-1}\mid\textbf{pa}_{k-1}; \sigma)\ldots P(c_{1}\mid\textbf{pa}_{1};\sigma)[U(d,\bm{c})-U(d^{\prime},\bm{c})] \tag{64}
$$

我们利用 $P(c^{\prime}_{k}\mid\textbf{pa}_{k};\sigma)=1-P(c^{\prime\prime}_{k}\mid
\textbf{pa}_{k};\sigma)$ 进行整理，得到：

$$
P(c^{\prime}_{k}\mid\textbf{pa}_{k};\sigma)=\frac{Q_{k}-\beta(c_{k}^{\prime \prime})}{\beta(c_{k}^{\prime})-\beta(c_{k}^{\prime\prime})}(65)
$$

假设我们已有近似值 $\hat{P}(c^{\prime}_{k-1}\mid\textbf{pa}_{k-1};\sigma),\ldots,\hat{P}(c^{\prime
}_{1}\mid\textbf{pa}_{1};\sigma)$，其中 $\hat{P}(c^{\prime}_{k}\mid\textbf{pa}_{k};\sigma)=P(c^{\prime}_{k}\mid\textbf{
pa}_{k};\sigma)+\mathcal{O}(\delta)$，即对于 $\delta\ll 1$，这些参数估计的误差随 $\delta$ 线性增长。
由于 $\beta(c_{k})$ 是这些参数估计的乘积之和，因此我们对 $\beta(c_{k})$ 的估计在 $\delta\ll 1$ 时也具有线性误差，即 $\hat{\beta}(c_{k})=\beta(c_{k})+\mathcal{O}(\delta)$，同理可得：

$$
\hat{P}(c^{\prime}_{k}\mid\textbf{pa}_{k};\sigma)=\frac{Q_{k}-\beta(c_{k}^{\prime\prime})+\mathcal{O}(\delta)}{\beta(c_{k}^{\prime})-\beta(c_{k}^{\prime \prime})+\mathcal{O}(\delta)}=P(c^{\prime}_{k}\mid\textbf{pa}_{k};\sigma)\left (1+\mathcal{O}(\delta)\right)(66)
$$

那么对于 $k=1$，我们精确地知道 $\beta(c_{1})=U(d,\bm{c})-U(d^{\prime},\bm{c})$，因此：

$$
\hat{P}(c^{\prime}_{1}\mid\textbf{pa}_{1};\sigma)=\frac{Q_{1}-\beta(c_{1}^{\prime\prime})+\mathcal{O}(\delta)}{\beta(c_{1}^{\prime})-\beta(c_{1}^{\prime \prime})}=P(c^{\prime}_{1}\mid\textbf{pa}_{1};\sigma)\left(1+\mathcal{O}( \delta)\right)(67)
$$

这满足了我们对于 $k=1$、$\delta\ll 1$ 时误差为 $\mathcal{O}(\delta)$ 的假设。因此，对于所有 $k$，当 $\delta\ll 1$ 时，误差随 $\delta$ 线性增长。

在 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 证明中的情况 2 下，$Q_{k},\alpha(c_{k})$ 的表达式是相似的，并且用同样的方法可以轻易证明，对于这些参数，当 $\delta\ll 1$ 时，误差同样随 $\delta$ 线性增长。

#### **学习图结构（Learning graph structure）**

在 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 中，我们从 $P(c_{k}\mid\textrm{do}(\bm{C}\setminus\{C_{k}\}))$ 确定 $\textbf{Pa}_{k}$。假设因果忠实性（Causal Faithfulness）——这对几乎所有 $P$ 都成立（Meek, 2013）——那么 $C_{j}\in\textbf{Pa}_{k}$ 当且仅当对于某些 $C_{j}=c_{j},C_{j}=c^{\prime}_{j}$，$P(c_{k}\mid\textrm{do}(\bm{C}\setminus\{C_{k}\}))$ 的值不同。然而，由于我们现在只有估计值 $\hat{P}(c_{k}\mid\textrm{do}(\bm{C}\setminus\{C_{k}\}))$，任何关于 $C_{j}=c_{j}$ 的变化都可能是由于这些估计值中变化的误差所致，而非条件概率本身的变化。

然而，我们已经证明，我们可以在误差范围内学习任何 $P(c_{i}\mid\textbf{pa}_{i})$，并且当 $\delta\ll 1$ 时，这些误差界随 $\delta$ 线性缩放。令 $C_{j}\in\textbf{Pa}_{i+n}$ 且 $\theta_{kj}=P(c_{k}\mid\textrm{do}(\bm{C}\setminus\{C_{k}\}))$，并将来自 [引理 6](https://arxiv.org/html/2402.10877v7#Thmlemma6) 的相应上下界记为 $\theta_{kj}^{\pm}$。如果 $\exists$ $\theta_{kj}\neq\theta_{kj^{\prime}}$ 并且要么 $\theta_{kj}^{+}<\theta_{kj^{\prime}}^{-}$，要么 $\theta_{kj^{\prime}}^{+}<\theta_{kj}^{-}$（即对于 $C_{j}=c_{j}$ 和 $C_{j}=c_{j}^{\prime}$ 存在不重叠的界），那么我们可以确定地知道 $C_{j}\in\textbf{Pa}_{k}$。如果对于所有 $j$ 都不存在这样的不重叠界，我们就不知道 $C_{j}\in\textbf{Pa}_{k}$，因此将其从集合中排除。

这种方法保证能识别出 $G$ 的一个子图（即没有假阳性——在近似因果贝叶斯网络（Causal Bayesian Network, CBN）中存在而环境中不存在的有向边）。此外，我们仅会在以下情况下遗漏一个父节点：在真实的底层因果模型中，对于所有 $\textbf{Pa}_{k}=\textbf{pa}_{k}$，干预以改变 $C_{j}$ 会使得 $|P(c_{k}\mid\textbf{pa}_{k},\textrm{do}(c_{j}))-P(c_{k}\mid\textbf{pa}_{k},
\textrm{do}(c_{j}^{\prime}))|<\mathcal{O}(\delta)$。因此，对于 $\delta\ll 1$，我们仅无法学习那些幅度很小（相对于遗憾 $\delta$ 而言）的因果关系，即父节点对子节点的因果效应为 $\mathcal{O}(\delta)$ 的情况。

在 [附录 F](#appendix-f) 中，我们使用模拟数据探讨了遗憾界与所学因果图中误差之间的关系，并发现即使产生相对较高遗憾的智能体，与随机基线相比，也能用于高精度地识别因果结构。
∎

<a id="appendix-e"></a>

## 附录 E Appendix: proof of Theorem 3 (附录 E：定理 3 的证明)

参见 [3](https://arxiv.org/html/2402.10877v7#Thmtheorem3)

###### 证明。

首先，我们考虑已知精确模型 $M=(P,G)$ 的情况。
由于 $M$ 是 **因果充分的（causally sufficient）** ，对于任何与 $G$ 兼容且仅涉及 $G$ 中变量（包括 $\textbf{Anc}_{U}\cup\{U\}$）的给定 **软干预（soft interventions）** $\sigma$，我们都可以识别 $\mathbb{E}[u\mid d,\textbf{pa}_{D};\sigma]$。
我们的 **策略预言机（policy oracle）** 通过以下步骤构建：i) 对输入的 $\sigma$ 估计 $\mathbb{E}[u\mid d,\textbf{pa}_{D};\sigma]$；ii) 计算 $d^{*}=\operatorname*{arg\,max}_{d}\mathbb{E}[u\mid d,\textbf{pa}_{D};\sigma]$ 并返回满足此条件的任意 $d^{*}$。

接下来，考虑我们已知近似模型 $M^{\prime}=(P^{\prime},G^{\prime})$ 的情况，其中 $\left|P^{\prime}(v_{i}\mid\textbf{pa}_{i})-P(v_{i}\mid\textbf{pa}_{i})\right|
\leq\epsilon\ll 1$，这意味着 $P^{\prime}(v_{i}\mid\textbf{pa}_{i})=P(v_{i}\mid\textbf{pa}_{i})+c_{i}\,\epsilon$，其中 $|c_{i}|\leq 1$。首先我们证明，对于任何软干预 $\sigma$，我们可以近似 **干预后联合分布（post-intervention joint distribution）** $P^{\prime}(\bm{Z}=\bm{z}\mid\textrm{do}(D=d),\textbf{Pa}_{D}=\textbf{pa}_{D};
\sigma)=P(\bm{Z}=\bm{z}\mid\textrm{do}(D=d),\textbf{Pa}_{D}=\textbf{pa}_{D};
\sigma)+k\epsilon+\mathcal{O}(\epsilon^{2})$，其中 $\bm{Z}=\bm{C}\setminus\textbf{Pa}_{D}$，$k$ 是模型参数的函数且与 $\epsilon$ 无关。
令 $\sigma=\sum_{j}q_{j}\sigma_{j}$，其中 $\sigma_{j}$ 是软干预。

$$
\displaystyle P^{\prime}(\bm{Z}=\bm{z}\mid\textrm{do}(D=d),\textbf{Pa}_{D}= \textbf{pa}_{D};\sigma)=\sum_{j}q_{j}\frac{P^{\prime}(\bm{C}=\bm{c}\mid\textrm {do}(D=d);\sigma)}{P^{\prime}(\bm{Z}=\bm{z}^{\prime},\textbf{Pa}_{D}=\textbf{pa}_{D};\sigma_{j})} (68) \displaystyle=\sum_{j}q_{j}\frac{\prod\limits_{i}P^{\prime}(C_{i}=c_{i}\mid \textrm{do}(D=d);\sigma_{j})}{\sum_{\bm{z}^{\prime}}\prod\limits_{i}P^{\prime} (C_{i}=c^{\prime}_{i}\mid\textrm{do}(D=d);\sigma_{j})} (69) \displaystyle=\sum_{j}q_{j}\frac{\prod\limits_{i}[P(C_{i}=c_{i}\mid\textrm{do} (D=d);\sigma_{j})+c_{i}\epsilon]}{\sum_{\bm{z}^{\prime}}\prod\limits_{i}[P(C_{i}=c^{\prime}_{i}\mid\textrm{do}(D=d);\sigma_{j})+c_{i}\epsilon]} (70) \displaystyle=\sum_{j}q_{j}\frac{\prod\limits_{i}P(C_{i}=c_{i}\mid\textrm{do}( D=d);\sigma_{j})(1+c^{\prime}_{ij}\epsilon)}{\sum_{\bm{z}^{\prime}}\prod \limits_{i}P(C_{i}=c^{\prime}_{ij}\mid\textrm{do}(D=d);\sigma_{j})(1+c^{\prime}_{i}\epsilon)} (71) \displaystyle=P(\bm{Z}=\bm{z}\mid\textrm{do}(D=d),\textbf{Pa}_{D}=\textbf{pa}_ {D};\sigma)+\epsilon f(\theta)+\mathcal{O}(\epsilon^{2})(72)
$$

其中 $c^{\prime}_{ij}:=c_{i}/P(C_{i}=c_{i}\mid\textrm{do}(D=d);\sigma_{j})$，$f(\theta)$ 是模型参数 $\theta_{i}=P(v_{i}\mid\textbf{pa}_{i})$ 的多项式。
因此，使用 $M^{\prime}$ 评估的干预 $\sigma$ 下的 **期望效用（expected utility）** 满足：

$$
\displaystyle\mathbb{E}_{P^{\prime}}[U\mid\textrm{do}(D=d),\textbf{Pa}_{D}= \textbf{pa}_{D}] \displaystyle=\sum_{\bm{z}}P^{\prime}(\bm{Z}=\bm{z}\mid\textrm{do}(D=d), \textbf{Pa}_{D}=\textbf{pa}_{D};\sigma) (73) \displaystyle=\sum_{\bm{z}}^{\prime}(\bm{Z}=\bm{z}\mid\textrm{do}(D=d),\textbf {Pa}_{D}=\textbf{pa}_{D};\sigma)+\epsilon g(\theta)+\mathcal{O}(\epsilon^{2}) (74) \displaystyle=\mathbb{E}[U\mid\textrm{do}(D=d),\textbf{Pa}_{D}=\textbf{pa}_{D} ]+\epsilon g(\theta)+\mathcal{O}(\epsilon^{2})(75)
$$

其中 $g(\theta)$ 是模型参数的多项式。
决策 $d^{*}=\operatorname*{arg\,max}_{d}\mathbb{E}_{P^{\prime}}[U\mid\textrm{do}(D=d
),\textbf{Pa}_{D}=\textbf{pa}_{D}]$ 最多产生 $\epsilon g(\theta)$ 的 **遗憾（regret）** ，因此遗憾是 $\epsilon$ 的线性函数。
∎

###### 证明。

<a id="appendix-f"></a>

## 附录 F 实验（Appendix F Experiments）

如[第 4 节](#section-4)所述，[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 的证明可被视为 **因果发现算法（Causal Discovery Algorithms）** ，其中我们假设：i) 已知环境变量集合 $\bm{C}$；ii) 已知效用函数 $U$；iii) 决策任务是无中介的；iv) 存在领域依赖性。
基于这些假设，我们仅根据智能体在干预 $\sigma$ 下的策略 $\pi(\sigma)$，即可学习底层 **因果贝叶斯网络（Causal Bayesian Network, CBN）** 的一个近似，当 $\pi(\sigma)$ 是最优策略时，该近似是精确的。

为了演示这一理论结果，我们采用[附录 B](#appendix-b) 中概述的简单二元决策任务的证明，并将其重新表述为一个因果发现算法（如下文算法 2）。
我们在[图 5](https://arxiv.org/html/2402.10877v7#A2.F5) 所示形式的 **因果影响图（Causal Influence Diagrams, CIDs）** 上对其进行测试，其中我们随机选择 $X,Y$ 的联合分布及其因果结构 $G$。
请注意，算法 2 比[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 证明中概述的通用方法要简单得多，因为它利用了 $D,X,Y$ 是二元变量且 $|\bm{C}|=2$ 这一事实。该因果发现算法要求我们能够对潜在变量 $X,Y$ 进行干预，但仅要求我们能够观察单个变量（决策）对这些干预的响应。
为了说明这一设定，我们可以设想一些潜在变量 $X,Y$ 无法直接观测但可以进行干预的情况。

**示例** 。许多疾病无法在患者生理层面直接观测，只能通过症状的存在间接观测。
令 $X,Y\in\{0,1\}$ 为两种此类疾病，它们有对应的治疗方法，即我们可以通过干预来“关闭” $X$ 和 $Y$，但无法观测它们。
$D\in\{0,1\}$ 表示提供特定止痛药物的决策，这会导致症状严重程度（效用）发生变化。
对止痛治疗的反应取决于疾病的存在与否（例如，对于 $X=T$ 的患者，止痛治疗非常有效；对于 $Y=T$ 的患者，效果中等；对于 $X=F,Y=F$ 的患者，效果较差）。
医生的目标是在避免不必要使用止痛药的同时，最小化症状严重程度，例如，$U(d,x,y)=d[s(x,y)-c]$，其中 $c$ 是与止痛治疗相关的某种成本，$s(x,y)$ 是对止痛治疗的反应。
在实施干预 $\sigma$（例如，治愈一种疾病 $\sigma=\textrm{do}(X=F)$）之后，医生在变化的人群中调整其治疗策略。
例如，这种调整可能通过试错法发生，医生随机选择治疗决策 $D$ 并观察症状严重程度的变化——这是一个 **无上下文老虎机问题（Context-free Bandit Problem）** 。
尽管我们无法直接观测疾病状态 $X,Y$，但通过对潜在疾病状态进行干预并观察医生的策略如何调整，我们可以学习联合分布 $P(X,Y)$ 以及 $X,Y$ 上的因果图。

![G_graph](images/G_graph.png)

> (a) G 的误分类率随遗憾界变化的情况

[图 7](https://arxiv.org/html/2402.10877v7#A6.F7) 展示了当 $\pi(\sigma)$ 满足不同的遗憾界时，学习到的参数 $P(x,y)$ 和 $G$ 的平均误差。
这些结果是基于 1000 个随机生成的 **因果贝叶斯网络（Causal Bayesian Network, CBN）** 平均得到的，其中：i) 联合分布 $P(x,y)$ 的参数是随机选择的，ii) 变量 $X, Y$ 上的 **有向无环图（Directed Acyclic Graph, DAG）** $G$ 是从 $X\rightarrow Y$ 和 $X\leftarrow Y$ 中随机选择的，iii) 效用函数 $U(d,x,y)\in[0,1]$ 是随机选择的（参数描述详见 [A.2 节](https://arxiv.org/html/2402.10877v7#A1.SS2)）。
为了模拟 **遗憾有界（regret-bounded）** 的智能体，我们为每个环境计算最优策略，如果次优决策满足遗憾界，则在从 [算法 1](#algorithm-1) 的策略预言机中采样时，从两个决策中随机选择。
我们还与一个随机基线算法进行了比较，该算法估计 $P(x,y)=1/4$，并以相等概率从 $X\rightarrow Y$ 或 $X\leftarrow Y$ 中随机选择。
在少数情况下，由于有限样本误差，算法 1 无法预测 $P(x,y)\in[0,1]$，对于这些情况，我们将因果发现算法的输出替换为随机猜测。

从 [图 7](https://arxiv.org/html/2402.10877v7#A6.F7) 可以看出，错误率随遗憾的增长是 **次线性（sub-linearly）** 的。
需要注意的是，遗憾的相关尺度是两个决策之间的期望效用差，因此我们绘制了归一化的遗憾界，即将 $\delta$ 除以这个期望效用差。
值得注意的是，即使对于相对较大的遗憾界（代表泛化能力较弱的智能体），我们仍然能以很高的准确率识别因果结构。
例如，当遗憾界是期望效用差的 30% 时，我们仍然能在约 $\sim 90\%$ 的随机生成的 **因果影响图（Causal Influence Diagram, CID）** 中识别出正确的因果结构。
这描述了一个智能体，其保证在领域偏移前，所遭受的遗憾至多是决策间期望效用差的 $30\%$。
如果领域偏移导致期望效用差小于未偏移时期望效用差的 $30\%$，该智能体可能会返回一个次优决策。

<a id="algorithm-2"></a>

**算法 2 简单 CID 的图学习器（Algorithm 2 Graph Learner for simple CID）**

1.  function graph learner ( $\Pi^{\delta}_{\Sigma}$ , $U$ , $\delta$ , $N$ )
2.  $d_{1},d_{2},x^{\prime},y^{\prime},q_{\text{crit}}\leftarrow\text{Algorithm 1}( U,\Pi^{\delta}_{\Sigma},N,\sigma_{1}=\textrm{do}(Y=0))$ $\triangleright$ 为 $\textrm{do}(Y=0)$ 识别 $q_{\text{crit}}$
3.  Exp. U difference $=(U(d_{2},x^{\prime},y^{\prime})-U(d_{1},x^{\prime},y^{\prime}))*(1-1/q_{\text {crit}})$
4.  $\Delta_{0}=U(0,0,d_{2})-U(0,0,d_{1})$
5.  $\Delta_{1}=U(1,0,d_{2})-U(1,0,d_{1})$
6.  $P(X_{Y=0}=0)=(\text{Exp. U difference}-\Delta_{1})/(\Delta_{0}-\Delta_{1})$
7.
8.  $d_{1},d_{2},x^{\prime},y^{\prime},q_{\text{crit}}\leftarrow\text{Algorithm 1}( U,\Pi^{\delta}_{\Sigma},N,\sigma_{1}=\textrm{do}(Y=1))$ $\triangleright$ 为 $\textrm{do}(Y=1)$ 识别 $q_{\text{crit}}$
9.  Exp. U difference $=(U(d_{2},x^{\prime},y^{\prime})-U(d_{1},x^{\prime},y^{\prime}))*(1-1/q_{\text {crit}})$
10. $\Delta_{0}=U(0,1,d_{2})-U(0,1,d_{1})$
11. $\Delta_{1}=U(1,1,d_{2})-U(1,1,d_{1})$
12. $P(X_{Y=1}=0)=(\text{Exp. U difference}-\Delta_{1})/(\Delta_{0}-\Delta_{1})$
13.
14. $d_{1},d_{2},x^{\prime},y^{\prime},q_{\text{crit}}\leftarrow\text{Algorithm 1}( U,\Pi^{\delta}_{\Sigma},N,\sigma_{1}=\textrm{do}(X=0))$ $\triangleright$ 为 $\textrm{do}(X=0)$ 识别 $q_{\text{crit}}$
15. Exp. U difference $=(U(d_{2},x^{\prime},y^{\prime})-U(d_{1},x^{\prime},y^{\prime}))*(1-1/q_{\text {crit}})$
16. $\Delta_{0}=U(0,0,d_{2})-U(0,0,d_{1})$
17. $\Delta_{1}=U(0,1,d_{2})-U(0,1,d_{1})$
18. $P(Y_{X=0}=0)=(\text{Exp. U difference}-\Delta_{1})/(\Delta_{0}-\Delta_{1})$
19.
20. $d_{1},d_{2},x^{\prime},y^{\prime},q_{\text{crit}}\leftarrow\text{Algorithm 1}( U,\Pi^{\delta}_{\Sigma},N,\sigma_{1}=\textrm{do}(X=1))$ $\triangleright$ 为 $\textrm{do}(X=1)$ 识别 $q_{\text{crit}}$
21. Exp. U difference $=(U(d_{2},x^{\prime},y^{\prime})-U(d_{1},x^{\prime},y^{\prime}))*(1-1/q_{\text {crit}})$
22. $\Delta_{0}=U(1,0,d_{2})-U(1,0,d_{1})$
23. $\Delta_{1}=U(1,1,d_{2})-U(1,1,d_{1})$
24. $P(Y_{X=1}=0)=(\text{Exp. U difference}-\Delta_{1})/(\Delta_{0}-\Delta_{1})$
25.
26. if $P(Y_{X=0}=0)=P(Y_{X=1}=0)$ then $\triangleright$ 从干预数据中识别 $G$ 和 $P$
27. if $P(X_{Y=0}=0)=P(X_{Y=1}=0)$ then
28. $G\leftarrow()$
29. $P(x,y)=P(X_{Y=0}=x)P(Y_{X=0}=y)$
30. else
31. $G\leftarrow(Y\rightarrow X)$
32. $P(x,y)=P(Y_{X=0}=y)P(X_{Y=y}=x)$
33. end if
34. else
35. $G\leftarrow(X\rightarrow Y)$
36. $P(x,y)=P(X_{Y=0}=x)P(Y_{X=x}=y)$
37. end if
38. return $G,P(x,y)$
39. end function

<a id="appendix-g"></a>

## 附录 G：可迁移性与珀尔的因果层级结构

#### 可迁移性（Transportability）。

在分布偏移下评估策略的问题已在 **可迁移性理论（Transportability theory）** 中得到广泛研究（Pearl & Bareinboim, 2011; Bareinboim & Pearl, 2016; Bellot & Bareinboim, 2022）。
对于[第 2.2 节](#section-2-2)中概述的决策任务，可迁移性旨在为识别分布偏移后的期望效用 $R=\mathbb{E}[u\mid d,\textbf{pa}_{D};\sigma]$ 提供 **必要和充分条件（necessary and sufficient conditions）** ，给定以下（部分）知识：i) 源域中的联合分布 $P$、因果图 $G$ 和干预分布 $I$；以及 ii) 目标域中的联合分布 $P^{*}$ 和因果图 $G^{*}$ 的（部分）知识（Pearl & Bareinboim, 2011; Bareinboim & Pearl, 2012b）。
因此，这些结果与[定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 的不同之处在于，它们仅限于所有关于数据生成过程（即归纳偏置）的假设都可以表示为对底层 **因果贝叶斯网络（Causal Bayesian Network, CBN）** 的（部分）知识的情况。
例如，Bareinboim & Pearl (2016) 声称，在“假设可以用 **有向无环图（Directed Acyclic Graph, DAG）** 形式表达”的情况下，问题已基本解决。
这并不限制那些利用非因果假设和启发式方法的领域泛化可能途径（^5^55 事实上，超越 DAG 形式可表达的因果假设的著名例子包括限制结构方程类别（Mooij et al., 2016）和假设因果不对称性（Mitrovic et al., 2018）），并且深度学习算法确实利用了比单独因果假设更广泛的归纳偏置集合（Neyshabur et al., 2014; Battaglia et al., 2018; Rahaman et al., 2019; Goyal & Bengio, 2022; Cohen & Welling, 2016）。
在许多现实任务中，这些可能足以识别“足够好”（即遗憾有界）的策略，而无需了解数据生成过程的因果结构。
我们的目标一直是确定学习因果模型对于一般的领域泛化是否是必要的。
因此，假设智能体仅限于使用等同于对底层 CBN 的（部分）知识的归纳偏置，将会陷入 **循环论证（begging the question）** 。

#### 因果层级定理（Causal hierarchy’s theorem, CHT）。

著名的 **因果层级定理（Causal Hierarchy Theorem, CHT）** （Bareinboim 等人，2022；Ibeling & Icard，2021）表明，对于几乎所有环境，都存在环境变量之间的某些因果关联，这些关联无法在没有任何额外假设的情况下仅从观测数据中识别出来。这是否意味着识别最优策略必须需要一个因果模型？

首先，请注意 CHT 是一个 **不充分性（insufficiency）** 结果，它仅能推出一些平凡的必要性结论。例如，识别所有环境变量间的因果和关联关系是否必须需要一个因果模型？答案是肯定的，但这仅仅是因为这组观测和干预分布本身就是一个因果模型。形式上，我们可以通过假设 **因果忠实性（causal faithfulness）** （该假设对几乎所有因果模型都成立（Meek，2013））来识别底层的因果模型（在潜在混杂因素范围内）。此处的区别在于，CHT 关注的是 **所有环境变量间因果和关联关系的可识别性（identifiability）** 。这设定了一个比 **领域泛化（domain generalisation）** 高得多的标准，后者只关注识别这些关系的一个严格子集（即 **遗憾有界策略（regret-bounded policies）** ）（[图 3](#figure-3)）。

其次，CHT 关注的是 **因果层级（causal hierarchy）** 的坍缩（或未坍缩）。例如，观测数据不足以识别所有因果查询。我们并不限制智能体只能拥有观测训练数据——事实上，在我们考虑的 **在线学习（online learning）** 设置中，通常假设智能体可以同时访问观测数据和干预数据（例如，根据假设，智能体可以进行干预以固定决策节点 $D$）。

最后，我们可以设想一个 CHT 的改进版本，它断言：若无额外假设，仅凭观测数据不足以识别遗憾有界策略，从而使其与 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 和 [定理 2](https://arxiv.org/html/2402.10877v7#Thmtheorem2) 保持一致。即使 CHT 隐含了这一结论，它也不会直接推出我们的结果，除非我们限制在 **所有假设都作为对因果结构的约束** 的情况下（类似于 **可迁移性（transportability）** ）。同样，很容易证明 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 并不隐含 CHT。在推导 [定理 1](https://arxiv.org/html/2402.10877v7#Thmtheorem1) 时，我们并未局限于观测分布（也未对智能体生成策略时可用的数据施加任何限制）。
