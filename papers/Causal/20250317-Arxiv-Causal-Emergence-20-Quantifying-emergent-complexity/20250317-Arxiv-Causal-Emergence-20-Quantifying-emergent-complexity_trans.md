# 标题：因果涌现 2.0：量化涌现复杂性（Causal Emergence 2.0: Quantifying emergent complexity）

- ArXiv: 2503.13395
- 作者：Erik Hoel，艾伦发现中心，塔夫茨大学，梅德福，马萨诸塞州，美国
- 章节：22
- 估计词元数：17.3k

## 目录

- 1 引言
- 2 因果关系的公理
  - 2.1 充分性与必要性
  - 2.2 决定论与简并性
- 3 在 CE 2.0 中量化宏观尺度因果关系
  - 3.1 通过微观 → 宏观路径遍历尺度层级
  - 3.2 沿路径的因果分配
  - 3.3 选择微观 → 宏观路径
- 4 涌现复杂性
- 5 局限性与启发式方法
- 6 与其他涌现理论的比较
  - 6.1 公理化基础
  - 6.2 CE 2.0 捕捉了所有宏观尺度因果关系
  - 6.3 与其他相关涌现理论的比较
  - 6.4 CE 2.0 的概念性启示
- 7 结论
- 8 致谢
- 补充信息
- 附录 S1 因果基元（及其推广）对噪声和共同原因敏感
- 附录 S2 跨尺度的动力学一致性计算
- 附录 S3 通过 SVD 为 CE 2.0 框架提供启发式方法
- 参考文献

## 摘要（Abstract）

###### 摘要（Abstract）

复杂系统可以在无数不同的尺度上进行描述，其因果运作通常具有多尺度结构（例如，一台计算机可以在其硬件电路的 **微观尺度（microscale）** 、其机器代码的 **介观尺度（mesoscale）** 以及其操作系统的 **宏观尺度（macroscale）** 上进行描述）。尽管科学家们研究并建模系统跨越其全部尺度层次——从微观物理学到宏观经济学，但对于系统的宏观尺度除了单纯的压缩之外还能提供什么，一直存在争论。为了解决这个长期存在的问题，本文引入了一种新的 **涌现理论（theory of emergence）** ，其中系统的不同尺度被视为一个更高维对象的切片。该理论能够区分这些尺度中哪些具有独特的因果贡献，哪些在因果上不相关。该理论基于一个公理化的 **因果概念（notion of causation）** 构建，其应用在 **马尔可夫链（Markov chains）** 的粗粒化中得到了展示。它识别了所有宏观尺度因果作用的案例：即那些可以还原到微观尺度，但在因果信息上存在损失的情况。此外，该理论提出了一个 **因果分配模式（causal apportioning schema）** ，用于计算每个尺度的因果贡献，展示每个尺度独特添加了什么。最后，它揭示了一种新颖的 **涌现复杂性（emergent complexity）** 度量：即一个系统的因果运作在其尺度层次中分布得有多广泛。

<a id="section-1"></a>

## 1 引言（Introduction）

复杂系统（Complex systems）在多个尺度上运行，因此呈现出数量巨大的可能描述 [1]。这导致了一种主张，即在复杂系统（例如生物学 [2] 中的系统）中“不存在具有特权的因果层级”。然而，所有可能尺度的集合——以 **降维（dimensionality reduction）** 的形式存在——极其庞大，即使对于小型系统也是如此，并且其中大多数都是对系统因果运作的拙劣描述（例如，随机地对计算机的逻辑门进行粗粒化处理）。

这种 **多重性困境（embarrassment of multiplicity）** 催生了对 **涌现（emergence）** 的正式数学理论的需求。这样的理论应该解释并量化 **宏观尺度（macroscales）** （即基于某种降维的系统高层级描述）如何对系统的因果运作做出贡献，以及哪些宏观尺度具有因果相关性。一个涌现理论甚至可能解释科学本身在时空上的层级结构，而不仅仅是其作为有用压缩工具的功能 [3]。

或许有人会反驳，科学中没有涌现存在的空间，因为理论上任何给定系统的未来都可以通过对其微观尺度的完全认知来预测，并且任何给定系统都可以被还原到其微观尺度。然而， **预测（prediction）** 并不等同于 **因果（causation）** [4]。一个简单的例子是恒温器与房间系统 [5, 6]。虽然在理论上，房间内所有单个粒子的微观尺度可以用来预测恒温器的读数，但从因果理解的角度看，这对于回答“是什么导致恒温器读数为 $20^{\circ}\text{C}$？”这个问题来说是一个拙劣的答案。事实上，所有粒子的精确 **微观状态（microstate）** 对于该读数在因果上并非必要，因为许多其他构型也可能导致该读数。与此同时， **宏观状态（macrostate）** （房间的温度）与恒温器的读数有着直接的因果关系，因为它是任何给定读数值的必要条件。

在另一个例子中，神经元树突的输入信号可以用来预测下游动作电位将会发生。然而，在因果分析中，输入信号不足以（作为原因）触发某种精确的离子交换，因为这种交换会因噪声（例如来自 **布朗运动（Brownian motion）** [7] 或 **量子效应（quantum effects）** [8] 的噪声）而不可预测地演变。与此同时，输入信号仍然可以确定性地足以（作为原因）触发下游神经元的“发放”这一宏观状态，而无论其底层的微观尺度细节如何。

基于这些直觉，在 2013 年，我与合著者提出了 **因果涌现理论（theory of causal emergence）** [9]。该理论利用了离散因果模型（属于逻辑门网络、 **有向无环图（Directed Acyclic Graphs, DAGs）** 和 **马尔可夫链（Markov chains）** 的范畴）以及一种因果度量—— **有效信息（effective information, EI）** [10]。该理论提供了一个工具包，用于搜索此类系统所有可能的维度约简，以找到能最大化 EI 的那一个（其中，EI 的评估是通过 **do(x) 算子（do(x) operator）** [11] 将系统扰动至所有可能状态，然后计算干预的概率分布与其效果的概率分布之间的 **互信息（mutual information）** 来完成的）。识别出能带来最大 EI 增量的宏观尺度，就量化了系统中的因果涌现程度。

因果涌现揭示了为什么系统的宏观尺度尽管可以还原为其底层微观尺度，却可以拥有更强的因果关系：由于宏观尺度是 **多重可实现（multiply-realizable）** 的，它们可以最小化因果关系中的不确定性，而像 EI 这样的因果度量对此很敏感。这在数学上类似于通过信息通道进行编码可以最小化通信噪声的方式 [10, 3]。

**因果涌现理论（Causal emergence theory）** 自此催生了大量研究，例如测量从 **元胞自动机（Cellular automata）** [12] 到 **功能磁共振成像数据（fMRI data）** [13] 再到 **基因调控网络（Gene regulatory networks）** [14, 15] 等各种数据中的因果涌现，以及开发启发式方法 [16]，例如使用训练好的 **人工神经网络（Artificial Neural Networks, ANNs）** 来检测因果涌现 [17]。该理论已与网络理论中的 **无标度性（Scale-freeness）** 和 **鲁棒性（Robustness）** 等现象联系起来 [18, 19]，被整合到 **整合信息理论（Integrated Information Theory, IIT）** 中 [20, 21]，并且已有关于应使用何种度量来量化因果涌现的替代方案提议，例如利用系统的 **动力学可逆性（Dynamical reversibility）** 来近似 **有效信息（Effective Information, EI）** [22]。完整综述请参见 Yuan 等人 2024 年的文章 [23]。

然而，因果涌现的初始版本（以下简称 **CE 1.0** ）由于两个悬而未决的问题，一直是不完整的。第一个问题是对 **有效信息（Effective Information, EI）** （及其近似方法）的依赖，用以检测因果涌现。虽然 EI 是一个相对构建良好的因果度量 [24]，但其计算中做出了背景假设（例如要求干预的均匀分布，这一点已受到一些批评）[25, 26, 27]。此外，如本文第 6.2 节所示，使用 EI 实际上会 **低估因果涌现** 。

第二个问题是， **CE 1.0 仅识别了一个因果相关的尺度（即 EI 的最大值）** ，忽略了所有的多尺度结构。然而，许多系统似乎跨尺度运行；一个突出的例子是大脑的不同功能尺度，范围从神经元到皮层微柱再到整个脑区 [28]。另一个例子是，计算机可以在其硬件电路的微观尺度、其机器码软件的中观尺度或其操作系统和应用程序的宏观尺度上进行描述 [29]；事实上，甚至正在发生何种计算也可能因描述尺度而异 [30, 31]。

为了发展一个能解决这些问题、普适且根基扎实的因果涌现理论，本文引入了一种新颖的形式化框架： **因果涌现 2.0（Causal Emergence 2.0, CE 2.0）** 。 **CE 2.0 的基本直觉是，一个系统并不局限于某个特定的描述尺度；相反，最好用那些对系统因果运作有贡献的尺度集合来描述它** 。任何一个单一尺度（即使是微观尺度）都类似于对一个三维物体取二维切片，因此无法完全捕捉系统的因果关系。CE 2.0 提出了一个跨尺度的 **因果分配模式（Causal apportioning schema）** ，用以检测各个尺度对多尺度整体的因果贡献（如果有的话）。这是通过定义一个从“顶”到“底”遍历系统尺度的路径来实现的，该理论沿着这条路径分配系统运作的因果关系。

**CE 2.0 建立在一个公理化的因果概念之上，而非一个孤立的度量（如 CE 1.0 中的 EI）** ，这使得该理论能够捕捉所有宏观尺度因果关系的案例，并且能够以前所未有的方式展开和量化系统的多尺度因果结构。这种关于系统如何跨尺度运作的新分类法引出了一个新颖的度量： **涌现复杂性（Emergent complexity）** ，即系统的因果运作在其各尺度上的分布有多广泛，其中拥有许多贡献尺度的系统更为复杂。

下文将概述 CE 2.0，首先定义一个在背景假设下具有公理性和鲁棒性的因果概念，然后利用它来计算模型 **马尔可夫链（Markov chains）** 的 **粗粒化（Coarse-grains）** 中的宏观因果关系程度，从而量化其因果涌现的程度，接着详细说明如何通过一条遍历尺度集合的路径来分配因果贡献，并进一步探讨这如何自然地导向涌现复杂性的概念。最后，将 CE 2.0 与其他相关的涌现理论进行直接比较，展示其优势并概述其概念内涵。

<a id="section-2"></a>

## 2 因果关系的公理

<a id="section-2-1"></a>

### 2.1 充分性与必要性

科学家们提炼并提取关于他们所研究系统的因果知识 [32]。对因果关系的科学理解取得的突破包括 R.A. 费希尔（R.A. Fisher）对 **随机对照试验（Randomized Controlled Trials）** 的形式化 [33]，以及朱迪亚·珀尔（Judea Pearl）近期引入的 **do(x) 算子（do(x) operator）** [11]。

许多研究者引入了特定的 **概率因果度量（Probabilistic measures of causality）** ，以捕捉原因与结果之间的因果程度。这类度量的目标可以用多种方式描述，例如，捕捉特定原因的效力、特定因果关系的强度、一个变量因果控制另一个变量的程度等。应用这类因果度量需要指定一个 **因果模型（Causal model）** ，然后使用 **反事实（Counterfactuals）** [34] 或 **干预（Interventions）** [11] 来将因果知识与单纯的观察区分开。

最近一项工作分析了不同作者提出的十几种概率因果度量 [27]，结果表明，在科学文献中存在 **因果一致性（Causal consilience）** ：从心理学到统计学再到哲学 [35] 等领域独立引入的因果度量，都关联着两个基本项。这些项被称为“ **因果基元（Causal primitives）** ” [27]——更常见的是，它们被称为 **充分性（Sufficiency）** 和 **必要性（Necessity）** 。这种一致性适用于从哲学家大卫·刘易斯（David Lewis）[34]、数学家朱迪亚·珀尔（Judea Pearl）[11] 到我和合著者最近对 **实际因果关系（Actual causation）** 的定义 [36] 等一系列因果度量。这些基元被多次独立地重新发现，构成了任何因果度量的公理化基础，并确保了不同因果度量在数学行为上具有显著的重叠。最终，每一项都代表了 **不确定性（Uncertainty）** 的逆：充分性是在给定原因下对结果的确定性，而必要性是在给定结果下对原因的确定性。

正如将要展示的，充分性和必要性这两个因果基元有进一步的信息论推广，即 **确定性（Determinism）** 和 **简并性（Degeneracy）** （分别对应）。 **CE 2.0** 正是明确基于这些基元及其进一步的推广。

首先，为了形式化地定义因果基元，需要一些术语。对于像 **马尔可夫链（Markov chain）** 、 **有向无环图（Directed Acyclic Graph, DAG）** 或一组 **逻辑门（Logic gates）** 这样的离散系统，评估因果基元（以下简称 CP）需要指定一个与系统相关的抽象空间 $\Omega$，它定义了进入因果分析的发生项集合（例如，状态、事件、变量等）。然后，对于任何发生项（如状态转移），我们可以定义潜在原因 $C \in \Omega$ 和潜在结果 $E \in \Omega$。

由于该理论将在模拟的马尔可夫链中阐述，这里的 $\Omega$ 仅仅是系统的 **状态空间（State space）** 。发生项则是状态转移，即对于马尔可夫链，总是存在某个系统先前的单个状态 $c$ 及其结果 $e$（下一个状态）。这些转移各自具有某个概率 $P$。

给定一个发生项（此处为状态转移），原因的充分性定义为

$$
\text{suff}(e,c)=P(e \mid c),
$$

它随着 $c$ 在概率上更可能带来 $e$ 而增加，并在 $c$ 对 $e$ 完全充分时达到 1。

原因的必要性定义为

$$
\text{nec}(e,c)=1-P(e \mid C, \neg c),
$$

它指定了 $e$ 在没有 $c$ 的情况下发生的概率。也就是说，给定系统内某个原因集合 $C$，并且在该集合内 $c$ 本身没有发生，$e$ 的概率的逆是什么？当存在许多共同原因时，这个概率很低；只有当 $c$ 对 $e$ 绝对必要时，它才为 $1$，这意味着 $C$ 中没有其他成员能产生 $e$（在马尔可夫链中，这意味着没有其他状态会在下一个时间步导向状态 $e$）。

由于目标是比较整个尺度之间的因果关系，因此术语“充分性”和“必要性”（以及它们合称为 CP）此后将意味着它们在所有 $t$ 到 $t_{+1}$ 转移上的系统范围平均值。该平均值由每个转移的概率 $P(e \mid c)$ 加权，给定 $P(C)$。

在因果分析中，$P(C)$ 特意不是观测到的分布 [37]。$P(C)$ 的选择可以被概念化为识别一组可行的反事实，或者等价地，指定系统因果模型中一组可能的干预 [37, 10]。由于以下值将在由某个 **转移概率矩阵（Transition Probability Matrix, TPM）** 及其转移（从 $t$ 到 $t_{+1}$）定义的马尔可夫链中计算，因此这里的 $P(C)$ 是在给定系统某个尺度下，在整个状态集合上的 **均匀分布（Uniform distribution）** 。

从概念上讲，这仅仅意味着在作为可行的干预或反事实来评估其他状态的因果关系时，一个尺度的各个状态被平等对待。例如，给定一个自循环概率 $p=1$ 的 **复制门（COPY gate）** ，这意味着我们在 $P(C)$ 中平等地考虑 0 和 1，因此可以正确地说，状态 COPY $=1$（在 $t$ 时刻）对于状态 COPY $=1$（在 $t_{+1}$ 时刻）是充分且必要的（注意，使用观测分布则无法做出这样的判断）。

<a id="section-2-2"></a>

### 2.2 确定性（Determinism）与简并性（Degeneracy）

充分性（Sufficiency）与必要性（Necessity）各自都有一个信息论层面的推广：系统的 **确定性（Determinism）** 与 **简并性（Degeneracy）** [9]。

具体而言， **确定性（Determinism）** 是状态转移概率分布中噪声（或随机性）的逆，因此是充分性的信息论推广。对于单个原因 $c$，可以将其定义为一个系数，该系数基于 $c$ 在系统效应集合 $E$ 上的转移概率分布的熵（归一化到 $[0,1]$ 范围）：

$$
\textit{determinism}=1-\frac{H(E\mid c)}{\log_{2}n},
$$

其中核心的熵项为：

$$
H(E\mid c)=\sum_{e\in E}\text{{suff}}(e,c)\log_{2}\frac{1}{\text{{suff}}(e,c)}.
$$

在下文中，“确定性（Determinism）”一词专指系统范围的确定性系数，它是在给定 $P(C)$ 下，上述单个原因确定性的平均值。 **仅当转移概率矩阵（Transition Probability Matrix, TPM）由“独热（one-hot）”行组成时，确定性才达到最大；仅当所有行都是均匀分布（即转移是随机的）时，确定性才为零。**

“ **简并性（Degeneracy）** ”同样是一个系统范围的简并性系数，定义为在给定 $P(C)$ 下，对完整效应集合的概率分布 $P(E)$ 的熵的逆：

$$
\textit{degeneracy}=1-\frac{H(E\mid C)}{\log_{2}(n)}.
$$

如果原因具有许多相似的效应，则简并性高。当所有原因都具有唯一的效应时，简并性为零。在马尔可夫链（Markov chains）中，一个完全确定且非简并的系统是指每个状态都以 $p=1$ 的概率转移到某个唯一的下一个状态，且状态间转移无重叠（即一个置换矩阵）。简并性充当了必要性的一个 **包容性逆（inclusive inverse）** ，因为：

$$
H(E\mid C)=\sum_{e\in E}P(e\mid C)\log_{2}\frac{1}{P(e\mid C)},
$$

其中 $P(e\mid C)$ 是必要性计算 $P(e\mid C,\neg c)$ 中核心项的包容形式。它不是移除原因（$\neg c$），而是针对任何给定的 $e$，计算从完整的 $P(C)$ 集合中，导致该 $e$ 的可能原因总体上有多必要。

为了避免围绕“逆”产生的语言混淆（因为较低的简并性表示更强的因果关系），在计算过程中，简并性常被转换为 **特异性（Specificity）** ，其值增加表示因果关系更强：

$$
\text{{specificity}}=1-\textit{degeneracy}.
$$

总结来说：正如从相同的基本概率项构建它们时所见， **确定性本质上是充分性的归一化熵，而简并性本质上是必要性的归一化熵（作为其逆，并以包容方式计算）。**

这些因果原语（Causal primitives）及其推广对因果模型中关于原因和效应关系的不确定性很敏感，并且在不确定性减少或增加时表现出相似的行为，这使得它们成为检测宏观尺度因果关系（Macroscale causation）的合适基础。在补充信息（Supplementary Information, SI）的 S1 部分中，通过系统内概率重分布的模拟，展示了它们的敏感性和相似性。

<a id="section-3"></a>

## 3 量化 CE 2.0 中的宏观尺度因果性（Quantifying macroscale causation in CE 2.0）

**CE 2.0** 将 **宏观尺度因果性（macroscale causation）** 定义为：当 **因果基元（Causal Primitives, CP）** 在系统的某个宏观尺度上共同增加时，表明该宏观尺度降低了对原因和结果的不确定性。需要注意的是，并非所有的 **降维（dimension reduction）** 都会导致宏观尺度反映出 CP 的增加——许多会导致零增益甚至减少（即 **因果还原（causal reduction）** 的情况）。为了识别正向增益，CE 2.0 利用了一个有序的 **微观→宏观路径（micro→macro path）** ，该路径遍历系统的尺度层次结构，揭示其 **多尺度结构（multiscale structure）** 。系统中的 **因果涌现（causal emergence）** 程度是沿微观→宏观路径的 CP 总增益，代表了所有尺度上宏观尺度因果性的总和。然后，这种因果性沿着路径进行分配，以追踪哪些尺度具有正向的因果贡献（即它们增加 CP 的程度）。

<a id="section-3-1"></a>

### 3.1 使用微观→宏观路径遍历尺度层次结构（Traversing the hierarchy of scales with a micro → macro path）

在 CE 2.0 中， **宏观尺度（macroscales）** 被定义为系统某种降维的结果。对于一个给定的 **马尔可夫链（Markov chain）** $S$ 及其关联的 **转移概率矩阵（Transition Probability Matrix, TPM）** （代表微观尺度），定义一个新的系统 $S_{M}$，它拥有自己关联的 TPM，其中 **宏观状态（macrostates）** 取代了一组 **微观状态（microstates）** ，并且进出给定宏观状态的转移是底层微观状态的汇总统计量。

为简单起见，本文仅考虑对微观状态进行 **粗粒化（coarse-graining）** 的降维方法。粗粒化会产生诸如 $(0,1,2),(3)$ 这样的宏观尺度，这表明对于某个 4 状态系统，微观状态 $(0,1,2)$ 被粗粒化为一个宏观状态，而 $(3)$ 保持其在微观尺度的原样。此方法遵循先前关于因果涌现的研究（关于如何基于微观尺度 TPM 的粗粒化推导宏观尺度 TPM 的完整细节，请参见 [9, 18]）。

所有粗粒化结果都经过检查，以确保其作为底层微观尺度准确描述的有效性；具体而言，与 [18] 类似，检查了每个宏观尺度的 **动力学一致性（dynamical consistency）** ，并丢弃了不一致的宏观尺度（参见 SI 中的 S2）。请注意，CE 2.0 也可以应用于其他类似于粗粒化的降维类型，例如 **黑箱化（black-boxing）** [38, 10] 或 **高阶宏观状态（higher-order macrostates）** [18]。

一个 **微观→宏观路径（micro→macro path）** 是一组有效的（即动力学一致的）尺度，它们从微观尺度通向一个最终的宏观尺度，该宏观尺度作为路径的终点。从概念上讲，路径只是指定了系统的哪些粗粒化是跨越系统尺度层次结构、通往其他粗粒化的“必经之路”。路径中的每一步都是所有可能尺度集合中的一个“切片”，微观→宏观路径从底部（全维度）遍历到顶部（最终降维）。

在一个假设的 4 状态系统中，粗粒化 $(0,1),(2),(3)$ 将位于通往更低维度的粗粒化 $(0,1,2),(3)$ 的路径上。沿着这样一条路径，微观状态 $(0)$ 和 $(1)$ 首先会被一起粗粒化为单个宏观状态 $(0,1)$，然后进一步粗粒化为 $(0,1,2)$。对于这样一个假设的 4 状态系统，从微观尺度开始的一个完整的微观→宏观路径可能是：$(0),(1),(2),(3)$ $\rightarrow$ $(0,1),(2),(3)$ $\rightarrow$ $(0,1,2),(3)$ $\rightarrow$ $(0,1,2,3)$，最终所有状态被粗粒化为一个宏观状态。形式上，这可以表示为：

$$
\pi^{(1)}\;\longrightarrow\;\pi^{(2)}\;\longrightarrow\;\cdots\; \longrightarrow\;\pi^{(k)},
$$

其中每个 $\pi^{(i)}$ 是原始 $n$ 个微观状态的某个有效 **划分（partition）** （代表一个粗粒化），而 $\pi^{(i+1)}$ 又是 $\pi^{(i)}$ 的一个粗粒化，最终在终点划分 $\pi^{(k)}$ 处结束。

作为一个示例，图 1 可视化的 8 状态马尔可夫链绘制了一条选定的微观→宏观路径。路径上的进展通过 **颜色传染（color contagion）** 在图 1 中显示。从微观尺度 $(0),(1),(2),(3),(4),(5),(6),(7)$ 开始，沿着路径被一起粗粒化的微观状态在每个粗粒化步骤中被更改为相同颜色，直到终点 $(0),(1,2,3,4,5,6,7)$，此时除 $(0)$ 外，所有微观状态都被粗粒化在一起。

<a id="figure-1"></a>

![Figure3](images/Figure3.png)

> 图 1：微观→宏观路径可视化。一个示例 8 状态马尔可夫链，其转移概率用灰度表示（其 TPM 见图 2A）。从 $(0),(1),(2),(3),(4),(5),(6),(7)$ 的完整划分（微观尺度）开始，状态被一起粗粒化，每个进一步的划分都是路径中的一步（因此也是系统中的一个尺度），最终到达 $(0),(1,2,3,4,5,6,7)$。变为相同颜色的变化指示了沿着选定路径状态被一起粗粒化的时刻（颜色传染）。

<a id="section-3-2"></a>

### 3.2 沿路径的因果分配（Causal apportioning along a path）

**因果涌现（Causal Emergence, CE）** 沿路径的分布，可通过一种因果分配方案找到。具体而言，对于给定的系统和选定的 **微观→宏观（micro→macro）** 路径，CE 2.0 计算路径中每一步相对于前一个尺度的 $\Delta\mathrm{CP}$。

具体来说，对于给定尺度（路径中的每一步），其 CP 值是宏观尺度（或路径起始处的微观尺度）的 **充分性（sufficiency）** 与 **必要性（necessity）** 之和，然后减去 1，以使其值域限定在 $[0,1]$ 之间。同样地，每一步的 **确定性（determinism）** 与 **特异性（specificity）** 的 CP 值也通过将它们相加并减去 1 来计算（这使得该值与 CE 1.0 [9] 中的 **有效性（effectiveness）** 等价）。

沿路径计算 $\Delta\mathrm{CP}$ 的过程在一个 8 状态 **马尔可夫链（Markov chain）** 中进行了演示，其起始微观尺度的 **转移概率矩阵（Transition Probability Matrix, TPM）** 如图 2A 所示。该系统存在一个直观的 **宏观状态（macrostate）** ，形式为微观状态 $(4,5,6,7)$ 上的一个等价类，这些微观状态共享完全相同的转移概率分布。其连接性（状态转移）与图 1 所示相同，并且使用了相同的选定微观→宏观路径。

<a id="figure-2"></a>

![Figure4](images/Figure4.png)

> 图 2：沿微观→宏观路径的因果基元。(A) 微观尺度的 TPM，单元格根据其概率着色（$p=1$ 为深蓝色）。(B) 宏观尺度的 TPM，经过该尺度后 $\Delta\mathrm{CP}$ 突变为零。(C) 同一宏观尺度可视化为马尔可夫链，并标注了粗粒化后的宏观状态（其 $p=1$ 的自环未显示）。(D) 沿维度增加路径的因果基元变化，CP 的总增益为 $0.33$（以确定性加特异性计），反映了因果涌现的程度。

值得注意的是，在该系统中，CP 显示出持续增益，直到直观的宏观尺度 $(4,5,6,7)$ 被 **粗粒化（coarse-grained）** 成一个具有自环的单一宏观状态（该宏观尺度的 TPM 如图 2B 所示，并在图 2C 中可视化）。随后，路径立即进入零增益区域（绘制于图 2D 中）。

以确定性加特异性表示，微观尺度起始于 CP $=0.66$，在过渡到零增益之前的宏观尺度处，该值增加了 $0.33$，达到最大值 CP$=1$，这表明在该宏观尺度上，因果关系变得最大程度地确定且非退化，进一步的粗粒化不会带来额外增益。因此，该系统的因果涌现程度为 CE $=0.33$，反映了沿路径的总增益。

<a id="section-3-3"></a>

### 3.3 选择微观 → 宏观路径（Choosing micro → macro paths）

在分析系统的因果涌现时，可能有先验理由选择某条特定路径；然而，也可以基于第一性原理来识别合适的微观→宏观路径。

具体而言，微观→宏观路径的终点可以选择为能带来可能最高 CP 总增益（即最大因果涌现）的宏观尺度。如果存在多个可能的终点具有相同的最高增益，那么代表维度缩减程度最低的宏观尺度是最佳终点，因为它标志着超过此点后，维度缩减不会带来 CP 增益。一旦确定了终点，用于分析 $\Delta\mathrm{CP}$ 的信息量最大的微观→宏观路径，就是从微观尺度到终点宏观尺度范围内，跨越所有 **一致宏观尺度（consistent macroscales）** （一致性定义见 SI 中的 S2）的最长路径。

作为终点的宏观尺度可以通过 **暴力搜索（brute-force search）** 来识别：首先生成所有可能的宏观尺度，丢弃不一致的尺度，计算它们的 CP，然后选择具有最高 CP 和最高维度的宏观尺度作为路径的终点。

在实践中，对于较大的系统，这并不可行，需要 **启发式方法（heuristics）** 。一种方法是沿着一条路径进行粗粒化，直到达到 **收益递减（diminishing returns）** 。正式地说，路径中每一步 $i$ 都存在一个增量增益序列 $\Delta\mathrm{CP}_{i}$。如果 $\Delta\mathrm{CP}_{i^{*}}<\varepsilon$（对于某个小的阈值 $\varepsilon>0$），或者比率 $\Delta\mathrm{CP}_{i+1}\!/\!\Delta\mathrm{CP}_{i}$ 在较长的路径长度上持续下降，则系统在步骤 $i^{*}$ 进入“收益递减”状态。

换句话说，一旦 $\Delta\mathrm{CP}$ 变得可忽略不计（低于 $\varepsilon$）或在路径的较长部分持续逐步缩小，就表明达到了收益递减的转折点和一个近似的终点，该终点不会遗漏 CP 的实质性增益。然而，必须注意不要仅仅因为选择了过小的 $\varepsilon$ 或过短的路径长度来评估收益递减，而简单地停留在 **局部最大值（local maxima）** （见第 4 节）。

为了在定义路径甚至任何宏观尺度之前，对系统中的因果涌现量进行界定，可以简单地测量微观尺度下的 CP。其与 1 的距离为 CE 提供了一个上界，无需跨尺度搜索，从而允许快速估计。如果微观尺度的 CP 显著小于 1，则很可能存在某种维度缩减（如粗粒化）能将其增加到接近或达到最大值；因此，对于许多系统，微观尺度的 CP 与 1 的差值可以近似作为 CE 值（但并未指明增益来自哪个宏观尺度）。甚至有可能，对于足够大且具有足够结构的系统，总存在某个宏观尺度使 CP 趋近于 1，尤其是在考虑完整的维度缩减集合时（如 **高阶宏观状态（Higher Order Macrostates）** [18]），或者当放宽一致性假设时。

<a id="section-4"></a>

## 4 The emergent complexity（涌现复杂性）

传统上，复杂性科学领域的一个主要动机是这样一个定性概念：系统在微观尺度之外，存在着直观的宏观或介观尺度结构。用于检测宏观或介观尺度结构的具体定量方法主要集中于 **可压缩性（compressibility）** 或 **效率（efficiency）** [39]，或者最近通过评估 **信息论惊奇（info-theoretic surprise）** [40] 来实现。然而，这意味着检测到的介观尺度可能只是方便的压缩结果，而 **没有因果相关性（causal relevance）** 。

相比之下，通过分析 **因果贡献（causal contributions）** 的分布， **CE 2.0** 可用于量化那些对系统运作真正具有因果相关性的 **涌现复杂性（emergent complexity）** 。具体来说，路径上每一步的 $\Delta\mathrm{CP}$ 代表了该尺度对总 **CP（Causal Power，因果力）** 的因果贡献（总 CP 完全决定了系统的因果运作）——因此可以评估其沿路径的分布。因此，CE 2.0 为系统因果运作的复杂程度提供了一种分类法：如果因果运作主要局限于单一尺度（例如要么是微观尺度，要么是由“头重脚轻”的宏观尺度主导），那么系统是简单的；而如果系统存在也具有实质性因果贡献的中间介观尺度，那么它就是复杂的。

<a id="figure-3"></a>

![Figure5](images/Figure5.png)

> 图 3：跨尺度的因果贡献。(A) 一个没有介观尺度结构的系统的微观尺度 TPM。(B) 同一系统的可视化表示。(C) CE 2.0 将该系统的因果贡献识别为“头重脚轻”，即最后一次维度约简贡献最大。(D) 一个在其他方面相似但具有介观尺度结构的系统的微观尺度 TPM。(E) 介观尺度系统的可视化表示。(F) 因果贡献向更小的维度约简方向偏移，表明主要是多尺度的因果结构；因此它具有更多的涌现复杂性。

为了展示 CE 2.0 的这一新颖特性，在图 [3](#figure-3) 中分析了两个因果涌现系统：一个没有介观尺度结构，另一个有，但在其他方面尽可能相似。

图 3A 展示了一个由直观宏观尺度组成、没有可区分介观尺度结构的系统的 **TPM（Transition Probability Matrix，转移概率矩阵）** ，其中 $(0,1,2,3)$ 和 $(4,5,6,7)$ 已被粗粒化为两个各自的 **宏观态（macrostates）** 。确实，粗粒化到这两个宏观态的微观态各自构成了一个 **等价类（equivalency class）** （可视化见图 3B）。

沿着微观→宏观路径的每个尺度，都追踪了 $\Delta\mathrm{CP}$。值得注意的是，对于第一个系统，沿着微观→宏观路径的 CP 增益集中在路径的终点（见图 3C）。这表明了一种“头重脚轻”的结构，主要由微观尺度的因果贡献（占总 CP 的 0.14）和跨越两个等价类的大宏观尺度终点的因果贡献（贡献 0.18）构成。微观尺度和宏观尺度共同占据了该系统全部 CP 值（0.41）的大部分，量化了其因果运作主要由这两个尺度主导。

为了正式捕捉“头重脚轻”或“底重脚轻”系统与具有实质性介观尺度结构的系统之间的分类，这里我引入了 **涌现复杂性（Emergent Complexity, EC）** 的概念。它基于长度为 $L$ 的路径上因果贡献的 **熵（Entropy）** 。给定每一步（不包括微观尺度）$i=1,2,\ldots,L$ 的一组增益 $\Delta\mathrm{CP}_{i}$，则

$$
p_{i}\;=\;\frac{\Delta\mathrm{CP}_{i}}{\sum_{j=1}^{L}\Delta\mathrm{CP}_{j}} \quad\text{对于}i=1,\dots,L
$$

这确保了 $\{p_{1},\dots,p_{L}\}$ 是 $L$ 个步骤上的概率分布。为了衡量增益的“分散”（多尺度）程度，计算熵：

$$
\text{EC}=-\sum_{i=1}^{L}p_{i}\log_{2}(p_{i})
$$

如果所有 $\Delta\mathrm{CP}_{i}$ 都相等（即 $p$ 是均匀分布），则 EC 等于 $\log_{2}(L)$；当路径中少数步骤主导了总增益时，EC 值减小；当单个涌现尺度具有唯一的因果贡献时，EC 降至零。为了比较长度显著不同的路径，这些值可以进一步用 $\log_{2}(L)$ 进行归一化。

为了演示此方法如何检测介观尺度结构，在图 3D-E 中对一个相似系统进行了建模，但做了如下改变：$(0)$ 和 $(4)$ 在状态转移方面，与它们在其终点宏观态 $(0,1,2,3)$ 和 $(4,5,6,7)$ 中粗粒化在一起的其他成员是可区分的。也就是说，对于这个系统，对 CP 产生增益的最大维度约简并非发生在一个纯等价类上。

在这个介观尺度系统中（在 $(0,1,2,3)$ 和 $(4,5,6,7)$ 为宏观态的终点处，总 CE 为 0.13），当绘制沿路径的因果贡献时，存在一个可见的更早的峰值（图 3F）。由于 $\Delta\mathrm{CP}$ 存在这个更早的峰值，介观尺度系统的涌现复杂性为 2.07 比特，而“头重脚轻”系统的涌现复杂性仅为 1.67 比特。也就是说，介观尺度是通过 $\Delta\mathrm{CP}$ 在比终点更高维度处达到峰值来揭示的。

<a id="section-5"></a>

## 5 局限性与启发式方法（Limitations and heuristics）

**因果涌现 2.0（Causal Emergence 2.0, CE 2.0）** 的一个局限性在于，其当前表述假设了一条特定的、选定的 **微观→宏观路径（micro→macro path）** 来定义尺度的层次结构。这种初始表述在实践和概念上都有其意义；特别是因为分析时通常只需要一条路径。然而，在开发一种能够整合所有 **不可公度的微观→宏观路径（non-commensurate micro→macro paths）** 的因果分配方案方面，仍有工作要做；潜在的机遇在于开发一种能在所有划分集合上运作的因果分配方案，利用诸如 **默比乌斯反演（Möbius inversion）** [41] 或 **沙普利值（Shapley values）** [42] 等工具。

CE 2.0 一个更实际的局限性在于，遍历系统的所有宏观尺度集合会导致 **组合爆炸（combinatorial explosion）** 。对于 CE 1.0 [17, 16]（以及在连续系统中 [43]）已有一些先前的启发式方法，其中最近包括 Zhang 等人 [22] 的工作，他们提出使用 **奇异值分解（Singular Value Decomposition, SVD）** 作为一种精确方法来估计在某个宏观尺度上可用的 **有效信息（Effective Information, EI）** 增益，而无需在所有尺度集合中进行搜索（从而避免了相关的组合爆炸）。这项工作也强调了 **动力学可逆性（dynamical reversibility）** 对因果涌现的重要性，因为他们提出的动力学可逆性度量与 **确定性（determinism）** 和 **简并性（degeneracy）** 的行为非常相似（这表明因果基元与可逆性存在联合关系）。

因此，在本文的 **补充信息（Supplementary Information）** 中，基于对其 SVD 方法的调整，探索了一种用于 CE 2.0 的新颖启发式方法。如 S3 节所述，该方法能够检测第 4 节中比较的系统的多尺度结构差异，而无需在宏观尺度上进行搜索，这表明 CE 2.0 分析（例如因果涌现和涌现复杂性）可以在没有组合爆炸的情况下进行估计。

<a id="section-6"></a>

## 6 与其他涌现理论的比较（Comparison to other theories of emergence）

作为一种理论， **因果涌现 2.0（Causal Emergence 2.0, CE 2.0）** 具有许多优势：(a) 它在因果分析最基础的术语上具有 **公理化基础（axiomatic grounding）** ，并对该分析中的假设具有鲁棒性；(b) 它涵盖了所有可能的宏观尺度因果案例，而 **因果涌现 1.0（Causal Emergence 1.0, CE 1.0）** 则不能；(c) 它以新颖的方式阐明了系统的 **多尺度因果结构（multiscale causal structure）** ，解决了围绕 **过度决定（over-determination）** 和 **因果排除（causal exclusion）** 的长期冲突。下文将对这些优势进行探讨和论证。

<a id="section-6-1"></a>

### 6.1 公理化基础（An axiomatic grounding）

如前所述，CE 1.0 中对 **有效信息（Effective Information, EI）** 的使用曾因其源自 **最大熵（maximum entropy）** （均匀）分布 [24]（此处由某个 $P(C)$ 表示）[26, 25] 而受到批评。关键问题在于，CE 1.0 需要 EI 的这一假设来检测因果涌现——而事实上，一个因果涌现理论不应依赖于 EI 计算背后的假设。这既会影响其实用性，也会影响其理论基础。在某些情况下，例如在 **整合信息理论（Integrated Information Theory, IIT）** 中，最大熵分布可以进一步被证明为对系统采取“内在视角（intrinsic perspective）”的函数 [21]，但这依赖于接受 IIT 的假设，包括其对意识的分析。

在 CE 2.0 中，同样建议对 $P(C)$ 使用均匀分布。这是因为不允许 $P(C)$ 在不同尺度间变化，将意味着宏观尺度的反事实和干预无法独立于其微观尺度进行计算；这对于大多数科学的因果模型而言是一个荒谬的结果（例如，评估电灯开关和灯泡之间因果关系的强度，将需要根据各自的总原子数进行加权等）。

然而，与 CE 1.0 不同，这一关于 $P(C)$ 的建议并非检测因果涌现所必需的。事实上，宏观尺度上 **因果力量（Causal Power, CP）** 的增益已被证明对 $P(C)$ 的选择具有鲁棒性，其增益程度甚至在微观和宏观尺度都使用观测分布时也会出现 [27]。这一优势也可以在图 4 中直接看到，其中微观和宏观尺度的 $P(C)$ 都是各自尺度上的观测分布，因此宏观尺度的 $P(C)$ 本身就是微观尺度 $P(C)$ 的 **粗粒化（coarse-grain）** （意味着干预分布的任何描述在尺度间是等价的）。然而，CE 2.0 在这些条件下仍然能检测到宏观尺度因果性。

总之，CE 2.0 在理论上比 CE 1.0 更具鲁棒性，因为它植根于历史上已被证明是因果本质基础的 **因果原语（causal primitives）** ，并且能够在因果分析执行方式的不同背景假设下检测到因果涌现。

<a id="section-6-2"></a>

### 6.2 CE 2.0 涵盖所有宏观尺度因果性（CE 2.0 captures all macroscale causation）

CE 2.0 能够检测到 CE 1.0 框架无法检测的宏观尺度因果性案例。

这种检测的示例展示在一个由两个等价类构成的 8 状态系统中，使其在微观尺度上成为一个“块模型（block model）”（见图 4A，左）。指定了一个单一的宏观尺度，其中两个等价类各自被粗粒化为一个具有自循环的相应宏观态，由粗粒化 $(0,1,2,3),(4,5,6,7)$ 表示。通过 CE 1.0（基于宏观尺度 EI 的增益）计算的因果涌现，与通过 CE 2.0（基于宏观尺度 CP 的增益）计算的因果涌现，在该系统的操控过程中被展示出来。

从图 4A（左）所示的初始 **转移概率矩阵（Transition Probability Matrix, TPM）** 开始，对每个状态 $s_{i}$ 在其所属等价类内的概率进行操控，使得在 50 个步骤内，先前转移到等价类内其他成员的概率被逐步加到 $s_{i}$ 的自循环概率上，最终达到 $p=1$。此操控的中点如图 4A（中）所示，最终的结束系统是图 4A（右）中的 TPM。这通过概率重新分配的离散步骤，将系统从初始的“块模型”配置推进到一组具有 $p=1$ 自循环的 8 个微观态形式的 **置换矩阵（permutation matrix）** 。

在每一步，都展示了基于 EI 的 CE 1.0 计算结果，并与 CE 2.0 中 CP 的增益（以固定的选定宏观尺度为终点）进行比较。与直觉相反的是，CE 1.0 中的 EI 没有检测到任何因果涌现，即使系统最初被分割成两个等价类。与此同时，从 CE 2.0 的新视角来看，图 4 中的系统合理地从一个显著程度的宏观尺度因果性开始。然后，该程度随着自循环概率的增加和微观尺度可区分性的增加而降低，随着宏观尺度的贡献逐渐减少而变得越来越弱，直到当微观尺度变得完全确定性和非退化时，宏观因果性完全消失。

<a id="figure-4"></a>

![Figure6_new](images/Figure6_new.png)

> 图 4 | 图注描述。

**图 4：CE 1.0 无法捕捉所有宏观尺度的因果关系。** (A) 一个具有两个 **宏观状态（Macrostates）** 的“块模型”系统，在其等价类上，于增加每个状态自循环概率的起始点、中点和结束点时的 **转移概率矩阵（Transition Probability Matrices, TPMs）** （概率以蓝色深浅表示）。概率重分配是通过以 $1/步骤$ 的增量从完整的转移集合中抽取概率来执行的，直到微观尺度完全由自循环概率 $p=1$ 的状态组成。(B) CE 2.0 检测到了宏观尺度的因果关系，并在概率重分配过程中随着微观尺度在因果上变得更可区分而合理地下降，而 **有效信息（Effective Information, EI）** 则没有。

在某些情况下，CE 1.0 和 CE 2.0 会重叠（例如，图 4 中 $\Delta\mathrm{CP}$ 转变为零之前的那个尺度，与通过搜索 EI 最大值所选取的尺度相同）。这是因为 CE 1.0 和 CE 2.0 在数学项上有着密切的联系，因为 EI 有一个分解式，其中：

$$
\textit{EI}=\textit{effectiveness}*{\log_{2}n}
$$

也就是说，EI 可以分解为确定性减去简并性（即有效性），然后乘以一个规模项 ${\log_{2}n}$，而该规模项正是给定尺度的维度（其状态数量）[9]。然而，在 CE 2.0 中，通过适当的 **多尺度结构分析（Multiscale Structure Analysis）** 和 **因果分配（Causal Apportioning）** ，规模项变得不再必要。

虽然 CE 2.0 侧重于 $\Delta\mathrm{CP}$，但如果为了因果建模或解释而需要一个具有高维度的单一因果相关尺度，其方法可以用来挑选出一个： **在维度缩减量最小的情况下最大化因果潜力（Causal Potential, CP）的单个宏观尺度** 。这可以通过分析 $\Delta\mathrm{CP}$ 的收益递减来实现，如第 3.3 节所述，甚至可以通过明确地重新引入规模项来权衡 CP 的增益，从而灵活地重新捕捉 CE 1.0 的分析方法。

<a id="section-6-3"></a>

### 6.3 与其他相关涌现理论的比较

对于检测 **因果涌现（Causal Emergence）** 或更广义的涌现，已有其他替代性建议，例如通过 **整合信息分解（Integrated Information Decomposition）** [44]，或通过检查 **动态依赖性（Dynamical Dependency）** [45]。然而，这两种建议都利用了 **互信息（Mutual Information）** 。这对于因果涌现来说是有问题的，因为因果关系的定义性特征在于它不仅仅依赖于被测量过程的 **数据分布（Data Distribution）** [37]。例如，在一个 COPY 门循环中，互信息的大小完全取决于初始状态的变化程度，而不是像 **因果基元（Causal Primitives）** 那样捕捉到每个门对于链中的下一步既是充分的也是必要的这一事实 [27, 22]。

近期其他关于涌现（Emergence）的研究侧重于识别 **宏观尺度（macroscales）** 与其底层 **微观尺度（microscales）** 保持一致，但在动力学上仍可独立描述的情况，例如计算机的软件 [29]。这些研究探讨了宏观尺度是否“因果闭合（causally closed）”，即能否将其视为自身的原因。这与本文基于随机游走者（random walkers）提出的宏观尺度一致性条件密切相关，并且（在较宽松程度上）也与先前的研究相关 [18]。然而，仅仅检查一致性（consistency）、可聚合性（lumpability）、因果闭合性等，并不能直接衡量 **因果涌现（Causal Emergence, CE）** ，因为它并未反映宏观尺度对系统因果运作的贡献如何超越微观尺度——这需要某种特定的因果度量（或者，在此处，是构成此类度量基础的 **因果基元（causal primitives）** 的增益）。相反，它仅仅识别出哪些宏观尺度是其微观尺度的有效描述，保留了其动力学，因此是适当的压缩。例如，对于图 4 所分析的系统，在预选路径末端实现的最大维度缩减是一个有效的宏观尺度，与其微观尺度完全一致，但其因果贡献却微不足道，为零。

<a id="section-6-4"></a>

### 6.4 CE 2.0 的概念性启示

除了微观物理学之外，所有科学领域都隐含地以存在因果涌现的方式运作，即在其模型、解释和实验中，理所当然地认为宏观尺度实体对系统的因果运作是有效的。这与名义上对 **普遍还原论（universal reductionism）** 的承诺相矛盾，后者似乎暗示所有因果力都会“流失”到任何系统的底层微观尺度 [46, 47]。这是 **因果排除论证（causal exclusion argument）** 的一个效应 [48]——对于任何给定的随附性宏观尺度，其效应也可以被描述为其底层微观尺度的原因，这使得宏观尺度的描述变得不必要。

**CE 1.0** 颠覆了排除论证，它指出，根据 **有效信息（Effective Information, EI）** ，宏观尺度拥有更大的因果力。类似的思想也构成了 **整合信息理论（Integrated Information Theory, IIT）** 中排除公设的基础（这可能是该理论中最具争议的公设 [49]）。但这意味着在 CE 1.0 中，存在一个反直觉的结果：即使宏观状态并非基于精确的等价类，底层微观尺度本身也可能被排除；这是一个令人惊讶的结果，其认识论和本体论含义尚不明确。

相比之下，在 **CE 2.0** 中，因果排除的处理方式更为优雅。从当前对单一路径的分析来看，宏观尺度并不会覆盖微观尺度的因果作用（尽管微观尺度的因果贡献可能仍然微乎其微）。相反，它们只是通过 **因果分配方案（causal apportioning schema）** 贡献了额外的因果力，从而导向一个更全面的 **分体论（mereology）** ，其中各个尺度是一个包含系统因果运作所有相关信息的高维对象的 **有损切片（lossy slice）** 。

尽管其他涌现理论常常假定某些宏观尺度属性或定律甚至在原则上也无法还原到微观尺度，因此物理学并非“因果闭合的”（例如，[50, 51, 52]），但这对于涌现理论而言仍然是一个有争议的要求 [53, 54]。相比之下，因果涌现即使在宏观尺度完全可以还原到其微观尺度时（如本文中的模型）也可能发生，因为宏观尺度无论如何都能减少因果关系中的不确定性。也就是说，即使宏观尺度本身是可还原的，它们在因果基元上的增益根据定义是不可还原的。这些超越微观尺度的增益来源并非神秘，而是基于不确定性的减少 [10, 55]，而这又源于 **宏观状态的多重可实现性（multiple-realizability of macrostates）** [3]。

由于在 CE 2.0 中，涌现是通过宏观尺度上这种不确定性的最小化而发生的，这就引出了一个更广泛的问题：当涉及到科学的因果模型及其所代表的系统时，不确定性（以噪声或共同原因的形式）是否仅仅存在于认识论层面。回答这个问题涉及对未知事物的推测，例如物理学的科学终态 [3]。即使是微小的真实不确定性来源（如 **非决定论（indeterminism）** ）也可能在混沌系统中被放大，甚至可能存在可证明的不可判定的物理系统 [56]。即使科学因果模型的因果关系中固有的所有不确定性最终被证明在原则上只是认识论的，这也仅适用于跨越整个宇宙的封闭因果模型，而在这样一个宇宙尺度的因果模型中，所有因果概念无论如何都会完全消失，因为不存在来自模型外部的可定义干预 [11]。因此，虽然因果涌现在微观尺度对原因的效果具有零不确定性且没有共同原因（例如，像置换矩阵那样）的条件下会消失，但这样的条件意味着与科学中的大多数因果模型相去甚远。

<a id="section-7"></a>

## 7 结论（Conclusion）

**因果涌现 2.0（Causal Emergence 2.0, CE 2.0）** 提供了一个概念上和数学上全新的涌现理论，它将系统视为一个 **尺度层级结构（hierarchy of scales）** 。单个尺度，甚至在大多数情况下是微观尺度，都仅仅是一个更高维对象的切片——但只有极少数的这些尺度具有因果相关性，而该理论能够识别它们，从而揭示对系统因果运作至关重要的层级结构。

具体而言，该理论展示了如何通过沿一条遍历系统所有可能 **降维（dimension reductions）** 的特定路径上， **因果基元（causal primitives）** （即充分性和必要性）在 **信息论（information-theoretic）** 层面上的泛化增益来度量宏观尺度因果性。 **因果涌现（Causal emergence）** 是这种增益的总和，并且因果贡献可以沿着该路径进行分配。该理论揭示了一种新颖的分类法，其中一些系统在因果性上是“头重脚轻的（top-heavy）”，例如当单一宏观尺度占主导时，而另一些系统则具有更多的 **中尺度（mesoscale）** 结构。 **涌现复杂性（emergent complexity）** 的计算量化了因果贡献在尺度层级结构中分布的广泛程度。

该理论的初始表述基于一条遍历降维的单一路径，确实存在一些局限性。它没有具体说明如何在多条路径之间分配贡献，并且在应用时会面临 **组合爆炸（combinatorial explosions）** 。然而，如何解决这些局限性是相对明确的（关于无路径方法的讨论，请参见第 5 节；关于为 CE 2.0 分析提出的启发式方法，请参见 S3）。

CE 2.0 在物理学、生物学、神经科学和经济学等领域具有重要的应用价值。在此，也值得强调一个特定的新用例：鉴于因果基元对于理解复杂系统中的因果性具有公理般的重要性，并且鉴于先前研究表明 **有效信息（Effective Information, EI）** 对人工神经网络学习后的变化有响应 [57]，CE 2.0 也可能为不断发展的 **人工智能可解释性（AI interpretability）** [58] 和 **人工智能安全（AI safety）** [59] 领域做出贡献，例如分析深度神经网络的多尺度结构。

<a id="section-8"></a>

## 8 致谢（Acknowledgments）

感谢 Softmax 公司为 CE 2.0 的开发及本文的撰写提供资金支持。同时感谢亚当·戈尔茨坦（Adam Goldstein）和埃米特·希尔（Emmett Shear）对草稿提出的宝贵反馈，也感谢迈克尔·莱文（Michael Levin）的鼓励与交流。

## 补充信息（Supplementary Information）

## **附录 S1 因果原语（及其推广）对噪声和共同原因敏感**

先前关于 **因果一致性（Causal Consilience）** 的研究已经表明，由于其紧密的数学关系，在一组两个转移上的 **充分性（Sufficiency）** 和 **必要性（Necessity）** ，在不确定性增加的情况下，其行为与它们在信息论中的对应概念—— **确定性（Determinism）** 和 **简并性（Degeneracy）** ——相似 [27]。在此，我们通过一组更大的转移来展示它们的相似性。

具体而言，我们指定了一个由 8 个状态组成的系统，每个状态的自环概率为 $p=1$（见图 S1，左上角）。为了改变关于原因和结果的不确定性，我们沿着两个独立的轴引入了 **噪声（Noise）** （关于结果的不确定性）和 **共同原因（Common Causes）** （关于原因的不确定性）。第一个轴通过将系统转变到其转移完全随机的情况（一个全连接的马尔可夫链，其中所有转移概率 = $1/n$，这意味着系统的行为尽可能不可预测），来增加关于结果的不确定性。第二个轴，即关于原因的不确定性，将系统移动到所有转移都具有相同结果集的情况（从而增加了共同原因的数量）。

沿着增加噪声的轴，每一步中，每个自环的概率被平均重新分配到系统中的其他状态，每一步重新分配的总概率为 $1/steps$。沿着增加共同原因数量的轴，每一步中，一个状态（TPM 中的一行）的完整转移集被逐一替换为第一行的副本，直到所有分布都相同。仅沿后一个轴改变模型，始于所有状态具有唯一的状态转移，终于所有状态都转移到单一状态（见图 S1，左下角）。最后，将这两个系统变化的轴结合起来，使得在每一步中，既引入了更多的结果噪声，也在同一步中引入了更多的共同原因（见图 S1，中间对角线）。

在每一步，我们计算了每个状态在系统范围内的充分性加必要性，以及确定性加特异性，分别沿着增加结果不确定性的轴（图 S2A）、增加共同原因的轴（图 S2B），以及两个轴结合的路径（图 S2C）。与正文中一样，这样做是为了确保相同的 $[0,1]$ 边界（见第 3.2 节）。

![Figure1](images/Figure1.png)

> **图 S1 | 增加一个 8 状态系统中因果关系的不确定性。** 从自环概率 $p=1$ 的状态开始，网络中的状态以三种方式在等于系统规模（状态数）的固定步数内发生变化。沿 x 轴，自环概率每步减少 $1/steps$ 并平均分配给其他状态（从而在给定原因的情况下，增加了特定结果的不确定性），直到系统成为一个具有随机转移的全连接网络。沿 y 轴，每一步将一个状态替换为另一个状态的转移分布（增加了共同原因的数量，从而在给定结果的情况下，增加了原因的不确定性），直到系统中的所有状态共享相同的转移。系统在每一步也同时经历这两种变化（中间对角线），最终再次达到具有随机转移的全连接状态。

值得注意的是，所有 **因果原语（Causal Primitives, CP）** 值在增加原因和结果不确定性的两个步进轴上均表现出相似的变化，即使在更大的系统中也是如此（100 个状态和 100 步的值也显示在图 2D 中）。

![Figure2](images/Figure2.png)

> **图 S2 | 因果原语随不确定性共同变化。** （A）当系统中自环的概率（如图 S1 所示）在递增的步数中重新分配时，充分性加必要性值以及确定性加特异性值在增加结果不确定性（噪声）的过程中表现出相似的行为。（B）对共同原因（重叠）增加表现出相似的行为。（C）对噪声和重叠同时增加表现出相似的行为。（D）随着系统变大、步数增加，这些度量仍表现出相似的行为。

## **附录 S2 跨尺度的动态一致性计算**

并非所有宏观尺度都是对其底层微观尺度的合理概括；事实上，一些宏观尺度在定义时可能是 **动态不一致（Dynamically Inconsistent）** 的 [60]。与 Klein 和 Hoel [18] 中的做法类似，在此我认为，如果一个宏观尺度与其底层微观尺度一致，那么它就是有效的。一致性被定义为：在马尔可夫链上，随机游走者的路径在微观尺度和宏观尺度上是否相同（即，宏观尺度是否充当了微观尺度动态的准确汇总统计量）。

具体而言，对于马尔可夫链，不一致性可以定义为 **库尔贝克-莱布勒散度（Kullback-Leibler Divergence）** [61]，该散度是在给定每个尺度上相同的起始状态时，随机游走者在时间步 $t$ $\rightarrow$ $t_{n}$ 内，在 $S$ 与 $S_{M}$ 中的期望分布之间计算的。虽然先前的研究仅检查了此类不一致性的平稳分布 [18]，但在此我强制执行一个严格的一致性概念：在每个可能的状态放置一个随机游走者，并将宏观尺度与微观尺度之间在未来 5 个时间步内所有移动的不一致性进行求和。任何非零值都意味着不一致的宏观尺度，这些尺度将被丢弃。因此，本文考虑的所有宏观尺度都与其底层微观尺度的动态完全一致。

## 附录 S3 通过 SVD 为 CE 2.0 框架提供的启发式方法

Zhang 等人 [22] 提出了一种基于对 **马尔可夫链（Markov chains）** 应用 **奇异值分解（Singular Value Decomposition, SVD）** 的“模糊”因果涌现计算。该方法基于所得奇异值（$\sigma$s）的平均值，他们称之为 $\gamma$，并证明其反映了系统状态间的平均动态可逆性。该方法还被证明能够近似本文所使用的 **确定性（Determinism）** 加 **特异性（Specificity）** （详见 [22]）。正如他们所指出的，这表明了因果基元与动态可逆性之间存在联系。

此外，他们的研究提供了一种在 CE 1.0 框架中近似（相当精确地）因果涌现程度的方法（类似于寻找具有最大有效信息（Effective Information, EI）的宏观尺度）。具体来说，他们首先在给定某个阈值 $\epsilon$ 的情况下，识别出一组非零 $\sigma$ 值，并将这些值平均（$\bar{\sigma}$）。然后取 $\gamma$（未经阈值处理的平均值）与 $\bar{\sigma}$ 之间的差值。当 $\epsilon$ 是一个小的非零值，从而包含了大多数非零 $\sigma$ 值时，这种方法可以优雅地追踪在某个可能的宏观尺度上 EI 可获得的最大增量，而无需在所有尺度集合中进行搜索，从而为 CE 1.0 框架中的因果涌现程度提供了一个精确的启发式方法，且没有任何组合爆炸问题（仅需要微观尺度的转移概率矩阵（Transition Probability Matrix, TPM））。

在此，我将展示如何调整 SVD 方法以适用于 CE 2.0 框架的步骤（在这种情况下，“模糊”的因果涌现成为一个特定值）。为此，奇异值集合可以用作 **粗粒化（Coarse-graining）** 方向的代理。然后，可以通过调整第 3.2 节详述的因果分配方案，来计算每个方向性（由每个 $\sigma$ 表示）的因果贡献。

这里与 [22] 不同，初始的平凡 $\sigma_{1}$ 值被舍弃，因为对于任何马尔可夫链，它总是大于或等于 1，因此不反映系统的任何因果特性（假设 $\sigma$ 值集合按降序排列）。剩余值的平均值，此处称为 $\gamma{*}$，能更近似地反映系统的因果基元。这可以通过先前在 S1 中详述的相同模拟来观察，即沿着其效应不确定性（通过噪声）和原因不确定性（通过共同原因）增加的轴来操纵系统。在与 S1 相同的系统中进行相同操作，图 S3A 绘制了来自 [22] 的原始状态平均动态可逆性 $\gamma$ 与新的 $\gamma{*}$ 的对比，并展示了它们与因果基元相比的行为。$\gamma{*}$ 的行为与确定性加特异性非常相似（例如，当系统处于完全随机状态时变为零，而 $\gamma$ 则不会）。

这些调整使 SVD 方法更符合 CE 2.0 框架。在针对 CE 2.0 调整的 SVD 方法中，因果涌现的总量可以估计为可获得的最高非平凡增益：$\sigma_{2}-\gamma{*}$。当在如正文图 4 所示的“块模型”系统中，按照相同的概率重分配方案计算时，该值与因果涌现（Causal Emergence, CP）的总增益行为相似（绘制于图 S3B）。事实上，在概率重分配之前的系统初始配置中，宏观尺度上 CP 的总增益（以充分性加必要性计）实际上等于 $\sigma_{2}-\gamma{*}$ 值。

为便于比较，图中展示了应用文献 [22] 中“模糊”因果涌现（Causal Emergence, CE）方法得到的数值。由于该方法近似估计有效信息（Effective Information, EI）的增益，它继承了 CE 1.0 框架的一些相同局限性。例如，在宏观尺度内进行此类概率重分布时，该度量在数学上是不稳定的，在宏观状态（Macrostate）内进行任何此类概率重分布后，其值都会降至零（参见图 S3B 中的绘图值，其中使用较低的 $\epsilon$ 以包含大多数非零 $\sigma$ 值，并标记为“CE 1.0 (SVD)”）。

![SI_SVD](images/SI_SVD.png)

> 图 S3 | 将 SVD 方法适配于 CE 2.0 在两个不同的概率重分布模型中得到验证。（A）在与图 S1 相同的系统中，针对增加噪声和共同原因（Common Causes）的相同步骤，再次绘制了因果原语（Causal Primitives, CP）值，同时也绘制了对平均动态可逆性（Averaged Dynamical Reversibility）的拟议更改，其中使用了 $\gamma_{*}$ 而非 $\gamma^{*}$。（B）针对另一种情况的概率重分布绘制度量值，此次情况与正文图 4 相同。其中，一个由两个等价类组成的“块模型（Block Model）”其概率被重分布，直至完全由自环（Self-loops）构成。黑色曲线表示使用文献 [22] 中详述的动态可逆性方法计算此概率重分布过程中的因果涌现时得到的值。正如使用 EI 时一样，该度量在大多数配置中检测不到宏观尺度因果性。然而，当使用本文提出的针对 CE 2.0 的 SVD 方法适配方案时，总增益（$\sigma_{2}-\gamma_{*}$）的表现与因果原语中的总增益相似（可与正文图 4 进行比较）。

此外，应用一个版本的因果分配方案（Causal Apportioning Schema）可以实现基于 SVD 的多尺度因果结构评估。具体而言，每个 $\sigma$（不包括 $\sigma_{1}$）都可以与剩余值的平均值 $\gamma_{*}$ 进行比较。可以对所有为正且因此满足下式的 $\sigma_{i}$（其中 $2\leq i\leq n$）评估其因果贡献：

$$
\sigma_{i}>\frac{1}{n-1}\sum_{j=2}^{n}\sigma_{j}
$$

这种利用 SVD 识别不同尺度独特因果贡献的方法，在正文第 4 节所用模型系统的多尺度结构上进行了测试。对于正文图 3 中第一个系统（图 3A-C），该系统缺乏多尺度结构，此方法从多尺度 SVD 分析中仅得到一个正值：0.61，表明存在一个单一的头重宏观尺度（Top-heavy Macroscale）（与正文中基于路径的分析结果一致）。同时，第二个系统（图 3D-F）确实具有中尺度（Mesoscale）结构，它有三个正值：0.489、0.06、0.06，表明存在一个或多个中尺度。

最终，这些结果表明，将文献 [22] 中为 CE 1.0 框架设计的 SVD 方法适配用于 CE 2.0 分析是一种有前景的启发式方法。它具有避免组合爆炸（Combinatorial Explosion）的优势，因为宏观尺度上因果原语的总增益以及不同尺度自身的因果贡献，都可以仅基于微观尺度的转移概率矩阵（Transition Probability Matrix, TPM）直接估计。然而，需要指出的是，这只是关于如何将 CE 2.0 适配到 SVD 框架的一种提议，未来的研究可能会完全阐明此方法，或提出替代方案。

## 参考文献（References）

- [1]
  杰弗里·韦斯特（Geoffrey West）。
  《尺度：生命、生长与死亡在生物体、城市和公司中的普适定律》（Scale: The universal laws of life, growth, and death in organisms, cities, and companies）。
  企鹅出版社（Penguin），2018年。
- [2]
  丹尼斯·诺布尔（Denis Noble）。
  《生物相对论：无特权的因果层次》（A theory of biological relativity: no privileged level of causation）。
  《界面聚焦》（Interface focus），第2卷，第1期，第55–64页，2012年。
- [3]
  埃里克·霍尔（Erik Hoel）。
  《世界背后的世界：意识、自由意志与科学的界限》（The world behind the world: Consciousness, Free Will, and the Limits of Science）。
  西蒙与舒斯特出版社（Simon and Schuster），2024年。
- [4]
  马泰奥·格拉索（Matteo Grasso），拉里萨·阿尔班塔基斯（Larissa Albantakis），乔纳森·P·朗（Jonathan P Lang），朱利奥·托诺尼（Giulio Tononi）。
  《因果还原论与因果结构》（Causal reductionism and causal structures）。
  《自然神经科学》（Nature neuroscience），第24卷，第10期，第1348–1355页，2021年。
- [5]
  埃里克·P·霍尔（Erik P Hoel）。
  《上为智能体，下为原子：智能体如何从其底层微观物理中因果涌现》（Agent above, atom below: how agents causally emerge from their underlying microphysics）。
  《漫游向目标：无心的数学定律如何产生目标与意图？》（Wandering towards a goal: how can mindless mathematical laws give rise to aims and intention?），第63–76页，2018年。
- [6]
  杰西卡·C·弗拉克（Jessica C Flack）。
  《粗粒化作为一种下向因果机制》（Coarse-graining as a downward causation mechanism）。
  《皇家学会哲学汇刊A辑：数学、物理与工程科学》（Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences），第375卷，第2109期，20160338，2017年。
- [7]
  汉斯·阿尔伯特·布劳恩（Hans Albert Braun）。
  《神经生物学中的随机性与确定性：从离子通道到“自由意志”问题》（Stochasticity versus determinacy in neurobiology: From ion channels to the question of the “free will”）。
  《系统神经科学前沿》（Frontiers in Systems Neuroscience），第15卷，629436，2021年。
- [8]
  彼得·耶德利奇卡（Peter Jedlicka）。
  《重访量子大脑假说：迈向量子（神经）生物学？》（Revisiting the quantum brain hypothesis: toward quantum (neuro) biology?）。
  《分子神经科学前沿》（Frontiers in molecular neuroscience），第10卷，366，2017年。
- [9]
  埃里克·P·霍尔（Erik P Hoel），拉里萨·阿尔班塔基斯（Larissa Albantakis），朱利奥·托诺尼（Giulio Tononi）。
  《量化因果涌现表明宏观可以胜过微观》（Quantifying causal emergence shows that macro can beat micro）。
  《美国国家科学院院刊》（Proceedings of the National Academy of Sciences），第110卷，第49期，第19790–19795页，2013年。
- [10]
  埃里克·P·霍尔（Erik P Hoel）。
  《当地图优于领土时》（When the map is better than the territory）。
  《熵》（Entropy），第19卷，第5期，188，2017年。
- [11]
  朱迪亚·珀尔（Judea Pearl）。
  《因果论》（Causality）。
  剑桥大学出版社（Cambridge university press），2009年。
- [12]
  托马斯·F·瓦利（Thomas F Varley）。
  《离散与连续动力系统中的因果涌现》（Causal emergence in discrete and continuous dynamical systems）。
  arXiv预印本，arXiv:2003.13075，2020年。
- [13]
  杨明哲（Mingzhe Yang），王志鹏（Zhipeng Wang），刘凯威（Kaiwei Liu），戎英琪（Yingqi Rong），袁冰（Bing Yuan），张江（Jiang Zhang）。
  《通过最大化有效信息在数据中发现涌现》（Finding emergence in data by maximizing effective information）。
  《国家科学评论》（National Science Review），第12卷，第1期，nwae279，2025年。
- [14]
  费德里科·皮戈齐（Federico Pigozzi），亚当·戈尔茨坦（Adam Goldstein），迈克尔·莱文（Michael Levin）。
  《基因调控网络模型中的联想性条件作用增加了整合性因果涌现》（Associative conditioning in gene regulatory network models increases integrative causal emergence）。
- [15]
  埃里克·霍尔（Erik Hoel），迈克尔·莱文（Michael Levin）。
  《生物系统中信息性更高尺度的涌现：用于最优预测与控制的计算工具包》（Emergence of informative higher scales in biological systems: a computational toolkit for optimal prediction and control）。
  《交流与整合生物学》（Communicative & Integrative Biology），第13卷，第1期，第108–118页，2020年。
- [16]
  罗斯·格里本诺（Ross Griebenow），布伦南·克莱因（Brennan Klein），埃里克·霍尔（Erik Hoel）。
  《寻找网络的正确尺度：通过谱聚类高效识别因果涌现》（Finding the right scale of a network: efficient identification of causal emergence through spectral clustering）。
  arXiv预印本，arXiv:1908.07565，2019年。
- [17]
  张江（Jiang Zhang），刘凯威（Kaiwei Liu）。
  《用于因果涌现的神经信息压缩器》（Neural information squeezer for causal emergence）。
  《熵》（Entropy），第25卷，第1期，26，2022年。
- [18]
  布伦南·克莱因（Brennan Klein），埃里克·霍尔（Erik Hoel）。
  《复杂网络中信息性更高尺度的涌现》（The emergence of informative higher scales in complex networks）。
  《复杂性》（Complexity），2020年，第1期，8932526，2020年。
- [19]
  布伦南·克莱因（Brennan Klein），埃里克·霍尔（Erik Hoel），安舒曼·斯温（Anshuman Swain），罗斯·格里本诺（Ross Griebenow），迈克尔·莱文（Michael Levin）。
  《进化与涌现：生命树中蛋白质相互作用组的高阶信息结构》（Evolution and emergence: higher order information structure in protein interactomes across the tree of life）。
  《整合生物学》（Integrative Biology），第13卷，第12期，第283–294页，2021年。
- [20]
  埃里克·P·霍尔（Erik P Hoel），拉里萨·阿尔班塔基斯（Larissa Albantakis），威廉·马歇尔（William Marshall），朱利奥·托诺尼（Giulio Tononi）。
  《宏观能胜过微观吗？跨时空尺度的整合信息》（Can the macro beat the micro? integrated information across spatiotemporal scales）。
  《意识神经科学》（Neuroscience of Consciousness），2016年，第1期，niw012，2016年。
- [21]
  威廉·马歇尔（William Marshall），格雷厄姆·芬德利（Graham Findlay），拉里萨·阿尔班塔基斯（Larissa Albantakis），朱利奥·托诺尼（Giulio Tononi）。
  《从微观单元到宏观单元：从系统内在视角识别其因果粒度的数学框架》（From micro to macro units: a mathematical framework for identifying the causal grain of a system from its intrinsic perspective）。
  bioRxiv，第2024–04页，2024年。
- [22]
  张江（Jiang Zhang），陶如意（Ruyi Tao），梁敬豪（Keng Hou Leong），杨明哲（Mingzhe Yang），袁冰（Bing Yuan）。
  《动力学可逆性与基于奇异值分解（SVD）的因果涌现新理论》（Dynamical reversibility and a new theory of causal emergence based on svd）。
  《npj复杂性》（npj Complexity），第2卷，第1期，3，2025年。
- [23]
  袁冰（Bing Yuan），张江（Jiang Zhang），吕傲博（Aobo Lyu），吴佳芸（Jiayun Wu），王志鹏（Zhipeng Wang），杨明哲（Mingzhe Yang），刘凯威（Kaiwei Liu），牟牧云（Muyun Mou），崔鹏（Peng Cui）。
  《复杂系统中的涌现与因果性：因果涌现及相关定量研究综述》（Emergence and causality in complex systems: A survey of causal emergence and related quantitative studies）。
  《熵》（Entropy），第26卷，第2期，108，2024年。
- [24]
  戴维·巴尔杜齐

* [38] 威廉·马歇尔（William Marshall）、拉里萨·阿尔班塔基斯（Larissa Albantakis）和朱利奥·托诺尼（Giulio Tononi）。《黑箱化与因果效应能力（Black-boxing and cause-effect power）》。PLoS 计算生物学（PLoS computational biology），14(4):e1006114，2018年。
* [39] 卡洛斯·格申森（Carlos Gershenson）和纳尔逊·费尔南德斯（Nelson Fernández）。《复杂性与信息：多尺度下的涌现、自组织和稳态测量（Complexity and information: Measuring emergence, self-organization, and homeostasis at multiple scales）》。复杂性（Complexity），18(2):29–44，2012年。
* [40] 埃米利亚诺·马尔凯塞（Emiliano Marchese）、圭多·卡尔达雷利（Guido Caldarelli）和蒂齐亚诺·斯夸尔蒂尼（Tiziano Squartini）。《通过意外性检测中尺度结构（Detecting mesoscale structures by surprise）》。通信物理学（Communications Physics），5(1):132，2022年。
* [41] 阿贝尔·扬斯马（Abel Jansma）。《复杂系统中高阶结构的部分论方法：从宏观到微观与莫比乌斯（Mereological approach to higher-order structure in complex systems: From macro to micro with möbius）》。物理评论研究（Physical Review Research），7(2):023016，2025年。
* [42] 埃亚尔·温特（Eyal Winter）。《沙普利值（The shapley value）》。博弈论与经济应用手册（Handbook of game theory with economic applications），3:2025–2054，2002年。
* [43] 帕维尔·奇维科夫（Pavel Chvykov）和埃里克·霍尔（Erik Hoel）。《因果几何（Causal geometry）》。熵（Entropy），23(1):24，2020年。
* [44] 费尔南多·E·罗萨斯（Fernando E Rosas）、佩德罗·AM·梅迪亚诺（Pedro AM Mediano）、亨里克·J·延森（Henrik J Jensen）、阿尼尔·K·塞斯（Anil K Seth）、亚当·B·巴雷特（Adam B Barrett）、罗宾·L·卡哈特-哈里斯（Robin L Carhart-Harris）和丹尼尔·博尔（Daniel Bor）。《调和涌现：一种识别多元数据中因果涌现的信息论方法（Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data）》。PLoS 计算生物学（PLoS computational biology），16(12):e1008289，2020年。
* [45] 莱昂内尔·巴尼特（Lionel Barnett）和阿尼尔·K·塞斯（Anil K Seth）。《动力学独立性：发现复杂动力系统中的涌现宏观过程（Dynamical independence: discovering emergent macroscopic processes in complex dynamical systems）》。物理评论 E（Physical Review E），108(1):014304，2023年。
* [46] 托马斯·D·邦特利（Thomas D Bontly）。《随附性论证的推广（The supervenience argument generalizes）》。哲学研究（Philosophical Studies），109:75–96，2002年。
* [47] 内德·布洛克（Ned Block）。《因果力会流失吗？（Do causal powers drain away?）》。哲学与现象学研究（Philosophy and Phenomenological Research），67(1):133–150，2003年。
* [48] 金在权（Jaegwon Kim）。《物理世界中的心灵：关于心身问题与心理因果性的论文（Mind in a physical world: An essay on the mind-body problem and mental causation）》。麻省理工学院出版社（MIT press），2000年。
* [49] 蒂姆·贝恩（Tim Bayne）。《论意识整合信息理论的公理化基础（On the axiomatic foundations of the integrated information theory of consciousness）》。意识神经科学（Neuroscience of consciousness），2018(1):niy007，2018年。
* [50] 菲利普·W·安德森（Philip W Anderson）。《多则异：破缺对称性与科学层级结构的本质（More is different: Broken symmetry and the nature of the hierarchical structure of science）》。科学（Science），177(4047):393–396，1972年。
* [51] 乔治·FR·埃利斯（George FR Ellis）。《现实世界语境中物理学的因果闭合性（The causal closure of physics in real world contexts）》。物理学基础（Foundations of Physics），50(10):1057–1097，2020年。
* [52] JM·弗里茨曼（JM Fritzman）。《瓦解强涌现的瓦解问题（Collapsing strong emergence’s collapse problem）》。欧洲科学哲学杂志（European Journal for Philosophy of Science），14(2):24，2024年。
* [53] 肖恩·卡罗尔（Sean Carroll）。《大图景：论生命、意义和宇宙本身的起源（The big picture: on the origins of life, meaning, and the universe itself）》。企鹅出版社（Penguin），2017年。
* [54] 肖恩·M·卡罗尔（Sean M Carroll）和阿丘斯·帕罗拉（Achyuth Parola）。《涌现可能意味着什么（What emergence can possibly mean）》。arXiv 预印本 arXiv:2410.15468，2024年。
* [55] 刘凯伟（Kaiwei Liu）、袁冰（Bing Yuan）和张江（Jiang Zhang）。《线性随机迭代系统因果涌现的精确理论（An exact theory of causal emergence for linear stochastic iteration systems）》。arXiv 预印本 arXiv:2405.09207，2024年。
* [56] 罗伯特·卡多纳（Robert Cardona）、埃娃·米兰达（Eva Miranda）、丹尼尔·佩拉尔塔-萨拉斯（Daniel Peralta-Salas）和弗朗西斯科·普雷萨斯（Francisco Presas）。《在三维空间中构造图灵完备的欧拉流（Constructing turing complete euler flows in dimension 3）》。美国国家科学院院刊（Proceedings of the National Academy of Sciences），118(19):e2026818118，2021年。
* [57] 西西亚·马罗（Scythia Marrow）、埃里克·J·米肖（Eric J Michaud）和埃里克·霍尔（Erik Hoel）。《使用信息论检验深度神经网络的因果结构（Examining the causal structures of deep neural networks using information theory）》。熵（Entropy），22(12):1429，2020年。
* [58] 阿德利·坦普尔顿（Adly Templeton）。《扩展单义性：从 Claude 3 Sonnet 中提取可解释特征（Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet）》。Anthropic，2024年。
* [59] 塞思·拉扎尔（Seth Lazar）和阿隆德拉·纳尔逊（Alondra Nelson）。《以谁的条件确保人工智能安全？（Ai safety on whose terms?）》，2023年。
* [60] 保罗·K·鲁宾斯坦（Paul K Rubenstein）、塞巴斯蒂安·魏希瓦尔德（Sebastian Weichwald）、斯蒂芬·邦格斯（Stephan Bongers）、约里斯·M·穆伊（Joris M Mooij）、多米尼克·扬青（Dominik Janzing）、莫里茨·格罗斯-文特拉普（Moritz Grosse-Wentrup）和伯恩哈德·舍尔科普夫（Bernhard Schölkopf）。《结构方程模型的因果一致性（Causal consistency of structural equation models）》。arXiv 预印本 arXiv:1707.00819，2017年。
* [61] 托马斯·M·科弗（Thomas M Cover）。《信息论基础（Elements of information theory）》。约翰·威利父子出版公司（John Wiley & Sons），1999年。
