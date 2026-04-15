# 标题：Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention（低精度Transformer训练为何失败：对Flash Attention的分析）

- **ArXiv（预印本）** ： 2510.04212
- **作者** ： Haiquan Qiu Quanming Yao, Department of Electronic Engineering, Tsinghua University（清华大学电子工程系）
- **章节数** ： 42
- **估计词元数** ： 21.5k

## 目录

- 1 引言
  - 符号说明
- 2 预备知识
  - 2.1 低精度训练
  - 2.2 Flash Attention
- 3 Flash Attention中不稳定的根本原因
  - 3.1 低精度Flash Attention的失败案例
  - 3.2 在Flash Attention中定位失败来源
    - 分块不是失败来源。
    - 失败源于单个层。
    - 失败与 𝜹 的计算有关。
    - 𝐎 中的数值误差是失败来源。
    - 失败定位于特定的注意力头。
      - 论断 1。
  - 3.3 训练失败的根本原因
    - 3.3.1 原因 1. 相似的低秩矩阵导致权重更新偏差
      - 高精度与低精度梯度之间的误差分析
      - 权重的相似低秩更新导致训练失败
        - 论断 2。
    - 3.3.2 原因 2. 有偏的舍入误差导致正的 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$
      - 定位 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 中的大误差
      - 有偏舍入误差分析
        - 备注 1。
      - 图 6 中的舍入误差分析
        - 论断 3。
- 4 实验：缓解舍入误差中的偏差
- 5 结论
  - 讨论
  - 局限性
- 参考文献
- 附录 A 相关工作
  - A.1 混合精度BF16训练。
  - A.2 稳定低精度训练
    - 梯度缩放。
    - 超低精度（FP8/INT8）训练。
    - 优化器和梯度稳定技术。
    - 激活和架构技术。
- 附录 B BF16加法
- 附录 C 缓解Flash Attention中有偏舍入误差的设计考量
  - 使用动态最大值而非固定偏移量
  - 有条件地应用动态最大值
  - 关于处理负的重复行最大值的说明

## 摘要

###### 摘要

对计算效率的追求推动了采用低精度格式来训练 **Transformer模型（Transformer models）** 。然而，这一进展常常受到众所周知的训练不稳定性的阻碍。本文首次为一种长期存在且未解决的失败案例提供了机制性解释：在低精度设置下使用 **Flash Attention** 进行训练会导致灾难性的损失爆炸。我们的深入分析表明，这种失败并非随机现象，而是由两个相互交织的现象引起的： **注意力机制（attention mechanism）** 内相似 **低秩表示（low-rank representations）** 的出现，以及低精度算术固有的 **有偏舍入误差（biased rounding errors）** 的复合效应。我们展示了这些因素如何创建一个误差累积的恶性循环，从而破坏权重更新，最终扰乱训练动态。为了验证我们的发现，我们对Flash Attention引入了一个最小的修改，以缓解舍入误差中的偏差。这一简单的改变稳定了训练过程，证实了我们的分析，并为这个持续存在的问题提供了一个实用的解决方案。代码可在 [https://github.com/ucker/why-low-precision-training-fails](https://github.com/ucker/why-low-precision-training-fails) 获取。

<a id="section-1"></a>

## 1 引言（Introduction）

训练更大、更强大的 **Transformer模型（Transformer models）** ，其驱动力是对计算效率的不懈追求 [1, 2]。实现这一目标的一个关键策略是采用 **低精度数值格式（low-precision numerical formats）** [3, 4, 5, 6]，这有望大幅减少内存占用并显著提升训练速度。在工业实践中，通常使用 **BF16** 来处理内存受限的操作（如 **Flash Attention（Flash Attention）** ），同时将计算受限的操作（如前馈网络，FFNs）推向更低的精度，如 **FP8** [6, 7]。这突显了 **注意力机制（attention mechanisms）** 对数值精度的高度敏感性。尽管已经开发了诸如 **QK 归一化（QK normalization）** [8, 7]、 **QK 裁剪（QK-clip）** [9] 和 **门控注意力（Gated Attention）** [10, 7] 等稳定化技术，但进一步降低精度的道路常常因缺乏对底层失效机制的理解而受阻。

本文通过剖析一个涉及 Flash Attention 的著名且长期存在的失效问题来应对这一挑战。通过将注意力机制相对于序列长度的内存复杂度从二次降低到线性，Flash Attention 已成为高效 Transformer 训练的基石算法，使其在处理现代大规模模型所需的长上下文时不可或缺 [11, 12, 13]。这种在低精度设置下出现的失效 [14, 15, 16, 17, 18, 19]，构成了一个显著的瓶颈。我们聚焦于社区报告的一个具体、可复现的失效案例 [15, 16]，该问题已悬而未决超过两年。我们的深入分析首次为这一失效提供了机理上的解释，揭示其并非随机产物，而是两个交织现象的直接后果： **在不同训练步和词元（tokens）间出现的相似低秩表示（low-rank representations）** ，以及 **低精度算术固有的有偏舍入误差（biased rounding errors）的复合效应** 。我们展示了这些有偏舍入误差如何充当低秩表示的系数，导致它们作为有偏的梯度更新累积到权重上。这推动权重和激活的 **谱范数（spectral norm）** 异常增大，最终压垮训练动态。为了验证我们的分析，我们引入了一个对 Flash Attention 的最小修改，以减轻舍入误差的偏差，使得低秩权重更新在训练过程中能够相互抵消。该实验证实了我们的分析并稳定了训练过程，为这一长期存在的问题提供了一个实用的解决方案。

##### 符号说明（Notations）

我们使用粗体小写字母表示向量（例如 ${\bm{\delta}},{\mathbf{m}},{\mathbf{r}}_{m}$），使用粗体大写字母表示矩阵（例如 ${\mathbf{Q}},{\mathbf{K}},{\mathbf{V}}$）。$\mathrm{diag}(\mathbf{v})$ 表示一个以向量 ${\mathbf{v}}$ 的元素为对角线元素的对角矩阵。$\circ$ 表示逐元素乘积。我们使用 Python 风格的索引，例如 ${\mathbf{M}}[i,:]$ 表示矩阵 ${\mathbf{M}}$ 的第 $i$ 行。下标 $lp$ 和 $hp$ 用于区分低精度（BF16）和高精度（FP32）计算。二进制字符串使用等宽字体表示（例如 $\mathtt{101010}$）。符号 $\equiv$ 表示逐元素相等比较，类似于 Python 的 `==` 运算符，它返回一个二进制张量。$\mathrm{where}$ 函数模仿 `torch.where`。除非另有说明，操作遵循 PyTorch 的广播规则。关键主张在部分章节末尾以浅青色框突出显示。

<a id="section-2"></a>

## 2 预备知识

<a id="section-2-1"></a>

### 2.1 低精度训练

**低精度训练（Low-precision training）** 是现代深度学习的基石，它通过减少内存使用和加速计算，使得开发越来越大的模型成为可能 [1, 2, 3]。这是通过使用比标准的 32 位单精度（FP32）更少比特数的数值格式来表示权重、激活值和梯度来实现的。 **混合精度训练（Mixed-precision training）** [4] 在实践中已被广泛采用，它将 FP16 或 bfloat16（BF16）等 16 位格式用于大部分计算，同时保留一个 FP32 的权重主副本以维持精度。虽然 FP16 提供了更高的精度，但其有限的动态范围常常导致 **梯度下溢（gradient underflow）** ，需要采用诸如 **损失缩放（loss scaling）** 等技术。相比之下，BF16 最初为谷歌的 TPU（Tensor Processing Units）开发，现已得到广泛支持，它提供了与 FP32 相同的动态范围，使其对下溢更具鲁棒性，并成为训练大型语言模型（Large Language Models, LLMs）的首选 [5, 6]。然而，BF16 降低的精度仍然可能引入导致训练失败的数值误差，这正是本文关注的焦点。

**bfloat16（BF16）** 格式是一种 16 位浮点数表示，包含 1 个符号位、8 个指数位和 7 个尾数位。它与 32 位单精度（FP32）的动态范围相匹配，但精度较低，使其成为深度学习中平衡计算和数值范围的热门选择。两个 BF16 数相加涉及对齐它们的指数、相加尾数、对和进行归一化，并将结果舍入以适应 7 位尾数。这个最终的舍入步骤，通常是“ **就近舍入，偶数优先（round to nearest, ties to even）** ”，是误差的主要来源之一。虽然这种舍入方法旨在对随机数据无偏，但对具有特定分布的数据进行一系列操作可能导致 **有偏的舍入误差（biased rounding error）** 。这种误差在一个方向上的累积是低精度设置下观察到的训练失败的关键因素。有关 BF16 加法的详细信息，请参见[附录 B](#appendix-b)。

<a id="section-2-2"></a>

### 2.2 Flash Attention

**Flash Attention（FA）** [7, 8, 9] 是一种 **I/O感知（I/O-aware）** 的精确注意力算法，旨在克服标准注意力的内存瓶颈。标准注意力定义为 ${\mathbf{O}}=\mathrm{softmax}(\alpha{\mathbf{Q}}{\mathbf{K}}^{\top}){\mathbf{V}}$，需要具体化 $N\times N$ 的注意力分数矩阵 ${\mathbf{S}}=\alpha{\mathbf{Q}}{\mathbf{K}}^{\top}$，导致相对于序列长度 $N$ 的内存复杂度为 $\mathcal{O}(N^{2})$。Flash Attention 通过将输入矩阵 ${\mathbf{Q}},{\mathbf{K}},{\mathbf{V}}\in\mathbb{R}^{N\times d}$ 划分为块并迭代处理，将复杂度降低到 $\mathcal{O}(N)$。这些块从 **高带宽内存（High-Bandwidth Memory, HBM）** 加载到快速的 **片上静态随机存取存储器（Static Random-Access Memory, SRAM）** 中，从而最小化昂贵的内存传输。

在本文中，我们重点分析 Flash Attention 2。FA 的前向传播使用一种 **在线 softmax（online softmax）** 方法（[算法 2](#algorithm-2)）计算输出 ${\mathbf{O}}$ 和对数和指数统计量 ${\mathbf{L}}$。它遍历 ${\mathbf{Q}}$ 的块（外循环）和 ${\mathbf{K}},{\mathbf{V}}$ 的块（内循环）。对于每个查询块 ${\mathbf{Q}}_{i}$，它维护运行统计量：最大分数 ${\mathbf{m}}_{i}$ 和归一化因子 ${\bm{\ell}}_{i}$。在每个内循环步骤中，它计算未归一化的注意力分数 $\bar{{\mathbf{P}}}_{i}^{(j)}=\exp({\mathbf{S}}_{i}^{(j)}-{\mathbf{m}}_{i}^{(j)})$，并通过累加乘积 $\bar{{\mathbf{P}}}_{i}^{(j)}{\mathbf{V}}_{j}$ 来更新一个未归一化的输出。 **这种累加是我们分析的一个关键点** 。在遍历所有键/值块之后，最终的输出块 ${\mathbf{O}}_{i}$ 被正确归一化。这种 **分块策略（tiling strategy）** 避免了具体化完整的 $N\times N$ 分数矩阵。

后向传播（[算法 3](#algorithm-3)）利用了相同的分块策略。 **它首先计算一个关键的中间项 ${\bm{\delta}}=\mathrm{rowsum}(d{\mathbf{O}}\circ{\mathbf{O}})$，这是我们研究的核心。** 然后，它即时重新计算注意力分数以计算分数的梯度 $d{\mathbf{S}}_{i}^{(j)}={\mathbf{P}}_{i}^{(j)}\circ(d{\mathbf{P}}_{i}^{(j)}-{\bm{\delta}}_{i})$，其中 $d{\mathbf{P}}_{i}^{(j)}=d{\mathbf{O}}_{i}{\mathbf{V}}_{j}^{\top}$。最终的梯度 $d{\mathbf{Q}},d{\mathbf{K}},d{\mathbf{V}}$ 以块为单位进行累加。这种方法在两个传播过程中都保持了 I/O 效率。

<a id="figure-1"></a>

![failure_process](images/failure_process.png)

> 图 1：不同章节的分析。我们的论文逆向追溯训练失败（蓝色框）的因果链，以识别根本原因。

<a id="section-3"></a>

## 3 Flash Attention 不稳定的根本原因（Root Causes of Instability in Flash Attention）

我们首先在 [3.1 节](#section-3-1) 介绍故障案例。
在 [3.2 节](#section-3-2) 中，我们缩小了 Flash Attention 内部数值误差的来源范围。 [3.3 节](#section-3-3) 的进一步分析表明，故障源于两个因素的结合： **低秩表示（low-rank representations）** 的出现以及 **BF16 算术（BF16 arithmetic）** 固有的 **有偏舍入误差（biased rounding errors）** 的累积。这种故障从根本原因到损失爆炸的完整过程如 [图 1](#figure-1) 所示。

<a id="section-3-1"></a>

### 3.1 低精度 Flash Attention 的故障案例（The Failure Case of Low-Precision Flash Attention）

<a id="figure-2"></a>

![stable_unstable_loss](images/stable_unstable_loss.png)

> 图 2 | 使用 BF16 和 Flash Attention 的故障案例导致损失突然爆炸，而稳定配置则能收敛。

我们的研究针对一个记录详尽且持续存在的故障： **在使用 BF16 精度训练带有 Flash Attention 的生成式预训练 Transformer 2（Generative Pre-trained Transformer 2, GPT-2）模型时发生的灾难性损失爆炸** （nanoGPT Issue 303, 2023; nanoGPT Issue 524, 2024; nanoGPT Issue 554, 2024）。这个已报告两年多的故障案例，表现为训练数千步后损失突然爆炸（参见 [图 8](https://arxiv.org/html/2510.04212v2#A3.F8)）。虽然诸如回退到标准注意力机制或使用更高精度（FP32）等经验性变通方法可以稳定训练，但它们都以牺牲效率为代价。这种不稳定性并非孤立事件；更广泛的社区在训练大型语言模型时也观察到了类似的故障（Kimi-Team, 2025; Qwen-Team, 2025）。这些故障在经验上常与权重的大谱范数、大激活值（Yang et al., 2023; Rybakov et al., 2024）以及注意力汇聚（attention sinks）（Xiao et al., 2023）等现象相关联，从而催生了一系列修复方案，包括 **QK 归一化（QK normalization）** （Henry et al., 2020; Qwen-Team, 2025）、 **QK 裁剪（QK-clipping）** （Kimi-Team, 2025）和 **门控注意力（Gated Attention）** （Qiu et al., 2025; Qwen-Team, 2025）。尽管有这些干预措施，对根本原因的基本理解仍然难以捉摸。由于缺乏从数值误差到损失爆炸的清晰因果链条，社区只能依赖临时补丁而非原则性解决方案，这阻碍了鲁棒低精度训练的进展。本文通过复现故障、剖析其根本原因并提出一个实用且基于原则的解决方案，首次提供了机制性的解释。

为了复现故障，我们采用了一个具有 12 层、12 个注意力头、嵌入维度为 768、上下文长度为 1024 的 GPT-2 架构。该模型在 **OpenWebText 数据集（OpenWebText dataset）** （Gokaslan et al., 2019）上进行预训练。
为了确保确定性的可复现性，我们偏离了标准的随机数据加载器，而是记录并重用了最初导致故障的运行中精确的数据批次序列。
这确保了所有后续实验以相同顺序处理相同的数据，从而将故障与数据相关的随机性隔离开来。

我们使用 **AdamW 优化器（AdamW optimizer）** 训练模型，其中 $\beta_{1}=0.9$，$\beta_{2}=0.95$，权重衰减为零。
学习率遵循余弦调度，经过 2000 次迭代的线性预热达到峰值 $1\times 10^{-3}$，然后衰减至 $1\times 10^{-5}$。
我们应用最大范数为 1.0 的全局梯度裁剪。
训练在 4 块 **NVIDIA A100（80GB）** GPU 上使用 **PyTorch 的分布式数据并行（Distributed Data Parallel, DDP）** 模块进行。
我们使用 **自动混合精度（automatic mixed precision）** ，前向传播使用 BF16，反向传播使用 FP32。
每个 GPU 处理的微批次大小为 32，经过 4 步的梯度累积后，
每个优化步骤的有效全局批次大小为 524,288 个词元。

<a id="section-3-2"></a>

### 3.2 在 Flash Attention 中定位故障源（Isolating the Source of Failure within Flash Attention）

为了精确定位 Flash Attention（闪存注意力）中的故障源，我们进行了一系列有针对性的实验。我们系统地修改算法——禁用分块处理（Tiling）、选择性地用标准实现替换 Flash Attention，以及在高精度下执行关键计算——以缩小不稳定的潜在原因。为了加速此分析，我们监测了权重谱范数（Spectral Norm）等经验指标，以快速识别导致失败的配置。

##### 分块处理并非故障源（Tiling is not the Source of Failure）。

为了确定 Flash Attention 中的分块处理（Block-wise Processing）是否导致了故障，我们进行了一项实验，通过将块大小（Block Size）设置为等于序列长度来禁用分块处理。这迫使算法一次性计算完整矩阵。训练过程仍然失败，导致了相同的损失爆炸（Loss Explosion）。这一发现排除了分块策略是问题原因的可能性。因此，在后续所有实验中，我们都使用这种非分块设置来简化分析，并专注于核心数值计算。

##### 故障源于单个层（Failure Originates in a Single Layer）。

我们首先分析了所有层的权重谱范数（Yang et al., 2023; Rybakov et al., 2024）。这揭示了第二层注意力机制中一个特定的异常峰值（参见 [图 9](https://arxiv.org/html/2510.04212v2#A3.F9)）。我们通过两个针对性实验证实了这一发现：

1.  仅在第二层使用 Flash Attention 就足以复现训练失败；
2.  在第二层用标准注意力（Standard Attention）替换 Flash Attention，同时在其他所有层保留 Flash Attention，恢复了训练稳定性。

这些结果最终确定第二层的 Flash Attention 是故障的根源。因此，后续分析将聚焦于该模块，以剖析故障机制。

##### 故障与 𝜹 的计算相关（Failure is Linked to the Computation of 𝜹）。

Flash Attention 的反向传播（Backward Pass）为了计算效率，使用了项 ${\bm{\delta}}=\mathrm{rowsum}(d{\mathbf{O}}\circ{\mathbf{O}})\in\mathbb{R}^{N}$。另一种数学上等价的公式将该项计算为 ${\bm{\delta}}=\mathrm{rowsum}(d{\mathbf{P}}\circ{\mathbf{P}})$，其中 $d{\mathbf{P}}=d{\mathbf{O}}{\mathbf{V}}^{\top}$。我们发现，用这种替代公式替换高效计算后，训练稳定性得以恢复。该实验表明，在 BF16（Brain Floating Point 16）精度下计算 ${\mathbf{O}}$ 时引入的数值误差很可能是故障的主要来源，因为避免训练失败的替代公式等价于使用在 FP32（单精度浮点数）下计算的 ${\mathbf{O}}$。

<a id="figure-3"></a>

![layer_1_wq_head_spectral_norms](images/layer_1_wq_head_spectral_norms.png)

> 图 3 | 第一层 Wq 头权重谱范数。

**图 3（Figure 3）** ：注意力头 8 的 ${\mathbf{W}}^{Q}$ 具有最大的谱范数。后续分析将聚焦于此头。

##### **${\mathbf{O}}$ 中的数值误差是失败的根源**

基于 ${\bm{\delta}}$ 的计算是关键这一发现，我们进一步将误差源定位到低精度计算中的输出矩阵 ${\mathbf{O}}_{lp}$，即 ${\bm{\delta}}_{lp}=\mathrm{rowsum}(d{\mathbf{O}}\circ{\mathbf{O}}_{lp})$。我们进行了两个关键实验。首先，我们不再使用前向传播中计算出的低精度 ${\mathbf{O}}$ 来计算 ${\bm{\delta}}$，而是在反向传播中将其重新计算为 FP32 精度的 ${\mathbf{P}}{\mathbf{V}}$；这一改变稳定了训练。其次，我们发现，在前向传播期间以高精度（FP32）计算 ${\mathbf{O}}$，即 ${\bm{\delta}}_{hp}=\mathrm{rowsum}(d{\mathbf{O}}\circ{\mathbf{O}}_{hp})$，同时保持所有其他操作为 BF16 精度，同样恢复了稳定性。这些证据确凿地证明，在 BF16 精度下计算 ${\mathbf{O}}$ 时引入的数值误差是导致失败的直接原因（参见 [1](https://arxiv.org/html/2510.04212v2#Thmclaim1)）。

##### **失败局限于特定的注意力头**

为了进一步缩小失败根源的范围，我们通过跟踪各注意力头查询投影矩阵（${\mathbf{W}}^{Q}$）的 **谱范数（Spectral norm）** （Yang 等人，2023；Rybakov 等人，2024）来分析单个注意力头，如 [图 3](#figure-3) 所示。结果显示，少数几个头表现出不成比例的大谱范数。我们通过选择性地为这些异常头（1、7、8、9、11 和 12）以高精度计算输出 ${\mathbf{O}}$ 来确认它们的作用，这足以恢复训练稳定性。_由于头 8 显示出最大的谱范数，我们将后续分析聚焦于此头，以剖析精确的失败机制。_

###### 声明 1（Claim 1）。

###### 声明 1（Claim 1）。

<a id="section-3-3"></a>

### 3.3 训练失败的根本原因

我们在本节中的调查揭示了导致训练失败的两个相互关联的根本原因。在 [第 3.3.1 节](https://arxiv.org/html/2510.04212v2#S3.SS3.SSS1) 中，我们展示了低精度的 ${\bm{\delta}}_{lp}$ 如何通过有偏的权重更新导致训练失败，并发现这种偏差源于相似的 **低秩表示（low-rank representations）** ${\mathbf{R}}$ 的出现，其系数 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 偏向正值，导致误差累积而非抵消。在 [第 3.3.2 节](https://arxiv.org/html/2510.04212v2#S3.SS3.SSS2) 中，我们将这些正系数 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 的起源追溯到 $\bar{{\mathbf{P}}}{\mathbf{V}}$ 乘积中 BF16 加法固有的有偏舍入误差。

#### 3.3.1 原因 1：相似的低秩矩阵导致权重更新有偏

本节将训练失败追溯到有偏的权重更新。我们首先分析分别使用 ${\bm{\delta}}_{hp}$ 和 ${\bm{\delta}}_{lp}$ 计算的高精度和低精度梯度之间的差异。然后我们发现，低秩表示和有偏的 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 导致了损失爆炸。

##### **高精度与低精度梯度间的误差分析**

为了理解数值误差如何传播到梯度中，我们分析了查询矩阵 $d{\mathbf{Q}}$ 的高精度（$hp$）与低精度（$lp$）梯度之间的差异。梯度 $d{\mathbf{Q}}$ 由注意力分数梯度 $d{\mathbf{S}}$ 计算得出：$d{\mathbf{Q}}=d{\mathbf{S}}{\mathbf{K}}$。分数梯度由 $d{\mathbf{S}}=\alpha{\mathbf{P}}\circ(d{\mathbf{P}}-{\bm{\delta}})$ 给出，其中 ${\bm{\delta}}=\mathrm{rowsum}(d{\mathbf{O}}\circ{\mathbf{O}})$ 是基于 [第 3.2 节](#section-3-2) 高精度和低精度反向传播之间唯一不同的项，$\alpha$ 是注意力机制中的一个缩放因子。

高精度与低精度查询梯度之间的差异可推导如下：

$$
\begin{split}&d{\mathbf{Q}}_{hp}-d{\mathbf{Q}}_{lp}=(d{\mathbf{S}}_{hp}-d{\mathbf{S}}_{lp}){\mathbf{K}}\\ =&\left(\alpha{\mathbf{P}}\circ(d{\mathbf{P}}-{\bm{\delta}}_{hp})-\alpha{\mathbf{P}}\circ(d{\mathbf{P}}-{\bm{\delta}}_{lp})\right){\mathbf{K}}=\left(\alpha{\mathbf{P}}\circ({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})\right){\mathbf{K}}\\ =&\alpha\cdot\mathrm{diag}({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})({\mathbf{P}}{\mathbf{K}}).\end{split}(1)
$$

在最后一步中，我们将按行缩放操作 ${\mathbf{P}}\circ({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})$（遵循广播规则）表示为矩阵乘法，其中 $\mathrm{diag}({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})$ 是一个对角矩阵，其对角线元素是向量差 ${\bm{\delta}}_{lp}-{\bm{\delta}}_{hp}$ 的元素。这个公式表明，梯度误差与 ${\bm{\delta}}$ 中的误差成正比，并由 ${\mathbf{P}}{\mathbf{K}}$ 项调制。

<a id="figure-4"></a>

![PK_at_H8_batch_idx_190_6610](images/PK_at_H8_batch_idx_190_6610.png)

> **图 4（Figure 4）** ：不同批次索引和训练步数下的 ${\mathbf{P}}{\mathbf{K}}$、${\mathbf{X}}$ 和 $({\mathbf{P}}{\mathbf{K}})[T]^{\top}{\mathbf{X}}[T]$。（c）和（f）显示，对于不同的词元和训练步数，$({\mathbf{P}}{\mathbf{K}})[T]^{\top}{\mathbf{X}}[T]$ 在输入特征 546 和 678 处有一些相似的列。

查询投影矩阵 ${\mathbf{W}}^{Q}$ 的梯度 $d{\mathbf{W}}^{Q}$ 由输入特征 ${\mathbf{X}}$ 和查询梯度 $d{\mathbf{Q}}$ 的外积给出。${\mathbf{W}}^{Q}$ 的高精度（$hp$）与低精度（$lp$）梯度之间的差异可表示为：

$$
\displaystyle d\mathbf{W}^{Q}_{hp} - d\mathbf{W}^{Q}_{lp} = (d\mathbf{Q}_{hp} - d\mathbf{Q}_{lp})^{\top} \mathbf{X} = \alpha (\mathbf{P}\mathbf{K})^{\top} \mathrm{diag}(\bm{\delta}_{lp} - \bm{\delta}_{hp}) \mathbf{X},
$$

$$
\displaystyle = \alpha \sum_{T=1}^{N} (\bm{\delta}_{lp} - \bm{\delta}_{hp})[T] \cdot (\mathbf{P}\mathbf{K})[T]^{\top} \mathbf{X}[T],
$$

(2)

其中 $(\mathbf{P}\mathbf{K})[T]$ 和 $\mathbf{X}[T]$ 分别是其矩阵的第 $T$ 个行向量。该方程表明，总梯度误差是多个秩-1（rank-1）矩阵的加权和，权重由 $\bm{\delta}$ 中的误差给出。

##### 权重的相似低秩更新导致训练失败

在 [图 4](#figure-4) 中，$\mathbf{P}\mathbf{K}$ 的行（图 a, d）和 $\mathbf{X}$ 的行（图 b, e）在不同的训练步骤和词元（token）位置上表现出强烈的结构相似性。这意味着由此产生的秩-1矩阵 $(\mathbf{P}\mathbf{K})[T]^{\top} \mathbf{X}[T]$ 彼此之间也高度相似。例如，[图 4](#figure-4)（图 c, f）分别展示了在训练步骤 6610 和 6619 时，词元 50 和 718 的这种相似性。由于这些秩-1误差分量在结构上是一致的，我们可以将总梯度差近似为

$$
d\mathbf{W}^{Q}_{hp} - d\mathbf{W}^{Q}_{lp} \approx \alpha \sum_{T=1}^{N} (\bm{\delta}_{lp} - \bm{\delta}_{hp})[T] \mathbf{R}
$$

(3)

其中 $\mathbf{R}$ 表示在不同词元和训练步骤中出现的共同低秩结构。

方程 ([3](https://arxiv.org/html/2510.04212v2#S3.E3)) 表明，低秩误差方向 $\mathbf{R}$ 的累积由标量项 $\sum_{T=1}^{N} (\bm{\delta}_{lp} - \bm{\delta}_{hp})[T]$ 控制。如果该和偏向于非零值，则不同训练步骤中的误差将累积而非相互抵消。我们追踪了在导致失败的一系列训练步骤（6580 到 6680）中 $\sum_{T=1}^{N} (\bm{\delta}_{lp} - \bm{\delta}_{hp})[T]$ 的累积和，如 [图 5](#figure-5)(a) 所示。该图显示该和始终为正，表明存在系统性偏差。这种偏差导致低秩方向 $\mathbf{R}$ 上的误差随着每个训练步骤而复合。由于 $\mathbf{R}$ 在不同步骤间也相似，这最终会破坏权重更新，增加谱范数（spectral norm）（[图 9](https://arxiv.org/html/2510.04212v2#A3.F9)）和激活值（Yang et al., 2023; Rybakov et al., 2024），并导致训练失败（参见 [2](https://arxiv.org/html/2510.04212v2#Thmclaim2)）。下一节通过分析训练步骤 6619（[图 5](#figure-5)(a) 中识别出的一个具有显著正向贡献的点）的权重和梯度，来找出这种正向偏差的根本原因。

###### 论断 2（Claim 2）.

![X_at_batch_idx_190_train_steps_6610](images/X_at_batch_idx_190_train_steps_6610.png)

> (a) 正向偏差的 $(\bm{\delta}_{lp}\!-\!\bm{\delta}_{hp})[T]$

###### 断言 2（Claim 2）.

#### 3.3.2 原因 2. 有偏的舍入误差导致正的 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$

本节研究 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 中正向偏差的起源。我们将此误差追溯到 $d{\mathbf{O}}$ 与 ${\mathbf{O}}_{lp}-{\bm{\delta}}_{hp}$ 中数值差异之间的相互作用，而该数值差异本身源于 $\bar{{\mathbf{P}}}{\mathbf{V}}$ 计算过程中 BF16（Bfloat16）加法产生的有偏舍入误差。

![PKX_at_H8_batch_idx_190_token50_train_steps_6610](images/PKX_at_H8_batch_idx_190_token50_train_steps_6610.png)

> (a) 大多数 ${\mathbf{V}}[:,i]$ 为负值

##### 定位 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 中的大误差

我们首先研究 $\sum_{T=1}^{N}({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 中正向偏差的来源。如[第 3.2 节](#section-3-2)所分析，${\bm{\delta}}$ 中的误差源于上游梯度 $d{\mathbf{O}}$ 与低精度输出中的数值误差 ${\mathbf{O}}_{lp}-{\mathbf{O}}_{hp}$ 的乘积。为了剖析这一点，我们聚焦于一个词元位置 $T=718$，该处的误差分量 $({\bm{\delta}}_{lp}-{\bm{\delta}}_{hp})[T]$ 为正。

在[图 5](#figure-5)(b) 和 (c) 中，我们观察到对于特定的特征维度（例如 20 和 29，在其他词元中也观察到类似情况），梯度 $d{\mathbf{O}}[T,:]$ 与输出误差 ${\mathbf{O}}_{lp}[T,:]-{\mathbf{O}}_{hp}[T,:]$ 之间存在强烈的符号相关性。在这些维度中，$d{\mathbf{O}}$ 和输出误差 ${\mathbf{O}}_{lp}-{\mathbf{O}}_{hp}$ 都一致为负。这种一致性确保了它们的乘积（即贡献给 ${\bm{\delta}}$ 误差的部分）为正。输出误差倾向于为负（${\mathbf{O}}_{lp}[T,i] < {\mathbf{O}}_{hp}[T,i]$）这一事实表明，${\mathbf{O}}$ 的低精度计算系统性地偏向于更负的值。因此，我们随后的分析重点在于找出这种计算偏差的起源。

输出 ${\mathbf{O}}$ 是从一个中间的非归一化输出 $\bar{{\mathbf{O}}}$ 计算得出的。该计算涉及一个安全的 softmax 函数，随后是矩阵乘法和归一化：

$$
\begin{aligned}
\bar{{\mathbf{P}}} &= \exp({\mathbf{S}}-\mathrm{rowmax}({\mathbf{S}})), \\
\bar{{\mathbf{O}}} &= \bar{{\mathbf{P}}}{\mathbf{V}}, \\
{\mathbf{O}} &= \bar{{\mathbf{O}}}/\mathrm{rowsum}(\bar{{\mathbf{P}}}).
\end{aligned}
$$

进一步的实验将故障根源定位到非归一化输出 $\bar{{\mathbf{O}}}=\bar{{\mathbf{P}}}{\mathbf{V}}$ 的计算上。我们发现，仅将此乘积计算切换为 FP32（单精度浮点数）就足以稳定训练。为了理解这种偏差的起源，我们检查了单个元素 $\bar{{\mathbf{O}}}[T,i]$（在我们的分析中，特征索引 $i=20$）的低精度与高精度计算之间的差异：

$$
\bar{{\mathbf{O}}}_{lp}[T,i]-\bar{{\mathbf{O}}}_{hp}[T,i]=(\bar{{\mathbf{P}}}_{lp}[T,:]{\mathbf{V}}[:,i])_{lp}-(\bar{{\mathbf{P}}}_{hp}[T,:]{\mathbf{V}}[:,i])_{hp} \tag{4}
$$

其中，输入 $\bar{{\mathbf{P}}}$ 和 ${\mathbf{V}}$ 本身是先前 BF16（Brain Floating Point 16）运算的结果。具体而言，此处的下标 $(\cdot)_{lp}$ 表示在 FP32（单精度浮点数）中计算点积，并将最终结果舍入为 BF16，而 $(\cdot)_{hp}$ 则表示完全在 FP32 中计算。

为了理解误差 $\bar{{\mathbf{O}}}_{lp}[T,i]-\bar{{\mathbf{O}}}_{hp}[T,i]$ 如何系统地变为负值，我们在 [图 6](#figure-6)(b) 和 (c) 中绘制了随着词元位置求和进行而累积的误差：

$$
\bar{{\mathbf{O}}}_{\text{error}}(t)=\left(\sum\nolimits_{t^{\prime}=1}^{t}\bar{{\mathbf{P}}}[T,t^{\prime}]{\mathbf{V}}[t^{\prime},i]\right)_{lp}-\left(\sum\nolimits_{t^{\prime}=1}^{t}\bar{{\mathbf{P}}}[T,t^{\prime}]{\mathbf{V}}[t^{\prime},i]\right)_{hp} \tag{5}
$$

该图显示，误差以显著的负向步长累积。这些步长出现在词元位置 $t$ 处，其中对应的注意力概率 $\bar{{\mathbf{P}}}[T,t]$ 恰好为 1（在其他词元位置也观察到此现象）。当 softmax 前的分数 ${\mathbf{S}}[T,t]$ 是其所在行的最大值时，就会发生这种情况，导致 $\exp({\mathbf{S}}[T,t]-\max({\mathbf{S}}[T,:]))$ 的计算结果为 $\exp(0)=1$。

##### 有偏舍入误差分析（Analysis of Biased Rounding Error）

此外，这种偏差源于这些单位值与值矩阵 ${\mathbf{V}}$ 的分布之间的相互作用。如 [图 6](#figure-6)(a) 所示，对于有问题的特征维度 $i=20$，${\mathbf{V}}[:,i]$ 的值主要为负。当 $\bar{{\mathbf{P}}}[T,t]=1$ 时，乘积 $\bar{{\mathbf{P}}}[T,t]{\mathbf{V}}[t,i]$ 简化为 ${\mathbf{V}}[t,i]$，这是一个负的 BF16 数。
_当两个这样的负 BF16 数相加时，就会发生系统误差。_ 在浮点运算中，将两个同号数相加可能导致结果的有效数（Significand）溢出（例如，$\mathtt{-1.xxxx}+\mathtt{-1.yyyy}=\mathtt{-10.zzzz}$），需要进行右移和指数递增以重新归一化。从 7 位 BF16 小数部分移出的比特决定了舍入方向。当两个负数相加时，舍入操作（例如，四舍五入）可能引入一致的偏差。

为了说明这种舍入偏差是如何发生的，考虑两个导致溢出的有效数相加，需要右移进行归一化。被移出的比特（舍入位）决定了舍入方向。我们展示了最后两个 2 比特数所有可能的加法，其中绿色比特表示将被移出的比特（舍入位）：

由于求和是在 FP32 中进行的，来自小数的低阶比特的累积可以激活粘滞位（Sticky bit）。因此，当后续的 BF16 数被加入时，这会强制进行向上舍入。所以，舍入位 ${\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}\mathtt{1}}$ 表明需要向上舍入。因为操作数 $\bar{{\mathbf{P}}}[T,t]{\mathbf{V}}[t,i]$ 是负数且具有较大的指数（因为 $\bar{{\mathbf{P}}}[T,t]=1$ 并不会使 $\bar{{\mathbf{P}}}[T,t]{\mathbf{V}}[t,i]$ 变小），向上舍入的误差被放大，导致负误差。当舍入位是 ${\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}\mathtt{0}}$ 时，结果向下舍入，引入正误差。
此外，与向上舍入产生的负误差相比，正误差更小，因为被相加的值通常非常小。这种不对称性导致舍入误差主要由向上舍入主导，从而在我们的分析中观察到负的舍入误差。这种在 $\bar{{\mathbf{O}}}$ 计算中系统性的负偏差是训练失败的最终根源。

###### 备注 1（Remark 1）.

$$
\bar{{\mathbf{O}}}_{lp}[T,i]-\bar{{\mathbf{O}}}_{hp}[T,i]=(\bar{{\mathbf{P}}}_{lp}[T,:]{\mathbf{V}}[:,i])_{lp}-(\bar{{\mathbf{P}}}_{hp}[T,:]{\mathbf{V}}[:,i])_{hp} \tag{4}
$$

其中，输入 $\bar{{\mathbf{P}}}$ 和 ${\mathbf{V}}$ 本身是先前 BF16 运算的结果。具体而言，此处的下标 $(\cdot)_{lp}$ 表示在 FP32 中计算点积，并将最终结果舍入为 BF16，而 $(\cdot)_{hp}$ 则表示完全在 FP32 中计算。

为了理解误差 $\bar{{\mathbf{O}}}_{lp}[T,i]-\bar{{\mathbf{O}}}_{hp}[T,i]$ 如何系统地变为负值，我们在 [图 6](#figure-6)(b) 和 (c) 中绘制了随着词元位置求和进行而累积的误差：

$$
\bar{{\mathbf{O}}}_{\text{error}}(t)=\left(\sum\nolimits_{t^{\prime}=1}^{t}\bar{{\mathbf{P}}}[T,t^{\prime}]{\mathbf{V}}[t^{\prime},i]\right)_{lp}-\left(\sum\nolimits_{t^{\prime}=1}^{t}\bar{{\mathbf{P}}}[T,t^{\prime}]{\mathbf{V}}[t^{\prime},i]\right)_{hp} \tag{5}
$$

该图显示，误差以显著的负向步长累积。这些步长出现在词元位置 $t$ 处，其中对应的注意力概率 $\bar{{\mathbf{P}}}[T,t]$ 恰好为 1（在其他词元位置也观察到此现象）。当 softmax 前的分数 ${\mathbf{S}}[T,t]$ 是其所在行的最大值时，就会发生这种情况，导致 $\exp({\mathbf{S}}[T,t]-\max({\mathbf{S}}[T,:]))$ 的计算结果为 $\exp(0)=1$。

##### 有偏舍入误差分析（Analysis of Biased Rounding Error）

此外，这种偏差源于这些单位值与值矩阵 ${\mathbf{V}}$ 的分布之间的相互作用。如 [图 6](#figure-6)(a) 所示，对于有问题的特征维度 $i=20$，${\mathbf{V}}[:,i]$ 的值主要为负。当 $\bar{{\mathbf{P}}}[T,t]=1$ 时，乘积 $\bar{{\mathbf{P}}}[T,t]{\mathbf{V}}[t,i]$ 简化为 ${\mathbf{V}}[t,i]$，这是一个负的 BF16 数。
_当两个这样的负 BF16 数相加时，就会发生系统误差。_ 在浮点运算中，将两个同号数相加可能导致结果的有效数溢出（例如，$\mathtt{-1.xxxx}+\mathtt{-1.yyyy}=\mathtt{-10.zzzz}$），需要进行右移和指数递增以重新归一化。从 7 位 BF16 小数部分移出的比特决定了舍入方向。当两个负数相加时，舍入操作（例如，四舍五入）可能引入一致的偏差。

为了说明这种舍入偏差是如何发生的，考虑两个导致溢出的有效数相加，需要右移进行归一化。被移出的比特（舍入位）决定了舍入方向。我们展示了最后两个 2 比特数所有可能的加法，其中绿色比特表示将被移出的比特（舍入位）：

由于求和是在 FP32 中进行的，来自小数的低阶比特的累积可以激活粘滞位。因此，当后续的 BF16 数被加入时，这会强制进行向上舍入。所以，舍入位 ${\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}\mathtt{1}}$ 表明需要向上舍入。因为操作数 $\bar{{\mathbf{P}}}[T,t]{\mathbf{V}}[t,i]$ 是负数且具有较大的指数（因为 $\bar{{\mathbf{P}}}[T,t]=1$ 并不会使 $\bar{{\mathbf{P}}}[T,t]{\mathbf{V}}[t,i]$ 变小），向上舍入的误差被放大，导致负误差。当舍入位是 ${\color[rgb]{0,1,0}\definecolor[named]{

当 $\bar{{\mathbf{P}}}[T,t]<1$ 时，其与 ${\mathbf{V}}[t,i]$ 的乘积的最低 16 位非零。当舍入到 BF16（Brain Floating Point 16）时，这不会引入有偏的舍入误差。

###### 备注 1（Remark 1）

##### 图 6 中的舍入误差分析（Analysis of Rounding Error in Fig. 6）

为了使分析具体化，我们现在分析导致 [图 6](#figure-6)(c) 中所示巨大负误差跳跃的特定 BF16 数值加法。由于求和是在 FP32（单精度浮点数）中执行的，在第二个 BF16 值被加入之前，第一个 BF16 值会先与其他词元（tokens）产生的一些小值相加。这可能会激活粘滞位（sticky bit），从而在加入第二个 BF16 值时强制进行向上舍入（round-up）。

对于这个例子，我们从 FP32 表示开始。
第一个操作数是先前项的累加和，它包括一个 BF16 值（$\mathtt{1}{{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\mathtt{10000000}}}{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathtt{0011010}}}{\color[rgb]{.5,.5,.5}\definecolor[named]{pgfstrokecolor}{rgb}{.5,.5,.5}\pgfsys@color@gray@stroke{.5}\pgfsys@color@gray@fill{.5}\mathtt{0000000000000000}};-2.40625$）加上激活了粘滞位的小残余值（$\sim 0.00087$）。第二个操作数是另一个 BF16 值。它们的 FP32 表示如下：

由于它们的指数相同，加法在其有效数（significands）上进行：

结果溢出了有效数的格式，需要进行归一化（normalization）。有效数右移一位，指数加一：

FP32 中的精确结果是 $\mathtt{1}\,{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\mathtt{10000001}}\,{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathtt{0010110}}{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}\mathtt{1}}{\color[rgb]{.5,.5,.5}\definecolor[named]{pgfstrokecolor}{rgb}{.5,.5,.5}\pgfsys@color@gray@stroke{.5}\pgfsys@color@gray@fill{.5}\mathtt{000011100010111}}$，对应数值 $-4.703990459442139$。要将此结果存储为 BF16，必须将其舍入到 7 位小数位（fraction bits）。有效数为 $\mathtt{1.{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}0010110}{\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}1}}$。7 位小数部分是 ${\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathtt{0010110}}$。由于舍入位（rounding bit）是 ${\color[rgb]{0,1,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,1,0}\mathtt{1}}$ 且舍入位之后存在非零位，根据“最近舍入（遇偶取整）”（round-to-nearest, ties-to-even）规则，进行向上舍入（在小数部分的最后一位加 1）：

最终的 BF16 结果是 $\mathtt{1}{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\mathtt{10000001}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathtt{0010111}}$，表示 $-4.71875$。这个舍入后的值比真实和 $-4.703990459442139$ 更负。由这一次加法引入的误差是 $\mathbf{-0.014759540557861328}$。当此类舍入事件在 $\bar{{\mathbf{P}}}{\mathbf{V}}$ 乘积的多次加法中系统性发生时，误差会累积，在 ${\mathbf{O}}$ 中产生负偏差，最终导致训练不稳定。

###### 声明 3（Claim 3）

###### 声明 3（Claim 3）

<a id="section-4"></a>

## 4 实验：缓解舍入误差中的偏差（Experiment: Mitigate Bias in Rounding Error）

我们的分析将训练失败归因于 $\bar{{\mathbf{P}}}{\mathbf{V}}$ 计算中存在偏差的舍入误差。当 **预归一化分数（pre-softmax scores）** ${\mathbf{S}}$ 的某一行出现多个相同的最大值时，会导致 $\bar{{\mathbf{P}}}$ 中的对应元素恰好变为 1。为了验证我们的发现，我们修改了 softmax 函数，使其能够检测到这种特定情况并调整归一化过程，确保 $\bar{{\mathbf{P}}}$ 的所有元素都严格小于 1。这防止了有偏差的舍入，并恢复了训练的稳定性。

为了防止有偏差的舍入，我们对 **安全 softmax（safe softmax）** 计算引入了一项有针对性的修改。其核心思想是，仅当分数矩阵 ${\mathbf{S}}$ 的某一行包含多个相同的最大值时，才动态调整归一化因子 ${\mathbf{m}}$。这种调整确保了指数函数的自变量 ${\mathbf{S}}-{\mathbf{m}}$ 在这些最大值位置上变为严格负数，进而保证了 $\bar{{\mathbf{P}}}=\exp({\mathbf{S}}-{\mathbf{m}})$ 的所有元素都小于 1。一种简单的方法（例如减去一个小的固定常数）是不够的，因为它会引入新的系统性舍入误差（参见 [附录 C](#appendix-c)）；因此，需要一种动态的最大值策略。我们的修改如上所述。

<a id="figure-7"></a>

![PK_at_H8_batch_idx_209_6619](images/PK_at_H8_batch_idx_209_6619.png)

> 图 7 | 稳定的 FA 与原始 FA 的对比。

这项修改防止了 $\bar{{\mathbf{P}}}$ 的元素变为恰好 1。如果某行的最大值 ${\mathbf{r}}_{m}$ 为正数且重复出现，则将归一化因子调整为 ${\mathbf{m}}=\beta{\mathbf{r}}_{m}$（其中 $\beta>1$）。这使得指数中的新最大值变为 $-(\beta-1){\mathbf{r}}_{m}$，这是一个严格负数。如果 ${\mathbf{r}}_{m}$ 为负数且重复出现，我们设 ${\mathbf{m}}=0$，这同样确保了指数中的最大值保持为负数。在这两种情况下，此调整都保证了 $\max({\mathbf{S}}-{\mathbf{m}})<0$，从而 $\max(\bar{{\mathbf{P}}})<1$，防止了导致有偏差舍入的条件。

**关键** 在于，在精确算术中，此修改在数学上等价于标准注意力机制，因为它利用了 softmax 函数的平移不变性（$\mathrm{softmax}(\mathbf{z})=\mathrm{softmax}(\mathbf{z}-c)$）。我们的方法只是选择了一个不同的行级常数 $c$ 来确保数值稳定性。在我们的实验中，我们设 $\beta\in[2,8]$，因为更小的值可能导致结果舍入回 1，而更大的值则存在 **下溢（underflow）** 风险。此修改被集成到标准的 **闪存注意力分块算法（flash attention tiling algorithm）** 中（[算法 1](#algorithm-1) 中洋红色标记的行），且不改变反向传播过程。如 [图 7](#figure-7) 所示，这个简单的改变（$\beta=7$）成功地稳定了训练，证实了我们的分析。更多设计细节见 [附录 C](#appendix-c)。

<a id="section-5"></a>

## 5 结论（Conclusion）

本文首次对 **低精度 Flash Attention（Flash Attention）** 训练中一个众所周知的 **损失爆炸（loss explosion）** 问题给出了机制性解释。我们将根本原因归结为 **涌现的低秩表示（emergent low-rank representations）** 与 **有偏的 BF16 舍入误差（biased BF16 rounding errors）** 之间的相互作用。通过对 Flash Attention 进行最小化、有针对性的修改，我们恢复了训练的稳定性，从而验证了我们的分析。我们的分析工作流程为诊断其他架构、规模以及低精度格式（如 FP8）中类似的数值不稳定性提供了蓝图，为更鲁棒、更高效的大规模模型训练铺平了道路。

##### 讨论（Discussion）

我们的发现在多种硬件（NVIDIA A100、RTX 4090、华为昇腾 910B）上均保持一致，并从机制上解释了训练不稳定性的经验观察结果。 **权重谱范数（weight spectral norms）** 的增长源于梯度中一个 **低秩误差矩阵（low-rank error matrix）** 的累积。我们还阐明了 **注意力汇（attention sinks）** 的作用：通过吸引高注意力分数，它们更可能产生值为 1 的注意力概率，从而在 $\bar{{\mathbf{P}}}{\mathbf{V}}$ 计算中触发有偏的舍入误差。这就在汇的架构行为与导致训练失败的算术不稳定性之间建立了直接的数值联系。

##### 局限性（Limitations）

我们的分析聚焦于 GPT-2 模型中的一个特定故障案例。我们的发现对其他架构、更大规模或不同低精度格式（如 FP8）的普适性需要进一步研究。此外，我们提出的缓解措施是针对已识别的特定舍入误差量身定制的，可能无法解决其他来源的数值不稳定性。

## 参考文献（References）

- Ali 等人 (2024)
  Sami Ben Ali, Silviu-Ioan Filip, 和 Olivier Sentieys。
  一种支持 **随机舍入（Stochastic Rounding）** 的低精度浮点乘累加单元，用于 **深度神经网络（Deep Neural Network, DNN）** 训练。
  收录于 _2024 年欧洲设计、自动化与测试大会暨展览会（2024 Design, Automation & Test in Europe Conference & Exhibition, DATE）_，第 1–6 页。IEEE，2024年。
- Balança 等人 (2024)
  Paul Balança, Sam Hosegood, Carlo Luschi, 和 Andrew Fitzgibbon。
  Scalify：用于高效低精度 **大型语言模型（Large Language Model, LLM）** 训练的尺度传播方法。
  _arXiv 预印本 arXiv:2407.17353_，2024年。
- Blake 等人 (2024)
  Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y Prince,
  Björn Deiseroth, Andres Felipe Cruz-Salinas, Carlo Luschi, Samuel
  Weinbach, 和 Douglas Orr。
  u-$\mu$p：单位尺度最大更新参数化方法。
  _arXiv 预印本 arXiv:2407.17465_，2024年。
- Brown 等人 (2020)
  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla
  Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
  等人。
  语言模型是 **小样本学习器（Few-shot Learners）** 。
  _神经信息处理系统进展（Advances in Neural Information Processing Systems）_，
  33:1877–1901，2020年。
- Chowdhery 等人 (2023)
  Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra,
  Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian
  Gehrmann, 等人。
  PaLM：基于 Pathways 的 **语言建模（Language Modeling）** 扩展。
  _机器学习研究杂志（Journal of Machine Learning Research）_，24(240):1–113，2023年。
- Dao (2024)
  Tri Dao。
  FlashAttention-2：具有更好并行性和工作划分的更快注意力机制。
  收录于 _国际学习表征会议（International Conference on Learning Representations, ICLR）_，2024年。
- Dao 等人 (2022)
  Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, 和 Christopher Ré。
  FlashAttention：具有 I/O 感知能力的快速且内存高效的精确实力机制。
  收录于 _神经信息处理系统进展（Advances in Neural Information Processing Systems, NeurIPS）_，2022年。
- Fishman 等人 (2024)
  Maxim Fishman, Brian Chmiel, Ron Banner, 和 Daniel Soudry。
  将 **FP8（8位浮点数）** 训练扩展到万亿词元规模的大型语言模型。
  _arXiv 预印本 arXiv:2409.12517_，2024年。
- flash-attention Issue 337 (2024)
  flash-attention Issue 337。
  关于 flash-attention 第 337 号议题的讨论：使用 Triton FlashAttention 时预训练损失爆炸。
  [https://github.com/Dao-AILab/flash-attention/issues/337](https://github.com/Dao-AILab/flash-attention/issues/337)，2024年。
  访问日期：2025-09-07。
- Gokaslan 等人 (2019)
  Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, 和 Stefanie Tellex。
  OpenWebText 语料库。
  [http://Skylion007.github.io/OpenWebTextCorpus](http://Skylion007.github.io/OpenWebTextCorpus)，2019年。
- Golden 等人 (2024)
  Alicia Golden, Samuel Hsia, Fei Sun, Bilge Acun, Basil Hosmer, Yejin Lee,
  Zachary DeVito, Jeff Johnson, Gu-Yeon Wei, David Brooks, 等人。
  FlashAttention 稳定吗？
  _arXiv 预印本 arXiv:2405.02803_，2024年。
- Hao 等人 (2025)
  Zhiwei Hao, Jianyuan Guo, Li Shen, Yong Luo, Han Hu, Guoxia Wang, Dianhai Yu,
  Yonggang Wen, 和 Dacheng Tao。
  大型语言模型的低精度训练：方法、挑战与机遇。
  _arXiv 预印本 arXiv:2505.01043_，2025年。
- He 等人 (2022)
  Xin He, Jianhua Sun, Hao Chen, 和 Dong Li。
  Campo：面向 **混合精度（Mixed-Precision）** 神经网络训练的成本感知性能优化。
  收录于 _2022 USENIX 年度技术会议（2022 USENIX Annual Technical Conference, USENIX ATC 22）_，
  第 505–518 页，2022年。
- Henry 等人 (2020)
  Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, 和 Yuxuan Chen。
  **Transformer** 模型的查询-键归一化。
  _arXiv 预印本 arXiv:2010.04245_，2020年。
- Hoffmann 等人 (2022)
  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor
  Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes
  Welbl, Aidan Clark, 等人。
  训练计算最优的大型语言模型。
  _arXiv 预印本 arXiv:2203.15556_，2022年。
- Huang 等人 (2025)
  Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, 和 Shiwei Liu。
  SPAM：用于稳定 LLM 训练的、具有动量重置的尖峰感知 Adam 优化器。
  _arXiv 预印本 arXiv:2501.06842_，2025年。
- Kalamkar 等人 (2019)
  Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal
  Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka,
  Jianyu Huang, Hector Yuen, 等人。
  **BFloat16（BF16）** 用于深度学习训练的研究。
  _arXiv 预印本 arXiv:1905.12322_，2019年。
- Kimi-Team (2025)
  Kimi-Team。
  Kimi K2：开放的 **智能体智能（Agentic Intelligence）** 。
  _arXiv 预印本 arXiv:2507.20534_，2025年。
- Lee 等人 (2024)
  Joonhyung Lee, Jeongin Bae, Byeongwook Kim, Se Jung Kwon, 和 Dongsoo Lee。
  从 FP8 到返回：量化降低精度对 LLM 训练稳定性的影响。
  _CoRR_，2024年。
- Liu 等人 (2024)
  Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang
  Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, 等人。
  DeepSeek-V3 技术报告。
  _arXiv 预印本 arXiv:2412.19437_，2024年。
- Mellempudi 等人 (2019)
  Naveen Mellempudi, Sudarshan Srinivasan, Dipankar Das, 和 Bharat Kaul。
  使用 8 位浮点数的混合精度训练。
  _arXiv 预印本 arXiv:1905.12334_，2019年。
- Micikevicius 等人 (2017)
  Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen,
  David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh
  Venkatesh, 等人。
  混合精度训练。
  _arXiv 预印本 arXiv:1710.03740_，2017年。
- Micikevicius 等人 (2022)
  Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey,
  Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John
  Kamalu, 等人。
  用于深度学习的 FP8 格式。
  _arXiv 预印本 arXiv:2209.05433_，2022年。
- Molybog 等人 (2023)
  Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman
  Goyal, Punit Singh Koura, Sharan Narang, Andrew Poulton,

- **nanoGPT Issue 303 (2023)**
  **nanoGPT Issue 303** 。
  关于 nanoGPT issue #303 的讨论：使用 bfloat16 训练时出现 **梯度爆炸（Gradient explosion）** 。
  [https://github.com/karpathy/nanoGPT/issues/303](https://github.com/karpathy/nanoGPT/issues/303)，2023。
  访问日期：2025-09-07。
- **nanoGPT Issue 524 (2024)**
  **nanoGPT Issue 524** 。
  关于 nanoGPT issue #524 的讨论：使用 bfloat16 时 **训练损失（Training loss）** 变为 nan。
  [https://github.com/karpathy/nanoGPT/issues/524](https://github.com/karpathy/nanoGPT/issues/524)，2024。
  访问日期：2025-09-07。
- **nanoGPT Issue 554 (2024)**
  **nanoGPT Issue 554** 。
  关于 nanoGPT issue #554 的讨论：使用 bfloat16 和 **闪存注意力（Flash Attention）** 时损失发散。
  [https://github.com/karpathy/nanoGPT/issues/554](https://github.com/karpathy/nanoGPT/issues/554)，2024。
  访问日期：2025-09-07。
- **Noune 等人 (2022)**
  Badreddine Noune, Philip Jones, Daniel Justus, Dominic Masters, 和 Carlo Luschi。
  用于深度神经网络的 8 位数值格式。
  _arXiv 预印本 arXiv:2206.02915_，2022。
- **Peng 等人 (2023)**
  Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, 等人。
  Fp8-lm：训练 FP8 大型语言模型。
  _arXiv 预印本 arXiv:2310.18313_，2023。
- **Perez 等人 (2023)**
  Sergio P Perez, Yan Zhang, James Briggs, Charlie Blake, Josh Levy-Kramer, Paul Balanca, Carlo Luschi, Stephen Barlow, 和 Andrew William Fitzgibbon。
  使用 8 位浮点数进行大型语言模型的训练和推理。
  _arXiv 预印本 arXiv:2309.17224_，2023。
- **Qiu 等人 (2025)**
  Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, 等人。
  大型语言模型的 **门控注意力（Gated Attention）** ：非线性、稀疏性和无注意力汇。
  _arXiv 预印本 arXiv:2505.06708_，2025。
- **Qwen-Team (2025)**
  Qwen-Team。
  Qwen3 技术报告，2025。
  网址 [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)。
- **Radford 等人 (2019)**
  Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, 和 Ilya Sutskever。

2019.

- Rae 等人 (2021)
  Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann,
  Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young,
  等人。
  Scaling language models: Methods, analysis & insights from training
  gopher.
  _arXiv preprint arXiv:2112.11446_, 2021.
- Rajbhandari 等人 (2020)
  Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He.
  Zero: Memory optimizations toward training trillion parameter models.
  In _SC20: International Conference for High Performance
  Computing, Networking, Storage and Analysis_, pp. 1–16. IEEE, 2020.
- Rybakov 等人 (2024)
  Oleg Rybakov, Mike Chrzanowski, Peter Dykas, Jinze Xue, and Ben Lanir.
  Methods of improving llm training stability.
  _arXiv preprint arXiv:2410.16682_, 2024.
- Shah 等人 (2024)
  Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri
  Dao.
  Flashattention-3: Fast and accurate attention with asynchrony and
  low-precision.
  _Advances in Neural Information Processing Systems_,
  37:68658–68685, 2024.
- Touvron 等人 (2023)
  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
  Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric
  Hambro, Faisal Azhar, 等人。
  Llama: Open and efficient foundation language models.
  _arXiv preprint arXiv:2302.13971_, 2023.
- Tseng 等人 (2025)
  Albert Tseng, Tao Yu, and Youngsuk Park.
  Training llms with mxfp4.
  _arXiv preprint arXiv:2502.20586_, 2025.
- Wang 等人 (2018)
  Naigang Wang, Jungwook Choi, Daniel Brand, Chia-Yu Chen, and Kailash
  Gopalakrishnan.
  Training deep neural networks with 8-bit floating point numbers.
  _Advances in neural information processing systems_, 31, 2018.
- Wang 等人 (2025)
  Ruizhe Wang, Yeyun Gong, Xiao Liu, Guoshuai Zhao, Ziyue Yang, Baining Guo,
  Zhengjun Zha, and Peng Cheng.
  Optimizing large language model training using fp4 quantization.
  _arXiv preprint arXiv:2501.17116_, 2025.
- Wang & Kanwar (2019)
  Shibo Wang and Pankaj Kanwar.
  Bfloat16: The secret to high performance on cloud tpus.
  [https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus),
  August 2019.
  Accessed: 2025-09-07.
- Wortsman 等人 (2023)
  Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari S. Morcos, Ali Farhadi,
  and Ludwig Schmidt.
  Stable and low-precision training for large-scale vision-language
  models.
  In _Thirty-seventh Conference on Neural Information Processing
  Systems_, 2023.
  URL [https://openreview.net/forum?id=sqqASmpA2R](https://openreview.net/forum?id=sqqASmpA2R).
- Xiao 等人 (2023)
  Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis.
  Efficient streaming language models with attention sinks.
  _arXiv preprint arXiv:2309.17453_, 2023.
- Yang 等人 (2022)
  Greg Yang, Edward J Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David
  Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao.
  Tensor programs v: Tuning large neural networks via zero-shot
  hyperparameter transfer.
  _arXiv preprint arXiv:2203.03466_, 2022.
- Yang 等人 (2023)
  Greg Yang, James B Simon, and Jeremy Bernstein.
  A spectral condition for feature learning.
  _arXiv preprint arXiv:2310.17813_, 2023.
- Zhao 等人 (2021)
  Ruizhe Zhao, Brian Vogel, Tanvir Ahmed, and Wayne Luk.
  Reducing underflow in mixed precision training by gradient scaling.
  In _Proceedings of the Twenty-Ninth International Conference on
  International Joint Conferences on Artificial Intelligence_, pp. 2922–2928, 2021.
- Zhou 等人 (2025)
  Jiecheng Zhou, Ding Tang, Rong Fu, Boni Hu, Haoran Xu, Yi Wang, Zhilin Pei,
  Zhongling Su, Liang Liu, Xingcheng Zhang, 等人。
  Towards efficient pre-training: Exploring fp4 precision in large
  language models.
  _arXiv preprint arXiv:2502.11458_, 2025.

<a id="appendix-a"></a>

## 附录 A 相关工作

### A.1 混合精度 BF16 训练

当代大型语言模型（Large Language Model, LLM）的预训练几乎普遍采用混合精度算术。Micikevicius 等人（2017）的早期工作表明，对于许多模型，使用 FP32 权重主副本和固定损失缩放的 FP16 训练可以达到与 FP32 相当的精度。然而，FP16 较窄的指数范围常常导致许多梯度下溢，需要仔细调优。 **bfloat16（BF16）** 格式具有 8 位指数和 7 位尾数，在将存储成本减半的同时，保留了 FP32 的宽动态范围。Kalamkar 等人（2019）表明，BF16 在大型模型上无需专门调优即可实现与 FP32 相当的收敛效果。此后，BF16 已成为许多大规模训练框架默认的 16 位格式，并在 PyTorch 和 TensorFlow 中得到原生支持（Wang & Kanwar, 2019）。

BF16 混合精度使得以空前规模训练具有里程碑意义的 LLM 成为可能，包括 GPT-3（175B）（Brown 等人, 2020）、Google PaLM（540B）（Chowdhery 等人, 2023）、DeepMind Gopher（280B）（Rae 等人, 2021）、Chinchilla（70B）（Hoffmann 等人, 2022）以及 Meta 的 LLaMA 系列（7B-65B）（Touvron 等人, 2023）。为了处理巨大的内存占用，Megatron 和 DeepSpeed 等并行训练框架将 BF16 训练与 **零冗余优化器（Zero Redundancy Optimizer, ZeRO）** （Rajbhandari 等人, 2020）等技术相结合。

尽管有其优势，BF16 训练仍可能表现出不稳定性。实证研究表明，即使使用损失缩放，FP16 也极不稳定，而 BF16 消除了大部分与精度相关的调优和故障（Wang & Kanwar, 2019）。然而，Lee 等人（2024）报告称，大约 10% 的 GPT-2 预训练在纯 BF16 下发散，而在 TF32 下则为 0%。这表明，虽然 BF16 显著提高了稳定性，但在大规模训练中，补充性的稳定技术仍然是必要的。

### A.2 稳定低精度训练

##### 梯度缩放。

Micikevicius 等人（2017）的早期工作引入了 FP16 混合精度训练，其中权重、激活值和梯度以半精度存储，同时维护一个 FP32 主副本。他们还提出了损失缩放以防止 FP16 下溢。然而，即使在损失缩放的情况下，深度网络中仍可能发生一些下溢。为了解决这个问题，Zhao 等人（2021）引入了梯度缩放，它动态计算每层的缩放因子，以避免下溢和上溢。

##### 超低精度（FP8/INT8）训练。

为了进一步降低成本，近期工作探索了使用 FP8 或 INT8 精度进行训练和推理。然而，朴素的 FP8 训练容易发散。Lee 等人（2024）指出，在没有额外稳定技术的情况下，将 FP8 直接应用于 LLM 训练是不稳定的。为了解决这个问题，Perez 等人（2023）提出了针对 FP8 矩阵乘法动态调整的每张量缩放因子。使用此方案，他们成功地在纯 FP8 下训练了参数高达 700 亿的 GPT 和 LLaMA 风格模型。类似地，Peng 等人（2023）引入了 FP8-LM，这是一个逐步将 FP8 应用于梯度、优化器状态和分布式通信的框架，与 BF16 相比，实现了 39% 的内存减少和 75% 的速度提升。Balança 等人（2024）提出了 Scalify，它在整个计算图中传播缩放因子，以确保稳定的 FP8 操作而无需手动调优。这些方法共同表明，仔细的缩放管理使得 FP8 或 INT8 训练能够匹配 BF16 的性能，同时减少内存和计算需求。

##### 优化器与梯度稳定。

优化器算法在训练稳定性中起着关键作用。Molybog 等人（2023）从理论上分析了 Adam，并表明当更新方向与大规模模型中的真实下降方向不相关时，常常会发生灾难性发散。为了解决梯度不稳定性，Huang 等人（2025）提出了 **SPAM（Spike-Aware Adam with Momentum Reset）** ，它通过重置动量并应用峰值感知裁剪来检测和缓解罕见但严重的“梯度峰值”。与此同时，Wortsman 等人（2023）研究了视觉-语言模型中的损失峰值，并表明 AdamW 在峰值发生前常常低估二阶矩。他们提出了一种混合的 AdamW-AdaFactor 优化器，能够自适应地校正二阶矩的低估，其效果优于单独的梯度裁剪。这些方法突显了优化器修改如何直接缓解低精度训练中的发散问题。

##### 激活与架构技术。

激活函数和初始化策略的选择也会影响稳定性。Fishman 等人（2024）观察到，SwiGLU 激活在长时间的 FP8 训练运行中会放大异常值。他们引入了 Smooth-SwiGLU，这是一种改进的激活函数，可以防止异常值放大，从而实现稳定的万亿词元 FP8 训练。在视觉-语言领域，Wortsman 等人（2023）表明，“层缩放零”初始化和精心设计的低精度线性层（例如 SwitchBack）进一步提高了 int8 训练的稳定性。

<a id="appendix-b"></a>

## **附录 B BF16 加法（Appendix B BF16 Addition）**

**bfloat16（Brain Floating-Point）格式** 是一种 16 位浮点数表示法，因其在计算效率和数值范围之间的平衡而被广泛应用于深度学习。它由 1 个符号位、8 个指数位和 7 个尾数位（或称小数位）组成。这种结构使得 bfloat16 具有与 32 位单精度格式（FP32）相同的动态范围，但精度显著降低。

两个 bfloat16 数（例如 $a$ 和 $b$）的加法遵循浮点运算的标准流程：

1.  **指数对齐（Exponent Alignment）** ：比较两个数的指数。指数较小的那个数，其有效数（由隐含的前导位和尾数组合而成）将向右移位，直到其指数与较大的指数匹配。每次右移会使指数加一。移出可用精度范围的比特位会丢失，这是误差的初始来源。
2.  **有效数相加（Significand Addition）** ：将对齐后的有效数相加。结果的符号由操作数的符号和大小决定。
3.  **规格化（Normalization）** ：对结果进行规格化，以确保其符合 $\mathtt{1.xxxx...}\times 2^{e}$ 格式。如果加法导致溢出（例如 $\mathtt{10.xxxx...}$），则有效数右移，指数加一。如果导致抵消（例如 $\mathtt{0.00xx...}$），则有效数左移，指数递减，直到前导位变为 $\mathtt{1}$。
4.  **舍入（Rounding）** ：规格化后的结果有效数可能超过 7 个尾数位，必须进行舍入。标准模式是“ **向最近偶数舍入（round to nearest, ties to even）** ”。这意味着，如果被截断的部分大于最低可存储位（LSB）值的一半，则向上舍入；如果小于，则向下舍入；如果恰好等于一半，则舍入到 LSB 为偶数的最近值。

数值误差的主要来源来自步骤 1 和 4。在指数对齐过程中，较小幅值的数会损失精度。加法之后，结果必须舍入回 7 位尾数，这引入了另一个舍入误差。虽然“向最近偶数舍入”的设计初衷是对随机数据无偏，但对具有特定分布的数据（例如，大部分是负数相加）进行一系列加法运算，可能导致 **有偏舍入误差（biased rounding error）** ，即累积的误差持续地将结果推向一个方向。这种有偏误差的累积是在低精度设置下观察到的训练失败的一个关键因素。

<a id="appendix-c"></a>

## 附录 C：缓解 Flash Attention 中有偏舍入误差的设计考量

##### 使用动态最大值而非固定偏移量

固定偏移量会导致 $\bar{{\mathbf{P}}}$ 的计算值在 BF16（Brain Floating Point 16-bit）转换过程中始终朝一个方向舍入，从而引入固定误差。由于 ${\mathbf{V}}$ 的元素通常具有相同的符号，$\bar{{\mathbf{P}}}$ 中的这种固定舍入误差在计算 $\bar{{\mathbf{P}}}{\mathbf{V}}$ 时不会平均为零。这会导致输出 ${\mathbf{O}}$ 中出现有偏误差，进而产生一个有偏项 ${\bm{\delta}}$，重新引入了我们旨在解决的失效问题。

##### 动态最大值有条件地应用

我们的修改是有条件应用的—— **仅当某一行包含多个相同的最大值时** ——以避免引入新的数值不稳定性。无条件调整并非更优选择。例如，如果某行有一个单一的、非常大的正最大值 ${\mathbf{r}}_{m}$，应用我们的规则意味着计算 $\exp({\mathbf{S}}-\beta{\mathbf{r}}_{m})$。指数中的最大项将变为 $-(\beta-1){\mathbf{r}}_{m}$，而 $\exp(-(\beta-1){\mathbf{r}}_{m})$ 可能下溢为零。这将导致归一化因子变为零，进而在计算输出 ${\mathbf{O}}$ 时引发除零错误。通过仅在导致有偏舍入的特定情况下应用此修改，我们在所有其他场景中保持了标准在线 Softmax（Online Softmax）的数值稳定性。

##### 关于处理负重复行最大值的说明

我们还探讨了针对负的、重复的行最大值（${\mathbf{r}}_{m}<0$）的替代稳定方法。一种方法涉及为某个 $\gamma\in(0,1)$ 设置归一化因子 ${\mathbf{m}}=\gamma{\mathbf{r}}_{m}$。这使得指数中的新最大值变为 $(1-\gamma){\mathbf{r}}_{m}$。然而，我们观察到，如果 $\gamma$ 接近 1，这个新最大值将趋近于零。在低精度算术中，$\exp((1-\gamma){\mathbf{r}}_{m})$ 可能被舍入为恰好 1，从而重新引入我们旨在防止的失效。因此，我们发现设置 $\gamma=0$（即 ${\mathbf{m}}=0$）是一个稳健的选择，因为它确保了指数中的最大值保持足够负。

<a id="algorithm-1"></a>

**算法 1：通过缓解有偏舍入误差实现稳定的 Flash Attention（Flash Attention）：前向传播**

- 0: 矩阵 ${\mathbf{Q}},{\mathbf{K}},{\mathbf{V}}\in\mathbb{R}^{N\times d}$ ，分块大小 $B_{c}$ ， $B_{r}$ ， $\beta>1$ 。
- 1: 将 ${\mathbf{Q}}$ 划分为 $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 个块 ${\mathbf{Q}}_{1},\dots,{\mathbf{Q}}_{T_{r}}$，每个大小为 $B_{r}\times d$；将 ${\mathbf{K}},{\mathbf{V}}$ 划分为 $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 个块 ${\mathbf{K}}_{1},\dots,{\mathbf{K}}_{T_{c}}$ 和 ${\mathbf{V}}_{1},\dots,{\mathbf{V}}_{T_{c}}$，每个大小为 $B_{c}\times d$。
- 2: 将输出 ${\mathbf{O}}\in\mathbb{R}^{N\times d}$ 划分为 $T_{r}$ 个块 ${\mathbf{O}}_{1},\dots,{\mathbf{O}}_{T_{r}}$，每个大小为 $B_{r}\times d$；将对数和指数（Log-Sum-Exp, LSE） ${\mathbf{L}}$ 划分为 $T_{r}$ 个块 ${\mathbf{L}}_{1},\dots,{\mathbf{L}}_{T_{r}}$，每个大小为 $B_{r}$。
- 3: for $1\leq i\leq T_{r}$ do
- 4: 初始化 ${\mathbf{O}}_{i}^{(0)}=(0)_{B_{r}\times d}\in\mathbb{R}^{B_{r}\times d},{\bm{\ell}}_{i}^{(0)}=(0)_{B_{r}}\in\mathbb{R}^{B_{r}},{\mathbf{m}}_{i}^{(0)}=(-\infty)_{B_{r}}\in\mathbb{R}^{B_{r}}$ 。
- 5: for $1\leq j\leq T_{c}$ do
- 6: 计算 ${\mathbf{S}}_{i}^{(j)}={\mathbf{Q}}_{i}{\mathbf{K}}_{j}^{T}\in\mathbb{R}^{B_{r}\times B_{c}}$ 。
- 7: ${\mathbf{r}}_{m}=\mathrm{rowmax}({\mathbf{S}}_{i}^{(j)})$ ， ${\mathbf{r}}_{s}=\mathrm{rowsum}({\mathbf{S}}_{i}^{(j)}\equiv{\mathbf{r}}_{m})$
- 8: ${\mathbf{m}}_{i}^{(j)\prime}=\mathrm{where}({\mathbf{r}}_{m}>0\land{\mathbf{r}}_{s}>1,\beta{\mathbf{r}}_{m},{\mathbf{r}}_{m})$
- 9: ${\mathbf{m}}_{i}^{(j)}=\mathrm{where}({\mathbf{r}}_{m}<0\land{\mathbf{r}}_{s}>1,0,{\mathbf{m}}_{i}^{(j)\prime})$
- 10: 计算 ${\mathbf{m}}_{i}^{(j)}=\mathrm{max}({\mathbf{m}}_{i}^{(j-1)},{\mathbf{r}}_{m})\in\mathbb{R}^{B_{r}}$ ， $\bar{{\mathbf{P}}}_{i}^{(j)}=\exp({\mathbf{S}}_{i}^{(j)}-{\mathbf{m}}_{i}^{(j)})\in\mathbb{R}^{B_{r}\times B_{c}}$（逐点）， ${\bm{\ell}}_{i}^{(j)}=e^{{\mathbf{m}}_{i}^{(j-1)}-{\mathbf{m}}_{i}^{(j)}}{\bm{\ell}}_{i}^{(j-1)}+\mathrm{rowsum}(\bar{{\mathbf{P}}}_{i}^{(j)})\in\mathbb{R}^{B_{r}}$ 。
- 11: 计算 ${\mathbf{O}}_{i}^{(j)}=\text{diag}(e^{{\mathbf{m}}_{i}^{(j-1)}-{\mathbf{m}}_{i}^{(j)}}){\mathbf{O}}_{i}^{(j-1)}+\bar{{\mathbf{P}}}_{i}^{(j)}{\mathbf{V}}_{j}$ 。
- 12: end for
- 13: 计算 ${\mathbf{O}}_{i}=\text{diag}({\bm{\ell}}_{i}^{(T_{c})})^{-1}{\mathbf{O}}_{i}^{(T_{c})}$ 。
- 14: 计算 ${\mathbf{L}}_{i}={\mathbf{m}}_{i}^{(T_{c})}+\log({\bm{\ell}}_{i}^{(T_{c})})$ 。
- 15: 将 ${\mathbf{O}}_{i}$ 写入 ${\mathbf{O}}$ 的第 $i$ 个块。
- 16: 将 ${\mathbf{L}}_{i}$ 写入 ${\mathbf{L}}$ 的第 $i$ 个块。
- 17: end for
- 18: 返回输出 ${\mathbf{O}}$ 和对数和指数 ${\mathbf{L}}$ 。

<a id="algorithm-2"></a>

**算法 2 Flash Attention：前向传播**

- 0: 矩阵 ${\mathbf{Q}},{\mathbf{K}},{\mathbf{V}}\in\mathbb{R}^{N\times d}$，块大小 $B_{c}$，$B_{r}$。
- 1: 将 ${\mathbf{Q}}$ 划分为 $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 个块 ${\mathbf{Q}}_{1},\dots,{\mathbf{Q}}_{T_{r}}$，每个大小为 $B_{r}\times d$；将 ${\mathbf{K}},{\mathbf{V}}$ 划分为 $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 个块 ${\mathbf{K}}_{1},\dots,{\mathbf{K}}_{T_{c}}$ 和 ${\mathbf{V}}_{1},\dots,{\mathbf{V}}_{T_{c}}$，每个大小为 $B_{c}\times d$。
- 2: 将输出 ${\mathbf{O}}\in\mathbb{R}^{N\times d}$ 划分为 $T_{r}$ 个块 ${\mathbf{O}}_{1},\dots,{\mathbf{O}}_{T_{r}}$，每个大小为 $B_{r}\times d$；将对数和指数（Logsumexp）${\mathbf{L}}$ 划分为 $T_{r}$ 个块 ${\mathbf{L}}_{1},\dots,{\mathbf{L}}_{T_{r}}$，每个大小为 $B_{r}$。
- 3: for $1\leq i\leq T_{r}$ do
- 4: 初始化 ${\mathbf{O}}_{i}^{(0)}=(0)_{B_{r}\times d}\in\mathbb{R}^{B_{r}\times d},{\bm{\ell}}_{i}^{(0)}=(0)_{B_{r}}\in\mathbb{R}^{B_{r}},{\mathbf{m}}_{i}^{(0)}=(-\infty)_{B_{r}}\in\mathbb{R}^{B_{r}}$。
- 5: for $1\leq j\leq T_{c}$ do
- 6: 计算 ${\mathbf{S}}_{i}^{(j)}={\mathbf{Q}}_{i}{\mathbf{K}}_{j}^{T}\in\mathbb{R}^{B_{r}\times B_{c}}$。
- 7: 计算 ${\mathbf{m}}_{i}^{(j)}=\mathrm{max}({\mathbf{m}}_{i}^{(j-1)},\mathrm{rowmax}({\mathbf{S}}_{i}^{(j)}))\in\mathbb{R}^{B_{r}}$，$\bar{{\mathbf{P}}}_{i}^{(j)}=\exp({\mathbf{S}}_{i}^{(j)}-{\mathbf{m}}_{i}^{(j)})\in\mathbb{R}^{B_{r}\times B_{c}}$（逐元素），${\bm{\ell}}_{i}^{(j)}=e^{{\mathbf{m}}_{i}^{(j-1)}-{\mathbf{m}}_{i}^{(j)}}{\bm{\ell}}_{i}^{(j-1)}+\mathrm{rowsum}(\bar{{\mathbf{P}}}_{i}^{(j)})\in\mathbb{R}^{B_{r}}$。
- 8: 计算 ${\mathbf{O}}_{i}^{(j)}=\text{diag}(e^{{\mathbf{m}}_{i}^{(j-1)}-{\mathbf{m}}_{i}^{(j)}}){\mathbf{O}}_{i}^{(j-1)}+\bar{{\mathbf{P}}}_{i}^{(j)}{\mathbf{V}}_{j}$。
- 9: end for
- 10: 计算 ${\mathbf{O}}_{i}=\text{diag}({\bm{\ell}}_{i}^{(T_{c})})^{-1}{\mathbf{O}}_{i}^{(T_{c})}$。
- 11: 计算 ${\mathbf{L}}_{i}={\mathbf{m}}_{i}^{(T_{c})}+\log({\bm{\ell}}_{i}^{(T_{c})})$。
- 12: 将 ${\mathbf{O}}_{i}$ 写入 ${\mathbf{O}}$ 的第 $i$ 个块。
- 13: 将 ${\mathbf{L}}_{i}$ 写入 ${\mathbf{L}}$ 的第 $i$ 个块。
- 14: end for
- 15: 返回输出 ${\mathbf{O}}$ 和对数和指数 $L$。

<a id="algorithm-3"></a>

**算法 3 Flash Attention：反向传播（Flash Attention: Backward Pass）**

- 0: 矩阵 ${\mathbf{Q}},{\mathbf{K}},{\mathbf{V}},{\mathbf{O}},d{\mathbf{O}}\in\mathbb{R}^{N\times d}$，向量 $L\in\mathbb{R}^{N}$，块大小 $B_{c}$，$B_{r}$。
- 1: 将 ${\mathbf{Q}}$ 划分为 $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 个块 ${\mathbf{Q}}_{1},\dots,{\mathbf{Q}}_{T_{r}}$，每个大小为 $B_{r}\times d$；将 ${\mathbf{K}},{\mathbf{V}}$ 划分为 $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 个块 ${\mathbf{K}}_{1},\dots,{\mathbf{K}}_{T_{c}}$ 和 ${\mathbf{V}}_{1},\dots,{\mathbf{V}}_{T_{c}}$，每个大小为 $B_{c}\times d$。
- 2: 将 ${\mathbf{O}}$ 划分为 $T_{r}$ 个块 ${\mathbf{O}}_{1},\dots,{\mathbf{O}}_{T_{r}}$，每个大小为 $B_{r}\times d$；将 $d{\mathbf{O}}$ 划分为 $T_{r}$ 个块 $d{\mathbf{O}}_{1},\dots,d{\mathbf{O}}_{T_{r}}$，每个大小为 $B_{r}\times d$；将 ${\mathbf{L}}$ 划分为 $T_{r}$ 个块 ${\mathbf{L}}_{1},\dots,{\mathbf{L}}_{T_{r}}$，每个大小为 $B_{r}$。
- 3: 初始化 $d{\mathbf{Q}}=(0)_{N\times d}$ 并将其划分为 $T_{r}$ 个块 $d{\mathbf{Q}}_{1},\dots,d{\mathbf{Q}}_{T_{r}}$，每个大小为 $B_{r}\times d$。将 $d{\mathbf{K}},d{\mathbf{V}}\in\mathbb{R}^{N\times d}$ 划分为 $T_{c}$ 个块 $d{\mathbf{K}}_{1},\dots,d{\mathbf{K}}_{T_{c}}$ 和 $d{\mathbf{V}}_{1},\dots,d{\mathbf{V}}_{T_{c}}$，每个大小为 $B_{c}\times d$。
- 4: 计算 ${\bm{\delta}}=\mathrm{rowsum}(d{\mathbf{O}}\circ{\mathbf{O}})\in\mathbb{R}^{N}$（逐元素相乘），并将其划分为 $T_{r}$ 个块 ${\bm{\delta}}_{1},\dots,{\bm{\delta}}_{T_{r}}$，每个大小为 $B_{r}$。
- 5: for $1\leq j\leq T_{c}$ do
- 6: 初始化 $d{\mathbf{K}}_{j}=(0)_{B_{c}\times d},d{\mathbf{V}}_{j}=(0)_{B_{c}\times d}$。
- 7: for $1\leq i\leq T_{r}$ do
- 8: 计算 ${\mathbf{S}}_{i}^{(j)}={\mathbf{Q}}_{i}{\mathbf{K}}_{j}^{T}\in\mathbb{R}^{B_{r}\times B_{c}}$。
- 9: 计算 ${\mathbf{P}}_{i}^{(j)}=\exp({\mathbf{S}}_{ij}-{\mathbf{L}}_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$。
- 10: 计算 $d{\mathbf{V}}_{j}\leftarrow d{\mathbf{V}}_{j}+({\mathbf{P}}_{i}^{(j)})^{\top}d{\mathbf{O}}_{i}\in\mathbb{R}^{B_{c}\times d}$。
- 11: 计算 $d{\mathbf{P}}_{i}^{(j)}=d{\mathbf{O}}_{i}{\mathbf{V}}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$。
- 12: 计算 $d{\mathbf{S}}_{i}^{(j)}={\mathbf{P}}_{i}^{(j)}\circ(d{\mathbf{P}}_{i}^{(j)}-{\bm{\delta}}_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$。
- 13: 更新 $d{\mathbf{Q}}_{i}\leftarrow d{\mathbf{Q}}_{i}+d{\mathbf{S}}_{i}^{(j)}{\mathbf{K}}_{j}\in\mathbb{R}^{B_{r}\times d}$。
- 14: 计算 $d{\mathbf{K}}_{j}\leftarrow d{\mathbf{K}}_{j}+{d{\mathbf{S}}_{i}^{(j)}}^{\top}d{\mathbf{Q}}_{i}\in\mathbb{R}^{B_{c}\times d}$。
- 15: end for
- 16: end for
- 17: 返回 $d{\mathbf{Q}},d{\mathbf{K}},d{\mathbf{V}}$。

![X_at_batch_idx_209_train_steps_6619](images/X_at_batch_idx_209_train_steps_6619.png)

> (a) nanoGPT Issue #303 中的损失

<a id="figure-9"></a>

![PKX_at_H8_batch_idx_209_token718_train_steps_6619](images/PKX_at_H8_batch_idx_209_token718_train_steps_6619.png)

> 图 9
