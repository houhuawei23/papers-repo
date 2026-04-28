## 对比散度（Contrastive Divergence, CD）算法全面介绍

对比散度（Contrastive Divergence, CD）是一种用于训练无向概率图模型（特别是受限玻尔兹曼机，Restricted Boltzmann Machine, RBM）的高效近似算法。它由 Geoffrey Hinton 于 2002 年提出，解决了传统最大似然估计（MLE）在 RBM 中计算量过大的问题，成为深度学习早期发展的关键技术之一。

### 1. 背景：受限玻尔兹曼机（RBM）与学习问题

RBM 是一种二分图结构的无向概率图模型，包含可见层（visible units）$\mathbf{v}$ 和隐藏层（hidden units）$\mathbf{h}$，层内无连接，层间全连接。其能量函数定义为：

$$
E(\mathbf{v}, \mathbf{h}) = -\mathbf{a}^\top \mathbf{v} - \mathbf{b}^\top \mathbf{h} - \mathbf{v}^\top \mathbf{W} \mathbf{h}
$$

其中 $\mathbf{a}, \mathbf{b}$ 为偏置，$\mathbf{W}$ 为权重矩阵。联合概率分布为玻尔兹曼分布：

$$
P(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} e^{-E(\mathbf{v}, \mathbf{h})}, \quad Z = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}
$$

学习目标是最大化观测数据 $\mathcal{D} = \{\mathbf{v}^{(1)}, \dots, \mathbf{v}^{(N)}\}$ 的对数似然：

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{n=1}^N \log P(\mathbf{v}^{(n)}; \theta) = \frac{1}{N} \sum_{n=1}^N \log \left( \sum_{\mathbf{h}} e^{-E(\mathbf{v}^{(n)}, \mathbf{h})} \right) - \log Z
$$

对参数 $\theta = \{\mathbf{W}, \mathbf{a}, \mathbf{b}\}$ 求导，得到梯度：

$$
\frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}
$$

其中 $\langle \cdot \rangle_{\text{data}}$ 表示在数据分布 $P(\mathbf{h}|\mathbf{v})$ 下的期望（容易计算，因为条件独立），$\langle \cdot \rangle_{\text{model}}$ 表示在模型联合分布 $P(\mathbf{v}, \mathbf{h})$ 下的期望（难以计算，需要遍历所有状态）。

### 2. 对比散度的核心思想

最大似然梯度中的“正相”（data term）容易计算，而“负相”（model term）需要从模型分布中采样，通常使用马尔可夫链蒙特卡洛（MCMC）方法（如吉布斯采样）。但 RBM 的吉布斯采样需要运行到混合（mixing），计算量极大。

Hinton 的关键洞察： **不需要等待 MCMC 收敛到模型分布，只需执行少数几步吉布斯采样（通常 1 步）就足以给出有效的梯度近似** 。这就是对比散度的基本思想。

具体来说，CD 使用一个 **从数据样本开始的短链 MCMC** （而不是从随机状态开始），经过 k 步吉布斯采样后得到的“重构”样本来近似负相期望。这样，梯度近似为：

$$
\Delta W_{ij} \approx \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{recon}}
$$

其中 $\langle \cdot \rangle_{\text{recon}}$ 表示从数据样本出发，经过 k 步吉布斯采样后得到的可见层和隐藏层状态的期望（通常使用单一样本或小批量平均）。

### 3. CD-k 算法步骤

以标准 CD-1（k=1）为例，训练一个 RBM 的每个小批量更新步骤如下：

**输入** ：一批训练样本 $\mathbf{v}^{(0)}$（可见层初始状态），RBM 参数 $\mathbf{W}, \mathbf{a}, \mathbf{b}$，学习率 $\eta$。

**步骤** ：

1.  **正相** ：给定 $\mathbf{v}^{(0)}$，计算隐藏层激活概率 $P(h_j=1|\mathbf{v}^{(0)}) = \sigma(b_j + \sum_i v_i^{(0)} W_{ij})$。采样得到隐藏状态 $\mathbf{h}^{(0)}$（或直接使用概率值作为期望近似）。
2.  **重构（负相）** ：

- 从 $\mathbf{h}^{(0)}$ 重构可见层：$P(v_i=1|\mathbf{h}^{(0)}) = \sigma(a_i + \sum_j W_{ij} h_j^{(0)})$，采样得 $\mathbf{v}^{(1)}$。
- 再从 $\mathbf{v}^{(1)}$ 计算隐藏层：$P(h_j=1|\mathbf{v}^{(1)}) = \sigma(b_j + \sum_i v_i^{(1)} W_{ij})$，采样得 $\mathbf{h}^{(1)}$。

3.  **计算梯度** ：

$$
   \Delta W_{ij} = \eta \left( \langle v_i^{(0)} h_j^{(0)} \rangle_{\text{batch}} - \langle v_i^{(1)} h_j^{(1)} \rangle_{\text{batch}} \right)
$$

偏置更新类似：$\Delta a_i = \eta ( \langle v_i^{(0)} \rangle - \langle v_i^{(1)} \rangle )$，$\Delta b_j = \eta ( \langle h_j^{(0)} \rangle - \langle h_j^{(1)} \rangle )$。 4. **更新参数** ：$\theta \leftarrow \theta + \Delta \theta$。

对于 CD-k，重复步骤 2（吉布斯采样）k 次，最后使用第 k 步的 $\mathbf{v}^{(k)}$ 和 $\mathbf{h}^{(k)}$ 计算负相统计量。

### 4. 数学分析与理论解释

#### 4.1 CD 与最大似然梯度的关系

定义模型分布 $P_\infty(\mathbf{v})$（吉布斯链的平稳分布），以及从数据分布 $Q^0(\mathbf{v})$ 开始、经过 k 步吉布斯转移 $T^k$ 后的分布 $Q^k(\mathbf{v}) = Q^0 T^k$。CD 的目标函数不是最大似然，而是 **最小化** ：

$$
\text{CD}_k = \text{KL}(Q^0 \| P_\infty) - \text{KL}(Q^k \| P_\infty)
$$

其中 KL 表示 Kullback-Leibler 散度。当 k→∞ 时，$Q^k \to P_\infty$，第二项为零，CD 退化为最大似然的 KL 梯度。对于有限 k，CD 近似于最大似然，但引入了 **偏差** 。有趣的是，CD 梯度是 **一致估计** 吗？不是，但实验表明它能引导参数朝正确方向更新。

#### 4.2 偏差分析

CD 梯度的期望可以写为：

$$
\text{CD}_k \text{gradient} = \text{MLE gradient} + \text{bias term}
$$

该偏差与吉布斯链的混合速度有关。如果模型分布与数据分布接近，或者 k 足够大，偏差小。Hinton 指出，即使 k=1，算法在实践中也能有效工作，因为：

- 正相推动模型分布靠近数据分布；
- 负相使用一步重构，相当于“推开”模型分布中与数据样本相近但非真实模式的部分。

#### 4.3 CD 与对比散度的名称由来

“对比”体现在 **比较数据分布与一步重构分布** ；“散度”指两者之间的差异（非对称的 KL 散度差）。

### 5. 变体与改进

#### 5.1 持久对比散度（Persistent CD, PCD）

标准 CD 每次更新后丢弃马尔可夫链状态，重新从数据样本开始。PCD（也称为“持续对比散度”） **维护一条（或多条）持续运行的马尔可夫链** ，在每次参数更新后，继续从上次的终点开始采样。这样可以更接近模型分布，减少偏差，尤其适用于学习更复杂的分布（如深层玻尔兹曼机）。

#### 5.2 快速持久对比散度（FPCD）

在 PCD 基础上，引入一组“快速权重”来加速链的混合，帮助模型逃离局部模式。

#### 5.3 平均对比散度（mean-field CD）

使用平均场近似代替吉布斯采样，用于某些特定结构。

### 6. 优缺点

#### 优点

- **计算高效** ：k=1 时，每次更新只需两次矩阵乘法（正相和重构），远快于运行到混合的 MCMC。
- **简单易实现** ：无需复杂的采样调度。
- **有效** ：在实践中能够训练出良好的 RBM 特征提取器，成为深度信念网络（DBN）预训练的基础。

#### 缺点

- **有偏估计** ：CD 梯度不是任何目标函数的真实梯度（除 k→∞ 外），可能在某些情况下收敛到非最大似然点。
- **对 k 敏感** ：k 太小可能无法充分探索模型分布，导致模式坍塌；k 太大会增加计算量。
- **不适合所有分布** ：对于高度多模态或稀疏分布，CD 可能失效（如学习稀疏散列编码的 RBM 需要更大的 k 或 PCD）。

### 7. 应用与影响

- **训练 RBM** ：作为标准算法，广泛用于无监督特征学习、协同过滤（如 Netflix 问题）、主题建模等。
- **深度信念网络（DBN）** ：堆叠 RBM，每层用 CD 训练，再微调。
- **生成建模** ：尽管后来被变分自编码器（VAE）和生成对抗网络（GAN）部分取代，但 CD 在非深度生成模型中仍有理论价值。

### 8. 示例：CD-1 的直观解释

假设数据分布有两个模式（簇），模型初始随机。正相拉高数据点的概率；负相从数据点出发一步吉布斯采样，得到类似数据但略有变形的“重构点”。如果模型已经能很好重建数据，重构点与数据点相似，正负相抵消，梯度为零。如果模型不好，重构点偏离数据点，梯度推动参数使数据点概率升高、重构点概率降低。这种“对比”过程相当于一次迭代的“吸引-排斥”机制。

### 9. 与最大似然梯度比较的数值示例

考虑一个简单的 RBM（2 个可见单元，1 个隐藏单元）。计算精确 MLE 梯度与 CD-1 梯度，可以发现方向大致相同，但幅度有差异。随着 k 增大，CD-k 梯度收敛到 MLE 梯度。

### 10. 总结

对比散度算法通过 **从数据样本开始的短链吉布斯采样** 近似最大似然学习中的负相期望，极大地简化了 RBM 的训练。虽然理论上存在偏差，但实践效果出色，是深度学习中无监督预训练的重要基石。后续的 PCD 等变体进一步改善了其性能。理解 CD 不仅有助于掌握 RBM，也体现了近似推断在统计学习中的核心思想： **用快速、有偏的近似替代精确但昂贵的计算** ，往往能带来更好的可扩展性与实用性。

---

**参考文献** （经典）：

- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. _Neural Computation_, 14(8), 1771-1800.
- Carreira-Perpinan, M. A., & Hinton, G. E. (2005). On contrastive divergence learning. _AISTATS_.
- Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient. _ICML_.
