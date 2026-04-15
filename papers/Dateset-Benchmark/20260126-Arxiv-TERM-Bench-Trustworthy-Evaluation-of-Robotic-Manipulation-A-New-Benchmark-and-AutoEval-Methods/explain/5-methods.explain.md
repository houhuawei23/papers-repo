## 说明

- **来源**：原论文 Markdown 中标题为「IV Method」的小节（对应标准章节：Methods (方法)）。
- **解析模板**：`prompts/paper/section-methods.md`
- **提示摘要**：你是学术论文解读助手。请针对下列 **Methods（方法）** 原文进行中文解析。

---

# 方法解析

## 方法管线概览
- **输入**：机器人操作轨迹（视觉关键帧序列 $\mathcal{F}$ 和/或关节状态矩阵 $\mathbf{Q}$）。
- **处理**：
  1. **Rank-Guided Weight Optimization**：使用遗传算法优化运动学评分函数的权重参数，生成与专家排序对齐的校准分数作为**高质量 Ground Truth (GT)**。
  2. **AutoEval 模型**：基于视觉语言模型 (VLM) 构建，输入视觉关键帧与运动学描述文本 $I_{phys}$，进行多任务推理。
- **输出**：
  - 动作质量分数 $\hat{S}$（回归任务）。
  - 任务成功与否 $\hat{O}$（二分类）。
  - 轨迹来源 $\hat{C}$（策略生成 vs. 遥操作，二分类）。
  - （AutoEval-P 额外输出）**Chain-of-Thought (CoT)** 推理过程。

## 关键模块、符号与假设

### 1. Rank-Guided Weight Optimization
- **目标**：自动化评分函数 $S_{raw}$ 的参数 $\boldsymbol{\theta}$ 应与人类专家排序 $R_{human}$ 一致。
- **关键符号**：
  - $\boldsymbol{\theta} = \{w_{vel}, \dots, w_{len}, \lambda_{coll}, \lambda_{fail}\}$：可调权重与惩罚因子。
  - $s_i$：归一化后的运动学指标（如速度、平滑度等）。
  - $s'_i$：违反安全/成功约束时，$s_i$ 会被除数 $\lambda$ 惩罚。
- **评分函数**：
  $$S_{raw}(\boldsymbol{\theta}) = \frac{\sum w_i \cdot s'_i}{\sum w_i}, \quad s'_i = \begin{cases} s_i / \lambda & \text{if violation} \\ s_i & \text{otherwise} \end{cases}$$
- **优化目标**：最小化平均绝对排名差异（Mean Absolute Rank Difference）：
  $$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{k=1}^{N} |R_{human}^{(k)} - R_{raw}^{(k)}(\boldsymbol{\theta})|$$
- **分布对齐**：优化后通过 Z-score 归一化将 $S_{raw}$ 映射到人类评分尺度：
  $$S_{final} = \mu_{human} + \sigma_{human} \cdot \left( \frac{S_{raw}(\boldsymbol{\theta}^*) - \mu_{raw}}{\sigma_{raw}} \right)$$
- **假设**：人类专家的相对排序是可靠的 GT；运动学指标的组合能有效反映操作质量。

### 2. AutoEval 任务建模
- **输入表示**：
  - 视觉：$N$ 个关键帧 $\mathcal{F} = \{f_i\}_{i=1}^N$。
  - 运动学：关节状态矩阵 $\mathbf{Q} \in \mathbb{R}^{T \times J}$（$J$ 为自由度）。
- **运动学特征提取**：
  - 角速度与角加速度（离散一阶差分）：
    $$\mathbf{v}_t = \mathbf{q}_t - \mathbf{q}_{t-1}, \quad \boldsymbol{\alpha}_t = \mathbf{v}_t - \mathbf{v}_{t-1}$$
  - 均匀性指标（捕获最差情况的不稳定性）与平均绝对速度：
    $$\mathcal{U}_v = \max_{j \in \{1 \dots J\}} (\mathbf{s}_v[j]), \quad \mathcal{U}_\alpha = \max_{j \in \{1 \dots J\}} (\mathbf{s}_\alpha[j]), \quad \mu_v = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} |v_{t,j}|$$
  - 序列化为文本描述符 $I_{phys}$，作为**运动学校准信号**输入 VLM。
- **模型输出**：
  $$(\hat{S}, \hat{O}, \hat{C}) = \Phi_\theta \left( \mathcal{F}, I_{phys}(\mathcal{U}_v, \mathcal{U}_\alpha, \mu_v) \right)$$

### 3. AutoEval-S（轻量版）
- **核心创新**：**时空聚合策略 (Spatio-Temporal Aggregation Strategy)**。
  - **问题**：增加采样帧数 $N$ 会显著增加视觉 token 数量与 VRAM 占用。
  - **解决方案**：将关键帧 $f_i$ 与 $f_{i+1}$ 之间的 $k$ 个中间帧**空间拼接**成复合图像，再缩放到标准分辨率，得到精炼序列 $\mathcal{F}'$。
  - **优点**：在固定 token 预算内压缩高频运动细节，提升时间信息密度。
- **训练目标**：监督微调 (SFT)，将多任务输出序列化为结构化文本 $\mathbf{Y}$，最小化负对数似然：
  $$\mathcal{L} = -\sum_{t=1}^{L} \log P_\theta (y_t \mid y_{<t}, \mathcal{F}', I_{phys})$$

### 4. AutoEval-P（增强版）
- **核心创新**：采用 **Group Relative Policy Optimization (GRPO)** 强化学习范式，提升 CoT 推理的物理一致性与逻辑性。
- **奖励函数设计**：
  1. **分数奖励 $R_{score}$**（基于高斯核的软回归机制）：
     $$R_{score} = \exp\left( -\frac{(S - \hat{S})^2}{2\sigma^2} \right)$$
  2. **分类奖励**：$R_{succ} = \mathbb{I}(\hat{O} = O)$, $R_{src} = \mathbb{I}(\hat{C} = C)$。
  3. **内容准确度奖励**（加权和）：
     $$R_{acc} = \omega_{score} \cdot R_{score} + \omega_{succ} \cdot R_{succ} + \omega_{src} \cdot R_{src}$$
  4. **格式奖励 $R_{fmt}$**：约束生成文本结构。
  5. **总奖励**：
     $$R_{total} = (1-\gamma) \cdot R_{acc} + \gamma \cdot R_{fmt}$$
- **GRPO 优化目标**：
  $$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \frac{\pi_\theta(y_i | x)}{\pi_{old}(y_i | x)} A_i - \beta \mathbb{D}_{KL} (\pi_\theta \| \pi_{ref}) \right) \right]$$
  - 其中优势值 $A_i$ 通过组内归一化计算：
    $$A_i = \frac{R_{total}(y_i) - \mu_{group}}{\sigma_{group} + \epsilon}$$
- **关键假设**：GRPO 的组内相对奖励比较能有效引导策略优化，且 KL 惩罚能防止灾难性遗忘。

## 与 Baseline 或常见做法的差异
1. **GT 生成方式**：
   - **常见做法**：依赖人工标注或简单启发式函数。
   - **本文**：提出 **Rank-Guided Weight Optimization**，使用遗传算法校准运动学评分函数，使其与专家排序对齐，生成**高质量、可扩展的 GT**。
2. **多模态输入**：
   - **常见 VLM 评估**：仅使用视觉帧。
   - **本文**：引入 **Kinematic Calibration Signal $I_{phys}$**，显式提供运动统计量（平滑度、强度等），补偿视频压缩伪影，提升运动量化精度。
3. **时序信息处理**：
   - **常见做法**：均匀采样或随机采样关键帧，可能丢失细节。
   - **AutoEval-S**：提出**时空聚合策略**，通过空间拼接保留中间帧信息，在不增加 token 数的前提下提升时间分辨率。
4. **复杂推理任务**：
   - **常见 CoT 生成**：仅依赖 SFT，可能产生幻觉或逻辑不一致。
   - **AutoEval-P**：引入 **GRPO 强化学习**，通过混合奖励函数（分数奖励+分类奖励+格式奖励）引导模型生成**物理一致、结构规范的 CoT**。
5. **训练效率**：
   - **标准 PPO**：需要额外训练价值网络。
   - **GRPO**：**无需价值网络**，通过组内归一化计算优势，降低计算开销。

## 实现或复现需注意的细节
1. **Rank-Guided 优化**：
   - 遗传算法采用**实值编码**与**锦标赛选择**；需合理设置种群大小、迭代次数与突变率。
   - 最终务必进行 **Z-score 分布对齐**，使 $S_{final}$ 与人类评分尺度一致。
2. **运动学特征计算**：
   - 关节状态矩阵 $\mathbf{Q}$ 的维度 $J$ 根据机器人自由度确定（文中为 7 或 14）。
   - 计算速度/加速度时，**假设单位时间间隔**；若实际采样率非均匀，需调整差分公式。
3. **时空聚合策略**：
   - 中间帧拼接时需保持**空间对齐**，避免扭曲变形。
   - 复合图像需** resize 到 VLM 编码器的标准输入分辨率**。
4. **AutoEval-P 训练**：
   - **奖励权重**（$\omega_{score}, \omega_{succ}, \omega_{src}, \gamma$）需根据任务重要性调优。
   - **GRPO 组大小 $G$** 影响优势估计的稳定性；文中未指定，需实验确定。
   - **KL 系数 $\beta$** 需谨慎设置，以平衡性能提升与语言流畅性保持。
5. **数据准备**：
   - 关键帧采样策略影响信息完整性；文中未详细说明采样准则，复现时需设计合理的关键帧提取方法（如基于运动显著性的采样）。
6. **基线对比**：
   - 文中强调 **zero-shot VLM 性能极差**（SRCC ≈ 0.02），凸显了其微调管道的必要性；复现时需包含 zero-shot 基线作为对比。
7. **计算资源**：
   - AutoEval-S 的时空聚合可**降低 VRAM 占用**，但拼接与 resize 操作增加前处理开销。
   - AutoEval-P 的 GRPO 训练需**多次采样生成 CoT**，比 SFT 更耗时。