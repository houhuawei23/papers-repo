# TERM-Bench / Eval-Actions / AutoEval：概念、公式与符号系统梳理

> 依据译文：`20260126-Arxiv-Trustworthy-Evaluation-of-Robotic-Manipulation-A-New-Benchmark-and-AutoEval-Methods_trans.md`  
> 论文：*Trustworthy Evaluation of Robotic Manipulation*（ArXiv: 2601.18723）

---

## 一、关键概念与术语

### 1.1 评估范式与问题动机

| 术语 | 英文/缩写 | 定义与上下文作用 |
|------|-----------|-------------------|
|  **可信评估**  | Trustworthy Evaluation | 不仅看任务是否成功，还要能 **诊断** 执行质量与行为来源，使评估结果可解释、可比对、难操纵。 |
|  **差距 1：执行质量模糊性**  | Gap 1 (Ambiguity in Execution Quality) | 二元成功率无法区分“抖动但成功”与“流畅成功”，掩盖安全风险与部署可信度。 |
|  **差距 2：来源真实性模糊性**  | Gap 2 (Ambiguity in Source Authenticity) | 视觉上的成功无法证明行为来自 **自主策略** 还是 **人类遥操作** ，评估易受操纵。 |
|  **来源模糊性**  | Source Ambiguity | 成功演示的 **来源不可验证** ，与“质量模糊性”共同削弱基准公平性。 |
|  **质量模糊性**  | Quality Ambiguity | 成功/失败二元标签无法反映流畅度、安全性、效率等过程质量。 |
|  **Jerky Success / Smooth Success**  | 抖动成功 / 流畅成功 | 用于图示同一“成功”标签下执行质量差异的典型对比。 |

### 1.2 数据集与标注（Eval-Actions）

| 术语 | 英文/缩写 | 定义与上下文作用 |
|------|-----------|-------------------|
|  **Eval-Actions**  | — | 面向 **评估** 而非单纯训练的数据集：含失败、混合来源（人类/策略）、密集多模态与细粒度标注。 |
|  **Eval-Actions Small (EAS)**  | EAS | 子集：专门用于 **区分遥操作与策略轨迹** 、验证任务结果，缓解策略分布偏移。 |
|  **混合收集策略**  | Hybrid Collection Strategy | 同时收集人类遥操作与多种 VA/VLA 策略轨迹，支撑 **来源真实性** 验证。 |
|  **失败数据**  | Failure Data | 含真实失败轨迹（文中约 37.4% 失败），与“仅成功演示”的训练集形成对比。 |
|  **细粒度动作质量**  | Fine-Grained Action Quality | 对执行过程除成功外的多维度刻画：流畅度、安全性、效率等，用于诊断性评估。 |
|  **细粒度质量雷达图**  | Fine-Grained Quality Radar Chart | 可视化四维度：成功、流畅度、安全性、效率。 |
|  **专家评分**  | Expert Grading (EG) | 多专家按平滑度/安全/效率分档（优秀/良好/差），取平均以减偏差；标签尺度如 1–10。 |
|  **排序引导偏好 / 排序引导权重优化**  | Rank-Guided preferences (RG) / Rank-Guided Weight Optimization | 专家对一批视频排序；用 **遗传算法** 优化运动学综合分权重，使 **算法排序** 与 **人类排序** 一致。 |
|  **思维链标注**  | Chain-of-Thought (CoT) | 除分外提供 **文字理由** ，增强可解释性，也用于 AutoEval-P 的推理训练。 |
|  **基准真值 / 真值**  | Ground Truth (GT) | EG、RG、CoT 三种监督信号在文中作为不同粒度的评估“标准答案”。 |

### 1.3 质量维度（标注协议）

| 术语 | 说明 |
|------|------|
|  **成功率 (Success Rate)**  | 二元任务是否完成。 |
|  **平滑度 (Smoothness)**  | 关节角速度、加速度方差等运动学指标量化。 |
|  **安全性 (Safety)**  | 碰撞、危险交互等物理异常监测与惩罚。 |
|  **效率 (Efficiency)**  | 完成时间；在 **空间区域内相对基线** 做归一化，以公平比较不同初始距离。 |

### 1.4 方法与模型（AutoEval）

| 术语 | 英文/缩写 | 定义与上下文作用 |
|------|-----------|-------------------|
|  **AutoEval-S**  | AutoEval Small | 面向 EG、RG： **时空聚合**  +  **运动学校准信号**  +  **SFT** （交叉熵）。 |
|  **AutoEval-P**  | AutoEval Plus | 面向 CoT： **GRPO**  强化学习，混合奖励（内容准确度 + 格式）。 |
|  **时空聚合策略**  | Spatio-Temporal Aggregation | 将中间帧与关键帧 **空间拼接** 成复合图，在固定词元预算内保留高频运动信息。 |
|  **运动学校准信号**  | Kinematic Calibration Signal | 由物理指标序列化成的文本提示 $I_{phys}$，减轻视频压缩伪影、校准平滑度量化。 |
|  **视觉语言模型**  | Vision-Language Model (VLM) | 多模态主干，用于评分、成功/失败、来源分类及 CoT 生成。 |
|  **组相对策略优化**  | Group Relative Policy Optimization (GRPO) | 每组采样多个 CoT，用组内相对优势更新策略， **无需单独价值网络** ；文中用于缓解幻觉、对齐物理推理与分数。 |
|  **信息稀释效应**  | Informational Dilution Effect | CoT 任务中长推理占用注意力，使纯回归/相关性相对下降的现象解释。 |

### 1.5 领域与相关方向

| 术语 | 英文/缩写 | 简述 |
|------|-----------|------|
|  **视觉-动作**  | Vision-Action (VA) | 观测→控制，端到端策略。 |
|  **视觉-语言-动作**  | Vision-Language-Action (VLA) | 融合语言与视觉的策略模型。 |
|  **模仿学习**  | Imitation Learning (IL) | 从演示学习策略。 |
|  **动作质量评估**  | Action Quality Assessment (AQA) | 评估“做得多好”，区别于动作识别分类。 |
|  **监督微调**  | Supervised Fine-Tuning (SFT) | 有监督微调 VLM。 |
|  **低秩适应**  | Low-Rank Adaptation (LoRA) | 实验中微调方式。 |

### 1.6 实验指标（分类）

| 术语 | 英文/缩写 | 用途 |
|------|-----------|------|
|  **斯皮尔曼等级相关系数**  | SRCC (Spearman’s $\rho$) | 衡量预测分数与人工标注的 **单调一致性** ，对尺度非线性更鲁棒。 |
|  **相对 L2 误差**  | $R_{\ell_2}$ | 按分数范围归一化后的均方误差，反映相对尺度误差。 |
|  **准确率 / F1 / AUC**  | Acc / F1 / AUC | 成功预测、来源预测等二分类任务； **类别不平衡** 时强调 F1。 |

---

## 二、核心公式解析

以下为译文中编号  **(1)–(14)**  及算法相关式。若文中同一符号在不同模块重复出现，以 **首次定义模块** 为准。

### 公式 (1)：原始综合分数 $S_{raw}$（排序引导权重优化 / RG）

 **形式：** 


$$
S_{raw}(\mathbf{\theta})=\frac{\sum w_{i}\cdot s^{\prime}_{i}}{\sum w_{i}},\quad
s^{\prime}_{i}=\begin{cases}s_{i}/\lambda&\text{if violation}\\ s_{i}&\text{otherwise}\end{cases}
$$


-  **含义** ：在可调权重 $\mathbf{\theta}$ 下，将 **归一化运动学分项**  $s_i$ 加权平均；若违反 **安全或成功** 约束，则用除数 $\lambda$  **惩罚** 后再聚合。
-  **变量** ：$\mathbf{\theta}=\{w_{vel},\dots,w_{len},\lambda_{coll},\lambda_{fail}\}$ 为可调参数向量；$w_i$ 为各分项权重；$s_i$ 为各原始算法分项；$\lambda$ 为 violation 对应的惩罚因子（属于 $\mathbf{\theta}$）。
-  **背景** ：把“运动学指标与惩罚”的相对重要性变成 **超参数搜索** 问题，使自动分数与人类 **排序** 一致。
-  **创新/对比** ：与单纯加权平均相比，显式引入 **约束违反时的惩罚分支** ，避免高分但危险/失败的轨迹。

### 公式 (2)：优化目标 $\mathcal{L}(\mathbf{\theta})$（平均绝对排序差）


$$
\mathcal{L}(\mathbf{\theta})=\frac{1}{N}\sum_{k=1}^{N}|R_{human}^{(k)}-R_{raw}^{(k)}(\mathbf{\theta})|
$$


-  **含义** ：最小化 **人类排序**  $R_{human}$ 与 **原始算法排序**  $R_{raw}$ 的 **平均绝对秩差** 。
-  **应用场景** ： **遗传算法 (GA)**  搜索 $\mathbf{\theta}^*$ 使排序一致。
-  **注意** ：目标只关心 **相对顺序** ，故 $S_{raw}$ 的 **绝对数值尺度** 未必与 0–10 专家分一致。

### 公式 (3)：Z 分数分布对齐 $S_{final}$


$$
S_{final}=\mu_{human}+\sigma_{human}\cdot\left(\frac{S_{raw}(\mathbf{\theta}^{*})-\mu_{raw}}{\sigma_{raw}}\right)
$$


-  **含义** ：将最优 $\mathbf{\theta}^*$ 下的 $S_{raw}$ 线性变换到与 **专家分** 同均值方差的尺度。
-  **变量** ：$\mu_{raw},\sigma_{raw}$ 为数据集上 $S_{raw}(\mathbf{\theta}^*)$ 的均值/标准差；$\mu_{human},\sigma_{human}$ 来自专家标注。
-  **作用** ：在 **排序** 正确后， **对齐数值分布** ，便于与 EG 尺度比较。

### 公式 (4)：角速度与角加速度（离散差分）


$$
\mathbf{v}_{t}=\mathbf{q}_{t}-\mathbf{q}_{t-1},\quad\boldsymbol{\alpha}_{t}=\mathbf{v}_{t}-\mathbf{v}_{t-1}
$$


-  **含义** ：在 **单位时间步** 假设下，由关节角序列 $\mathbf{q}_t$ 得一阶差分速度、二阶差分加速度。
-  **变量** ：$\mathbf{q}_t\in\mathbb{R}^J$ 为 $t$ 时刻关节构型；$J\in\{7,14\}$ 为 DoF（单臂/双臂）。

### 公式 (5)：平滑度相关统计（最坏情况 + 平均强度）


$$
\begin{split}\mathcal{U}_{v}&=\max_{j\in\{1\dots J\}}(\mathbf{s}_{v}[j]),\quad \mathcal{U}_{\alpha}=\max_{j\in\{1\dots J\}}(\mathbf{s}_{\alpha}[j]),\\ \mu_{v}&=\frac{1}{T\cdot J}\sum_{t=1}^{T}\sum_{j=1}^{J}|v_{t,j}|.\end{split}
$$


-  **含义** ：$\mathbf{s}_v,\mathbf{s}_\alpha$ 为各关节速度/加速度时间方差；用 **跨关节最大值** 刻画最坏抖动；$\mu_v$ 为平均绝对速度，表征整体运动强度。
-  **设计** ： **最坏情况** 公式化捕捉“任一关节异常”即可反映不稳定。

### 公式 (6)：多模态评估器 $\Phi_\theta$


$$
(\hat{S},\hat{O},\hat{C})=\Phi_{\theta}\left(\mathcal{F},I_{phys}(\mathcal{U}_{v},\mathcal{U}_{\alpha},\mu_{v})\right)
$$


-  **含义** ：同时预测 **质量分数**  $\hat{S}$、 **成功**  $\hat{O}$、 **来源**  $\hat{C}$（策略 vs 遥操作）。
-  **输入** ：关键帧 $\mathcal{F}$ 与物理提示 $I_{phys}$。

### 公式 (7)：AutoEval-S 的负对数似然（SFT）


$$
\mathcal{L}=-\sum_{t=1}^{L}\log P_{\theta}(y_{t}\mid y_{<t},\mathcal{F}^{\prime},I_{phys})
$$


-  **含义** ：将 $S,O,C$ 序列化为目标文本 $\mathbf{Y}$，自回归生成， **最小化 NLL** 。
-  **变量** ：$L$ 为目标序列长度；$\mathcal{F}'$ 为时空聚合后的帧序列；$\theta$ 为 VLM 参数。

### 公式 (8)：高斯核软回归奖励 $R_{score}$


$$
R_{score}=\exp\left(-\frac{(S-\hat{S})^{2}}{2\sigma^{2}}\right)
$$


-  **含义** ：预测分 $\hat{S}$ 越接近真值 $S$，奖励越接近 1；$\sigma$ 控制 **灵敏度** 。
-  **动机** ：缓解 RL 中 **二元稀疏奖励** 对连续回归的不利问题。

### 公式 (9)：内容准确度奖励 $R_{acc}$


$$
R_{acc}=\omega_{score}\cdot R_{score}+\omega_{succ}\cdot R_{succ}+\omega_{src}\cdot R_{src}
$$


-  **含义** ：连续分数 + 成功/来源二分类指示奖励的 **加权组合** 。
-  **消融** ：文中 4:3:3 优于 1:1:1，因 **回归更难** 需更高 $\omega_{score}$。

### 公式 (10)：全局奖励 $R_{total}$


$$
R_{total}=(1-\gamma)\cdot R_{acc}+\gamma\cdot R_{fmt}
$$


-  **含义** ： **内容** 与 **格式** 奖励的凸组合；$\gamma$ 控制格式约束强度（文中 $\gamma=0.2$ 最优）。

### 公式 (11)：组内优势 $A_i$（GRPO）


$$
A_{i}=\frac{R_{total}(y_{i})-\mu_{group}}{\sigma_{group}+\epsilon}
$$


-  **含义** ：同一输入 $x$ 下 $G$ 个 CoT 输出中，第 $i$ 个 **相对组均值** 的归一化优势；$\epsilon$ 防除零。
-  **注** ：算法 1 步骤中另写 $A_i=(R_i-\text{mean}(R))/(\text{std}(R)+\epsilon)$，与 (11)  **等价表述** ，仅记号 $\mu_{group},\sigma_{group}$ 与 mean/std 对应。

### 公式 (12)：GRPO 目标 $\mathcal{J}_{GRPO}$


$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{x\sim\mathcal{D}}\bigg[\frac{1}{G}\sum_{i=1}^{G}\bigg(&\frac{\pi_{\theta}(y_{i}|x)}{\pi_{old}(y_{i}|x)}A_{i} -\beta\mathbb{D}_{KL}(\pi_{\theta}||\pi_{ref})\bigg)\bigg]
$$


-  **含义** ： **重要性采样比率**  × 优势 +  **KL 到参考策略**  $\pi_{ref}$，抑制 **灾难性遗忘** 。
-  **对比 PPO** ：GRPO  **无需价值网络** ，计算更省。

### 公式 (13)：斯皮尔曼相关（与秩的皮尔逊相关等价写法）


$$
\rho=\frac{\sum_{i=1}^{N}(r_{g,i}-\bar{r}_{g})(r_{p,i}-\bar{r}_{p})}{\sqrt{\sum_{i=1}^{N}(r_{g,i}-\bar{r}_{g})^{2}}\sqrt{\sum_{i=1}^{N}(r_{p,i}-\bar{r}_{p})^{2}}}
$$


-  **含义** ：在 **秩**  $r_{g,i},r_{p,i}$ 上计算皮尔逊相关，即 SRCC。
-  **适用** ：人类评分主观、非线性时，比 MSE 更关注 **序** 。

### 公式 (14)：相对 L2 误差 $R_{\ell_2}$


$$
R_{\ell_{2}}=\frac{100}{N}\sum_{i=1}^{N}\left(\frac{s_{g,i}-s_{p,i}}{s_{max}-s_{min}}\right)^{2}
$$


-  **含义** ：按 **真实分范围** 归一化后的均方误差，再乘 100（百分比量级）。
-  **变量** ：$s_{g,i},s_{p,i}$ 为真值与预测；$s_{max},s_{min}$ 为真值分数范围边界。

---

## 三、符号表（规范化）

### 3.1 数据集与任务符号

| 符号 | 中文释义 | 上下文/出现位置 |
|------|----------|-----------------|
| $T$ | 片段时间长度（离散步数） | §IV-B，轨迹长度 |
| $N$ | 关键帧数量（$N\le T$） | $\mathcal{F}=\{f_i\}_{i=1}^N$ |
| $\mathcal{F}$ | 视觉关键帧集合 | 输入 (6)(7) |
| $\mathcal{F}'$ | 时空聚合后的帧序列 | AutoEval-S，(7) |
| $f_i, f'_i$ | 第 $i$ 个原始/聚合帧 | 时空聚合 |
| $k$ | 与关键帧拼接的中间帧数 | 聚合策略描述 |
| $J$ | 关节自由度（7 或 14） | $\mathbf{Q}\in\mathbb{R}^{T\times J}$ |
| $\mathbf{Q}$ | 关节状态矩阵 | §IV-B |
| $\mathbf{q}_t$ | 时刻 $t$ 关节向量 | (4)(5) |
| $EAS$ | Eval-Actions Small 子集 | §III-A |

### 3.2 运动学与物理提示（$I_{phys}$ 模块）

| 符号 | 中文释义 | 上下文/出现位置 |
|------|----------|-----------------|
| $\mathbf{v}_t$ | 时刻 $t$ 关节角速度（差分） | (4) |
| $\boldsymbol{\alpha}_t$ | 时刻 $t$ 角加速度 | (4) |
| $\mathbf{s}_v,\mathbf{s}_\alpha$ | 各关节速度/加速度时间方差向量 | (5) 前句 |
| $\mathcal{U}_v,\mathcal{U}_\alpha$ | 跨关节最大方差（最坏抖动） | (5)(6) |
| $\mu_v$ | 平均绝对速度（全局运动强度） | (5)(6) |
| $v_{t,j}$ | 关节 $j$ 在 $t$ 的速度标量 | (5) |
| $I_{phys}$ | 运动学文本提示（校准信号） | (6)(7) 等 |

### 3.3 排序引导权重优化（RG / GA）

| 符号 | 中文释义 | 上下文/出现位置 |
|------|----------|-----------------|
| $\mathbf{\theta}$ | 可调参数向量（权重与 $\lambda$） | (1)(2) |
| $w_i$ | 第 $i$ 个分项权重 | (1) |
| $w_{vel},\dots,w_{len}$ | 分项示例（速度、轨迹长度等） | $\mathbf{\theta}$ 列举 |
| $\lambda_{coll},\lambda_{fail}$ | 碰撞、失败等惩罚因子 | $\mathbf{\theta}$ 列举 |
| $s_i$ | 第 $i$ 个归一化运动学分项 | (1) |
| $s'_i$ | 违反约束时惩罚后的分项 | (1) |
| $\lambda$ | 泛化 violation 的除数（与 $\lambda_{coll}$ 等同类） | (1) 分支 |
| $S_{raw}$ | 原始算法综合分数 | (1)(3) |
| $S_{final}$ | 分布对齐后的最终分数 | (3) |
| $R_{human}^{(k)}$ | 第 $k$ 个样本的人类排序秩 | (2) |
| $R_{raw}^{(k)}(\mathbf{\theta})$ | 算法分数导出的排序秩 | (2) |
| $N$ | 排序样本数（与关键帧 $N$  **不同语境** ，易混） | (2) 中优化样本数 |
| $\mu_{raw},\sigma_{raw}$ | $S_{raw}$ 在数据集上的均值/标准差 | (3) |
| $\mu_{human},\sigma_{human}$ | 专家分分布的均值/标准差 | (3) |

 **易混说明** ：$N$ 在文中既表示 **关键帧数** （§IV-B），又表示 **排序优化中的样本数** （(2)）；阅读时按章节区分。

### 3.4 模型输出与预测

| 符号 | 中文释义 | 上下文/出现位置 |
|------|----------|-----------------|
| $\Phi_\theta$ | 参数为 $\theta$ 的多模态评估器 | (6)(7) |
| $\hat{S},\hat{O},\hat{C}$ | 预测分数、成功、来源 | (6) |
| $S,O,C$ | 真值分数、成功、来源 | 奖励与监督 |
| $\mathbf{Y}$ | 序列化目标文本（$S,O,C$） | §IV-B AutoEval-S |
| $y_t$ | 目标序列第 $t$ 个词元 | (7) |
| $L$ | 目标文本序列长度 | (7) |

### 3.5 AutoEval-P / GRPO 与奖励

| 符号 | 中文释义 | 上下文/出现位置 |
|------|----------|-----------------|
| $R_{score}$ | 高斯软回归分数奖励 | (8) |
| $R_{succ},R_{src}$ | 成功/来源指示奖励 | (9) |
| $R_{acc}$ | 内容准确度奖励 | (9) |
| $R_{fmt}$ | 格式奖励 | (10) |
| $R_{total}$ | 全局奖励 | (10)(11) |
| $\sigma$ | $R_{score}$ 灵敏度超参 | (8) |
| $\omega_{score},\omega_{succ},\omega_{src}$ | 各子任务权重 | (9) |
| $\gamma$ | 格式奖励混合系数 | (10) |
| $\pi_\theta,\pi_{ref},\pi_{old}$ | 当前策略、参考策略、旧策略 | (12) |
| $G$ | 每组 CoT 采样数 | 算法 1 |
| $x$ | 输入 $(\mathcal{F},I_{phys})$ | 算法 1 |
| $y_i$ | 第 $i$ 个 CoT 输出 | (11)(12) |
| $A_i$ | 第 $i$ 个输出的优势 | (11)(12) |
| $\mu_{group},\sigma_{group}$ | 组内奖励均值/标准差 | (11) |
| $\epsilon$ | 数值稳定小常数 | (11) |
| $\beta$ | KL 惩罚系数 | (12) |
| $\mathcal{D}$ | 训练数据集 | 算法 1 |
| $\mathbb{D}_{KL}$ | KL 散度 | (12) |

### 3.6 实验评估指标

| 符号 | 中文释义 | 上下文/出现位置 |
|------|----------|-----------------|
| $\rho$ | 斯皮尔曼等级相关系数（SRCC） | (13) |
| $r_{g,i},r_{p,i}$ | 真值/预测分数的秩 | (13) |
| $\bar{r}_g,\bar{r}_p$ | 秩的均值 | (13) |
| $s_{g,i},s_{p,i}$ | 真值/预测分数（连续） | (14) |
| $s_{max},s_{min}$ | 真值分数范围边界 | (14) |
| $R_{\ell_2}$ | 相对 L2 误差 | (14)，表 II 等 |

### 3.7 缩写与运算符

| 类别 | 条目 | 说明 |
|------|------|------|
| 缩写 | VA, VLA, AQA, IL, SFT, RL, GRPO, PPO, LoRA, CoT, EG, RG, GT, SRCC, MSE, ROC, AUC, DoF, VRAM | 见第一节术语表 |
| 运算符 | $\mathbb{I}(\cdot)$ | 指示函数，二分类命中为 1（§IV-B 描述 $R_{succ},R_{src}$） |
| 运算符 | $\mathbb{E}_{x\sim\mathcal{D}}$ | 对数据分布期望 | (12) |

---

## 四、工作流内符号聚合（便于检索）

### 4.1 管线：轨迹 → $I_{phys}$ → $\Phi_\theta$

$\mathbf{Q}$ → $\mathbf{q}_t$ → (4) $\mathbf{v}_t,\boldsymbol{\alpha}_t$ → 方差 $\mathbf{s}_v,\mathbf{s}_\alpha$ → (5) $\mathcal{U}_v,\mathcal{U}_\alpha,\mu_v$ → $I_{phys}$ → 与 $\mathcal{F}$ 一并输入 (6)。

### 4.2 管线：RG 分数生成

$\mathbf{\theta}$ → (1) $S_{raw}$ → GA 最小化 (2) → $\mathbf{\theta}^*$ → (3) $S_{final}$ 对齐专家尺度。

### 4.3 管线：AutoEval-P 训练

采样 $y_{1..G}$ → (10) $R_{total}$ → (11) $A_i$ → 最大化 (12) $\mathcal{J}_{GRPO}$。

---

*文档由译文自动梳理生成，公式编号与原文 (1)–(14) 一致。*
