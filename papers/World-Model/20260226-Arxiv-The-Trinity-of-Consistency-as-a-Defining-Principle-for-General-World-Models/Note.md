# The Trinity of Consistency as a Defining Principle for General World Models

三重一致性作为通用世界模型的定义原则

上海人工智能实验室，中国科学院大学，西湖大学，新加坡国立大学

模态一致性（语义接口）、空间一致性（几何基础）和时间一致性（因果引擎）三重约束。

问题：

- 到底什么是世界模型？
- 各种机器学习、深度学习模型都或多或少学习到了世界的某些知识？
- 视频生成模型算不算世界模型？
- 像素模拟和预测 vs 满足物理约束的模拟推理
- 为什么要让大模型去学习物理规律，大模型本身就是概率的无约束的，与确定性的运动推理本性不相符
  - 是否应该考虑使用其他方法

内容摘要：

- 统一多模态模型（Unified Multimodal Model, UMM）
- 一个世界模型必须基于 **一致性三元组（Trinity of Consistency）** ：
  - **模态一致性（Modal Consistency）** 作为语义接口
    - 将异质信息（文本、图像、触觉）对齐到统一语义空间的能力，作为指令和反馈的认知接口
    - 多模态语义对齐
  - **空间一致性（Spatial Consistency）** 作为几何基础
    - 构建一个尊重几何、遮挡和物体恒存性的三维感知表示的能力，确保模拟世界的静态合理性
    - 静态模拟时 几何关系正确
  - **时间一致性（Temporal Consistency）** 作为因果引擎
    - 随时间推移遵循物理定律和因果逻辑，确保动态演化遵循可预测且逻辑合理的轨迹
    - 按照物理规律进行演化
- 理论框架的验证：**CoW-Bench（世界模型一致性基准，Consistency of World-models Benchmark）**
  - 以多帧推理和约束满足为核心的统一评估套件
  - 严格测试模型在复杂、开放场景下维持一致性三元组的能力，迫使其证明自己理解了世界，而不仅仅是知道如何描绘它

## 2 一致性的基础探索

### 2.1 通用世界模型剖析

- 三种一致性的发展～互相渗透：来自模态对齐的统一表示空间为空间几何的重建提供了语义先验，而空间一致性的三维流形则为时间演化建立了物理约束。

### 2.2 Modal Consistency

- 多模态对齐/一致性问题，本质上是一个求解**高维异构流形对齐**的问题？？什么叫“高维异构流形对齐”？流形在这里指的是什么？
- 两个基本的理论假设：
  - 柏拉图假说 **（Platonic Representation Hypothesis）**
  - 超球面几何假说 **（Hypersphere Geometry Hypothesis）**
- from direct feed-forward mapping to iterative reasoning and planning
- 从直接前馈映射（feed-forward mapping）到迭代推理与规划（iterative reasoning and planning）的认知架构演进？？

#### 2.2.1 理论基础

- 柏拉图洞穴假说：各种模态 $​\{x_{img},x_{txt}\}$ 是高维实体 $z$ 在不同低维子空间上的投影
  - 本质：联合逆投影问题，即通过观察到的影子 x 重建共享的潜在变量 z
  - 病态 ～ 熵不对称性：视觉投影 $\mathcal{P}_{img}$ 保留了大量的高频物理熵，而文本投影 $\mathcal{P}_{txt}$ 则高度抽象了离散的符号逻辑
- 超球面假说：强制特征向量均匀分布在单位超球面 $\mathbb{S}^{d-1}$ 上
- 计算范式演进，从摊销推理到测试时计算

什么是“参数内化”策略？理论瓶颈指的是？

#### 2.2.2 离散序列与连续流形

Discrete Sequences vs Continuous Manifolds

离散符号序列 vs 连续欧几里德向量场

##### **离散自回归** AR

离散自回归：试图通过统一的离散符号接口，将视觉生成转化为序列预测问题。

生成过程：通过 VQ-GAN 将连续图像量化为离散符号，随后利用 Transformer 的因果注意力掩码最大化序类对数似然？？？

> 将图像编码为离散的 token 序列，token 词表（码本）的大小是有限的。用有限大小的码本去覆盖高维连续空间时，本质是在做**向量量化**？
> 什么是 VQ-GAN？

问题：指数级漂移 + 码本坍塌

- 维度灾难：离散化过程受 **狄利克雷过程（Dirichlet process）** 支配；**随着码本维度增加，有效利用率呈指数级衰减，导致高频纹理丢失**
- 误差累积动态：

#### **连续流匹配** CM

放弃随机漫步，学习一张从噪声直接通往数据的“导航地图”（概率流速度场）。

#### 2.2.3 架构演进

几何隔离 -> 早期融合 -> 正交解耦

双塔结构 -> 适配器 -> 原生统一多模态
基于连接器的范式

##### （1）双塔结构 and 连接器范式

- 双塔结构：
  - 利用 **对比学习（contrastive learning）** 将异构模态投影到共享的超球面上
  - CLIP ALIGN
  - 在检索任务中表现出色
  - 独立编码器对图像和文本的分离处理导致了**几何拓扑上的天然不对称性**，缺乏深度的细粒度交互
- 基于连接器的范式：Frozen Visual Backbone & Lightweight Connector
  - 冻结预训练的视觉编码器，引入科学系的桥接模块，用于对齐视觉特征和 llm 的语义空间
  - 降低训练成本 建立标准架构模板

![CLIP Main Diagram](https://raw.githubusercontent.com/houhuawei23/papers-repo/main/papers/Misc/20210226-ICML2021-CLIP-Learning-Transferable-Visual-Models-From-Natural-Language-Supervision/images/main-diagrams.png)


![](https://raw.githubusercontent.com/houhuawei23/papers-repo/main/papers/Misc/20210211-Arxiv-Scaling-Up-Visual-and-Vision-Language-Representation-Learning-With-Noisy-Text-Supervision/images/align_diagram.png)


##### （2）早期融合与统一优化的挑战

##### （3）正交解耦的主流范式
