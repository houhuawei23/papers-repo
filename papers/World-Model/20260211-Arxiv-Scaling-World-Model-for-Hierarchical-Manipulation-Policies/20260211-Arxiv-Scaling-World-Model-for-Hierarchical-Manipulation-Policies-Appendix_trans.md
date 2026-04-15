[A-A 世界模型规划器（World Model Planner）](#a-a-world-model-planner)

- [A-B 目标条件视觉语言动作模型（Goal-conditioned VLA）](#a-b-goal-conditioned-vla)

### A-A 世界模型规划器（World Model Planner）

- [模型架构与分词（Model Architecture and Tokenization）](#model-architecture-and-tokenization)
- [序列格式化（Sequence Formatting）](#sequence-formatting)
- [模型训练（Model Training）](#model-training)

##### 模型架构与分词（Model Architecture and Tokenization）

我们构建的 **世界模型（World Model）** 总参数量为 341 亿，其中 **Transformer** 层包含 312 亿参数， **嵌入层（embedding layers）** 包含 29 亿参数。对于分词，我们采用一个包含 282,926 个词元的统一词汇表。对于视觉部分，我们使用来自 EMU3.5 [16] 的预训练 **IBQ-Tokenizer** ，其词汇表大小为 131,072，并将每个 $16\times 16$ 的图像块量化为离散词元。特别是在我们的设置中，输入被调整为 $512\times 512$ 的分辨率，从而每张图像产生 1024 个视觉词元。对于语言部分，我们使用 **Qwen3 分词器（Qwen3 tokenizer）** [1]，其词汇表大小为 151,854。

##### 序列格式化（Sequence Formatting）

我们将最大序列长度设置为 16,384，每个序列使用交错的文本和视觉词元进行格式化。在监督方面，我们仅提供指令和初始观察结果，模型需要以自回归方式预测整个交错序列。由于一个序列最多包含 16 张图像，我们在随机目标图像时间戳的 1 秒范围内采样时间戳，并让模型预测后续步骤。因此，序列可以从任意子任务阶段开始，并继续生成子任务和目标图像序列，直到任务完成或达到提示中要求的阶段数。这种滑动窗口式的操作不仅增加了数据集的大小和多样性，还实现了 **无休止的闭环生成（endless closed-loop generation）** 。

##### 模型训练（Model Training）

在训练设置方面，我们利用 **Megatron-LM** [39] 来训练世界模型， **张量并行（tensor-parallel）** 大小设置为 8， **上下文并行（context-parallel）** 大小设置为 1。全局批大小为 512，学习率为 $1\times 10^{-5}$。我们在开源的 EMU3.5 [16] 检查点上进行了 2000 步的继续训练。我们使用 128 块 Nvidia H100 GPU 对 VISTA 进行了为期 2 天的后训练，采用了带线性预热的余弦学习率调度器。

### A-B 目标条件视觉语言动作模型（Goal-conditioned VLA）

- [模型架构（Model Architecture）](#model-architecture)
- [两阶段训练设置（Two-Stage Training settings）](#two-stage-training-settings)
- [推理细节（Inference Details）](#inference-details)

##### 模型架构（Model Architecture）

我们采用类似于 $\pi_{0}$ [5] 的 **混合专家（Mixture of Experts, MoE）** 架构，以 **PaliGemma-3B** [3] 作为主干，并同时构建了一个 0.3B 规模的动作专家。我们采用 **块级因果掩码（Block-wise causal masking）** ，其中 **视觉语言模型（Vision-Language Model, VLM）** 块关注其自身特征， **本体感知块（proprioception block）** （与动作块共享权重）关注其自身特征以及 VLM 块的特征，而 **动作块（action block）** 关注所有块；每个单独的块内部是完全双向的。在每个推理步骤中，模型接收六张图像作为输入，包括三张当前观察图像和三张目标图像，这些图像通过 **SigLIP** [52] 编码为词元。它还接收当前子任务提示和代表机器人臂末端执行器 6D 位姿的本体感知信号作为输入。随后，执行一个 10 步的 **流匹配（flow matching）** 解码过程，以生成长度为 30 步的动作块。

##### 两阶段训练设置（Two-Stage Training settings）

我们为 **GoalVLA** 采用了两阶段训练范式。在第一阶段，我们利用来自 AgiBot-Beta 数据集 [9] 的 200,000 个轨迹样本，使用标注的子任务文本和目标图像作为额外的条件。此阶段包含 100,000 个训练步，全局批大小为 512，初始学习率为 $5\times 10^{-5}$。在第二阶段，我们在 737 个自行收集的机器人轨迹上对 GoalVLA 进行微调。我们设置全局批大小为 128，学习率为 $2\times 10^{-5}$，将模型训练 10 个周期（约 20k 步）。

##### 推理细节（Inference Details）

在真实机器人部署和测试阶段，为了减轻运动误差的影响并使模型能够以更细的粒度调整机器人，我们仅执行模型推断出的动作块中 30 步里的 10 步。我们采用 **闭环绝对末端执行器（End-Effector, EE）位姿控制（closed-loop absolute end-effector pose control）** 来操作机器人臂。由于模型输出的是 **增量末端执行器位姿（delta EE poses）** ，我们根据当前读取的本体感知信号计算绝对末端执行器位姿。为了更平滑的控制，我们选择第 5 步和第 10 步的绝对末端执行器位姿作为执行的目标路径点。

<a id="appendix-b"></a>

[2]\*\* 将抽象任务类型细化为详细指令，并将原始技能描述拆分为细粒度的子任务。我们在该框架中使用的提示词完整展示在以下代码块中。

![teaser](images/teaser.png)

> **图 A.1：** 我们构建的数据集样本可视化。我们的数据集由两个主要部分组成： **具身子任务与目标图像数据集（Embodied Sub-task and Goal Image Dataset）** 和 **任意到图像（Any-to-Image, X2I）数据集** 。在底部面板中，我们展示了来自 X2I 数据集的 **相机视角编辑（Camera-View Edit）** 样本和其他通用编辑样本。

作为一种基础的 **单步生成任务（single-step generation task）** ， **X2I 生成（X2I generation）** 通常涉及由文本和任意数量图像组成的任意多模态交错输入，要求模型输出单张图像作为响应。这对模型的 **关键能力（key capabilities）** 提出了重大要求，特别是在 **多模态指令跟随（multimodal instruction following）** 、保持 **主体/背景一致性（subject/background consistency）** 、遵守 **世界知识与规则（world knowledge and rules）** 以及控制 **图像风格与纹理（image style and texture）** 方面。掌握这些具有挑战性的 X2I 能力将促进模型向更通用的 **任意到任意（Any-to-Any, X2X）生成范式** 稳健演进，从而使其进化为更复杂、更强大的 **世界模型（world model）** 。为此，我们构建了一个大规模的 X2I 数据集（包含 152 亿词元）用于训练，以克服现有开源数据在 **多样性（diversity）** 、 **质量（quality）** 和 **规模（scale）** 方面的限制。内部 X2I 数据集还整合了多个开源数据集的部分内容，包括 SEED-Data-Edit [22]、WeatherStream [53]、PromptFix [51]、OmniGen-X2I [45]、ShareGPT-4o-Image [11]、ImgEdit [50]、OmniGen2-X2I2 [43]、MultiRef [12] 和 GPT-IMAGE-EDIT-1.5M [42]。为了增强 **空间理解（spatial understanding）** 和 **多视角一致性（multi-view consistency）** ，我们进一步使用 **mast3r [30]** 标注了同一场景拍摄的多张图像的相对位置，并构建了一个 **相机视角编辑数据集（camera-view edit dataset）** 。

用于子任务文本分割与标注的提示词：

[A.4](https://arxiv.org/html/2602.10983v2#A4.F4) 中展示了域内场景的设置。我们训练数据的分布显示在图 [A.4](https://arxiv.org/html/2602.10983v2#A4.F4) 的左侧。训练场景包含五个固定物体，训练集由五个任务（即，将其中一个物体放到盘子上）组成。每个任务包含大约 150 条轨迹，总计 737 条轨迹。可以看出，我们的训练数据在物体放置和场景构成方面都高度同质。在图 [A.4](https://arxiv.org/html/2602.10983v2#A4.F4) 的右侧，我们展示了测试中使用的 **未见干扰物（unseen distractors）** 和 **未见目标（unseen targets）** 的设置。对于未见干扰物设置，我们首先与训练集中存在的物体放置布局对齐，然后用未见物体替换 $1–3$ 个干扰物。在此设置中，目标物体的位置保持不变，因此主要评估模型对干扰物物体的鲁棒性。对于未见目标设置，我们也与训练集中的物体放置布局对齐，然后用未见物体替换目标物体。在此设置中，目标物体的位置也是固定的，主要评估模型对未见物体的 **接地（grounding）** 和抓取的泛化能力。

图 [A.5](https://arxiv.org/html/2602.10983v2#A4.F5) 展示了 **新场景（novel scenarios）** 的设置，其中包括 21 个未见物体、三种不同图案的桌布和未见颜色的盘子，总共形成了 63 个用于评估的新场景。鉴于训练数据中物体放置位置的多样性不足，我们在新场景的设置中始终将目标物体放置在机械臂可达且可抓取的范围内，同时在其具体位置上引入随机性，以确保与训练集有所区别。

![teaser](images/teaser.png)

> 图 A.4：训练数据集及域内场景设置的可视化。

![method](images/method.png)

> 图 A.5：新场景设置的可视化。

<a id="appendix-e"></a>

## 附录 E 新场景生成目标图像的可视化

我们在图 [A.6](https://arxiv.org/html/2602.10983v2#A5.F6) 中展示了 **VISTA** 在新场景上生成的更多目标图像。可以观察到，生成的目标图像几乎都是 **时序准确（timing-accurate）** 的，有效地捕捉了拾取和放置的关键时刻，从而为物体抓取和放置提供了精确的空间位置指导。得益于 **Emu3.5** 强大的图像生成能力，VISTA 即使在面对先前未见过的场景时，也能在背景和物体外观上保持高度一致性。此外，在 **大规模跨具身机器人数据集（large-scale cross-embodiment robotic dataset）** 上进行微调后，VISTA 展现出卓越的 **多视角一致生成（multi-view consistent generation）** 能力。如图 [A.6](https://arxiv.org/html/2602.10983v2#A5.F6) 第四行所示，在生成的黄瓜放置图像中，从左手腕摄像头可以清晰地看到盘子里的黄瓜和右臂的夹爪，它们的相对空间位置几乎完全一致。这种强大的多视角空间一致性使得 VISTA 能够为 **目标条件视觉语言动作模型（goal-conditioned VLA）** 提供清晰可靠的空间关系线索，从而增强其感知物体空间位置的能力。

此外，我们发现我们的模型能够为不同的物体放置配置生成相应的合理抓取模式。如图 [A.6](https://arxiv.org/html/2602.10983v2#A5.F6) 最后一行所示，当为抓取苏打水瓶生成目标图像时，夹爪采用了水平向前抓取，而不是其他物体常见的垂直向下抓取。这种合成多样化且针对特定物体的抓取策略的能力对于处理广泛的未见物体至关重要。随着更多机器人操作数据集的加入，我们的模型有望为各种复杂的操作任务生成合理且关键的目标图像。

![teaser](images/teaser.png)

> 图 A.6：新场景生成目标图像的可视化。

<a id="appendix-f"></a>

## 附录 F 涌现能力分析

我们进一步研究了 VISTA 在生成目标图像方面的泛化边界。图 [A.7](https://arxiv.org/html/2602.10983v2#A6.F7)、[A.8](https://arxiv.org/html/2602.10983v2#A6.F8) 和 [A.9](https://arxiv.org/html/2602.10983v2#A6.F9) 展示了 VISTA 生成的 **组合任务（compositional tasks）** 、 **空间理解任务（spatial understanding tasks）** 和 **语义理解任务（semantic understanding tasks）** 的定性可视化结果。

对于组合任务，VISTA 能够准确解释给定的指令，并顺序生成操作多个物体的目标图像，同时保持物体位置的时空一致性，如图 [A.7](https://arxiv.org/html/2602.10983v2#A6.F7) 所示。这些结果表明，我们的方法有潜力扩展到更复杂的 **长视野操作任务（long-horizon manipulation tasks）** ，并支持子任务的灵活组合。此外，这种组合式视觉指导使得目标条件 VLA 能够实现操作技能的可控组合。

对于空间理解任务，VISTA 展示了理解涉及简单空间关系指令的能力。如图 [A.8](https://arxiv.org/html/2602.10983v2#A6.F8) 所示，VISTA 正确解释了放置目标和抓取目标的空间描述。此外，我们观察到 VISTA 可以泛化到新的任务类型，例如“将雪碧放在芒果附近”，其中生成的目标图像显示机器人根据指令将物体放置在盘子外部。

对于语义理解任务，VISTA 同样基于语义线索识别被操作物体和放置目标，并生成相应的目标图像。图 [A.9](https://arxiv.org/html/2602.10983v2#A6.F9) 展示了几个代表性示例，包括描述盘子形状和颜色的指令，以及需要对桌面上显示的图片进行语义理解的更具挑战性的案例。

我们将这些指令跟随能力归因于 Emu3.5 强大的 **文生图（text-to-image generation）** 和编辑能力。在机器人数据集上微调后，我们发现一部分这种指令跟随能力得以保留，尽管仍然有限且通常表现出 **幻觉（hallucinations）** 。我们相信，随着更多样化的机器人数据，VISTA 有巨大潜力在这些任务上进一步提高其生成质量。

通过利用 **世界模型（world model）** 的指令理解能力， **GoalVLA** 可以更专注于动作生成和目标图像跟随，从而缓解其在指令理解方面的局限性。我们在图 [A.10](https://arxiv.org/html/2602.10983v2#A6.F10)、[A.11](https://arxiv.org/html/2602.10983v2#A6.F11) 和 [A.12](https://arxiv.org/html/2602.10983v2#A6.F12) 中展示了几个真实世界执行示例。如图所示，通过将具有更强指令理解能力的世界模型集成进来，VLA 的操作能力可以进一步提升，不仅能够实现操作技能的灵活组合，还能执行需要指令级推理的任务。

![teaser](images/teaser.png)

> 图 A.7：VISTA 在组合任务上生成序列的可视化。

![method](images/method.png)

> 图 A.8：VISTA 在空间理解任务上生成序列的可视化。

![robot_dataset_v3.001](images/robot_dataset_v3.001.jpeg)

> 图 A.9：VISTA 在语义理解任务上生成序列的可视化。

![real_illustrate_single_col](images/real_illustrate_single_col.png)

> 图 A.10：组合任务的目标条件执行轨迹可视化。对于重置阶段，我们的 GoalVLA 通过语言指令进行训练，无需目标图像。

![bench_gen_short.001](images/bench_gen_short.001.jpeg)

> 图 A.11：空间理解任务的目标条件执行轨迹可视化。

![real_ood_all_mixture_new](images/real_ood_all_mixture_new.png)

> 图 A.12：语义理解任务的目标条件执行轨迹可视化。

<a id="appendix-g"></a>

## 附录 G 执行结果的可视化与分析

我们可视化了更多的执行结果，如图 [A.13](https://arxiv.org/html/2602.10983v2#A7.F13) 所示。通过比较每个子任务阶段的最后一帧与其对应的目标图像，我们观察到机械臂姿态在很大程度上与目标图像中显示的姿态一致。这表明 GoalVLA 能够准确捕捉目标图像提供的与机器人空间配置相关的视觉特征，然后稳健地生成相应的 **动作块（action chunks）** 以达到指定位置并执行期望的操作，即使在存在显著视觉干扰的新场景中也是如此。

![teaser](images/teaser.png)

> 图 A.13：目标条件执行轨迹的可视化。

我们还分析了在执行过程中观察到的几个失败案例。我们识别出由目标图像生成质量不佳引起的两类失败。第一类源于生成的目标图像时序不准确，图像未能精确对应拾取或放置的关键时刻。如图 [A.14](https://arxiv.org/html/2602.10983v2#A7.F14) 左下角示例所示，目标图像生成于物体抓取之前的时刻，此时机械臂尚未充分下降且夹爪尚未完全包围物体。在执行过程中，GoalVLA 过度专注于匹配目标图像中描绘的机器人姿态，导致向下运动不足并最终抓取失败。这表明 GoalVLA 严重依赖目标图像提供的准确抓取位置线索，而在适应在线观察到的实际执行状态方面能力有限。图 [A.14](https://arxiv.org/html/2602.10983v2#A7.F14) 左上角示例显示了另一个失败案例，其中目标图像对应的是物体已被抓取并轻微抬起后的时刻。在这种情况下，目标图像未能为抓取阶段提供精确的空间指导，导致位置偏差和抓取失败。为了减轻由目标图像时序不精确引起的失败，我们在 GoalVLA 的训练中采用了 **随机目标图像偏移（random goal image offset）** ，这可以部分缓解此问题。然而，在某些情况下，GoalVLA 的表现仍然欠佳。我们将对此方面的进一步改进留待未来工作。

第二类失败源于生成的目标图像中的 **空间错位（spatial misalignment）** 。如图 [A.14](https://arxiv.org/html/2602.10983v2#A7.F14) 右上角示例所示，来自左手腕摄像头的目标图像与目标物体未完全对齐，略微向左偏移，导致执行过程中左夹爪与物体碰撞，造成抓取失败。类似地，在图 [A.14](https://arxiv.org/html/2602.10983v2#A7.F14) 右下角示例中，来自左手腕摄像头的目标图像表现出类似的向左偏移，再次导致执行失败。在这种情况下，GoalVLA 需要平衡来自目标图像的指导与当前的视觉观察，而不是完全依赖目标图像来生成动作。实现这种平衡需要更精细的网络设计，我们计划在未来工作中进行探索。

最后，受限于训练数据的多样性，即使目标图像本身是准确生成的，GoalVLA 也难以准确跟随那些指定目标位置明显超出训练分布的目标图像。我们认为，增加训练数据的空间多样性对于提高 GoalVLA 的泛化能力至关重要，我们将在未来工作中探索使用更多样化的数据集进行训练。

![method](images/method.png)

> 图 A.14：失败执行轨迹的可视化。左侧显示了由生成的目标图像时序不准确引起的失败，右侧显示了由生成的目标图像空间错位引起的失败。
