[logo5](images/logo5.png)

代码仓库（Code Repository）: [https://github.com/amap-cvlab/ABot-PhysWorld](https://github.com/amap-cvlab/ABot-PhysWorld)

 **目录（Contents）** 

- [A. 视觉-动作对齐验证（Vision-Action Alignment Verification）](#Sx1.SSx1)
- [B. 两阶段物理感知字幕生成流程（Two-Stage Physics-Aware Captioning Pipeline）](#Sx1.SSx2)
- [C. 补充定性结果（Additional Qualitative Results）](#Sx1.SSx3)

- [A. 视觉-动作对齐验证（Vision-Action Alignment Verification）](#a-vision-action-alignment-verification)
- [B. 两阶段物理感知字幕生成流程（Two-Stage Physics-Aware Captioning Pipeline）](#b-two-stage-physics-aware-captioning-pipeline)
- [C. 补充定性结果（Additional Qualitative Results）](#c-additional-qualitative-results)

### A. 视觉-动作对齐验证（Vision-Action Alignment Verification）

由传感器标定漂移、时钟同步误差或坐标系不一致引起的 **未对齐动作-视频对（misaligned action-video pairs）** 会引入虚假相关性，阻碍 **世界模型（world model）** 学习真实的物理动力学。为了检测并移除此类样本，我们将标定后的动作信号（关节位置、末端执行器笛卡尔位姿和夹爪状态）渲染为半透明的彩色编码动作图，并将其叠加到相应的视频帧上（图 [6](#Sx1.F6)）。

生成的合成图像允许人工标注者和基于 **视觉语言模型（Vision-Language Model, VLM）** 的自动验证器（Qwen3-VL）检查投影的动作轨迹是否与像素空间中观察到的机器人运动相匹配。具有显著空间偏差的片段——例如，末端执行器路径与视觉观察到的夹爪轨迹偏离，或夹爪开/闭状态与视觉证据矛盾——会被标记并丢弃。

<a id="figure-6"></a>

![logo5](images/logo5.png)

> 图 6：视觉-动作对齐验证。标定后的动作信号被渲染为半透明的动作图并叠加到视频帧上。投影轨迹与观察到的机器人运动之间存在空间偏差的样本被识别并移除。

### B. 两阶段物理感知字幕生成流程（Two-Stage Physics-Aware Captioning Pipeline）

第 [2.3](#section-2-3) 节描述了我们 **两阶段字幕生成流程（two-stage captioning pipeline）** 。在此，我们为每个阶段提供更多细节和代表性示例。

- [阶段 1：结构化感知与属性提取（Stage 1: Structured Perception and Attribute Extraction）](#stage-1-structured-perception-and-attribute-extraction)
- [阶段 2：基于物理的叙事合成（Stage 2: Physics-Grounded Narrative Synthesis）](#stage-2-physics-grounded-narrative-synthesis)

##### 阶段 1：结构化感知与属性提取（Stage 1: Structured Perception and Attribute Extraction）

一个 **视觉语言感知模块（vision-language perception module）** （Qwen3-VL 32B）处理每个视频片段，并提取结构化的物理属性：机器人形态和 **具身类型（embodiment type）** 、被操纵物体及其属性（颜色、形状、材料、尺寸）、空间布局和物体相对位置，以及整个操作序列中的 **接触事件（contact events）** 和 **状态转换（state transitions）** 。输出是一个结构化的中间表示，捕捉了场景的“是什么”和“在哪里”，为后续的写作阶段奠定了基础。阶段 1 的代表性输出如图 [7](#Sx1.F7)–[9](#Sx1.F9) 所示。

<a id="figure-7"></a>

![logo5](images/logo5.png)

> 图 7：阶段 1 字幕生成示例 1。结构化感知输出，显示了提取的物理属性、物体身份和空间关系。

<a id="figure-8"></a>

![data2](images/data2.png)

> 图 8：阶段 1 字幕生成示例 2。感知模块从视频序列中识别出机器人形态、物体属性和接触事件。

<a id="figure-9"></a>

![overview](images/overview.png)

> 图 9：阶段 1 字幕生成示例 3。在整个操作轨迹上的空间布局解析和状态转换检测。

##### 阶段 2：基于物理的叙事合成（Stage 2: Physics-Grounded Narrative Synthesis）

一个 **语言模型（language model）** （Qwen3 32B FP8）接收阶段 1 的结构化输出，并生成一个四阶段的自然语言字幕：（1） **场景设置（Scene Setup）** ——初始配置、机器人类型和物体排列；（2） **动作细节（Action Detail）** ——细粒度的操作动作，包括笛卡尔轨迹、夹爪操作和接触动力学；（3） **状态转换（State Transition）** ——物理状态变化，如物体位移、变形和包含关系；（4） **摄像机摘要（Camera Summary）** ——视点、摄像机运动和视觉构图。通过将感知与写作分离，字幕在事实层面保持基于视觉证据，同时捕捉世界模型训练所需的因果动力学。阶段 2 的代表性输出如图 [10](#Sx1.F10)–[11](#Sx1.F11) 所示。

<a id="figure-10"></a>

![logo5](images/logo5.png)

> 图 10：阶段 2 字幕生成示例 1。写作模块从结构化感知输出中合成一个四阶段的叙事，涵盖场景设置、动作细节、状态转换和摄像机摘要。

<a id="figure-11"></a>

![data2](images/data2.png)

> 图 11：阶段 2 字幕生成示例 2。基于物理的叙事合成，捕捉细粒度的操作动力学和因果状态转换。

### C. 补充定性结果（Additional Qualitative Results）

我们在三种评估设置下展示补充的定性比较。

- [在 EZSbench 上的零样本定性比较（Zero-Shot Qualitative Comparison on EZSbench）](#zero-shot-qualitative-comparison-on-ezsbench)
- [零样本测试集案例研究（Case Study on Zero-Shot Test Set）](#case-study-on-zero-shot-test-set)
- [动作到视频的定性比较（Action-to-Video Qualitative Comparison）](#action-to-video-qualitative-comparison)

##### 在 EZSbench 上的零样本定性比较（Zero-Shot Qualitative Comparison on EZSbench）

图 [12](#Sx1.F12) 显示了在  **EZSbench**  上的零样本结果。当前的视频生成基线模型在处理需要复杂逻辑推理的 **长时程操作任务（long-horizon manipulation tasks）** 时存在困难。Wan-2.5、Veo 3.1 和 WoW 未能将物体颜色属性映射到正确的目标容器，产生了放置错误。Sora v2 生成了物理上不可行的 **无接触抓取（contactless grasping）** ，而 Giga R0 和 Veo 3.1 在接触丰富的交互过程中出现“ **生成崩溃（generation collapse）** ”——末端执行器和物体几何形状完全扭曲。我们的方法正确地遵循了组合指令，并在整个长时程抓取-放置轨迹中保持了时空连贯性，没有此类伪影。

<a id="figure-12"></a>

![logo5](images/logo5.png)

> 图 12：在 EZSbench 上的零样本定性比较。基线模型表现出放置错误（Wan-2.5, Veo 3.1, WoW）、无接触抓取（Sora v2）以及在接触交互期间的几何崩溃（Giga R0, Veo 3.1）。我们的方法（底行）正确遵循组合指令并保持物理合理性。

##### 零样本测试集案例研究（Case Study on Zero-Shot Test Set）

图 [13](#Sx1.F13) 展示了在零样本测试集上针对各种未见任务的生成结果。在长时程任务“红色刀 $\rightarrow$ 红色盒子，黑色勺子 $\rightarrow$ 黑色盒子”中，模型正确地将物体属性绑定到目标容器，并执行了连续的空间推理。对于双臂毛巾折叠，它在生成协调的双臂轨迹的同时，处理了可变形物体的拓扑变化。模型还为 **铰接物体交互（articulated-object interaction）** （关门）、 **刚体放置（rigid-body placement）** （放置积木）、 **接触密集型擦拭（contact-intensive wiping）** （去除污渍）和 **物体重定位（object relocation）** （移动苹果）生成了物理一致的结果。在这些任务中，生成的视频遵循语言指令，并在长时程中保持物理合理性。

<a id="figure-13"></a>

![logo5](images/logo5.png)

> 图 13：零样本测试集案例研究。我们的模型处理了多种未见操作任务，包括多物体属性绑定、双臂协调的可变形物体操作、铰接物体交互、刚体放置、接触密集型擦拭和物体重定位——所有这些都保持了严格的物理合理性和时空连贯性。

##### 动作到视频的定性比较（Action-to-Video Qualitative Comparison）

图 [14](#Sx1.F14) 比较了 **动作条件视频生成（action-conditioned video generation, A2V）** 的结果。我们的方法生成了接触密集型操作视频，同时在整个交互过程中保持了物体几何形状和视觉完整性。相比之下，Genie-Envisioner 和 Enerverse-AC 产生了明显的物体变形、无接触抓取伪影和目标定位错误，导致输出扭曲或任务失败。

<a id="figure-14"></a>

![logo5](images/logo5.png)

> 图 14：动作到视频的定性比较。我们的方法在接触密集型操作过程中保持了物体几何形状和视觉完整性。基线模型（Genie-Envisioner, Enerverse-AC）产生了物体变形、无接触抓取和定位错误。