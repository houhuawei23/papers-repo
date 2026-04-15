# Title: WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models

- ArXiv: 2602.08971
- Authors: Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, Chen Gao, Wei Wu, Xihui Liu, Dhruv Shah, Zhaoxiang Zhang, Zhibo Chen, Jun Zhu, Yonghong Tian, Tat-Seng Chua, Wenwu Zhu, Yong Li
- Sections: 44
- Estimated tokens: 44.8k

## Contents

- 1 Introduction
- 2 Related Works
  - 2.1 Embodied World Models
  - 2.2 World Model Benchmarks
- 3 The WorldArena Benchmark
  - 3.1 Video Quality Evaluation
    - 3.1.1 Visual Quality
    - 3.1.2 Motion Quality
    - 3.1.3 Content Consistency
    - 3.1.4 Physics Adherence
    - 3.1.5 3D Accuracy
    - 3.1.6 Controllability
  - 3.2 Embodied Task Evaluation
  - 3.3 Human Evaluation
  - 3.4 EWMScore Metric
- 4 Experiments
  - 4.1 Experimental Setup
  - 4.2 Results
    - 4.2.1 Visual Quality Evaluation
    - 4.2.2 Embodied Task Evaluation
    - 4.2.3 Human Evaluation
  - 4.3 Inter-metric Analysis
- 5 Conclusion and Future Work

## Abstract

###### Abstract

While world models have emerged as a cornerstone of embodied intelligence by enabling agents to reason about environmental dynamics through action-conditioned prediction, their evaluation remains fragmented. Current evaluation of embodied world models has largely focused on perceptual fidelity (e.g., video generation quality), overlooking the functional utility of these models in downstream decision-making tasks. In this work, we introduce WorldArena, a unified benchmark designed to systematically evaluate embodied world models across both perceptual and functional dimensions. WorldArena assesses models through three dimensions: video perception quality, measured with 16 metrics across six sub-dimensions; embodied task functionality, which evaluates world models as data engines, policy evaluators, and action planners integrating with subjective human evaluation. Furthermore, we propose EWMScore, a holistic metric integrating multi-dimensional performance into a single interpretable index. Through extensive experiments on 14 representative models, we reveal a significant perception–functionality gap, showing that high visual quality does not necessarily translate into strong embodied task capability. WorldArena benchmark with the public leaderboard is released at https://world-arena.ai, providing a framework for tracking progress toward truly functional world models in embodied AI.

Equal contribution^\* Equal contribution. Equal contribution^$\ddagger$ Equal contribution. Equal contribution^$\S$ Project lead.

<a id="section-1"></a>

## 1 Introduction

![EWMScore_14models](images/EWMScore_14models.png)

> (a)

In recent years, world models (Ding et al., 2025; Kong et al., 2025; Zhu et al., 2024b) have emerged as a foundational component of embodied intelligence. A world model (WM) learns to predict future environment states conditioned on current observations and actions, enabling agents to reason about dynamics and interaction outcomes. Embodied World Model (EWM) forecasts future states based on robot actions and external instructions, effectively functioning as a mental simulator guiding robot action planning and decision-making, or an environment proxy to support scalable robotic training and evaluation (Shang et al., 2025a; Long et al., 2025). Unlike general-purpose video generation models, EWMs must capture not only perceptual fidelity but also physically grounded, action-consistent dynamics that are critical for downstream embodied tasks.

However, existing evaluation protocols suffer from significant limitations. First, they lack comprehensive evaluations oriented toward embodied tasks, including the role of world models as environment proxies and embodied agents. Current benchmarks (Yue et al., 2025; Li et al., 2025a; Lu et al., 2025) mainly focus on video-level quality metrics, which fail to reflect the real-world value of embodied world models for practical embodied applications, as shown in the comparison of Table [1](#table-1). Although some recent studies (Qin et al., 2024; Zhang et al., 2025; Fan et al., 2026) evaluate embodied world models through closed-loop action execution, broader embodied capabilities such as their roles as synthetic data engines or tools for policy evaluation remain largely unassessed. Second, the coverage of evaluated models is insufficient. Most existing benchmarks (Yue et al., 2025; Fan et al., 2026) focus on general text-conditioned video generation models (Wan et al., 2025; Yang et al., 2024b), while many recent robot-specialized world models (Chi et al., 2025; Team et al., 2025; Liao et al., 2025; Zhen et al., 2025; Guo et al., 2025), have received little systematic attention and remain largely unevaluated.

To bridge this gap, we present WorldArena, the first embodied world model benchmark that integrates perceptual and three functional evaluations, combining both objective and subjective assessments. WorldArena provides a holistic evaluation framework across three complementary aspects: (1) multi-faceted video quality, comprising 16 numerical metrics across 6 key sub-dimensions, including visual quality, motion quality, content consistency, physics adherence, 3D accuracy, and controllability; (2) embodied task utility, which evaluates model performance in data synthesis, policy evaluation, and action planning; and (3) human evaluation, which complements automated metrics by capturing qualitative aspects of model behavior that are difficult to quantify, such as physical plausibility and instruction adherence. Additionally, we introduce EWMScore, a unified metric that combines multi-dimensional metrics into a single index, offering a comprehensive assessment of embodied world models’ generative performance. An overview of the evaluation result is shown in Figure [1](#figure-1).

For the evaluation data, we select bimanual robotic manipulation as a representative embodied scenario and conduct evaluations based on the RobotTwin 2.0 dataset (Chen et al., 2025), which covers 50 diverse robotic scenarios, ensuring both scenario diversity and evaluation reliability. We perform a unified evaluation on 14 representative world models, including both general video generation world models and specialized embodied world models. The results reveal a significant gap between visual fidelity and embodied task performance, indicating that current visual quality has not yet reached the level required to effectively support embodied tasks. Overall, our main contributions can be summarized as follows:

- We introduce the first comprehensive benchmark tailored for embodied world models, enabling a unified evaluation of their perceptual and functional capabilities.
- We propose EWMScore, a unified objective metric for embodied world models, and conduct extensive human studies to validate its effectiveness. Results demonstrate that EWMScore highly aligns with subjective judgment, serving as a reliable and interpretable index.
- We conduct a systematic evaluation of 14 representative embodied world models and provide a multi-dimensional analysis of their strengths and limitations, offering insights and guidance for future research.

<a id="section-2"></a>

## 2 Related Works

<a id="table-1"></a>

> Table 1: Comparison of existing world model benchmarks and WorldArena across three key evaluation dimensions.

| Benchmark                           | Video Quality | Embodied Tasks | Human        |              |              |              |              |              |              |              |
| ----------------------------------- | ------------- | -------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| Benchmark                           | Video Quality | Embodied Tasks | Human        |              |              |              |              |              |              |              |
| Benchmark                           | Video Quality | Embodied Tasks | Human        |              |              |              |              |              |              |              |
| Benchmark                           | Video Quality | Embodied Tasks | Human        |              |              |              |              |              |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
| Visual                              | Motion        | Content        | Physics      | Control      | 3D           | Data         | Policy       | Action       |              |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
|                                     | Quality       | Quality        | Consist.     | Adher.       | ability      | Acc.         | Engine       | Eval.        | Planner      |              |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldModelBench (Li et al., 2025a)  | $\times$      | $\times$       | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldSimBench (Qin et al., 2024)    | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldScore (Duan et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| 4DWorldBench (Lu et al., 2025)      | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| EWMBench (Yue et al., 2025)         | $\times$      | $\checkmark$   | $\checkmark$ | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| WorldEval (Li et al., 2025b)        | $\times$      | $\times$       | $\times$     | $\times$     | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| World-in-World (Zhang et al., 2025) | $\checkmark$  | $\checkmark$   | $\times$     | $\times$     | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\times$     |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WoW-World-Eval (Fan et al., 2026)   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$     | $\times$     | $\times$     | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| WorldArena (Ours)                   | $\checkmark$  | $\checkmark$   | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |

<a id="section-2-1"></a>

### 2.1 Embodied World Models

Embodied world models are generative models that predict future observations of physical scenes involving robot locomotion and manipulation. These models can be broadly categorized into three types: video generation-based models (Liao et al., 2025; Shang et al., 2025b; Team et al., 2025), 3D reconstruction-based models (Huang et al., 2025; Qian et al., 2025), and latent-space world models (Assran et al., 2025; Liu and Chen, 2025; Hafner et al., 2025). In practice, embodied world models serve three key roles: (1) as data synthesis engines (Jang et al., 2025) for generating video-action sequences to augment robot policy training; (2) as policy evaluation environments (Shang et al., 2025b; Li et al., 2025b) for scalable virtual testing through policy-world model interaction; and (3) as action planners (Hu et al., 2024; Fan et al., 2026), where predicted states are decoded into executable actions for robot control. Given the diversity of model paradigms and functional roles, embodied world models are inherently challenging to evaluate comprehensively, underscoring the need for a holistic benchmark that systematically assesses them across both perceptual and functional dimensions, driving their future development.

<a id="section-2-2"></a>

### 2.2 World Model Benchmarks

Existing benchmarks for world models can be broadly categorized into general-purpose and embodied benchmarks. General-purpose benchmarks (Duan et al., 2025; Li et al., 2025a; Lu et al., 2025) primarily evaluate world models from a perceptual and generative perspective, focusing on video quality aspects such as visual fidelity, motion realism, content consistency, and, in some cases, physical plausibility or geometric consistency. While these benchmarks are effective for standardizing generative evaluation, they largely treat world models as video generators and do not assess their functional roles in decision-making or interaction. More recent embodied world model benchmarks (Yue et al., 2025; Li et al., 2025b; Zhang et al., 2025; Fan et al., 2026) extend evaluation to controllability, action conditioning, and limited closed-loop interaction. However, existing embodied benchmarks remain limited in scope, often focusing on a single embodied role and predominantly targeting text-conditioned video models, with insufficient coverage of action-conditioned and robot-centric world models. Moreover, most existing benchmarks evaluate fewer than ten models, which further limits the scope and comprehensiveness. In contrast, WorldArena provides a unified benchmark that systematically evaluates embodied world models across both perceptual and functional dimensions, integrating objective metrics with human subjective assessments.

<a id="section-3"></a>

## 3 The WorldArena Benchmark

The evaluation framework of WorldArena consists of three key components. First, we assess video quality from 6 dimensions with 16 metrics, focusing on the world model’s open-loop prediction ability (Section [3.1](#section-3-1)). Second, we evaluate the world model’s closed-loop performance across 3 typical embodied downstream tasks (Section [3.2](#section-3-2)). Third, to complement objective measurements with subjective judgment, we collect human annotations to assess qualitative aspects of model performance (Section [3.3](#section-3-3)). Finally, we integrate multi-dimensional video metrics into an interpretable index EWMScore to reflect overall performance (Section [3.4](#section-3-4)).

<a id="section-3-1"></a>

### 3.1 Video Quality Evaluation

We begin by evaluating the quality of the videos generated by different embodied world models, considering 16 video metrics across six sub-dimensions, as shown in Figure [1](#figure-1) (b). The detailed metric explanations can be found in Appendix [A](#appendix-a) and the case visualization is shown in Appendix [C](#appendix-c).

#### 3.1.1 Visual Quality

Visual quality assesses whether generated videos are perceptually reliable for embodied scenarios, considering low-level fidelity, perceptual appeal, and similarity to real data. We evaluate it using three metrics:

Image Quality measures clarity and sharpness of frames using the MUSIQ (Ke et al., 2021) model, which detects distortions such as overexposure, noise, and compression artifacts (Huang et al., 2024). Higher scores indicate cleaner and more coherent images.

Aesthetic Quality evaluates the visual appeal of the video, considering lighting and color composition. Using the LAION aesthetic predictor (LAION-AI, 2022), we map frames to an aesthetic feature space and derive an average score (Huang et al., 2024), capturing both perceptual consistency and artistic quality.

JEPA Similarity quantifies similarity between feature distributions extracted by the pretrained V-JEPA encoder (Bardes et al., 2023), using maximum mean discrepancy (MMD) with a second-order polynomial kernel (Luo et al., 2024). Higher values indicate greater similarity to the ground-truth video.

#### 3.1.2 Motion Quality

Motion quality reflects whether a model captures physically meaningful and temporally coherent dynamics. We assess both the strength of motion and its temporal continuity. To this end, we introduce the following three metrics:

Dynamic Degree quantifies the motion intensity within the video. Using the RAFT (Teed and Deng, 2020) optical flow model, we extract motion vector fields between consecutive frames and focus on the top 5% of active pixels (Huang et al., 2024). A higher dynamic degree score indicates more pronounced and meaningful movement in the video, capturing the intensity of motion in key areas such as robotic arm gestures.

Flow Score measures the overall intensity of motion across the video by averaging optical flow magnitudes over time (Liu et al., 2023). This score reflects the degree of dynamic interaction, where a higher value indicates greater motion intensity and more physically meaningful dynamics throughout the video.

Motion Smoothness evaluates the temporal coherence of motion, assessing whether movements between consecutive frames are smooth and consistent with physical inertia. Using a frame interpolation model (Zhang et al., 2024), we predict intermediate frames and compare them to real frames (Duan et al., 2025). This approach incorporates motion magnitude as a weighting factor to prevent overestimating static backgrounds and ensure rapid motion sequences are not unfairly penalized.

#### 3.1.3 Content Consistency

Content consistency measures the stability of objects and scenes throughout the video, evaluated at both semantic and appearance levels using three metrics:

Subject Consistency assesses object consistency across frames by calculating cosine similarity between DINO (Caron et al., 2021) features from the first, current, and previous frames (Huang et al., 2024). Higher similarity scores indicate better consistency.

Background Consistency evaluates scene stability using CLIP (Radford et al., 2021) features, measuring cosine similarity between the current frame and the first and previous frames to assess scene stability (Huang et al., 2024).

Photometric Consistency measures pixel-level texture stability by calculating the average end-point error (AEPE) using optical flow (Duan et al., 2025). A higher AEPE indicates poorer alignment, while a higher score reflects better consistency.

#### 3.1.4 Physics Adherence

Physics adherence evaluates whether generated behaviors conform to real-world physical constraints rather than merely appearing visually plausible. We therefore assess both local interaction realism and global motion correctness with the following two metrics:

Interaction Quality evaluates the physical plausibility of interactions between the robot and objects. We use Qwen3-VL (Bai et al., 2025a) to assess factors such as contact behavior and force transmission, checking whether the interactions are physically realistic. The interaction quality score is based on a 1–5 scale, normalized to [0,1], showing how well the robot’s actions align with expected physical behaviors.

Trajectory Accuracy quantifies the accuracy of the robotic arm’s grasping trajectory. Using the SAM 3 (Carion et al., 2025) model, we extract bounding boxes for the arm in each frame and compute the normalized dynamic time warping (NDTW) distance to evaluate alignment with the ground-truth trajectory (Yue et al., 2025). A higher score reflects better spatial-temporal alignment and more accurate trajectory prediction.

#### 3.1.5 3D Accuracy

3D accuracy assesses whether generated videos preserve real-world spatial structure beyond image appearance. We evaluate geometric consistency and perspective plausibility with the following two metrics:

Depth Accuracy evaluates whether the generated video preserves real-world spatial geometry by comparing depth maps between the generated and ground-truth videos. We use monocular depth estimation and apply a median-based scaling strategy to address scale ambiguity. A higher depth accuracy score indicates better geometric consistency with the real-world scene.

Perspectivity evaluates the 3D plausibility of the video, focusing on factors such as scale variation with depth, lighting consistency, and occlusion relationships. We use Qwen3-VL as a judge to assess the perspective, judging whether the video adheres to realistic 3D geometry. A higher score reflects better perspective alignment with real-world scenes.

#### 3.1.6 Controllability

Controllability measures the model’s ability to respond to external instructions. We evaluate whether generated videos align with intended actions and instructions using three metrics:

Instruction Following assesses the model’s accuracy in following instructions regarding action type, target object, and task state, measured by a VLM-based judge (Qwen3-VL) and scores normalized to [0,1].

Semantic Alignment measures how well the generated video matches the semantic meaning of the instruction by computing cosine similarity between Qwen2.5-VL-generated (Bai et al., 2025b) descriptions of the generated and reference videos.

Action Following evaluates video diversity in response to different instructions. For a given initial frame, we automatically generate three distinct instructions and then use the world model to generate corresponding videos. The diversity score is the average pairwise feature dissimilarity, with higher values indicating greater diversity.

<a id="figure-2"></a>

![combined_radar_charts](images/combined_radar_charts.png)

> Figure 2: Illustrations of the video quality evaluations across six dimensions: visual quality, motion quality, content consistency, physics adherence, 3D accuracy, and controllability.

<a id="section-3-2"></a>

### 3.2 Embodied Task Evaluation

In this section, we evaluate the capabilities of world models through three embodied tasks, as illustrated in Figure [3](#figure-3).

Embodied Data Engine. World models can generate future observations based on external instructions, enabling synthetic data generation to supplement training data for downstream embodied policy models and alleviate the scarcity of real-world data. In this part, we treat world models as embodied data synthesis engines and evaluate their performance by measuring the gain they provide to policy models. We employ a two-phase training procedure. In the first phase, we fine-tune the world model on the RobotTwin 2.0 dataset and generate synthetic videos conditioned on the first frame and external instructions. In the second phase, we freeze the world model’s weights and integrate an inverse dynamics model (IDM) to extract actions from video features. Specifically, we follow the VPP (Hu et al., 2024) design of the diffusion policy head, guiding an action denoising head with intermediate world model features for action prediction. This process produces paired video-action sequences. We then evaluate the impact of world model–generated synthetic data by training a baseline $\pi_{0.5}$ (Intelligence et al., 2025) policy model with varying amounts of synthetic data. The performance gain of the policy model reflects the world model’s capability to enhance policy learning.

Embodied Policy Evaluator. In this section, we assess the capability of world models as environment proxies for evaluating policy performance. We train a series of policy models ($\pi_{0.5}$) with varying capabilities using the RoboTwin 2.0 dataset. These models are evaluated by interacting with an action-controllable world model, generating observation videos through a rollout process that continues until it exceeds 20% more frames than the corresponding ground truth video. Task success is evaluated using a VLM, which determines whether the embodied task was executed successfully. The used prompt for the VLM is shown in Appendix [B](#appendix-b). The success rate from the world model’s evaluation is compared to that from the RoboTwin simulator. A high correlation between the two suggests effective simulation of real-world dynamics, while a low correlation indicates a mismatch in environmental transition simulation.

Embodied Action Planner. By predicting future state transitions, world models can function as the action-planning ”brain” of an embodied agent. In this part, we investigate the ability of world models to execute embodied tasks in a closed-loop manner. Similar to the data synthesis engine setup, we pair the world model with an inverse dynamics model, where the world model takes textual instructions and the initial frame as input and outputs the corresponding action sequence for future operations. This sequence is then executed in the RoboTwin simulator, and the task success rate is measured to evaluate the world model’s performance in closed-loop action execution.

<a id="figure-3"></a>

![video_metric_part2](images/video_metric_part2.png)

> Figure 3: Overview of the embodied task evaluation systems, including the assessment of world models as embodied data engines (measuring success rate of trained downstream policies), policy evaluators (measuring correlation between world model and real-world evaluation results), and action planners (measuring success rate of world model-based policies).

<a id="section-3-3"></a>

### 3.3 Human Evaluation

Since video quality metrics alone cannot fully capture aspects like physical plausibility and instruction adherence, we incorporate two types of human evaluations. The first type involves scoring three key dimensions: overall video quality, instruction following, and physical adherence on a 1 to 5 scale, then normalizing to a 0-100 range. The second type is a head-to-head comparison, where annotators choose the superior video generated by two different models from the same prompt, yielding a win-rate metric. We recruited 70 annotators who evaluated a total of 3500 videos.

<a id="section-3-4"></a>

### 3.4 EWMScore Metric

After computing the 16 video quality metrics spanning six perceptual dimensions, we apply a linear normalization based on empirically defined metric boundaries to map all scores into the range, and subsequently scale them to [0,100]. We then compute the arithmetic mean across all normalized metrics to obtain a single composite score, referred to as EWMScore. EWMScore serves as an objective and automated metric for assessing the overall generative quality of embodied world models.

<a id="section-4"></a>

## 4 Experiments

<a id="section-4-1"></a>

### 4.1 Experimental Setup

Dataset. We focus on robotic manipulation scenarios, using the RoboTwin 2.0 (Chen et al., 2025) dataset and simulator for evaluation, which includes 50 task scenarios and 2500 videos. For video quality evaluation, we use 2000 videos to train the world model and 500 videos for testing. For the embodied data engine task, we train the $\pi_{0.5}$ policy model with 10%, 20%, 30%, 50%, and 100% of the data, resulting in a series of policy models with varying performance. For policy evaluation and action planning tasks, we conduct evaluations within the RoboTwin simulator environment.

Tested Models. We evaluate 14 representative world models, covering both general-purpose video world models and embodied-specific models. The evaluated general video world models include CogvideoX (Yang et al., 2024b), Wan 2.2 (Wan et al., 2025), Wan 2.6 (Wan et al., 2025), and Veo 3.1(^1^11https://aistudio.google.com/models/veo-3). The text-conditioned embodied world models consist of Genie Envisioner (Liao et al., 2025), GigaWorld (Team et al., 2025), TesserAct (Zhen et al., 2025), Cosmos-Predict 2.5 (Gu, 2025), WOW (Chi et al., 2025), RoboMaster(^2^22https://huggingface.co/datasets/robomaster2025/RoboMaster) , Cosmos-Predict 2.5 (text) (Gu, 2025), and Vidar (Feng et al., 2025). In addition, we include action-conditioned embodied world models, namely IRASim (Zhu et al., 2024a), Cosmos-Predict 2.5 (action) (Gu, 2025) and CtrlWorld (Guo et al., 2025). For fair comparison, all models with available training code are post-trained on the used dataset following their official implementations.

<a id="table-2"></a>

> Table 2: Video quality evaluation results across visual quality, motion quality and content consistency dimensions.

| Models                      | Visual Quality    | Motion Quality  | Content Consistency |            |                   |                  |                     |                      |        |
| --------------------------- | ----------------- | --------------- | ------------------- | ---------- | ----------------- | ---------------- | ------------------- | -------------------- | ------ |
| Models                      | Visual Quality    | Motion Quality  | Content Consistency |            |                   |                  |                     |                      |        |
| Models                      | Visual Quality    | Motion Quality  | Content Consistency |            |                   |                  |                     |                      |        |
| Models                      | Visual Quality    | Motion Quality  | Content Consistency |            |                   |                  |                     |                      |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| Image Quality               | Aesthetic Quality | JEPA Similarity | Dynamic Degree      | Flow Score | Motion Smoothness | Subject Consist. | Background Consist. | Photometric Consist. |        |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| GigaWorld-0                 | 0.5041            | 0.3991          | 0.4413              | 0.6709     | 0.3118            | 0.7811           | 0.7303              | 0.8563               | 0.1756 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| Genie Envisioner            | 0.2305            | 0.3289          | 0.3340              | 0.6930     | 0.0855            | 0.6966           | 0.7760              | 0.9024               | 0.2006 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| TesserAct                   | 0.3322            | 0.4590          | 0.4579              | 0.5150     | 0.2447            | 0.7579           | 0.8250              | 0.9238               | 0.2491 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| RoboMaster                  | 0.3487            | 0.3842          | 0.2966              | 0.6124     | 0.1484            | 0.6940           | 0.8295              | 0.9123               | 0.3356 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Vidar                       | 0.4145            | 0.4068          | 0.5608              | 0.2767     | 0.1426            | 0.7973           | 0.7629              | 0.8300               | 0.2350 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (text)   | 0.6668            | 0.4501          | 0.3126              | 0.5911     | 0.4302            | 0.7882           | 0.7488              | 0.8511               | 0.1383 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| Cosmos-Predict 2.5 (action) | 0.4489            | 0.3576          | 0.9296              | 0.3994     | 0.0573            | 0.7100           | 0.8197              | 0.8894               | 0.3528 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| WoW                         | 0.4587            | 0.3868          | 0.7440              | 0.4608     | 0.2706            | 0.7692           | 0.8161              | 0.9025               | 0.2170 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| CtrlWorld                   | 0.3522            | 0.3893          | 0.9185              | 0.4257     | 0.3449            | 0.7377           | 0.8411              | 0.9057               | 0.1729 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| Wan 2.2                     | 0.3884            | 0.3963          | 0.7575              | 0.4349     | 0.1269            | 0.7019           | 0.8388              | 0.9042               | 0.4776 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| CogvideoX                   | 0.3582            | 0.3777          | 0.9384              | 0.3166     | 0.2189            | 0.7391           | 0.8083              | 0.8773               | 0.3580 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| IRASim                      | 0.3489            | 0.3623          | 0.9330              | 0.4139     | 0.2083            | 0.7052           | 0.8312              | 0.9068               | 0.3522 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Veo 3.1                     | 0.6605            | 0.4632          | 0.5694              | 0.5450     | 0.1396            | 0.6989           | 0.7878              | 0.8710               | 0.3247 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |
| Wan 2.6                     | 0.6824            | 0.4433          | 0.7229              | 0.7421     | 0.4532            | 0.8539           | 0.7517              | 0.8687               | 0.1904 |

<a id="table-3"></a>

> Table 3: Video quality evaluation results across physics adherence, 3D accuracy and controllability dimensions.

| Models                     | Physics Adherence | 3D Accuracy | Controllability |                       |                    |                  |        |
| -------------------------- | ----------------- | ----------- | --------------- | --------------------- | ------------------ | ---------------- | ------ |
| Models                     | Physics Adherence | 3D Accuracy | Controllability |                       |                    |                  |        |
| Models                     | Physics Adherence | 3D Accuracy | Controllability |                       |                    |                  |        |
| Models                     | Physics Adherence | 3D Accuracy | Controllability |                       |                    |                  |        |
| Interaction Quality        | Trajectory Acc.   | Depth Acc.  | Perspectivity   | Instruction Following | Semantic Alignment | Action Following |        |
| Interaction Quality        | Trajectory Acc.   | Depth Acc.  | Perspectivity   | Instruction Following | Semantic Alignment | Action Following |        |
| Interaction Quality        | Trajectory Acc.   | Depth Acc.  | Perspectivity   | Instruction Following | Semantic Alignment | Action Following |        |
| Interaction Quality        | Trajectory Acc.   | Depth Acc.  | Perspectivity   | Instruction Following | Semantic Alignment | Action Following |        |
| Interaction Quality        | Trajectory Acc.   | Depth Acc.  | Perspectivity   | Instruction Following | Semantic Alignment | Action Following |        |
| Interaction Quality        | Trajectory Acc.   | Depth Acc.  | Perspectivity   | Instruction Following | Semantic Alignment | Action Following |        |
| Interaction Quality        | Trajectory Acc.   | Depth Acc.  | Perspectivity   | Instruction Following | Semantic Alignment | Action Following |        |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| GigaWorld-0                | 0.5368            | 0.1552      | 0.6316          | 0.7596                | 0.6156             | 0.8591           | 0.1134 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| Genie Envisioner           | 0.2052            | 0.0679      | 0.8663          | 0.5284                | 0.2028             | 0.8544           | 0.0109 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| TesserAct                  | 0.5800            | 0.1396      | 0.7159          | 0.7920                | 0.6152             | 0.8783           | 0.0311 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| RoboMaster                 | 0.5364            | 0.1158      | 0.8335          | 0.7588                | 0.5772             | 0.8761           | 0.0352 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Vidar                      | 0.5348            | 0.1928      | 0.7872          | 0.7592                | 0.5912             | 0.8826           | 0.0819 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5 (text)  | 0.3872            | 0.0816      | 0.7051          | 0.7964                | 0.2664             | 0.7733           | 0.1418 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| Cosmos-Predict 2.5(action) | 0.5500            | 0.2945      | 0.8862          | 0.7644                | 0.5840             | 0.8879           | 0.0133 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| WoW                        | 0.5564            | 0.2058      | 0.7283          | 0.7672                | 0.5692             | 0.8842           | 0.0434 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| CtrlWorld                  | 0.6212            | 0.4766      | 0.9300          | 0.7960                | 0.7272             | 0.8912           | 0.0210 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| Wan 2.2                    | 0.5184            | 0.1627      | 0.7768          | 0.7660                | 0.5376             | 0.8877           | 0.0512 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| CogvideoX                  | 0.5940            | 0.3526      | 0.9097          | 0.7828                | 0.7268             | 0.8977           | 0.0076 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| IRASim                     | 0.5656            | 0.3639      | 0.9312          | 0.7788                | 0.6604             | 0.8849           | 0.0526 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Veo 3.1                    | 0.7872            | 0.1231      | 0.7421          | 0.8276                | 0.9328             | 0.8607           | 0.0852 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |
| Wan 2.6                    | 0.7280            | 0.1182      | 0.7144          | 0.8032                | 0.8536             | 0.8728           | 0.0992 |

<a id="section-4-2"></a>

### 4.2 Results

#### 4.2.1 Visual Quality Evaluation

Tables [2](#table-2) and [3](#table-3) summarize video quality evaluation results across six evaluation dimensions. Overall, embodied world models exhibit stronger performance on structure- and interaction-related metrics, while general-purpose video models mainly excel in perceptual quality. Among embodied models, CtrlWorld and TesserAct score highly in subject consistency, background stability, and trajectory accuracy, indicating better alignment with manipulation dynamics. WoW shows strong action-following ability, while RoboMaster and Vidar maintain balanced performance across motion smoothness and content consistency. The open-source video model CogvideoX excels in visual quality and content consistency but lags in physics adherence and motion quality. Closed-source commercial models (Veo 3.1 and Wan 2.6) achieve the highest visual and aesthetic scores, though they show limited improvements in embodied-specific metrics. Qualitative results suggest that visually strong models tend to suffer from semantic drift, while embodied world models produce more coherent and goal-consistent action sequences.

#### 4.2.2 Embodied Task Evaluation

<a id="table-4"></a>

> Table 4: Task success rate of downstream policy models trained with generated data from different world models.

| Model                                             | Task 1 | Task 2 |
| ------------------------------------------------- | ------ | ------ |
| Model                                             | Task 1 | Task 2 |
| Model                                             | Task 1 | Task 2 |
| $\pi_{0.5}$ policy model (zero-shot)              | 2%     | 5%     |
| $\pi_{0.5}$ policy model (zero-shot)              | 2%     | 5%     |
| $\pi_{0.5}$ policy model (zero-shot)              | 2%     | 5%     |
| $\pi_{0.5}$ policy model (trained with real data) | 77%    | 66%    |
| $\pi_{0.5}$ policy model (trained with real data) | 77%    | 66%    |
| $\pi_{0.5}$ policy model (trained with real data) | 77%    | 66%    |
| Genie Envisioner (Liao et al., 2025)              | 7%     | 21%    |
| Genie Envisioner (Liao et al., 2025)              | 7%     | 21%    |
| Genie Envisioner (Liao et al., 2025)              | 7%     | 21%    |
| TesserAct (Zhen et al., 2025)                     | 1%     | 35%    |
| TesserAct (Zhen et al., 2025)                     | 1%     | 35%    |
| TesserAct (Zhen et al., 2025)                     | 1%     | 35%    |
| RoboMaster (37)                                   | 7%     | 68%    |
| RoboMaster (37)                                   | 7%     | 68%    |
| RoboMaster (37)                                   | 7%     | 68%    |
| Vidar (Feng et al., 2025)                         | 13%    | 53%    |
| Vidar (Feng et al., 2025)                         | 13%    | 53%    |
| Vidar (Feng et al., 2025)                         | 13%    | 53%    |
| WoW (Chi et al., 2025)                            | 45%    | 71%    |
| WoW (Chi et al., 2025)                            | 45%    | 71%    |
| WoW (Chi et al., 2025)                            | 45%    | 71%    |
| Wan 2.2 (Wan et al., 2025)                        | 15%    | 41%    |
| Wan 2.2 (Wan et al., 2025)                        | 15%    | 41%    |
| Wan 2.2 (Wan et al., 2025)                        | 15%    | 41%    |

<a id="figure-4"></a>

![embtask](images/embtask.png)

> Figure 4: Correlation of policy evaluation results from world models and the physical simulator.

<a id="table-5"></a>

> Table 5: Task success rate of different world models directly as action planners in the RoboTwin simulator.

| Model                                | Task 1 | Task 2 |
| ------------------------------------ | ------ | ------ |
| Model                                | Task 1 | Task 2 |
| Model                                | Task 1 | Task 2 |
| $\pi_{0.5}$ policy model             | 77%    | 66%    |
| $\pi_{0.5}$ policy model             | 77%    | 66%    |
| $\pi_{0.5}$ policy model             | 77%    | 66%    |
| Genie Envisioner (Liao et al., 2025) | 10%    | 20%    |
| Genie Envisioner (Liao et al., 2025) | 10%    | 20%    |
| Genie Envisioner (Liao et al., 2025) | 10%    | 20%    |
| TesserAct (Zhen et al., 2025)        | 1%     | 35%    |
| TesserAct (Zhen et al., 2025)        | 1%     | 35%    |
| TesserAct (Zhen et al., 2025)        | 1%     | 35%    |
| RoboMaster (37)                      | 8%     | 20%    |
| RoboMaster (37)                      | 8%     | 20%    |
| RoboMaster (37)                      | 8%     | 20%    |
| Vidar (Feng et al., 2025)            | 2%     | 19%    |
| Vidar (Feng et al., 2025)            | 2%     | 19%    |
| Vidar (Feng et al., 2025)            | 2%     | 19%    |
| WoW (Chi et al., 2025)               | 20%    | 21%    |
| WoW (Chi et al., 2025)               | 20%    | 21%    |
| WoW (Chi et al., 2025)               | 20%    | 21%    |
| Wan 2.2 (Wan et al., 2025)           | 12%    | 20%    |
| Wan 2.2 (Wan et al., 2025)           | 12%    | 20%    |
| Wan 2.2 (Wan et al., 2025)           | 12%    | 20%    |

<a id="figure-5"></a>

![simulator_vs_worldmodel_correlation](images/simulator_vs_worldmodel_correlation.png)

> Figure 5: Correlation between EWMScore with human evaluation and embodied task performance results.

In this section, we evaluate the capabilities of world models through three embodied tasks.

Embodied Data Engine. We evaluate six representative world models as data synthesis engines by measuring their impact on downstream policy learning. The evaluation is conducted on two manipulation tasks: adjust bottle (Task 1) and click bell (Task 2), each executed 100 times, with the success rate averaged. For each task, we train a $\pi_{0.5}$ policy using 25 synthetic trajectories generated by each world model. As shown in Table [4](#table-4), we observe that synthetic data from most world models provides some performance gains across both tasks but still lags behind real data. Only generated data from RoboMaster and WoW surpass real-data training on Task 2. These results suggest that the quality of generated data remains insufficient for effective policy training, indicating that current embodied world models are not yet reliable data sources for downstream learning.

Embodied Policy Evaluator. We investigate whether world models can serve as proxy simulation environments for policy evaluation. To this end, we train five policy models $\pi_{0.5}$ with varying performance levels. Each policy is then evaluated by interacting with an action-controllable world model, which generates observation rollouts conditioned on the policy’s actions. As shown in Figure [4](#figure-4), CtrlWorld exhibits a strong correlation with the evaluation results from the RoboTwin simulator, indicating that it effectively captures meaningful environment transition dynamics. In contrast, Cosmos-Predict 2.5 shows a weaker correlation, suggesting that it struggles to accurately model the environment dynamics. Moreover, both models have consistently higher success rates than those measured in the simulator, suggesting partial overfitting to successful trajectories.

Embodied Action Planner. Similar to the data engine task setting, we evaluate six representative world models as end-to-end action planners by executing their predicted action sequences in the RoboTwin simulator. As shown in Table [5](#table-5), while several world models achieve non-trivial success rates across tasks, their overall performance remains substantially lower than that of VLA policies such as $\pi_{0.5}$. These results indicate that, although current embodied world models capture useful predictive structure, they still struggle to reliably support closed-loop task execution, particularly over long horizons. This indicates significant room for improvement in leveraging world models for autonomous embodied control.

#### 4.2.3 Human Evaluation

As shown in Figure [1](#figure-1) (b), human evaluations reveal that commercial and large-scale general video models (e.g., Veo 3.1 and Wan 2.6) consistently achieve the highest scores across overall quality, instruction following, and physical adherence, indicating strong perceptual realism and semantic alignment. Among embodied world models, action-conditioned approaches such as CtrlWorld demonstrate notably better physical adherence and higher win rates than text-only counterparts, suggesting that explicit action modeling plays a critical role in producing physically plausible interactions. In contrast, earlier text-conditioned embodied models (e.g., Genie Envisioner) receive substantially lower scores across all dimensions, reflecting persistent gaps in long-horizon coherence and instruction compliance.

<a id="section-4-3"></a>

### 4.3 Inter-metric Analysis

Figure [5](#figure-5) presents a cross-dimensional analysis relating EWMScore to both human evaluation and embodied task performance. We observe a strong correlation between EWMScore and human judgments (Pearson $r=0.825$), indicating a high degree of alignment with subjective perceptual assessments. In contrast, EWMScore exhibits only moderate correlation with data synthesis performance ($r=0.600$) and a weak correlation with action planning performance ($r=0.360$). These results suggest that while perceptual realism is a necessary condition for favorable human evaluation, it does not directly translate into proportional gains in downstream embodied tasks. In particular, the limited correlation with action planning indicates that current synthetic data, despite achieving high visual fidelity, remains insufficient to provide strong predictive or decision-relevant signals for complex embodied reasoning.

<a id="section-5"></a>

## 5 Conclusion and Future Work

In this work, we present WorldArena, a unified benchmark for systematically evaluating embodied world models from both perceptual and functional perspectives, integrating multi-dimensional video quality metrics, embodied task evaluations, and human assessments. Through an extensive evaluation of 14 representative models, we reveal consistent gaps between perceptual quality and embodied task performance, highlighting that strong visual generation alone is insufficient for reliable embodied decision-making. We further demonstrate that EWMScore effectively captures overall generative capability and correlates well with human judgments. In the future, we will continue to expand WorldArena, incorporating more models to support the advancement of perceptually strong and functionally reliable embodied world models.
