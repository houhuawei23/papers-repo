# Title: ABot-PhysWorld: Interactive World Foundation Model for

Robotic Manipulation with Physics Alignment

- ArXiv: 2603.23376
- Authors: Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Dekang Qi, Haoyun Liu, Tong Lin, Shuang Zeng, Junjin Xiao, Xinyuan Chang, Feng Xiong, Xing Wei, Zhiheng Ma, Mu Xu
- Sections: 32
- Estimated tokens: 24.4k

## Contents

- [Title: ABot-PhysWorld: Interactive World Foundation Model for](#title-abot-physworld-interactive-world-foundation-model-for)
  - [Contents](#contents)
  - [Abstract](#abstract)
  - [1 Introduction](#1-introduction)
  - [2 Data Curation](#2-data-curation)
    - [2.1 Embodied-Specific Data Filtering](#21-embodied-specific-data-filtering)
    - [2.2 Hierarchical Distribution Balancing](#22-hierarchical-distribution-balancing)
    - [2.3 Physics-Aware Video Captioning](#23-physics-aware-video-captioning)
  - [3 Method](#3-method)
    - [3.1 Embodied Video Generation Backbone](#31-embodied-video-generation-backbone)
    - [3.2 Physical Preference Alignment](#32-physical-preference-alignment)
      - [3.2.1 Decoupled VLM Discriminator](#321-decoupled-vlm-discriminator)
      - [3.2.2 Diffusion-DPO Training](#322-diffusion-dpo-training)
    - [3.3 Action-Conditioned Video Generation](#33-action-conditioned-video-generation)
      - [3.3.1 Action Map Construction](#331-action-map-construction)
      - [3.3.2 Action Injection](#332-action-injection)
  - [4 Embodied-ZeroShot Benchmark](#4-embodied-zeroshot-benchmark)
    - [4.1 Evaluation Set](#41-evaluation-set)
    - [4.2 Evaluation Method](#42-evaluation-method)
  - [5 Experiments](#5-experiments)
    - [5.1 Implementation Details](#51-implementation-details)
    - [5.2 Evaluation Setup](#52-evaluation-setup)
    - [5.3 Evaluation Results](#53-evaluation-results)
  - [6 Conclusion](#6-conclusion)
  - [7 Contributions](#7-contributions)

## Abstract

Video-based world models offer a powerful paradigm for embodied simulation and planning, yet state-of-the-art models often generate physically implausible manipulations—such as object penetration and anti-gravity motion—due to training on generic visual data and likelihood-based objectives that ignore physical laws. We present ABot-PhysWorld, a 14B Diffusion Transformer model that generates visually realistic, physically plausible, and action-controllable videos. Built on a curated dataset of three million manipulation clips with physics-aware annotation, it uses a novel DPO-based post-training framework with decoupled discriminators to suppress unphysical behaviors while preserving visual quality. A parallel context block enables precise spatial action injection for cross-embodiment control. To better evaluate generalization, we introduce EZSbench, the first training-independent embodied zero-shot benchmark combining real and synthetic unseen robot-task-scene combinations. It employs a decoupled protocol to separately assess physical realism and action alignment. ABot-PhysWorld achieves new state-of-the-art performance on PBench and EZSbench, surpassing Veo 3.1 and Sora v2 Pro in physical plausibility and trajectory consistency. We will release EZSbench to promote standardized evaluation in embodied video generation.

Date: March 20, 2026

Correspondence: xumu.xm@alibaba-inc.com

Project Page: [https://github.com/amap-cvlab/ABot-PhysWorld](https://github.com/amap-cvlab/ABot-PhysWorld)

![logo5](images/logo5.png)

<a id="section-1"></a>

## 1 Introduction

An embodied world model needs to generate future predictions that adhere to real-world physical laws in order to be effective for simulation, planning, and policy learning. Video generation presents a promising paradigm: such models can serve as simulators for Vision-Language-Action (VLA) policies [kim2024openvla, intelligence2025pi_, abot-m0, lingbot-vla], provide interpretable trajectory previews, or function directly as World Action Models (WAMs) [ye2026worldactionmodelszeroshot, lingbot-va, kim2026cosmos-policy] by predicting action-conditioned dynamics—forming critical infrastructure for embodied intelligence.

Despite significant advances in visual fidelity, however, state-of-the-art models like Veo 3.1 [google2026veo31] and Sora v2 Pro [openai2025sora2] frequently produce manipulation sequences that violate basic physics, including object penetration, contactless motion, and unnatural deformations. These are not mere rendering artifacts but fundamental failures in physical reasoning, limiting their reliability in downstream robotic applications.

This gap arises from two core limitations: (i) training on general visual data lacking rich embodied interaction signals, which hinders the acquisition of fine-grained physical dynamics such as friction, collision response, and mass distribution; and (ii) reliance on standard maximum likelihood objectives during fine-tuning, which treat all prediction errors uniformly and fail to distinguish physically valid from invalid transitions. The absence of both embodied experience and physics-aware supervision results in a systematic disconnect between visual realism and physical plausibility.

To address this, we present ABot-PhysWorld, a physically grounded and action-controllable world model based on a 14B Diffusion Transformer [wan2025wan, wan2025open], built upon a carefully designed data curation pipeline. We integrate three million real-world manipulation clips from five major open-source embodied datasets, enhancing data diversity and balance through curated sampling, ratio optimization, and physics-aware annotation—improving generalization across robots, objects, and environments. Building on this foundation, we introduce a physics-inspired DPO-based [qian2025rdpo, wang2025physcorr, cai2025phygdpo, qian2025rdpo] post-training framework with decoupled discriminators that suppress unphysical behaviors (e.g., object penetration, anti-gravity motion) while preserving visual quality and improving dynamic consistency. A parallel context block enables multi-channel spatial action injection, supporting precise cross-embodiment control and action-aligned motion synthesis. Together, ABot-PhysWorld generates visually realistic, physically plausible, and highly controllable manipulation sequences—serving as a high-fidelity interface for robot simulation and planning.

<a id="figure-1"></a>

![logo5](images/data2.png)

> Figure 1: Overview of the data curation pipeline.
>
>  (a) shows the multi-stage filtering and balancing flow from raw aggregation ($\sim$ 3M clips) to training-ready splits (SFT, RL, and A2V data).
>
>  (b) Task-aware quota allocation: head tasks are capped at 8–15%, body tasks are uniformly sampled at 40–50%, and long-tail tasks are fully preserved to maximize task diversity.
>
> (c) Dataset and robot type distribution: the left ring shows the original composition and the right ring shows the rebalanced result after hierarchical sampling. 
>
> (d) Physics-aware video captioning pipeline: a perception module (Qwen3-VL 32B) extracts structured physical attributes, followed by a writing module (Qwen3 32B FP8) that generates four-phase captions covering scene setup, action detail, state transition, and camera summary.

However, evaluating such advances remains challenging: existing benchmarks often emphasize visual quality or in-distribution accuracy, with little emphasis on physical consistency or zero-shot generalization. To enable more rigorous and realistic assessment, we propose EZSbench, the first training-independent Embodied Zero-Shot Benchmark combining both real and synthetic scenarios involving unseen combinations of robots, tasks, and scenes. Unlike existing benchmarks biased toward in-distribution fidelity, EZSbench is specifically designed to assess three key capabilities: **action controllability, physical consistency, and zero-shot generalization.** It employs a decoupled dual-model evaluation protocol that separately scores physical realism and action alignment, enabling fine-grained diagnosis of model behavior. We will publicly release EZSbench to promote standardized and meaningful progress in embodied video generation.

Our model achieves new state-of-the-art results on both PBench and EZSbench, surpassing Veo 3.1 and Sora v2 Pro in physical plausibility and action trajectory consistency. More details are provided in Section [5](#section-5). Our primary contributions are:

- Data: We design a principled data curation pipeline that improves diversity and balance in embodied video data through curated sampling and physics-aware annotation—enabling scalable and robust training on real-world interactions.
- Model: We propose ABot-PhysWorld, a unified framework that jointly optimizes visual realism, physical plausibility, and action controllability through physics-aware DPO and parallel spatial action injection.
- Evaluation: We introduce EZSbench, the first training-independent zero-shot benchmark for embodied video generation, with a decoupled protocol to assess physical fidelity and action alignment under distribution shift.

<a id="section-2"></a>

## 2 Data Curation

<a id="figure-2"></a>

![overview](images/overview.png)

> Figure 2: Two-stage training pipeline. Stage 1: SFT on the DiT to predict future frames from observations and instructions. Stage 2: generate $N$ candidates, score via physics checklist, and apply DPO via LoRA on frozen DiT weights.

Data is essential for high-quality embodied world models. Following [halevy2009unreasonable], we adopt a data-driven approach through a systematic infrastructure that enhances data scale and diversity to address complex human-robot interaction modeling. As illustrated in Figure [1](#figure-1), our data curation pipeline consists of three stages: embodied-specific filtering (§[2.1](#section-2-1)), hierarchical distribution balancing (§[2.2](#section-2-2)), and physically grounded caption generation (§[2.3](#section-2-3)).

- [2.1 Embodied-Specific Data Filtering](#21-embodied-specific-data-filtering)
- [2.2 Hierarchical Distribution Balancing](#22-hierarchical-distribution-balancing)
- [2.3 Physics-Aware Video Captioning](#23-physics-aware-video-captioning)

<a id="section-2-1"></a>

### 2.1 Embodied-Specific Data Filtering

To build a physically consistent world model for embodied manipulation, we construct a foundational dataset of nearly three million real-world video clips by integrating five public datasets: AgiBot [bu2025agibot], RoboCoin [wu2025robocoin], RoboMind [wu2024robomind], Galaxea [jiang2025galaxea], and OXE [o2024open].

General-domain curation pipelines such as Cosmos-Curate [cosmos2025curate] and VideoX-Fun [videox2024fun] are misaligned with embodied data: they rely on scene-cut detectors unsuitable for static-background manipulation videos, and prioritize visual aesthetics over physical causality. To resolve noise introduced by raw aggregation, we apply a video-level quality gate followed by three semantic filtering stages.

Video-level quality gate. Clips with abnormal resolutions or moving cameras are discarded. Sequences are constrained to 80–500 frames; longer videos are segmented temporally by task index into training-compliant clips to ensure relevance and efficiency.

Optical-flow-based motion filtering. We extract grayscale frames at 2 FPS and compute Farnebäck dense optical flow [farneback2003two] to capture pixel-level motion. By averaging the polar magnitudes of displacement vectors across each frame, we derive a global kinematic score and remove clips with near-zero motion or unphysical oscillations.

CLIP-based temporal coherence. To eliminate visual corruption (e.g., black screens, cuts, stitching errors), we assess temporal continuity using CLIP-based embeddings [radford2021learning]. Eight equidistant frames are sampled per clip, and their 768D features are extracted; samples with low average cosine similarity between consecutive frames are discarded.

Vision-action alignment verification. Calibrated action maps, encoding joint actions, end-effector poses, and gripper states, are projected onto video frames. Qwen3-VL verifies spatiotemporal alignment between visual motion and control signals, filtering out mismatches from sensor calibration or synchronization errors.

The resulting dataset provides a robust foundation for training generalizable, dynamics-aware world models across diverse embodied tasks.

<a id="figure-3"></a>

![logo5](images/logo5.png)

> Figure 3: Construction pipeline of the EZSbench. Top: dual-source image augmentation—Branch 1 generates synthetic initial observations via text-to-image (Nano Banana) by varying robot morphology, scene, task, and viewpoint; Branch 2 applies VLM-guided background editing to real-world images while preserving foreground interactions. Down: three-stage dense description synthesis—visual anchoring grounds the scene layout and object coordinates, action simulation infers kinematically compliant trajectories with micro-physical interactions, and narrative synthesis produces a documentary-style caption integrating initial state, trajectory, and final state.

<a id="section-2-2"></a>

### 2.2 Hierarchical Distribution Balancing

Recent studies indicate that data diversity, not just volume, is key to scalable world models and generalist robotic policies [kang2024how, ye2026worldactionmodelszeroshot]; scaling repetitive data often leads to memorization rather than out-of-distribution generalization. To address this, we design a hierarchical dynamic sampling strategy spanning four levels: video, sub-dataset or robot type, task, and macro-dataset. This approach balances data distribution while preserving long-tail features.

Level 1: Intra-dataset diversity preservation. Several source datasets are themselves aggregations of smaller collections; for example, small sub-datasets within OXE [o2024open] are retained entirely to preserve unique interaction patterns before any cross-dataset operations are applied.

Level 2: Cross-robot rebalancing. Operating across the five source datasets, this level addresses imbalances among robot embodiment types. Underrepresented robot types are upweighted to retain rare interaction patterns (e.g., non-standard kinematics or dual-arm coordination), enhancing cross-platform generalization and mitigating head-category dominance (Figure [1](#figure-1)c).

Level 3: Task-aware quota allocation. Rather than applying a fixed sampling threshold, we partition tasks into three tiers based on data volume and assign tier-specific strategies (Figure [1](#figure-1)b). Head tasks (high data volume) are capped at 8–15% of their original size to prevent overfitting to dominant categories. Body tasks (medium volume) are uniformly sampled at 40–50%, preserving representative coverage without excessive redundancy. Long-tail tasks (rare tasks) are fully preserved to maximize task diversity.

Level 4: Macro-dataset scale regulation. Finally, at the coarsest granularity, large-scale macro-datasets (e.g., AgiBot, OXE) are capped via uniform subsampling , while micro-datasets (e.g., RoboMind) are guaranteed minimum coverage through a mandatory lower bound. When activated, a three-round supplementation strategy allocates: (1) a base quota uniformly across tasks, (2) reallocates unused quotas proportionally, and (3) fills residual gaps via random fallback sampling from the global filtered pool, preventing single-task over-extraction and improving long-tail balance. This hierarchical framework improves combinatorial diversity and distributional balance, providing a robust foundation for training general-purpose embodied world models.

<a id="section-2-3"></a>

### 2.3 Physics-Aware Video Captioning

Training embodied world models with text-to-video (T2V) objectives requires captions that go beyond surface-level scene descriptions. An effective annotation must capture three progressively deeper aspects of robotic manipulation: *what* the robot does (action semantics), *how* it interacts with the physical world (spatial and contact precision), and *why* the observed outcome occurs (causal reasoning). We design a multi-level annotation system that addresses each of these requirements.

Multi-level action semantics. Adopting a robot-centric “annotating for action” philosophy, we structure each caption across four granularities: macroscopic task intent in natural language; mesoscopic verb-noun action segmentation for long-horizon planning; microscopic details including Cartesian trajectories, relative motion, and gripper states; and scene-level descriptions of physical relations (contact, support, containment) and task outcomes (success, failure, partial accidents).

Grounded spatial precision. Purely template-driven annotation tends to produce hallucinated spatial relations and imprecise grasp descriptions. To suppress these errors, we introduce three mechanisms: few-shot in-context learning with explicit positive and negative examples for richer physical detail, dynamic vocabularies for precise grasp-type specification, and a visible-fact baseline that restricts descriptions to observable evidence.

Causal physical modeling. Beyond describing what happens, effective captions for world models must explain why it happens. We explicitly annotate physical causality, including gravity-induced dropping, surface deformation, and force feedback. A four-stage narrative structure (scene construction, action flow, final state confirmation, and camera summary) organizes each caption into a temporally coherent account of the manipulation episode.

This annotation system delivers physically grounded language supervision that captures not only events but their underlying causes, providing the semantic foundation for training world models with causal understanding.

<a id="section-3"></a>

## 3 Method

<a id="table-1"></a>

> Table 1: Quantitative comparison on the PAI-Bench robot domain subset.

| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| ------------------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------------- | ------------ | ------ |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| Wan 2.5            | 0.5477 | 0.8985 | 0.6458 | 0.9623 | 0.2190 | 0.8784 | 0.9438 | 0.9428 | 0.7548        | 0.8644       | 0.8096 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| GigaWorld-0        | 0.4757 | 0.9219 | 0.6506 | 0.9908 | 0.1944 | 0.9111 | 0.9673 | 0.9607 | 0.7591        | 0.8583       | 0.8087 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Veo 3.1            | 0.5458 | 0.9216 | 0.7244 | 0.9712 | 0.2214 | 0.9146 | 0.9317 | 0.9614 | 0.7740        | 0.8350       | 0.8045 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| Wan2.1_14B         | 0.4723 | 0.9315 | 0.7118 | 0.9917 | 0.1921 | 0.9185 | 0.9745 | 0.9451 | 0.7672        | 0.8391       | 0.8032 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| WoW-wan 14B        | 0.4664 | 0.9295 | 0.7027 | 0.9858 | 0.1941 | 0.9149 | 0.9613 | 0.9292 | 0.7605        | 0.8301       | 0.7953 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Cosmos-Predict 2.5 | 0.4897 | 0.9166 | 0.7405 | 0.9906 | 0.1911 | 0.8973 | 0.9304 | 0.9030 | 0.7574        | 0.8021       | 0.7797 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| Sora v2 Pro        | 0.5324 | 0.9285 | 0.6956 | 0.9702 | 0.2203 | 0.9163 | 0.9507 | 0.9290 | 0.7679        | 0.7626       | 0.7652 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| UnifoLM-WMA-0      | 0.4547 | 0.9423 | 0.6564 | 0.9875 | 0.1878 | 0.9412 | 0.9638 | 0.9403 | 0.7593        | 0.6693       | 0.7143 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model          | 0.4620 | 0.9373 | 0.6906 | 0.9916 | 0.1927 | 0.9406 | 0.9777 | 0.9498 | 0.7678        | 0.8785       | 0.8232 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |
| Our Model + DPO    | 0.4667 | 0.9365 | 0.6916 | 0.9908 | 0.1942 | 0.9355 | 0.9768 | 0.9483 | 0.7676        | 0.9306       | 0.8491 |

- [3.1 Embodied Video Generation Backbone](#31-embodied-video-generation-backbone)
- [3.2 Physical Preference Alignment](#32-physical-preference-alignment)
- [3.3 Action-Conditioned Video Generation](#33-action-conditioned-video-generation)

<a id="section-3-1"></a>

### 3.1 Embodied Video Generation Backbone

Generating physically plausible manipulation videos requires a backbone that captures both the visual diversity of real-world scenes and the fine-grained spatiotemporal dynamics of robot-object interactions. To meet this requirement, we build upon Wan2.1-I2V-14B [wan2025open], and fully fine-tune it on our curated embodied dataset.

<a id="section-3-2"></a>

### 3.2 Physical Preference Alignment

While SFT teaches the model to reproduce training distributions, it treats all samples equivalently and cannot distinguish physically correct predictions from those containing violations such as object penetration or anti-gravity motion. To explicitly suppress these violations, we propose a post-training preference alignment pipeline (Figure [2](#figure-2)) that pairs a decoupled VLM discriminator with Diffusion-DPO.

- [3.2.1 Decoupled VLM Discriminator](#321-decoupled-vlm-discriminator)
- [3.2.2 Diffusion-DPO Training](#322-diffusion-dpo-training)

#### 3.2.1 Decoupled VLM Discriminator

For a given prompt $x$ and initial state, we generate $N$ candidate video variants. Evaluating physical plausibility with a single VLM risks self-evaluation hallucinations, where the same model that generates questions also judges answers. To prevent this, we decouple the evaluation into two roles.

The Qwen3-VL 32B Thinking model acts as the *proposer*. It observes the first frame and text instruction to dynamically generate a task-specific physical checklist based on a hierarchical evaluation system. This system applies single-vote veto power to Tier 1 metrics (fatal violations such as penetration and anti-gravity) and uses Tier 2 metrics (micro-physical fidelity and contact dynamics) to differentiate compliant samples. Generating specific questions prevents hallucinations caused by vague queries. For example, given the instruction to grasp and place an apple, the proposer asks whether the gripper penetrates the apple, whether the apple penetrates the bag, and whether it is firmly grasped rather than magnetically attached. The proposer also explicitly constructs a balanced mix of positive and negative questions to prevent the scoring model from sycophantically predicting the absence of violations.

The Gemini 3 Pro model [gemini3_2025] then acts as the *scorer*. It uses explicit Chain-of-Thought reasoning, including global scanning, marking suspicious frames, and backtracking confirmation, to evaluate the $N$ variants against the generated checklist. To efficiently resolve score ties and isolate the optimal ($y_{w}$) and worst ($y_{l}$) samples within $\mathcal{O}(N)$ complexity, we apply a multi-round tournament-based sampling strategy: a knockout tournament first selects the optimal sample, followed by a loser-bracket round to identify the worst sample. This two-stage mechanism avoids full permutation comparisons and yields highly discriminative DPO [rafailov2023direct] training triplets $(x,y_{w},y_{l})$ with clear margins.

#### 3.2.2 Diffusion-DPO Training

Given the discriminative triplets $(c,v_{w},v_{l})$ produced by the decoupled discriminator, where $c$ is the condition, $v_{w}$ the physics-compliant video, and $v_{l}$ the physics-violating video, we adapt the Diffusion-DPO framework to fine-tune the video diffusion model directly in the latent space. For a video latent $z$, we inject Gaussian noise $\epsilon\sim\mathcal{N}(0,I)$ at time step $t\sim\mathcal{U}(0,T)$ to obtain $z_{t}$. The single-step denoising mean squared error for model $\epsilon_{\theta}$ is $L(\theta,z)=\|\epsilon_{\theta}(z_{t},t,c)-\epsilon\|_{2}^{2}$. Letting $L_{\theta}(\cdot)$ and $L_{ref}(\cdot)$ denote the denoising errors of the policy model $\pi_{\theta}$ and reference model $\pi_{ref}$ (the SFT baseline) respectively, the physical preference alignment loss is:

$$
\mathcal{L}_{DPO}=-\mathbb{E}_{z,\epsilon,t}\Bigg[\log\sigma\Bigg(-\frac{\beta}{2}\Big[\underbrace{(L_{\theta}(z_{w})-L_{\theta}(z_{l}))}_{\text{Policy Diff.}}\\ -\underbrace{(L_{ref}(z_{w})-L_{ref}(z_{l}))}_{\text{Ref.\ Diff.}}\Big]\Bigg)\Bigg],(1)
$$

where $\beta$ controls distribution divergence, and $z_{w},z_{l}$ are the latents of $v_{w},v_{l}$. This objective actively reduces the prediction error for $z_{w}$ while increasing it for $z_{l}$ at each timestep.

Standard DPO requires maintaining two complete computation graphs ($\pi_{\theta}$ and $\pi_{ref}$), causing out-of-memory errors for a 14B DiT. To resolve this, we freeze the DiT backbone and inject Low-Rank Adaptation [hu2021lora] (LoRA) modules with a rank of 64 into the self-attention (query, key, value, output) and feed-forward layers, so that the reference model loss $L_{ref}$ can be computed by temporarily disabling the LoRA weights with zero additional memory.

<a id="section-3-3"></a>

### 3.3 Action-Conditioned Video Generation

Beyond text-conditioned prediction, a world model for embodied intelligence must support controllable generation: given the current observation and a future action sequence, it should produce physically plausible videos that faithfully follow the commanded trajectory. Directly injecting low-dimensional robotic commands (*e.g*., end-effector poses) into high-dimensional visual pipelines creates a semantic gap. To bridge this gap, we convert discrete action commands into spatially structured action maps and inject them through parallel context blocks that preserve the backbone’s pre-trained physical knowledge.

- [3.3.1 Action Map Construction](#331-action-map-construction)
- [3.3.2 Action Injection](#332-action-injection)

#### 3.3.1 Action Map Construction

The input action is a 7D vector $\boldsymbol{a}\in\mathbb{R}^{7}$ (3D position, 3D orientation, gripper openness), extending to 14 dimensions for dual-arm systems. Using camera intrinsics and extrinsics, we project the 3D position $(x,y,z)$ to a 2D center $(u,v)$. Orientation is encoded as the three principal axes of the corresponding rotation matrix, projected into the image plane and rendered as colored arrows whose length encodes depth. The gripper state is mapped to a circular mask at $(u,v)$, with opacity linearly indicating openness. For dual-arm robots, we distinguish left and right arms via red and blue channels, yielding a multi-channel action map.

#### 3.3.2 Action Injection

Existing action injection methods either use Adaptive Layer Normalization (AdaLN) [dit] for MLP-encoded actions [cosmos, zhu2025irasim], which hinders cross-embodiment generalization, or concatenate action maps directly with noisy latents for full fine-tuning [evac, liao2025genie, team2025gigaworld], causing catastrophic forgetting of pre-trained physical priors.

<a id="figure-4"></a>

![logo5](images/logo5.png)

> Figure 4: Architecture of the action-conditioned video generation model. We selectively duplicate DiT blocks as parallel context blocks to process action maps, and fuse their outputs residually into the main DiT.

To address these challenges, as shown in Figure [4](#figure-4), we clone selective blocks from the main DiT [wan2025wan] to form a parallel set of context blocks that process the action maps [vace]. The output of each context block is projected via zero-initialized convolution layers and added residually to the corresponding main DiT block:

$$
\mathbf{x}_{i}={\mathrm{DiT}}_{i}(\mathbf{x}_{i-1})+\alpha\cdot W_{\mathrm{zero}}^{(i)}\mathbf{h}_{i},(2)
$$

where $\mathbf{h}_{i}$ is the $i$-th context block output, $W_{\mathrm{zero}}$ is the zero-initialized convolution layer, and $\alpha$ is the control scale. Following VACE, we instantiate context blocks selectively, replicating only every fifth DiT block. Because the zero initialization ensures that the context branch contributes no signal at the start of training, the backbone weights remain undisturbed, preserving pre-trained physical priors while gradually learning action controllability.

<a id="table-2"></a>

> Table 2: Quantitative comparison on EZSbench.

| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| ------------------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------------- | ------------ | ------ |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| Model              | AQ     | BC     | IQ     | MS     | OC     | SC     | I2VB   | I2VS   | Quality Score | Domain Score | Avg.   |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| WoW-wan 14B        | 0.4764 | 0.9412 | 0.7514 | 0.9922 | 0.1236 | 0.9347 | 0.9495 | 0.9178 | 0.7609        | 0.7951       | 0.7780 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| GigaWorld-0        | 0.4466 | 0.8893 | 0.6565 | 0.9889 | 0.1222 | 0.8678 | 0.9325 | 0.9139 | 0.7272        | 0.7826       | 0.7549 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| Cosmos-Predict 2.5 | 0.4148 | 0.8835 | 0.6810 | 0.9878 | 0.0970 | 0.8366 | 0.9054 | 0.8653 | 0.7089        | 0.7698       | 0.7394 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| UnifoLM-WMA-0      | 0.4861 | 0.9575 | 0.7390 | 0.9925 | 0.1046 | 0.9452 | 0.8421 | 0.8170 | 0.7355        | 0.5232       | 0.6294 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |
| Our Model          | 0.4789 | 0.9494 | 0.7483 | 0.9910 | 0.1301 | 0.9442 | 0.9680 | 0.9453 | 0.7694        | 0.8366       | 0.8030 |

<a id="figure-5"></a>

![data2](images/data2.png)

> Figure 5: Qualitative comparison on PAI-Bench.

<a id="section-4"></a>

## 4 Embodied-ZeroShot Benchmark

Existing embodied video generation benchmarks draw test samples from the same distribution as training data, making it difficult to assess genuine zero-shot generalization. We introduce EZSbench to evaluate physical fidelity and cross-embodiment generalization under fully out-of-distribution conditions, where diverse robot morphologies, environments, and tasks are composed into previously unseen combinations with no overlap with training data.

- [4.1 Evaluation Set](#41-evaluation-set)
- [4.2 Evaluation Method](#42-evaluation-method)

<a id="section-4-1"></a>

### 4.1 Evaluation Set

We construct the initial observation pool via a dual-branch strategy. The first branch generates synthetic images using the text-to-image model Nano Banana [google2025nanobanana], controlled by four orthogonal variables: robots, scenes, tasks, and perspectives. This approach targets morphological, scene, and task generalization by varying arm structures, backgrounds, and task complexity from basic pick-and-place to long-horizon manipulations. The second branch uses a large VLM for controllable scene editing on real-world mechanical arm images, dynamically altering backgrounds while preserving foreground physical interactions.

To generate reliable physical descriptions, we propose a physics-heuristic dense description synthesis framework that progresses through visual anchoring, kinematically compliant action simulation, and narrative synthesis, producing text that integrates the initial state, action trajectory, and final state. Each initial image paired with its dense description forms a core benchmark sample.

<a id="section-4-2"></a>

### 4.2 Evaluation Method

A key challenge in evaluating physical consistency is that a single model acting as both question generator and answer judge introduces self-evaluation bias. To address this, we propose a decoupled dual-model evaluation paradigm. The Qwen3-VL-32B-Thinking model dynamically generates physical checklist questions based on the initial state and text instructions. It employs a System 2 reasoning protocol for scene decoding and action parsing, guided by few-shot examples covering nine criteria across spatial, temporal, and physical dimensions. We mandate that 30–50% of the checklist comprises negative questions, such as asking whether a red apple is green, to prevent shortcut learning via random guessing. To eliminate self-evaluation bias, the Qwen2.5-VL-72B-Instruct model serves as the answering end. The final physical score $S_{v}$ for a video $v$ measures the consistency between the visual question answering (VQA) predictions and the checklist ground truth (GT): $S_{v}=\frac{1}{|Q_{v}|}\sum_{q\in Q_{v}}\mathbb{I}(\text{VQA}(v,q)=\text{GT}(q))$, where $Q_{v}$ is the generated checklist and $\mathbb{I}(\cdot)$ is the indicator function.

<a id="section-5"></a>

## 5 Experiments

- [5.1 Implementation Details](#51-implementation-details)
- [5.2 Evaluation Setup](#52-evaluation-setup)
- [5.3 Evaluation Results](#53-evaluation-results)

<a id="section-5-1"></a>

### 5.1 Implementation Details

We conduct all experiments on a cluster of 128 Nvidia H20 GPUs. The training pipeline consists of three stages: TI2V foundational training, DPO, and A2V training. For TI2V, we use Wan2.1-I2V-14B-480P with inputs cropped to $480\times 832$ and 81 frames uniformly sampled. Trained for 6,000 steps with a global batch size of 128 and learning rate 1e-5. For DPO, we apply LoRA-based fine-tuning to the Diffusion Transformer, inserting rank-64 adapters (scaling factor 64) into self-attention and feed-forward layers (q, k, v, o, ffn.0, ffn.2). Optimized with AdamW, lr=1e-6, 10-step warmup. To strengthen preference signals in diffusion models, we set $\beta=5000$. Training employs BF16 mixed precision, gradient checkpointing, per-device batch size 1, and runs for 500 steps/epoch over 100 epochs. For A2V, we adopt the VACE framework on the fine-tuned TI2V model. We duplicate specific Diffusion Transformer layers (0, 5, 10, 15, 20, 25, 30, 35) as a trainable context branch while keeping the backbone frozen. Data is augmented via random frame sampling with variable stride. Training uses batch size 16, learning rate 5e-5, and 20,000 steps.

<a id="section-5-2"></a>

### 5.2 Evaluation Setup

Text-Conditioned Generation. We evaluate physical plausibility and visual quality using PAI-Bench [zhou2025paibench] and its PBench dataset, focusing on the robot domain subset with 174 complex manipulation videos from BridgeData V2 [walke2023bridgedata], AgiBot, and Open X-Embodiment. We employ an MLLM-as-Judge approach with Qwen2.5-VL-72B-Instruct [bai2025qwen25vl] for binary visual question answering. The Domain Score evaluates accuracy across 886 questions in three dimensions: spatial (36.3%, geometry and contact), temporal (28.6%, causal logic), and physical (34.1%, object attributes and state changes). We also use PAI-Bench’s multidimensional quality metrics: subject [caron2021emerging]/background [fu2023dreamsim] consistency, Overall consistency [wang2024internvid], Aesthetic quality (LAION aesthetic head), Imaging quality [ke2021musiq], motion smoothness, and i2v subject/background consistency. For zero-shot evaluation, we use EZSbench with the decoupled dual-model protocol from Section [5](#section-5).

Action-Conditioned Generation. We construct the action-conditioned evaluation set by uniformly sampling 200 instances from the action-to-video dataset, each containing an initial frame and a structured action sequence (end-effector pose and gripper state). Visual alignment is evaluated frame-by-frame using PSNR for pixel accuracy and SSIM for local texture fidelity. For trajectory accuracy, we use nDTW: a fine-tuned YOLO detector locates the gripper in each frame, and the extracted trajectory is compared to ground truth via nDTW.

Baselines. We compare with Cosmos-Predict 2.5-2B [agarwal2025cosmos], GigaWorld-0 [gigaai2025gigaworld], UnifoLM-WMA-0 [unitree2025unifolm], WoW-wan 14B [chi2025wow], Veo 3.1 [google2026veo31], Sora v2 Pro [openai2025sora2], and Wan 2.5 [alibaba2025wan25] for text-conditioned generation, and with Enerverse-AC [evac] and Gen-Sim [liao2025genie] for action-conditioned generation.

<a id="table-3"></a>

> Table 3: Quantitative results on action-conditioned generation.

| Model        | PSNR  | SSIM   | Traj. Consis. |
| ------------ | ----- | ------ | ------------- |
| Model        | PSNR  | SSIM   | Traj. Consis. |
| Model        | PSNR  | SSIM   | Traj. Consis. |
| Model        | PSNR  | SSIM   | Traj. Consis. |
| Enerverse-AC | 20.42 | 0.7542 | 0.8157        |
| Enerverse-AC | 20.42 | 0.7542 | 0.8157        |
| Enerverse-AC | 20.42 | 0.7542 | 0.8157        |
| Enerverse-AC | 20.42 | 0.7542 | 0.8157        |
| Gen-Sim      | 18.05 | 0.7413 | 0.6195        |
| Gen-Sim      | 18.05 | 0.7413 | 0.6195        |
| Gen-Sim      | 18.05 | 0.7413 | 0.6195        |
| Gen-Sim      | 18.05 | 0.7413 | 0.6195        |
| Ours         | 21.09 | 0.8126 | 0.8522        |
| Ours         | 21.09 | 0.8126 | 0.8522        |
| Ours         | 21.09 | 0.8126 | 0.8522        |
| Ours         | 21.09 | 0.8126 | 0.8522        |

<a id="section-5-3"></a>

### 5.3 Evaluation Results

PBench Evaluation. As shown in Table [1](#table-1), our DPO-augmented model achieves the highest average score (0.8491) and sets a new state-of-the-art Domain Score (0.9306), outperforming the base model (0.8785) and all baselines. Existing methods show a trade-off between visual quality and physical fidelity: Veo 3.1 and Sora v2 Pro achieve high Quality Scores (0.7740, 0.7679) due to strong imaging and aesthetics, but lag in Domain Score (0.8350, 0.7626), favoring perception over physics. Our model maintains competitive visual quality (Quality Score: 0.7676) while enforcing physical constraints, proving that alignment with physical laws does not sacrifice perceptual quality. The base model also shows strong spatiotemporal stability (I2VB: 0.9777; MS: 0.9916).

EZSbench Evaluation. On the out-of-distribution EZSbench (Table [2](#table-2)), our model achieves the highest overall average score (0.8030), establishing state-of-the-art results for both Quality Score (0.7694) and Domain Score (0.8366). This confirms that the physical fidelity improvements generalize beyond the training distribution.

Qualitative Analysis. Figure [5](#figure-5) shows qualitative PBench comparisons. Baselines violate physical laws in complex interactions: Sora v2 Pro and Veo 3.1 show gripper or object distortion during dense contact; GigaWorld-0 and Cosmos exhibit grasping penetration; WoW produces non-contact grasping and geometric distortion; UnifoLM and Wan 2.5 misidentify targets (e.g., spatula instead of rag). Our method correctly identifies targets, maintains spatiotemporal coherence, and avoids deformation and penetration.

Action-Conditioned Generation Results. As shown in Table [3](#table-3), our method outperforms baselines in both visual quality and action fidelity. Our method consistently outperforms the baselines by substantial margins.

<a id="section-6"></a>

## 6 Conclusion

We introduce ABot-PhysWorld, a physically grounded and action-controllable world model for embodied manipulation based on a 14B Diffusion Transformer. It integrates curated data, physical alignment through Diffusion-DPO, and spatial action injection to reduce physical violations while maintaining control across different embodiments. We also propose EZSbench, a zero-shot benchmark featuring out-of-distribution scenarios and a decoupled evaluation protocol. Experimental results show state-of-the-art physical fidelity and improved trajectory consistency compared to Veo 3.1 and Sora v2 Pro. The model currently relies on fixed-viewpoint data and lacks closed-loop evaluation. Future work will explore multi-view generation and real-world deployment.

<a id="section-7"></a>

## 7 Contributions

Author contributions in the following areas are as follows:

- Data Curation: Yuzhi Chen, Ronghan Chen, Dongjie Huo, Haoyun Liu, Yandan Yang, Dekang Qi, Tong Lin, Shuang Zeng, Junjin Xiao
- Model Training: Yuzhi Chen, Ronghan Chen
- Evaluation: Yuzhi Chen, Ronghan Chen, Dongjie Huo
- Writing: Yuzhi Chen, Yandan Yang, Ronghan Chen, Dongjie Huo, Dekang Qi
- Project Lead: Xinyuan Chang, Feng Xiong
- Advisor: Zhiheng Ma, Xing Wei, Mu Xu^†
