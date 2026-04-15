# Title: OpenWorldLib: A Unified Codebase and Definition of Advanced World Models

- ArXiv: 2604.04707
- Authors: DataFlow Team, Bohan Zeng, Daili Hua, Kaixin Zhu, Yifan Dai, Bozhou Li, Yuran Wang, Chengzhuo Tong, Yifan Yang, Mingkun Chang, Jianbin Zhao, Zhou Liu, Hao Liang, Xiaochen Ma, Ruichuan An, Junbo Niu, Zimo Meng, Tianyi Bai, Meiyi Qiang, Huanyao Zhang, Zhiyou Xiao, Tianyu Guo, Qinhan Yu, Runhao Zhao, Zhengpin Li, Xinyi Huang, Yisheng Pan, Yiwen Tang, Yang Shi, Yue Ding, Xinlong Chen, Hongcheng Gao, Minglei Shi, Jialong Wu, Zekun Wang, Yuanxing Zhang, Xintao Wang, Pengfei Wan, Yiren Song, Mike Zheng Shou, Wentao Zhang
- Sections: 29
- Estimated tokens: 22.6k

## Contents

- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background and Related Works](#2-background-and-related-works)
  - [2.1 World Model Related Tasks](#21-world-model-related-tasks)
    - [Interactivate Video Generation.](#interactivate-video-generation)
    - [Multimodal Reasoning.](#multimodal-reasoning)
    - [Vision-Language-Action.](#vision-language-action)
  - [2.2 The Role of 3D and Simulators in World Models](#22-the-role-of-3d-and-simulators-in-world-models)
  - [2.3 Methods Not Considered World Models](#23-methods-not-considered-world-models)
- [3 OpenWorldLib Framework Design](#3-openworldlib-framework-design)
  - [3.1 Operator](#31-operator)
  - [3.2 Synthesis Module](#32-synthesis-module)
    - [3.2.1 Visual Synthesis](#321-visual-synthesis)
    - [3.2.2 Audio Synthesis](#322-audio-synthesis)
    - [3.2.3 Other Signal Synthesis](#323-other-signal-synthesis)
  - [3.3 Reasoning Module](#33-reasoning-module)
  - [3.4 Representation Module](#34-representation-module)
  - [3.5 Memory Module](#35-memory-module)
  - [3.6 Pipeline](#36-pipeline)
- [4 Discussion](#4-discussion)
- [5 Evaluation](#5-evaluation)
  - [5.1 Experimental Setting](#51-experimental-setting)
  - [5.2 Experimental Results](#52-experimental-results)
    - [5.2.1 Interactive Video Generation.](#521-interactive-video-generation)
    - [5.2.2 Multimodal Reasoning.](#522-multimodal-reasoning)
    - [5.2.3 3D Generation.](#523-3d-generation)
    - [5.2.4 Vision-Language-Action Generation.](#524-vision-language-action-generation)
- [6 Conclusion](#6-conclusion)

## Abstract

World models have garnered significant attention as a promising research direction in artificial intelligence, yet a clear and unified definition remains lacking. In this paper, we introduce OpenWorldLib, a comprehensive and standardized inference framework for Advanced World Models. Drawing on the evolution of world models, we propose a clear definition: a world model is a model or framework centered on perception, equipped with interaction and long-term memory capabilities, for understanding and predicting the complex world. We further systematically categorize the essential capabilities of world models. Based on this definition, OpenWorldLib integrates models across different tasks within a unified framework, enabling efficient reuse and collaborative inference. Finally, we present additional reflections and analyses on potential future directions for world model research.

[*]Equal Contribution \contribution[†]Project Leader \contribution[‡]Corresponding author \checkdata[ Correspondence ] \checkdata[ Source Code ] [https://github.com/OpenDCAI/OpenWorldLib](https://github.com/OpenDCAI/OpenWorldLib) \checkdata[ Documentation ] [https://wcny4qa9krto.feishu.cn/wiki/XtPJwf5XQipP7RkeVv0ckyWlnNd](https://wcny4qa9krto.feishu.cn/wiki/XtPJwf5XQipP7RkeVv0ckyWlnNd)

<a id="section-1"></a>

## 1 Introduction

<a id="figure-1"></a>

![email_logo](images/email_logo.png)

> Figure 1: Overview of our OpenWorldLib. Our OpenWorldLib establishes a unified framework for existing world model-related tasks, encompassing perception, understanding, memory, and generation of physical world inputs.

With the gradual advancement of LLMs and Agents [[7], [119], [99], [166], [86], [106], [124], [66], [107], [77], [37], [21], [36], [102], [101], [9], [35], [25], [75], [80], [8], [4]], models are urgently required to transition from virtual-world usage to real-world applications. As a result, world models have begun to enter the spotlight, with researchers increasingly focusing on the ability of large models to function in the physical world, moving beyond virtual environments.

The concept of world models was initially introduced by [[40]], and later works such as [[42], [12], [5]] began to next-frame-predict tasks like video generation and 3D generation as forms of world modeling. As the field has evolved, subsequent studies have explored the applications of world models, and numerous survey [[167], [46], [26], [131], [58], [82], [146], [70], [91], [32], [88], [145], [120], [30], [11], [47], [148]] and position [[59], [139], [10], [137], [152]] papers have provided summaries and analysis. However, despite these efforts, the definition and scope of world models remain diverse, and a broadly accepted consensus has yet to be established.

To provide a standardized definition of world models, it is helpful to start with their core objective: the ability to continuously learn from and interact with the real world. Accordingly, we define a world model as a model or framework centered on building internal representations from perception, equipped with action-conditioned simulation and long-term memory capabilities, for understanding and predicting the dynamics of a complex world. As noted in [[152]], a world model is not tied to any specific task or particular architecture, but rather represents a level of capability that a model or framework should aim to achieve, specifically, the ability to perceive, interact with, and remember a complex world.

This article primarily defines which tasks fall within the capabilities a world model should possess, and which tasks are often mistakenly considered as what world models should achieve. Moreover, because world models require a diverse set of capabilities, a more systematic approach to invoking them is needed. To this end, we build OpenWorldLib, a unified world model inference framework that standardizes the invocation of tasks such as interactive video generation, 3D generation, multimodal reasoning, and vision-language-action (VLA) under a single framework.

The main contributions of this paper are as follows:

- Contribution 1. We provide a standardized definition of world models, clarifying which tasks should be considered part of a world model’s capabilities.
- Contribution 2. We propose OpenWorldLib, a unified world model inference framework, to help structure and standardize research in this area.
- Contribution 3. We offer further analysis and discussion on the future development of world models.

<a id="section-2"></a>

## 2 Background and Related Works

World models [[40], [41]] are typically defined by three core conditional probability distributions:

$$
\displaystyle\text{State transition model:}\quad p(s_{t+1}\mid s_{t},a_{t}) (1) \displaystyle\text{Observation model:}\quad p(o_{t}\mid s_{t}) \displaystyle\text{Reward model:}\quad r_{t}\sim p(r_{t}\mid s_{t},a_{t})
$$

where $s_{t}$ denotes the latent state, which intrinsically incorporates memory storage to manage long-horizon dependencies for complex tasks; $a_{t}$ represents the action at time step $t$, drawn from an action space that has been broadened to encompass diverse operations and task-specific outputs such as generation and manipulation; $o_{t}$ is the perceptual observation (e.g., vision, audio, or proprioception); and $r_{t}$ is the reward obtained through interactions between actions and the environment.

Despite the wide use of these formulations, many tasks formally satisfy such conditional probability distributions without actually serving the core purpose of world models. These tasks are often conflated with or loosely labeled as world model research. Therefore, in this section, we draw on definitions proposed in prior work alongside the perspective advocated in this paper to clearly delineate which tasks fall within the scope of genuine world model research and which do not.

- [2.1 World Model Related Tasks](#21-world-model-related-tasks)
- [2.2 The Role of 3D and Simulators in World Models](#22-the-role-of-3d-and-simulators-in-world-models)
- [2.3 Methods Not Considered World Models](#23-methods-not-considered-world-models)

<a id="section-2-1"></a>

### 2.1 World Model Related Tasks

- [Interactivate Video Generation.](#interactivate-video-generation)
- [Multimodal Reasoning.](#multimodal-reasoning)
- [Vision-Language-Action.](#vision-language-action)

##### Interactivate Video Generation.

Next-frame prediction is widely regarded the most recognized paradigm by world model researchers [[40]], establishing interactive video generation as the main focus of research in this field. Early approaches primarily relied on regression-based models [[40], [42], [134], [93]] to predict subsequent frames. More recently, the field has shifted towards leveraging diffusion models [[44], [48]] to achieve higher-quality interactive video generation, with unified multimodal approaches [[135], [136], [23], [61], [129]] further advancing generation fidelity and controllability. With the accelerated inference speeds of diffusion models, game video generation [[65], [111], [138]] and camera-controlled video generation [[6], [92]] has emerged as a particularly prominent area of interest. Furthermore, the video prediction paradigm has been successfully integrated into Vision-Language-Action (VLA) models [[13], [53], [117]] and autonomous driving systems [[153], [45], [104]]. By incorporating next-frame prediction estimation, these models achieve significantly enhanced stability and robustness in their predictive capabilities. However, while interactive video generation remains a cornerstone of current world model research, it is important to note that next-frame prediction is not the sole implementation paradigm. Considering that the ultimate objective of a world model is to facilitate long-term interactions within complex environments, exploring alternative or complementary representational paradigms is equally crucial [[137]].

##### Multimodal Reasoning.

A critical capability of a world model lies in its profound understanding of the complex physical world; thus, multimodal reasoning serves as a key reflection of a world model’s capabilities. Multimodal reasoning tasks closely associated with world models encompass not only spatial reasoning [[94], [64], [133], [17], [105], [108], [142], [33], [24]] and omni reasoning [[140], [141], [132], [50], [19], [163], [109], [165], [84], [3], [18], [27]], but also temporal reasoning [[98], [100], [14], [72], [73]], causal reasoning [[56], [31]]. Recently, beyond traditional explicit reasoning methods, utilizing latent reasoning [[5], [126]] to analyze complex dynamics in the real world has emerged as a prominent research hotspot. By shifting away from the traditional text-centric pre-training paradigms of Large Language Models (LLMs), latent reasoning mechanisms enable models to more efficiently ingest and process the complex, high-dimensional, and continuous information inherent in the real world.

##### Vision-Language-Action.

The ultimate goal of a world model is to enable agents to interact with the physical world, and embodied devices serve as the primary representatives for interacting with complex environments. Consequently, Vision-Language-Action (VLA) has become a crucial capability that world models must support. In the domain of robotic arm manipulation, recent research primarily follows two trajectories: utilizing Multimodal Large Language Models (MLLMs) to directly predict actions [[13], [53], [117], [113], [154], [83], [5], [16], [34]], or combining action prediction with video generation [[2], [114], [67], [20], [130], [95]] to facilitate action planning through future frame prediction. Furthermore, this VLA paradigm is being broadly applied to more complex embodied scenarios, including mobile robots with highly complex and difficult-to-control dynamics [[49], [62]], and autonomous driving systems operating in vastly broader environments [[155], [45], [104], [168], [60], [127], [159], [69], [158]], thereby advancing the closed-loop interactive capabilities of models in the real world.

<a id="section-2-2"></a>

### 2.2 The Role of 3D and Simulators in World Models

Beyond tasks that rely on directly observable perception, a key part of world models involves processing virtual environments. To ensure that physical space remains consistent during long-term interactions, researchers often use simulators to let models learn in a structured way. While interactive video generation creates a visual guess of the future, 3D representations provide a verifiable environment where physical rules can be strictly followed [[63], [89], [160], [68], [118], [122], [85], [150], [143], [151]].

In this context, 3D generation and reconstruction are essential for keeping a stable world state. Recent works like VGGT [[123]], InfiniteVGGT [[147]], and OmniVGGT [[103]] use visual geometry grounded transformers to link image inputs with real geometric structures. To handle continuous data from the real world, some models now maintain a persistent 3D state [[125]] or utilize hybrid memory for long-context reconstruction [[156]], making sure the environment stays the same even when the agent moves. Additionally, new methods in metric 3D reconstruction [[55]], depth estimation [[81]], and large view synthesis [[54]] allow world models to recover exact physical spaces from any camera angle. By learning permutation-equivariant visual geometry [[128]], these models can work better across different types of physical settings.

Furthermore, simulators act as a "sandbox" for world models, helping them move from abstract thinking to real physical action. For these simulators to work in real-time, fast scene generation is necessary. For example, FlashWorld [[71]] and the Hunyuan series [[115], [144], [162], [52]] can create high-quality 3D scenes or assets in a very short time, giving the world model an immediate place to test its ideas. Recent investigations also explore the potential of reinforcement learning in these 3D generation processes [[112]]. By using these explicit 3D representations and simulation tools, world models can go beyond just predicting pixels and truly understand the physical rules of the real world.

<a id="section-2-3"></a>

### 2.3 Methods Not Considered World Models

Besides world model-related tasks, certain applications do not truly reflect a world model’s capabilities, yet they frequently appear in similar discussions. Based on the formulation and our specific definition of world models, this section clarifies which tasks fall outside this category.

A prominent example of this misconception is text-to-video generation. When Sora was released, many label it a "world simulator." However, [[167]] argues that Sora does not constitute a complete world simulator. While next-frame prediction frequently associates with world models, our definition emphasizes that the key lies not in the output format, but in whether the model utilizes multimodal inputs to analyze and recognize the environment. Next-frame prediction serves merely as one format. What truly matters is whether the model accurately understands complex physical rules and interacts with the world. Text-to-video generation lacks this complex perceptual input. Even though generating videos demonstrates some understanding of physics, it remains outside the core tasks of world models.

Similarly, some tasks, such as the code generation or web search [[22], [29]], borrow the long-term interaction structure of world models for other domains. However, these tasks typically lack multimodal inputs and do not involve understanding the physical world. While applying this structure to new areas presents interesting opportunities, these tasks do not qualify as true world models.

Even applications that actually involve multimodal [[28]] and long-term interaction, such as avatar video generation [[51], [149], [38]], do not necessarily fit the definition. These tasks primarily focus on entertainment. Because they have little to do with exploring or understanding the complex physical world, they do not represent a primary focus for world models.

<a id="figure-2"></a>

![email_logo](images/email_logo.png)

> Figure 2: Illustration of our OpenWorldLib framework.

<a id="figure-3"></a>

![github_logo](images/github_logo.png)

> Figure 3: Demonstration of world model Implicit representation and explicit representation.

<a id="section-3"></a>

## 3 OpenWorldLib Framework Design

Based on Section [2](#section-2), a world model requires the following capabilities: receiving inputs from the complex physical world, understanding the physical world, maintaining long-term memory during interactions, and supporting multimodal outputs. Although [[152]] proposed a design for a unified world model framework, it lacks a concrete engineering implementation and even a unified standard. This section details the specific design of our OpenWorldLib framework, as shown in Fig. [2](#figure-2).

- [3.1 Operator](#31-operator)
- [3.2 Synthesis Module](#32-synthesis-module)
- [3.3 Reasoning Module](#33-reasoning-module)
- [3.4 Representation Module](#34-representation-module)
- [3.5 Memory Module](#35-memory-module)
- [3.6 Pipeline](#36-pipeline)

<a id="section-3-1"></a>

### 3.1 Operator

In the OpenWorldLib framework, the Operator module serves as the crucial bridge between raw user inputs (or environmental signals) and the core execution modules (Synthesis, Reasoning, and Representation). Because a world model must handle complex, multimodal inputs from the physical world—such as text prompts, images, continuous control actions, and audio signals—the Operator is designed to standardize these diverse data streams.

Specifically, when the Pipeline is called, it routes the raw input through the Operator’s process() method. The Operator is responsible for two primary functions:

- Validation: Ensuring that the input data formats, shapes, and types meet the requirements of the downstream models.
- Preprocessing: Transforming raw signals into standardized tensor representations or structured formats (e.g., resizing images, tokenizing text, or normalizing action spaces).

To facilitate the integration of new world model methods, we define a unified Operator template. All task-specific operators inherit from this base class, ensuring a consistent API across the entire codebase. The definition of Operator is shown in Listing LABEL:lst:base_operator.

<a id="section-3-2"></a>

### 3.2 Synthesis Module

As shown in the implicit representation part of Fig. [3](#figure-3), a core capability of world models is using internal learned dynamics to generate visual, auditory, and other sensory outcomes as environmental feedback. We define this implicit generative process as the model’s implicit representation. In the OpenWorldLib framework, the Synthesis module serves as the generative bridge between standardized conditioning from upstream pipelines and the multimodal outputs(visual, auditory, and embodied) that users, simulators, or robotic stacks actually consume. Because a world model must realize predictions not only as internal states but as observable media and executable commands, Synthesis hosts heterogeneous generative backends while preserving a coherent integration pattern across modalities.

Specifically, when the Pipeline runs a generation path, it passes operator-aligned inputs to the appropriate synthesis backend, which performs inference under modality-specific controls and returns structured artifacts together with concise metadata for export, evaluation, or memory. The following subsections unpack this module along its visual, audio, and other physical-signal synthesis branches.

- [3.2.1 Visual Synthesis](#321-visual-synthesis)
- [3.2.2 Audio Synthesis](#322-audio-synthesis)
- [3.2.3 Other Signal Synthesis](#323-other-signal-synthesis)

#### 3.2.1 Visual Synthesis

The visual synthesis layer covers image and video oriented generation in OpenWorldLib: it turns structured conditioning, such as text prompts, reference images, or scene-level specifications, into raster outputs (frame tensors, decoded clips, or API-returned assets) together with metadata for export, evaluation, or optional memory hooks. In this way, the framework can furnish visible predictions of how scenes evolve over time, which is essential for interactive simulation, qualitative inspection, and comparing alternative futures or camera paths at a glance. These outputs also anchor multimodal storytelling and dataset-style recording when visual evidence must accompany text or control signals.

In practice, the visual synthesis layer is organized around the following responsibilities:

- Generative stack composition: Combining text encoders, latent decoders, and diffusion- or flow-based cores with schedulers or solvers appropriate to each task, and exposing knobs for spatial resolution, temporal extent (frame budget), and guidance-style parameters.
- Integration surfaces: Supporting checkpoint-driven pipelines (unified construction from pretrained resources and no-gradient inference) alongside hosted-service wrappers that authenticate via endpoints and credentials, so that local and remote generators share the same conceptual call pattern.

#### 3.2.2 Audio Synthesis

The audio synthesis layer focuses on continuous waveform generation under structured conditioning, commonly text, optional video-derived features, and timing or batch metadata, and returns waveforms with sampling rates and compact result records for downstream saving or metrics. Its role is to supply the auditory side of multimodal outputs so that scenarios are not limited to silent video or text-only feedback, which matters for perception-rich environments and for judging alignment between sound and visuals [[39]].

Concretely, the audio synthesis layer fulfills the following roles:

- Resource assembly: Instantiating the neural audio generator and any auxiliary modules (e.g., feature encoders) from pretrained sources through a single factory-style entry point, with explicit device and reproducibility-related settings.
- Conditional waveform synthesis: Mapping operator-prepared tensors and prompts to audio outputs via a unified inference entry point, with user-facing controls such as duration, random seeds, guidance strength, and sampling-step budgets.

#### 3.2.3 Other Signal Synthesis

Beyond visual and audio modalities, comprehensive interaction with the environment requires world models to generate a diverse spectrum of physical signals. Among these, action control proves especially critical, as it constitutes the fundamental mechanism through which embodied agents actively manipulate the physical world. OpenWorldLib therefore places a primary emphasis on Vision-Language-Action (VLA) signal generation within this module. This synthesis layer is tailored for embodied tasks and fulfills the following functions:

- Policy Initialization and Space Alignment: Loading specialized physical policies (e.g., VLA foundation models) from pretrained weights. Crucially, it maps diverse action representations—from discrete language-like tokens to continuous kinematic states—into unified interfaces compatible with target simulators or robotic hardware.
- Context-Conditioned Action Synthesis: Translating rich, multimodal contexts (such as real-time visual streams, textual goals, and proprioceptive histories) into grounded physical commands. The module yields executable action sequences and essential control metadata to drive closed-loop environmental interactions.

<a id="section-3-3"></a>

### 3.3 Reasoning Module

From the implicit representation part of Fig. [3](#figure-3), a world model must transcend mere perception to understand the physical world: inferring spatial relationships, integrating multimodal contexts, and generating grounded semantic interpretations prior to any downstream generation or action takes place. To address this necessity, OpenWorldLib introduces a dedicated Reasoning module, designed to equip the world model with structured understanding capabilities essential for complex physical inference. Concretely, the Reasoning module is organized into three sub-categories that reflect the distinct perceptual channels a world model must handle:

- General Reasoning: Multimodal large language models (MLLMs) capable of processing text, images, audio, and video in a unified manner.
- Spatial Reasoning: Models specialized in 3D spatial understanding and object localization from visual observations.
- Audio Reasoning: Models that interpret and reason over auditory signals.

To facilitate the integration of new reasoning-oriented world model methods, we define a unified BaseReasoning template. All task-specific reasoning classes inherit from this base class, ensuring a consistent API across the entire codebase. The definition of BaseReasoning is shown in Listing LABEL:lst:base_reasoning.

<a id="section-3-4"></a>

### 3.4 Representation Module

Apart from models that use internal capabilities to understand the world, some methods aim to build human-defined simulators, such as 3D meshes. These simulators provide a testable environment for the world model framework. Since these structured representations are different from the perception data that can be directly collected from the world, we design the Representation module separately from the Synthesis module to handle these explicit representations (e.g., 3D structures).

Specifically, the Representation module is designed to bridge the gap between raw perception and structured simulation. Its main functions include:

- 3D Reconstruction: It transforms input data into explicit 3D outputs, providing structured information such as point clouds, depth maps, and camera poses.
- Simulation Support: It creates a manual environment where the world model can test its reasoning and validate if its predicted actions are correct in a coordinate system.
- Service Integration: It supports both local inference and cloud-based APIs to help export these explicit representations to external physics engines.

To standardize how these models are used, we provide a unified BaseRepresentation template. All task-specific representation classes inherit from this base class to ensure a consistent API. The definition of BaseRepresentation is shown in Listing LABEL:lst:base_representation.

<a id="section-3-5"></a>

### 3.5 Memory Module

Long-term contextual memory is essential for interactive world models to maintain historical observations, reasoning chains, and interaction states. OpenWorldLib designs a unified Memory module to manage multimodal interaction history.

The Memory module serves as the persistent state center of the framework. It records structured information from perception, reasoning, generation, and action, and provides efficient context retrieval for multi-turn interactive tasks. Specifically, the Memory module fulfills the following functions:

- Historical Storage: Storing text, visual features, action trajectories, and scene states across interactions.
- Context Retrieval: Selecting relevant history to support consistent reasoning and generation.
- State Update: Recording new interaction results after each pipeline execution.
- Session Management: Supporting independent memory for different tasks and sessions.

To unify memory management, we define a unified BaseMemory template. All task-specific memory classes inherit from this base class. The definition of BaseMemory is shown in Listing LABEL:lst:base_memory.

<a id="section-3-6"></a>

### 3.6 Pipeline

To integrate the Operator, Reasoning, Synthesis, Representation, and Memory modules into a cohesive and usable system, OpenWorldLib provides a unified Pipeline module as the top-level scheduling and execution entry. The Pipeline encapsulates model initialization, data flow, module invocation, memory interaction, and result post-processing, enabling end-to-end world model inference with a simple, consistent API.

The Pipeline follows a standard forward-execution workflow: it receives raw user or environmental input, routes it to the Operator for validation and preprocessing, queries the Memory module for historical context, coordinates the Reasoning, Synthesis, and Representation modules for core computation, and finally returns structured outputs while updating the memory. This design fully decouples module implementations while ensuring efficient and reliable data transmission. The core responsibilities of the Pipeline include:

- Unified Model Initialization: Loading pretrained weights, configuring devices, and instantiating all submodules through a single from_pretrained() interface.
- End-to-End Inference: Implementing one-click forward inference via the **call**() method for single-turn world model tasks.
- Multi-turn Interactive Execution: Supporting stateful, continuous interaction through the stream() method with automatic memory reading and writing.
- Modular Orchestration: Dynamically invoking Reasoning, Synthesis, or Representation according to task type without modifying internal module logic.
- Result Structuring: Organizing outputs into standardized formats for visualization, evaluation, logging, or downstream control systems.

To maintain framework-wide consistency, all task-specific pipelines inherit from a unified BasePipeline template. Its definition is shown in Listing LABEL:lst:base_pipeline.

<a id="section-4"></a>

## 4 Discussion

OpenWorldLib is designed to provide a clearer and more standardized definition and framework for world models. Its goal is to promote the development of world models so that AI can better assist humans in complex environments. In this section, we discuss the future development directions of world models.

Many current world model architectures focus on next-frame prediction. This approach aligns with how humans process high-density sensory inputs, as humans are essentially "pre-trained" in the physical world, whereas large models are pre-trained on massive internet text corpora [[79], [78]]. However, based on existing architectures, VLMs might offer a practical solution. For example, Bagel [[161]] successfully achieves both multimodal reasoning and multimodal generation using the Qwen architecture. This demonstrates that Large Language Models (LLMs) pre-trained on internet data can possess all the capabilities required for a world model, showing their potential to serve as the foundational base. Therefore, before focusing entirely on the specific structural design of world models, we should first consider how to implement all their necessary functions to enable true and effective interaction with the complex world. Moreover, as LLMs serve as the foundational backbone for world models, data-centric methodologies—including multimodal data synthesis [[90], [74]], domain-specific data augmentation [[15], [164]], dynamic training [[80]], and training data quality evaluation [[76]]—will play an increasingly important role in strengthening the foundational models that underpin world model capabilities.

During real-world interactions, next-frame prediction retains more information compared to next-token prediction, but its efficiency needs significant improvement. To boost this efficiency, improvements must start at the hardware level. Current computer byte organization naturally favors next-token prediction. Even when models attempt next-frame prediction, the data is still processed as tokens during actual computation. To achieve the ideal world model, we need hardware iterations, changes to the foundational model structure (token-based Transformers may need to evolve), and the comprehensive realization of complex physical world interaction tasks.

<a id="figure-4"></a>

![email_logo](images/email_logo.png)

> Figure 4: Demonstration of interactive video generation results.

<a id="figure-5"></a>

![github_logo](images/github_logo.png)

> Figure 5: Demonstration of 3D scene generation results.

<a id="section-5"></a>

## 5 Evaluation

Following the introduction of the OpenWorldLib framework design, this section presents the testing process and evaluation results of OpenWorldLib.

- [5.1 Experimental Setting](#51-experimental-setting)
- [5.2 Experimental Results](#52-experimental-results)

<a id="section-5-1"></a>

### 5.1 Experimental Setting

We primarily conduct our experiments using NVIDIA A800 (80GB VRAM) and H200 (141GB VRAM) GPUs. In the future, we plan to evaluate our framework on a wider range of hardware devices.

<a id="section-5-2"></a>

### 5.2 Experimental Results

- [5.2.1 Interactive Video Generation.](#521-interactive-video-generation)
- [5.2.2 Multimodal Reasoning.](#522-multimodal-reasoning)
- [5.2.3 3D Generation.](#523-3d-generation)
- [5.2.4 Vision-Language-Action Generation.](#524-vision-language-action-generation)

#### 5.2.1 Interactive Video Generation.

For video generation, our evaluation primarily encompasses tasks such as navigation video generation and interactive video generation. The significance of video generation for a world model lies in assessing its understanding and memory of the complex world, while simultaneously assisting other sequential reasoning tasks in making accurate predictions. When executing video generation tasks, the world model must produce videos that conform to the precise visual evolution required by the specific task.Specifically, the inputs for these tasks consist of visual conditions (e.g., single images or image sequences) paired with diverse interaction signals, which include textual instructions, directional movement controls (forward, backward, left, right), and camera rotation commands.

As shown in Fig. [4](#figure-4), we evaluate and analyze the generation performance of various methods. In the context of navigation video generation, early approaches like Matrix-Game-2 [[157], [43]] offer fast generation speeds but suffer from noticeable color shifting during long-horizon generation. In contrast, recent models such as Lingbot-World [[116]], Hunyuan-GameCraft [[65], [111]], and YUME-1.5 [[97], [96]] successfully support high-quality navigation video generation, with Hunyuan-WorldPlay [[110]] achieving the best overall visual performance. Regarding interactive video generation, although Wan-IT2V [[121]] can execute basic interactive generation, it struggles with maintaining physical consistency. Furthermore, for generating complex interactive operations, while WoW [[20]] supports a diverse range of functionalities, its generation quality and physical realism are significantly inferior to those of Cosmos [[1]].

#### 5.2.2 Multimodal Reasoning.

In OpenWorldLib, the Reasoning module groups high-level cognitive tasks that require a world model to interpret multimodal evidence and produce explicit, verifiable conclusions. It covers spatial reasoning [[94], [64]] (e.g., answering geometry- and layout-centric queries, resolving object relations, and performing step-by-step spatial deductions from visual inputs) as well as omni/general reasoning [[141]] that operates over mixed modalities (text, images, audio, and videos) to support broad instruction following and multimodal understanding. The importance of this module for a world model lies in making internal perception and memory actionable: it turns observations into grounded decisions, explanations, and plans that can guide downstream generation or control. When executing reasoning tasks, inputs typically consist of an instruction or question paired with optional perceptual signals such as images, video clips, or audio segments, encoded into model-ready representations; the outputs are primarily decoded natural-language responses, and for certain omni reasoning settings may additionally include generated audio alongside the text.

<a id="figure-6"></a>

![email_logo](images/email_logo.png)

> Figure 6: Demonstration of simulator generation results.

#### 5.2.3 3D Generation.

The 3D generation pipeline in OpenWorldLib supports 3D scene reconstruction, enabling robust representations of complex real-world environments. The perceptual inputs for this pipeline typically consist of single images or image sequences, while the interaction signals involve movement controls or camera viewpoint adjustments (e.g., polar, azimuth, and yaw angles). As shown in Fig. [5](#figure-5), although VGGT [[123]] and InfiniteVGGT [[147]] can generate 3D scenes from different views, they still have clear limitations. For instance, when the camera moves significantly, these models often struggle with geometric inconsistency and show texture blurring in complex areas, which affects the overall realism. While faster methods like FlashWorld [[71]] speed up the process, balancing steady shapes with sharp details remains a major challenge. Nevertheless, as a crucial technique for realistic physical simulation, 3D generation remains fundamentally important for the development of world models.

#### 5.2.4 Vision-Language-Action Generation.

Similar to 3D generation, simulation environments constitute an indispensable component of world model evaluation, serving as controllable testbeds for both embodied video synthesis and action generation. To this end, OpenWorldLib incorporates two complementary simulation-based paradigms: AI2-THOR [[57]] for embodied video generation, enabling photorealistic scene rendering and dynamic agent-environment interaction; and LIBERO [[87]] for Vision-Language-Action (VLA) evaluation, providing reproducible and physically grounded manipulation environments. Together, these paradigms rigorously assess the world model’s capacity to couple semantic understanding with physical dynamics and fine-grained action planning across diverse interactive scenarios.

As shown in Fig. [6](#figure-6), we present representative evaluation cases from both LIBERO and AI2-THOR simulation environments, showcasing diverse manipulation tasks and embodied interaction scenarios. Furthermore, our framework supports the evaluation of a comprehensive suite of VLA methods. Prominent examples include $\pi_{0}$ [[13]] and $\pi_{0.5}$ [[53]], which leverage the PaliGemma vision-language backbone augmented with mixture-of-experts (MoE) action heads to achieve robust multi-task generalization. We also incorporate LingBot-VA [[67]], which approaches the task from a generative perspective by employing a video diffusion architecture to jointly model visual future predictions and continuous action synthesis.

<a id="section-6"></a>

## 6 Conclusion

In conclusion, OpenWorldLib presents a standardized workflow and evaluation pipeline for world models. By providing unified interfaces for core tasks such as interactive video generation and 3D scene reconstruction, the framework standardizes the integration of multimodal perceptual inputs and diverse interaction controls. Ultimately, we hope OpenWorldLib can serve as a practical reference for the research community, facilitating future explorations and fair comparisons in world model research.
