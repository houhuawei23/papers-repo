# Title: VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks

- ArXiv: 2412.18194
- Authors: Shiduo Zhang, , Zhe Xu, , Peiju Liu, , Xiaopeng Yu, , Yuan Li, , Qinghui Gao, ,, Zhaoye Fei, , Zhangyue Yin, , Zuxuan Wu, , Yu-Gang Jiang, , Xipeng Qiu, School of Computer Science, Fudan University, Project Website:
- Sections: 34
- Estimated tokens: 28.8k

## Contents

- 1 Introduction
- 2 Related Works
- 3 VLABench
  - 3.1 Task Description
  - 3.2 Benchmark
  - 3.3 Simulation
  - 3.4 Dataset Construction
- 4 Experiments
  - 4.1 Generalization Ability of VLAs
  - 4.2 Performance of Workflow Utilizing Foundation Model
  - 4.3 Comprehensive Ability of VLMs
- 5 Conclusion
- References
- 6 Benchmark Implementation
  - 6.1 Task Descriptions
  - 6.2 Task Observation
  - 6.3 Domain Randomization
- 7 Simulation and Framework
  - 7.1 Scenes
  - 7.2 Cross Embodiment
  - 7.3 Assets
- 8 Dataset Building
  - 8.1 Skill Library as Domain Specific Language
  - 8.2 Data Collection Progress
  - 8.3 Prompt for Interactive Instruction
- 9 Experiment Implementation
  - 9.1 VLA Setting
  - 9.2 Evaluation of Worksflows
  - 9.3 Evaluation of VLMs
    - Precise Matching Rate (PM).
- 10 Detailed Analysis and Case Study
  - 10.1 Ablations and Analysis for VLAs
  - 10.2 Further Analysis for Workflows
  - 10.3 Ablations and Analysis for VLMs

## Abstract

###### Abstract

General-purposed embodied agents are designed to understand the users’ natural instructions or intentions and act precisely to complete universal tasks. Recently, methods based on foundation models especially Vision-Language-Action models (VLAs) have shown a substantial potential to solve language-conditioned manipulation (LCM) tasks well. However, existing benchmarks do not adequately meet the needs of VLAs and relative algorithms. To better define such general-purpose tasks in the context of LLMs and advance the research in VLAs, we present VLABench, an open-source benchmark for evaluating universal LCM task learning. VLABench provides 100 carefully designed categories of tasks, with strong randomization in each category of task and a total of 2000+ objects. VLABench stands out from previous benchmarks in four key aspects: 1) tasks requiring world knowledge and common sense transfer, 2) natural language instructions with implicit human intentions rather than templates, 3) long-horizon tasks demanding multi-step reasoning, and 4) evaluation of both action policies and language model capabilities. The benchmark assesses multiple competencies including understanding of mesh&texture, spatial relationship, semantic instruction, physical laws, knowledge transfer and reasoning, etc. To support the downstream finetuning, we provide high-quality training data collected via an automated framework incorporating heuristic skills and prior information. The experimental results indicate that both the current state-of-the-art pretrained VLAs and the workflow based on VLMs face challenges in our tasks.

<a id="section-1"></a>

## 1 Introduction

Language-conditioned manipulation represents a fundamental challenge in embodied AI and a stepping stone toward Artificial General Intelligence [4, 13, 1].
Such tasks require agents to master multiple capabilities: interpreting natural language instructions, understanding complex environments, making decisions, formulating plans, and executing precise actions.
The rapid advancement of Large Language Models (LLMs) and Vision-Language Models (VLMs) [1, 14] has revolutionized the field with their impressive general abilities in semantic understanding, coding, planning, and reasoning.
The strong generalization capabilities has inspired two main approaches in language-conditioned manipulation: pre-training vision-language-action models using large-scale robotics data, as demonstrated by RT-2 and Palm-E [4, 13, 46], and integrating foundation models into agent workflows, like VoxPoser and Copa [23, 22], which combine LLM/VLM outputs with grasp prediction [15, 16] and motion planning algorithms [25].

While real-world robotics experiments provide valuable insights, their complexity and environmental variability often challenge reproducibility. Simulation-based evaluation has emerged as a fair and practical alternative. Existing benchmarks like RLBench, Calvin, and LIBERO [24, 41, 35] offer diverse task sets but fall short in addressing the unique requirements of foundation model-based methods. Tasks designed to align with the capabilities of foundation model-based algorithms should encompass nuanced semantic understanding of user intent, the integration of common-sense knowledge, and a robust ability to interpret diverse visual scenes, as well as require sophisticated multi-step reasoning. Such tasks demand a sophisticated integration of multimodal understanding to effectively interpret and respond to complex, real-world contexts. For example, one of the tasks in RT-2 [4] is “move the Coke can to Taylor Swift”, while another task in CoPA [22] is “make me a cup of pour-over coffee”. The first task challenges the robot to use common sense to identify Taylor, a capability of knowledge transfer that previous policies struggled to achieve. The second task further intensifies the difficulty, requiring the robot to decompose the task into subtasks and execute the steps to operate a coffee machine—a long–horizon challenge that has previously been difficult for a single policy to accomplish.

To better define the types of language-conditioned manipulation tasks suited for foundation models and provide a standardized evaluation suite to advance robotics research, we introduce VLABench. VLABench is an open-source benchmark specifically designed for methods utilizing foundation models. The tasks in VLABench are carefully divided into several dimensions to evaluate models across various aspects, including 1) Mastery of common sense and world knowledge, 2) Understanding of mesh and texture, 3) Comprehension of semantically rich instructions, 4) Spatial understanding, 5) Grasp of physical rules, and 6) Reasoning ability. For benchmarking purposes, VLABench offers 100 task categories with comprehensive evaluations across various methods. With a diverse collection of over 2,000 3D objects and scenes, VLABench creates a wide range of visual contexts and tasks. It enables the assessment of generalization capabilities through learning across multiple skills, providing thorough evaluations spanning visual, linguistic, planning, knowledge transfer, and action dimensions.

To ensure fair comparison and evaluation, we develop an automated data collection framework to construct standardized datasets for each task, supporting model training and fine-tuning. Using this dataset, we conduct extensive experiments to evaluate and benchmark three distinct types of approaches: pre-trained VLA, workflows integrating foundation models, and vision-language models (VLMs). The experimental results indicate that existing VLA methods perform poorly on our tasks and don’t exhibit the level of generalization abilities or the “emergent” phenomena observed in large models [58]. We summarize contributions as follows:

- •
  We propose VLABench, the first benchmark designed to comprehensively evaluate the capabilities of VLAs and VLMs in robotics manipulation tasks, covering multiple dimensions such as skills, vision, language, task execution, common sense, and reasoning.
- •
  We define 100 novel LCM tasks tailored to the capabilities of foundation models within a standardized evaluation framework. These tasks require a deep understanding of semantics, vision, spatial reasoning, and physical laws, as well as the ability to plan long-horizon tasks and transfer world knowledge and common sense into task execution.
- •
  We provide a scalable data construction framework and a standardized evaluation dataset. This automated data construction approach facilitates future research on pretraining robotics data.
- •
  Our experiments demonstrate that current pre-trained VLAs have yet to exhibit the strong generalization capabilities observed in LLMs, and existing SOTA VLMs also show limitations in embodied scenarios.

<a id="section-2"></a>

## 2 Related Works

<a id="table-1"></a>

> Table 1: Comparison of Popular Benchmarks in Robot Learning. SemLang: Semantically rich language instructions. LogiReason: Task logic and relevant information reasoning. Knowledge: Tasks require the application of common sense and world knowledge. DR: Strong task domain randomization. N-task: Total number of tasks. Cate-obj: Categories of assets used in the simulation. N-obj: Total number of objects in the asset library. AI-Gen: Use of generative AI models for asset library creation. MultiCam: Use of multiple cameras. PCD: Support point cloud data in 3D methods. Cross Emb: Support for cross-embodiment. Auto Traj: Supporting automated data collection

| Benchmarks            | SemLang | LogiReason | Knowledge | DR  | N-task | Cate-obj | N-obj | AI-Gen | MultiCam | PCD | Cross Emb | Auto Traj |
| --------------------- | ------- | ---------- | --------- | --- | ------ | -------- | ----- | ------ | -------- | --- | --------- | --------- |
| Alfred[49]            | ✗       | ✗          | ✗         | ✗   | 7      | -        | 3578  | ✗      | ✗        | ✗   | ✓         | ✗         |
| Rlbench[24]           | ✗       | ✗          | ✗         | ✗   | 100    | 28       | 28    | ✗      | ✓        | ✗   | ✗         | ✓         |
| Calvin[41]            | ✗       | ✗          | ✗         | ✗   | 34     | 5        | 30    | ✗      | ✓        | ✗   | ✗         | ✗         |
| ManiSkill[19, 42, 53] | ✗       | ✗          | ✗         | ✓   | 20     | 100      | 2600  | ✗      | ✓        | ✓   | ✓         | ✓         |
| LIBERO[35]            | ✗       | ✗          | ✗         | -   | 130    | 51       | 75    | ✗      | ✗        | ✗   | ✗         | ✗         |
| RoboCASA[43]          | ✗       | ✗          | ✗         | ✓   | 100    | 153      | 2509  | ✓      | ✓        | ✗   | ✓         | ✓         |
| ARNOLD[18]            | ✗       | ✗          | ✗         | -   | 8      | -        | 40    | ✓      | ✓        | ✓   | ✗         | ✓         |
| Behavior-1K[29]       | ✗       | ✗          | ✗         | ✓   | 1000   | 2211     | 9331  | ✗      | ✗        | ✗   | ✓         | ✗         |
| Habitat 2.0[52]       | ✗       | ✗          | ✗         | -   | 3      | 46       | 169   | ✗      | ✗        | ✗   | ✗         | ✗         |
| VLABench              | ✓       | ✓          | ✓         | ✓   | 100    | 163      | 2164  | ✓      | ✓        | ✓   | ✓         | ✓         |

Benchmarks and Datasets.
Numerous benchmarks such as RLBench and LIBERO [24, 41, 35, 63, 32] have been proposed to evaluate language-conditioned manipulation policies in realistic physical settings. A comparison of these benchmarks is provided in Table [1](#table-1). However, most of these focus on skill learning and fail to sufficiently address long-horizon planning capabilities.
Meanwhile, some benchmarks [49, 52, 60] address room-scale mobile manipulation tasks that require long-term memory or reasoning. Yet, these interactions typically occur through interfaces, rather than direct physical manipulation, limiting the transferability of learned policies to real-world scenarios. Additionally, while efforts [43, 29, 35, 19] have made strides in task format, difficulty, and scale, these benchmarks have largely overlooked the guiding role of language in tasks, often relying on template instructions that explicitly specify the robot’s actions. VLABench is the first to introduce features such as natural human interaction, implicit goal-oriented semantics, and requirements based on common sense into robot manipulation tasks, as shown in Figure [2](#figure-2). In terms of generalization evaluation, previous works [24, 41, 35] typically assess models at the instance level within the same category, which limits their ability to evaluate generalization across diverse object categories or different tasks within the same skill set. In contrast, VLABench is the first benchmark to evaluate generalization capabilities across a wide range of tasks, object types, and task categories, providing a more comprehensive assessment of model versatility.

Large-scale datasets have been built in both real and simulation [46, 57, 43, 5] for large-scale imitation learning for manipulation. However, real-world data faces challenges related to scalability, making it difficult to gather sufficient data at scale [3]. Simulated datasets, while more scalable, often suffer from limited diversity in scenarios and tasks [24, 41], and still require teleoperation [35, 18] for data collection. VLABench addresses these limitations by offering a broader range of tasks that are more closely aligned with real-world conditions, covering diverse aspects of vision, language, tasks, and skills. Furthermore, it introduces an efficient and robust process for the automated generation of simulated data, significantly enhancing task diversity and scalability.

<a id="figure-2"></a>

![Figure1_overview](images/Figure1_overview.png)

> Figure 2: Long horizon task requiring reasoning. This task involves a request for a latte in an interactive scenario. The agent needs to recognize the requirement for coffee with milk and integrate multiple skills, including picking, placing, tool use, pressing, and pouring.

Pretrained Vision-Language-Action Models.
The recent rise of multimodal models [1, 62, 12, 37] and the collection and organization of operational datasets[45, 57], have led to the integration of vision-language-action models (VLAs) [3, 13, 4, 27] into language-conditioned manipulation tasks. While the term VLA generally refers to models that combine visual and language inputs for policy learning, we focus specifically on approaches leveraging pre-trained models. Several works [13, 3, 27] have applied further training to pre-trained vision-language models (VLMs) for language-conditioned manipulation. These models demonstrate impressive generalization to unseen objects and tasks, yet their control precision is somewhat limited by the discretization of actions [47]. To address this limitation, some approaches have explored using diffusion models [47, 8] as policy networks or using diffusion decoders [31, 59]. Pre-trained models based on diffusion models [40, 34] have shown promising advancements in improving continuous space distribution learning. VLABench includes a selection of these representative methods for comprehensive evaluation.

Framework Utilizing Foundation Models.
Pre-trained language models[5, 50, 14] and vision-language models[1, 39] have demonstrated strong generalization and versatility. Some researchers[23, 22, 36] combine the general perception and cognitive abilities of these pre-trained models with traditional planning and control algorithms to create agent workflows.
These frameworks allow robots to perform complex zero-shot manipulation tasks without requiring additional training.
To harness the capabilities of foundation models for manipulation, some works [33, 23] utilize the code comprehension and generation abilities of large language models alongside motion planning optimization algorithms to tackle fundamental manipulation tasks.
Additionally, some methods[22, 21] leverage large models to decompose long-horizon tasks into subtasks, then integrate perception and trajectory generation modules to construct the entire manipulation pipeline. However, most zero-shot methods of this kind heavily rely on prompt design [23], the accuracy of each module, and even the specific parameters of the models being invoked [22]. While these methods demonstrate strong generalization, they often face challenges with accuracy. VLABench provides a zero-shot evaluation framework to assess the performance of such workflows and offers insights into their effectiveness.

<a id="section-3"></a>

## 3 VLABench

<a id="section-3-1"></a>

### 3.1 Task Description

<a id="figure-3"></a>

![get_coffee_with_sugar](images/get_coffee_with_sugar.png)

> Figure 3: Task examples in each dimension. The first row showcases examples of primitive tasks from Section [3.1](#section-3-1), while the second row presents examples of composite tasks.

VLABench is composed of 60 primitive and 40 composite tasks, categorized by task difficulty and required timesteps. These tasks are designed to encompass a rich variety of skills while covering ample visual and language semantic information. For skill learning, 100 tasks in VLABench cover a wide range including 1) Pick&place, 2) Open&close door, 3) Open&close drawer, 4) Hang objects on the wall, 5) Use tool e.g. Hammer nail, 6) Press button, 7) Insert, 8) Pour, 9) Twist, and 10) Explore. In addition, VLABench places greater emphasis on real-life scenarios and essential daily tasks, representing more interactive language instructions, a wider variety of task settings, the integration of common sense and societal knowledge, and long-horizon tasks requiring logical planning, as shown in Figure [3](#figure-3). Notably, VLABench adopts a stricter definition of task generalization, which will be elaborated on in the Section [3.2](#section-3-2). The whole task list can be found in supplementary material.

Primitive Tasks.
Primitive tasks are divided into five dimensions, each corresponding to the assessment of a specific ability dimension.

- •
  Mesh&Texture Understanding. This type of task requires the model to recognize different meshes and understand various texture features. Take the SelectToy task shown in Figure [3](#figure-3) (a) as an example, the robot is directly required to place a specific toy e.g. Aquaman into a receptacle. The model must possess strong visual capabilities to accurately recognize such complex meshes and textures.
- •
  Spatial Understanding. Spatial understanding tasks involve various spatial relationships, such as the nth left/right position, inside/outside of a receptacle, the mth row and nth column, near/far, and beside a specific object, representing relative positional relationships. Figure [3](#figure-3) (b) shows one case in task in PullBook. Such a complex relative positional relationship imposes extremely high demands on the model’s multimodal understanding [7].
- •
  Common Sense & World Knowledge. Tasks relative to common sense/world knowledge require the agent to transfer the knowledge gained in the pre-train stage to solve the problem. The task shown in Figure [3](#figure-3) (c) requires the agent not only to recognize different types of flowers from visual information but also to leverage world knowledge to determine that “the tulip is the national flower of the Netherlands”.
- •
  Semantic Understanding. This type of task emphasizes the complexity, subtlety, and natural interactivity of language instructions. Task objectives are often implicitly conveyed through a natural conversation. To perform well in GetDrink task as Figure [3](#figure-3) (d) shown, the agent must capture the implied request from a lengthy instruction: to take out a chilled cola from the refrigerator.
- •
  Physical Law. This type of task expects the robot to integrate visual information and take correct actions based on physical principles and real-time observation. In UseSeesaw task in Figure [3](#figure-3) (e), the robot is commanded to grasp an object that can not be achieved directly. The agent must recognize the need to apply the principle of leverage by using sufficient weight to lift the target object on the other end.

Composite Tasks.
Composite tasks in VLABench involve the combination of multiple skills, long-term task planning, and multi-step logical reasoning from instructions, scenes, and even game rules. Figure [3](#figure-3) (f) showcases a variety of challenging complex tasks. Composite tasks have a significantly longer trajectory horizon, with an average episode length exceeding 500 timesteps—considerably more than the average of 120 timesteps for primitive tasks. In Figure [3](#figure-3) (f1), The agent must not only correctly identify all poker cards from visual information and use world knowledge of poker rules to select the best hand, but also flip face-down cards to acquire complete information. This type of task, requiring the agent to consciously fulfill prerequisite conditions, has never been modeled before. Composite tasks also require extracting the user’s implicit needs from natural dialogue. The case in Figure [3](#figure-3) (f2) needs the agent to place the Python textbook on the table and open the laptop, without direct instruction.

<a id="section-3-2"></a>

### 3.2 Benchmark

Evaluation.
VLABench organizes evaluations into three main categories: assessments of pretrained or fine-tuned vision-language-action (VLA) models, heuristic workflows that integrate foundation models with various algorithms, and multi-dimensional evaluations of vision-language models (VLMs).

- •
  Generalization Ability of VLAs. For trained vision-language-action (VLA) models, the evaluation in VLABench includes two settings: seen objects and unseen objects. The seen objects evaluation closely aligns with the data distribution of the training set, primarily testing the model’s skill acquisition. Meanwhile, the unseen objects evaluation presents a greater challenge, requiring the model to exhibit strong generalization capabilities. Unlike previous benchmarks [24, 41], VLABench defines unseen objects as entirely different categories. For instance, in the PickFruit task, target objects for seen evaluation include apples, bananas, pears, and oranges, while unseen objects include kiwis, mangos, strawberries, lemons, and other distinct fruits. This setup requires the model to demonstrate not only strong visual generalization capabilities but also to handle the vastly differing common-sense knowledge associated with different categories of objects, as well as the challenge of processing lengthy instructions with unfamiliar tokens.
- •
  Zero-shot Transfer Ability of Heuristic Workflow. Training-free workflow methods are evaluated under a single setting but in many ability dimensions. Apart from the capability points for primitive tasks mentioned in Section [3.1](#section-3-1), we extend the evaluation to cover various skills and long-horizon tasks to assess the overall capability and execution robustness of the workflow.
- •
  Comprehensive Evaluation of VLMs’ Capabilities. Similar to heuristic workflows, the evaluation of VLMs is also comprehensive. Since VLMs lack intrinsic action capabilities, we organized a skill library and integrated it into a domain-specific language (DSL) [44, 51], leveraging annotated asset information as prior knowledge. This DSL functions as a straightforward API that VLMs can call, enabling efficient interaction. The whole evaluation pipeline will be discussed in Section [4.3](#section-4-3).

Metric.
Our evaluation focuses on generalization capabilities, but the task success metric, limited to a 0/1 score, is better suited for assessing straightforward skill learning. Thus, we introduce Progress Score (PS) as a graduated metric for more nuanced assessment. The computation equation of PS is:

$$
PS=\alpha\cdot\frac{n_{correct}}{N}+(1-\alpha)\cdot\frac{m_{done}}{M}(1)
$$

where $N$ indicates the total number of target objects and receptacles, $n_{correct}$ is the number of those selected correctly. $M$ represents the total number of sub-steps in the task, with $m_{done}$ indicating the number of completed sub-steps. Here, $\alpha$ is the weight assigned to correct decisions, default set to 0.2, while $1-\alpha$ represents the weight assigned to task progress. For the evaluation of VLMs, we employed a more detailed scoring method, the metric includes Skill Recall Rate, Parameter Recall Rate, Skill&Parameter Recall Rate, and Precise Matching Rate. Please refer to Section [9.3](#section-9-3) for further details.

<a id="section-3-3"></a>

### 3.3 Simulation

Simulator. VLABench is built based on Mujoco[55] and its control suite dm_control[56]. We selected Mujoco as the core simulation platform for our benchmark due to its lightweight design, high performance, and exceptional physical realism. These advances enable convenient, rapid evaluation of diverse algorithms. The VLABench framework is highly modular, meaning various object entities can be flexibly combined to create large-scale and diverse tasks and scenarios.

Assets.
To meet the requirements of diverse tasks and capability assessments, we built an asset library centered around multiple task themes. We inherited some annotated assets from Robocasa [43] and retrieved numerous 3D models from Objaverse[11]. For novel tasks, such as the series of tasks we created around the toy theme, we carefully gathered a variety of high-quality character models from online 3D model sites. These models were then converted to MJCF format using the obj2mjcf [61] tool. Similarly to previous work [43, 29], we expanded the dataset of common simple objects using generative AI models. Specifically, we utilized Tripo.aI’s text-to-3D and image-to-3D features to construct additional 3D objects, and Runaway.ai to generate multiple material textures.
Ultimately, the asset library we constructed contains 163 categories of objects, totaling 2164 items. Most of the assets are listed in Section [7.3](#section-7-3).

Robots.
To ensure versatility and broad applicability, we integrated a range of embodiment types. These include, but are not limited to, various models of 6-axis and 7-axis robotic arms, dual-arm robots, and humanoid robots. In the standard evaluation process, VLABench employs a 7-DoF Franka Emika Panda manipulator equipped with a parallel gripper. We represent the position and orientation of the robot’s end-effector in Euclidean space $\mathbb{R}^{3}$ using 3D coordinates for position and quaternions for orientation. Using inverse kinematics, we then resolve these end-effector poses into the corresponding rotational angles for the seven joints.

<a id="section-3-4"></a>

### 3.4 Dataset Construction

Domain Randomization.
To ensure data diversity and richness, we implemented various types of domain randomization. These randomizations include object position and orientation, mesh scale, scene layout, background and object textures (such as walls, floors, and tabletops), as well as lighting parameters. Details can be found in Section [6.3](#section-6-3).

Trajectory Generation.
As human teleoperation is time-consuming and not scalable [35, 43], we developed an efficient, scalable automated data collection pipeline based on our custom skill library. Inspired by [18], our data collection framework leverages the prior information including point clouds of the environment, entities’ grasp-points, target entity at the current step, etc. The data collection framework includes multiple task-specific motion planners. These motion planners call upon the skills in the skill library based on the current task progress and determine parameters by incorporating prior information. Subsequently, the selected skills generate trajectories using RRT [26], with quaternion interpolation achieved through Spherical Linear Interpolation (SLERP). The final trajectory is smoothed using a Bezier curve to optimize path quality. To enhance sample efficiency during data collection, reject sampling and failure-triggered early termination are applied.

Instruction Augmentation.
We use GPT-4 [1] to generate descriptions that incorporate target-specific characteristics and interactive instructions that encompass a variety of contexts and intentions. The supplementary material provides details on the generation process and the complete prompts.

<a id="section-4"></a>

## 4 Experiments

Following Section [3.2](#section-3-2), we conducted experiments centered on pre-trained VLA models, workflows incorporating multiple algorithmic modules, and various VLMs. The remainder of this section provides a detailed description of the experimental setup.

<a id="section-4-1"></a>

### 4.1 Generalization Ability of VLAs

> (a) Evaluation of visual generalization and knowledge transfer.

| Model        | Task Name | Add Condiment | Insert Flower | Select Book | Select Drink | Select Toy | Select Tube | Select Painting | Select Fruit | Average |      |        |       |        |       |        |       |        |      |
| ------------ | --------- | ------------- | ------------- | ----------- | ------------ | ---------- | ----------- | --------------- | ------------ | ------- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ------ | ---- |
|              | Seen      | Unseen        | Seen          | Unseen      | Seen         | Unseen     | Seen        | Unseen          | Seen         | Unseen  | Seen | Unseen | Seen  | Unseen | Seen  | Unseen | Seen  | Unseen |      |
| Octo         | Base      | 3.08          | 3.08          | 1.54        | 0.00         | 0.00       | 1.54        | 0.00            | 0.00         | 0.00    | 0.00 | 1.54   | 0.00  | 6.15   | 1.54  | 0.00   | 0.00  | 1.34   | 0.77 |
| Common Sense | 1.54      | 3.08          | 0.00          | 0.00        | 0.00         | 0.00       | 0.00        | 0.00            | 3.08         | 1.54    | 1.54 | 3.08   | 3.08  | 0.00   | 0.00  | 0.00   | 1.16  | 0.96   |      |
| OpenVLA      | Base      | 12.38         | 8.23          | 13.85       | 7.69         | 7.69       | 4.62        | 8.46            | 4.61         | 3.08    | 4.62 | 7.69   | 6.15  | 40.20  | 28.26 | 4.62   | 3.07  | 11.74  | 7.93 |
| Common Sense | 8.23      | 3.08          | 9.24          | 4.61        | 0.00         | 0.00       | 8.46        | 4.61            | 0.00         | 0.00    | 6.15 | 3.08   | 34.06 | 25.48  | 1.54  | 0.00   | 8.46  | 5.11   |      |
| RDT-1B       | Base      | 21.54         | 14.46         | 21.54       | 16.92        | 3.08       | 1.54        | 7.69            | 3.08         | 7.69    | 4.62 | 12.38  | 6.15  | 35.16  | 19.72 | 13.85  | 6.15  | 15.37  | 9.08 |
| Common Sense | 16.92     | 4.61          | 14.46         | 3.08        | 0.00         | 0.00       | 7.69        | 0.00            | 4.62         | 1.54    | 7.69 | 0.00   | 32.08 | 16.64  | 12.32 | 3.07   | 11.97 | 3.61   |      |

Pretrained VLAs are expected to possess robust generalization and versatility similar to LLMs. Experiments about are set to address the following research questions:

Q1: Do pre-trained VLAs exhibit stronger general abilities with unseen categories of objects?

Q2: Can pre-trained VLAs transfer their general knowledge and behavioral abilities to similar but unseen tasks?

Q3: Can pre-trained VLAs understand natural user interactions and implicit goal requirements?

Q4: Do pre-trained VLAs have the potential to transfer their world knowledge to related tasks?

Q5: Can existing VLA architectures accurately support the completion of long-horizon tasks?

Experiment Setup.
To investigate the questions outlined above, we fine-tuned various pre-trained VLA architectures, including OpenVLA, Octo, and RDT-1B [27, 54, 40], on our high-quality dataset. Our composite tasks demand generalization across language, vision, common sense, and long-horizon reasoning, requiring the integration of multiple skills. To assess generalization ability, we selected primitive tasks as the foundation for evaluation. Within each category of primitive tasks, the Mesh&Texture (base) tasks, Common sense & World knowledge tasks, and Semantic tasks share similar task setups and trajectories. Therefore, we opt for joint training on base and common sense data across each task category and evaluate in different settings. During the fine-tuning stage, we sample 100 trajectories from each task category, resulting in a total of 1,600 trajectories to ensure balanced representation across tasks. For complex tasks, we perform fine-tuning separately within the domain of each task and conduct evaluations independently.

Result and Analysis.
In the evaluation stage, different task settings are applied to cover multiple generalization abilities. In Table [2(a)](https://arxiv.org/html/2412.18194v1#S4.T2.st1), we present experimental results comparing the generalization capabilities of vision and common sense by evaluating seen and unseen categories of objects. The experimental results indicate that the current large-scale pre-trained VLAs did not exhibit the expected rapid adaptation to downstream tasks. The fine-tuned models performed poorly in primitive tasks especially involving the Pick&Place skill, the findings are similar to [43]. Limited by its discretization process and single-frame input architecture, OpenVLA’s skill-learning capability is lower than that of RDT-1B. However, benefiting from pre-trained VLMs, OpenVLA achieves higher scores than RDT-1B on common-sense tasks involving unseen objects. Our analysis suggests that although OpenVLA only fits trajectory data during pre-training, its foundation on Llama2-7B provides it with greater generalization potential.

In Table [2(b)](https://arxiv.org/html/2412.18194v1#S4.T2.st2), [2(c)](https://arxiv.org/html/2412.18194v1#S4.T2.st3), and [2(d)](https://arxiv.org/html/2412.18194v1#S4.T2.st4), evaluations were conducted on out-of-domain semantically rich language, unseen but similar tasks, and composite tasks respectively. These experimental results indicate that current architectures and pre-training approaches are insufficient for equipping VLA models with stronger semantic understanding, skill transfer, and long-horizon planning capabilities. Analogous to the classic paradigm of pretraining-finetuning in large language models during the GPT-3 era [48], it is still difficult to determine how much gain VLA has achieved from pretraining on the scarce, quality-varying dataset of only a few million samples. Moreover, this becomes even more challenging if the backbone has already undergone large-scale vision-language training. Drawing an analogy to the development trajectory of large language models, the present state of VLAs is still far from reaching a level comparable to GPT-2. Further ablation studies and analysis are presented in Section [10.1](#section-10-1).

<a id="figure-4"></a>

![Figure3_task_example](images/Figure3_task_example.png)

> Figure 4: Evaluation results for Voxposer and CoPA. Voxposer w/o refers to the version without visual perception, where ground truth labels are directly provided for object selection. Voxposer w uses GPT-4V as the visual perception module.

<a id="section-4-2"></a>

### 4.2 Performance of Workflow Utilizing Foundation Model

For our evaluation of foundation model-based algorithms, we reviewed two state-of-the-art frameworks, Voxposer [23] and CoPA [22], and the comparison results are shown in Figure [5](#figure-5). Given Voxposer’s dependence on large language models (LLMs), we assessed Voxposer’s performance with and without visual perception capabilities. While Voxposer performed adequately on basic tasks and achieved the Progress Scores of 30–40, its reliance on LLM-driven motion planning often led to grasping failures due to limited information for effective grasp planning, especially when interpreting rotation in non-visual contexts, resulting in low overall scores.

Interestingly, the foundational LLM alone maintained relatively stable scores in semantic understanding and reasoning tasks without visual input. However, adding visual perception slightly reduced performance in these areas while significantly improving spatial reasoning, where LLM-only setups struggle with spatial accuracy due to lack of spatial information.

The lack of closed-loop feedback limits these models’ ability to perform physical reasoning tasks, particularly those involving dynamic interactions, leading to lower scores in this dimension. Both models struggle with high-complexity tasks, succeeding mainly in entity recognition but performing poorly in long horizon task reasonable breaking down. This finding underscores the requirements for advancing foundational model-based frameworks to address complex reasoning. In fact, although the methods mentioned above emphasize their zero-shot capabilities and generalization to new scenarios, their modular design often limits the upper bound of their performance. A more detailed discussion of this can be found in Section [10.2](#section-10-2).

<a id="figure-6"></a>

![Figure_Agent_Result](images/Figure_Agent_Result.png)

> Figure 6: Evaluation pipeline for VLMs. Step 1: Sample the required four-view images, as well as those segmented with numerical information, from the simulation. Meanwhile, save the corresponding instructions and operation sequences. Step 2: Input the original and annotated images and the instructions into the model. Then, obtain the model’s output and extract the generated operation sequence. Step 3: The two operation sequences are evaluated using four metrics, which are weighted and summed to produce the final score.The red part represents the error output of the model, where the red solid arrows represent the dependencies generated by the error, and red dashed arrows represent the dependencies lost by the error.

<a id="section-4-3"></a>

### 4.3 Comprehensive Ability of VLMs

We referred to the evaluation results of multiple series of Vision-Language Models (VLMs) provided by OpenCompass [10] and selected several models from different families with strong overall performance. These models include: GPT-4-turbo-2024-0409, GPT-4o-2024-08-06 [1], GLM-4V-9B [17], MiniCPM-V2.6 [20], Qwen2-VL-7B [2], InterVL2-8B [6], and LLaVA-NeXT [38]. We evaluate the comprehensive performance of these models with the dataset derived from naturally self-contained information within a simulated environment. This dataset consists of a complex set of tasks designed to assess the VLM’s ability to perceive visual stimuli and comprehend verbal instructions. There are two types of evaluation approaches for VLMs: interactive and non-interactive. In the following, we will provide a detailed introduction to both approaches.

Non-interactive Evaluation.
Figure [6](#figure-6) illustrates the simplified evaluation process specifically designed for VLMs in VLABench. Firstly the evaluation dataset is generated by initializing a series of task scenarios, each associated with two four-view diagrams: one annotated with masks and labels to identify distinct entity segments, and the other serving as a reference image without annotations, as shown in Data Production module in Figure [6](#figure-6). A randomly selected linguistic instruction from GPT4 relevant to the task accompanies these diagrams, forming the input to the Vision-Language Model (VLM).

During inference time, we provide a detailed description of the skill library, the requirements of output format, and several few-shot examples in different settings. These elements collectively form the system prompt for querying the VLM. The VLM is required to generate DSL output consisting of a sequence of skills, where each skill includes a name and associated parameters, conforming to predefined patterns to enable systematic evaluation.

Then, the generated skill sequences are constructed into a directed graph based on their logical dependencies. Subsequently, these DAGs are matched with the reference ones and scored in four metrics. Finally the scores are combined using weighted aggregation to calculate a total score for each model. Please refer to Section
[9.3](#section-9-3) in the supplementary material for more detailed metric computation.

Interactive Evaluation.
Similar to the VLA and workflow evaluation process mentioned in previous sections, interactive evaluation computes a task progress score based on the interaction with the environment. VLABench provides a controller that parses the DSL action sequences output by the VLM into executable actions, which are then applied in a simulation environment to interact with real-world objects. This approach is one of the key metrics for evaluating robotic manipulation tasks. However, it is more time-consuming compared to non-interactive approaches, and its evaluation dimension is relatively limited, as it cannot distinguish between errors in skill selection and those in parameter generation.

Performance Comparison and Analysis.
In line with the evaluation framework outlined in the previous section, we evaluate the models across six dimensions of task performance. The results for each VLM model under the 1-shot(^1^11As GLM-4V-9B does not support multiple image inputs, the 0-shot method is employed.) setting are summarized in Figure [5](#figure-5).

Although these VLMs perform well on most multimodal tasks and even some embodied tasks [10, 30], their performance, including that of GPT-4o, falls short when faced with more complex scenarios, instructions, and more challenging tasks. We surprisingly find that the open-source model Qwen2-VL-7B-Instruct performed competitively, surpassing GPT-4-turbo-2024-04-09 on certain dimensions. However, all models struggled with complex tasks, especially those requiring long-term task decompose and logic reasoning. Only GPT-4o achieved a score in the reasoning dimension comparable to those in other dimensions, while the other models scored around 20 points. Besides, performance declines significantly when linguistic instructions transition from direct semantics to abstract meanings, as shown in semantic dimension. Different models appear to have distinct areas of expertise, e.g. LLaVA-NeXT exhibits weaker spatial perception ability, GLM-4V-9B excels in spatial and even physical law dimensions but lag in semantic comprehension. More ablations and discussions are in Section [10.3](#section-10-3). Overall, while the models demonstrate promising capabilities, their understanding and planning in embodied environments remain limited, highlighting the need for further advancements.

<a id="section-5"></a>

## 5 Conclusion

We propose VLABench, a large-scale benchmark designed for tasks with long-horizon and multi-dimensional reasoning. Such reasoning and evaluation involves many dimensions, including from vision to knowledge gained in pretrain stage, implicit semantic goal extracting ability, combining the requirement and interactive scene to make reasonable decision, logical reasoning and the ability to make long horizon plan. One of the most important thing we do is to provide a positive definition of the capabilities that intelligent agents with true cognitive abilities should possess and the tasks they should be able to perform, through the provision of 100 standardized task settings. Additionally, VLABench constructed a scalable automated data collection framework for future’s potential larger scale pertaining and a standardized dataset for fair comparison of VLAs, both in the present and in future developments. Our diverse and multiple experiments revealed that current VLAs and VLMs face significant challenges in our tasks, and there remain substantial uncertainties in research on robotics scaling. We hope that VLABench will inspire both the future research on robotics pertaining recipe and promote more robust VLA architectures development.

## References

- Achiam et al. [2023]
  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
  Gpt-4 technical report.
  _arXiv preprint arXiv:2303.08774_, 2023.
- Bai et al. [2023]
  Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou.
  Qwen-vl: A frontier large vision-language model with versatile abilities.
  _arXiv preprint arXiv:2308.12966_, 2023.
- Brohan et al. [2022]
  Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al.
  Rt-1: Robotics transformer for real-world control at scale.
  _arXiv preprint arXiv:2212.06817_, 2022.
- Brohan et al. [2023]
  Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich.
  Rt-2: Vision-language-action models transfer web knowledge to robotic control.
  In _arXiv preprint arXiv:2307.15818_, 2023.
- Brown [2020]
  Tom B Brown.
  Language models are few-shot learners.
  _arXiv preprint arXiv:2005.14165_, 2020.
- Chen et al. [2024]
  Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al.
  Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks.
  In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 24185–24198, 2024.
- Cheng et al. [2024]
  An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu.
  Spatialrgpt: Grounded spatial reasoning in vision language model.
  _arXiv preprint arXiv:2406.01584_, 2024.
- Chi et al. [2023]
  Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song.
  Diffusion policy: Visuomotor policy learning via action diffusion.
  _The International Journal of Robotics Research_, page 02783649241273668, 2023.
- Cobbe et al. [2021]
  Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al.
  Training verifiers to solve math word problems.
  _arXiv preprint arXiv:2110.14168_, 2021.
- Contributors [2023]
  OpenCompass Contributors.
  Opencompass: A universal evaluation platform for foundation models.
  [https://github.com/open-compass/opencompass](https://github.com/open-compass/opencompass), 2023.
- Deitke et al. [2023]
  Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi.
  Objaverse: A universe of annotated 3d objects.
  In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 13142–13153, 2023.
- Dosovitskiy [2020]
  Alexey Dosovitskiy.
  An image is worth 16x16 words: Transformers for image recognition at scale.
  _arXiv preprint arXiv:2010.11929_, 2020.
- Driess et al. [2023]
  Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al.
  Palm-e: An embodied multimodal language model.
  _arXiv preprint arXiv:2303.03378_, 2023.
- Dubey et al. [2024]
  Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al.
  The llama 3 herd of models.
  _arXiv preprint arXiv:2407.21783_, 2024.
- Fang et al. [2020]
  Hao-Shu Fang, Chenxi Wang, Minghao Gou, and Cewu Lu.
  Graspnet-1billion: A large-scale benchmark for general object grasping.
  In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 11444–11453, 2020.
- Fang et al. [2023]
  Hao-Shu Fang, Minghao Gou, Chenxi Wang, and Cewu Lu.
  Robust grasping across diverse sensor qualities: The graspnet-1billion dataset.
  _The International Journal of Robotics Research_, 2023.
- GLM et al. [2024]
  Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al.
  Chatglm: A family of large language models from glm-130b to glm-4 all tools.
  _arXiv preprint arXiv:2406.12793_, 2024.
- Gong et al. [2023]
  Ran Gong, Jiangyong Huang, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu, Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, et al.
  Arnold: A benchmark for language-grounded task learning with continuous states in realistic 3d scenes.
  In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 20483–20495, 2023.
- Gu et al. [2023]
  Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su.
  Maniskill2: A unified benchmark for generalizable manipulation skills.
  In _International Conference on Learning Representations_, 2023.
- Hu et al. [2024]
  Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al.
  Minicpm: Unveiling the potential of small language models with scalable training strategies.
  _arXiv preprint arXiv:2404.06395_, 2024.
- Hu et al. [2023]
  Yingdong Hu, Fanqi Lin, Tong Zhang, Li Yi, and Yang Gao.
  Look before you leap: Unveiling the power of gpt-4v in robotic vision-language planning.
  _arXiv preprint arXiv:2311.17842_, 2023.
- Huang et al. [2024]
  Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao.
  Copa: General robotic manipulation through spatial constraints of parts with foundation models.
  _arXiv preprint arXiv:2403.08248_, 2024.
- Huang et al. [2023]
  Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei.
  Voxposer: Composable 3d value maps for robotic manipulation with language models.
  _arXiv preprint arXiv:2307.05973_, 2023.
- James et al. [2020]
  Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison.
  Rlbench: The robot learning benchmark & learning environment.
  _IEEE Robotics and Automation Letters_, 5(2):3019–3026, 2020.
- Karaman and Frazzoli [2011]
  Sertac Karaman and Emilio Frazzoli.
  Sampling-based algorithms for optimal motion planning.
  _The international journal of robotics research_, 30(7):846–894, 2011.
- Karaman et al. [2011]
  Sertac Karaman, Matthew R Walter, Alejandro Perez, Emilio Frazzoli, and Seth Teller.
  Anytime motion planning using the rrt.
  In _2011 IEEE international conference on robotics and automation_, pages 1478–1483. IEEE, 2011.
- Kim et al. [2024]
  Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al.
  Openvla: An open-source vision-language-action model.
  _arXiv preprint arXiv:2406.09246_, 2024.
- Kirillov et al. [2023]
  Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.
  Segment anything.
  In _Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 4015–4026, 2023.
- Li et al. [2023]
  Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al.
  Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation.
  In _Conference on Robot Learning_, pages 80–93. PMLR, 2023.
- Li et al. [2024a]
  Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Li Erran Li, Ruohan Zhang, et al.
  Embodied agent interface: Benchmarking llms for embodied decision making.
  _arXiv preprint arXiv:2410.07166_, 2024a.
- Li et al. [2024b]
  Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al.
  Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation.
  _arXiv preprint arXiv:2411.19650_, 2024b.
- Li et al. [2024c]
  Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao.
  Evaluating real-world robot manipulation policies in simulation.
  _arXiv preprint arXiv:2405.05941_, 2024c.
- Liang et al. [2023]
  Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng.
  Code as policies: Language model programs for embodied control.
  In _2023 IEEE International Conference on Robotics and Automation (ICRA)_, pages 9493–9500. IEEE, 2023.
- Lin et al. [2024]
  Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao.
  Data scaling laws in imitation learning for robotic manipulation.
  _arXiv preprint arXiv:2410.18647_, 2024.
- Liu et al. [2024a]
  Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone.
  Libero: Benchmarking knowledge transfer for lifelong robot learning.
  _Advances in Neural Information Processing Systems_, 36, 2024a.
- Liu et al. [2024b]
  Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine.
  Moka: Open-vocabulary robotic manipulation through mark-based visual prompting.
  In _First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024_, 2024b.
- Liu et al. [2023]
  Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
  Visual instruction tuning, 2023.
- Liu et al. [2024c]
  Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee.
  Llava-next: Improved reasoning, ocr, and world knowledge, 2024c.
- Liu et al. [2024d]
  Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
  Visual instruction tuning.
  _Advances in neural information processing systems_, 36, 2024d.
- Liu et al. [2024e]
  Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu.
  Rdt-1b: a diffusion foundation model for bimanual manipulation.
  _arXiv preprint arXiv:2410.07864_, 2024e.
- Mees et al. [2022]
  Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard.
  Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks.
  _IEEE Robotics and Automation Letters_, 7(3):7327–7334, 2022.
- Mu et al. [2021]
  Tongzhou Mu, Zhan Ling, Fanbo Xiang, Derek Yang, Xuanlin Li, Stone Tao, Zhiao Huang, Zhiwei Jia, and Hao Su.
  Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations.
  _arXiv preprint arXiv:2107.14483_, 2021.
- Nasiriany et al. [2024]
  Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu.
  Robocasa: Large-scale simulation of everyday tasks for generalist robots.
  In _Robotics: Science and Systems_, 2024.
- Nordmann et al. [2014]
  Arne Nordmann, Nico Hochgeschwender, and Sebastian Wrede.
  A survey on domain-specific languages in robotics.
  In _International conference on simulation, modeling, and programming for autonomous robots_, pages 195–206. Springer, 2014.
- O’Neill et al. [2023]
  Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, et al.
  Open x-embodiment: Robotic learning datasets and rt-x models.
  _arXiv preprint arXiv:2310.08864_, 2023.
- Padalkar et al. [2023]
  Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, et al.
  Open x-embodiment: Robotic learning datasets and rt-x models.
  _arXiv preprint arXiv:2310.08864_, 2023.
- Pearce et al. [2023]
  Tim Pearce, Tabish Rashid, Anssi Kanervisto, Dave Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, et al.
  Imitating human behaviour with diffusion models.
  _arXiv preprint arXiv:2301.10677_, 2023.
- Qiu et al. [2020]
  Xipeng Qiu, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning Dai, and Xuanjing Huang.
  Pre-trained models for natural language processing: A survey.
  _Science China technological sciences_, 63(10):1872–1897, 2020.
- Shridhar et al. [2020]
  Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox.
  Alfred: A benchmark for interpreting grounded instructions for everyday tasks.
  In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 10740–10749, 2020.
- Sun et al. [2024]
  Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li, Qinyuan Cheng, Xiangyang Liu, Hang Yan, Yunfan Shao, Qiong Tang, Shiduo Zhang, Xingjian Zhao, Ke Chen, Yining Zheng, Zhejian Zhou, Ruixiao Li, Jun Zhan, Yunhua Zhou, Linyang Li, Xiaogui Yang, Lingling Wu, Zhangyue Yin, Xuanjing Huang, Yu-Gang Jiang, and Xipeng Qiu.
  Moss: An open conversational large language model.
  _Machine Intelligence Research_, 2024.
- Sutherland and MacDonald [2019]
  Craig J Sutherland and Bruce MacDonald.
  Robolang: a simple domain specific language to script robot interactions.
  In _2019 16th International Conference on Ubiquitous Robots (UR)_, pages 265–270. IEEE, 2019.
- Szot et al. [2021]
  Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets, et al.
  Habitat 2.0: Training home assistants to rearrange their habitat.
  _Advances in neural information processing systems_, 34:251–266, 2021.
- Tao et al. [2024]
  Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, and Hao Su.
  Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai.
  _arXiv preprint arXiv:2410.00425_, 2024.
- Team et al. [2024]
  Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al.
  Octo: An open-source generalist robot policy.
  _arXiv preprint arXiv:2405.12213_, 2024.
- Todorov et al. [2012]
  Emanuel Todorov, Tom Erez, and Yuval Tassa.
  Mujoco: A physics engine for model-based control.
  In _2012 IEEE/RSJ International Conference on Intelligent Robots and Systems_, pages 5026–5033. IEEE, 2012.
- Tunyasuvunakool et al. [2020]
  Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Siqi Liu, Steven Bohez, Josh Merel, Tom Erez, Timothy Lillicrap, Nicolas Heess, and Yuval Tassa.
  dm_control: Software and tasks for continuous control.
  _Software Impacts_, 6:100022, 2020.
- Walke et al. [2023]
  Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al.
  Bridgedata v2: A dataset for robot learning at scale.
  In _Conference on Robot Learning_, pages 1723–1736. PMLR, 2023.
- Wei et al. [2022]
  Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus.
  Emergent abilities of large language models.
  _Transactions on Machine Learning Research_, 2022.
  Survey Certification.
- Wen et al. [2024]
  Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, et al.
  Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation.
  _arXiv preprint arXiv:2409.12514_, 2024.
- Yenamandra et al. [2023]
  Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, et al.
  Homerobot: Open-vocabulary mobile manipulation.
  _arXiv preprint arXiv:2306.11565_, 2023.
- Zakka [2023]
  Kevin Zakka.
  obj2mjcf: Cli for processing composite wavefront obj files for use in mujoco.
  [https://github.com/kevinzakka/obj2mjcf](https://github.com/kevinzakka/obj2mjcf), 2023.
  Version 1.0.
- Zeng et al. [2023]
  Yihan Zeng, Chenhan Jiang, Jiageng Mao, Jianhua Han, Chaoqiang Ye, Qingqiu Huang, Dit-Yan Yeung, Zhen Yang, Xiaodan Liang, and Hang Xu.
  Clip2: Contrastive language-image-point pretraining from real-world point cloud data.
  In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_, pages 15244–15253, 2023.
- Zheng et al. [2024]
  Liming Zheng, Feng Yan, Fanfan Liu, Chengjian Feng, Zhuoliang Kang, and Lin Ma.
  Robocas: A benchmark for robotic manipulation in complex object arrangement scenarios.
  _arXiv preprint arXiv:2407.06951_, 2024.

<a id="section-6"></a>

## 6 Benchmark Implementation

<a id="section-6-1"></a>

### 6.1 Task Descriptions

All Tasks. VLABench includes both 60 primitive and 40 composite tasks. These tasks encompass a wide range of manipulation skills and involve many high-level capabilities. The skills include 1) Pick&place, 2) Open&close door, 3) Open&close drawer, 4) Hang objects on the wall, 5) Use tool e.g. Hammer nail, 6) Press button, 7) Insert, 8) Pour, 9) Twist, and 10) Explore. For higher-level intelligence, VLAbench’s evaluation dimensions encompass complex scene understanding, implicit semantic analysis, world knowledge transfer, understanding of physical laws, relative spatial perception, long-term task planning, and even multi-step logical reasoning. Table LABEL:table:task_description provides a detailed introduction to the 100 tasks involved in VLAbench, including the type of each task, the manipulation skills involved, the scope of high-level intelligence examined, the average episode length at a control frequency of 10Hz, as well as a detailed description of the task and an explanation of its challenges. For the sake of clarity in the table, we will use abbreviations to represent the various intelligence dimensions. M&T corresponds to Mesh & Texture Understanding, SP corresponds to Spatial Understanding, C&W corresponds to Common Sense & World Knowledge, SEM corresponds to Semantic Conversation Understanding, PHY corresponds to Physical Laws Understanding, and L&R corresponds to (Logistic) Reasoning.

Long-horizon Design with Multistep Reasoning. Compared to previous benchmarks, VLAbench places more emphasis on comprehensive long-term reasoning. The reasoning defined here includes associating world knowledge with visual mesh or texture information to solve tasks, understanding latent task requirements through emotional language interpretation, mapping spatial descriptions to target states, subtask planning for multi-step operations, logical understanding, calculations, and result derivation, among others. Figure [9](#figure-9) presents a detailed comparison of the average episode length of overall tasks. VLABench exhibits the longest horizon among both Primitive and Composite tasks, surpassing RoboCasa Atomic and RoboCasa Composite by 27.0% and 35.1%, respectively. Furthermore, VLABench demonstrates significantly greater multi-step reasoning depth compared to other task sets, including sub-task numbers, inferring users’ hidden semantics, integrating visual and commonsense information, spatial reasoning, and even logical reasoning, as exemplified by solving math problems.

One-to-Many Mapping from Task Types to Instances.
In VLABench, a task represents a broad category of activities designed around specific assets and the actions an agent performs. These tasks are centered on object-related themes and require diverse visual information, relevant common-sense knowledge, and rich semantic input from the user. To ensure variability, each rollout introduces different target objects and receptacles, creating unique task instances. Unlike previous benchmarks [35] where similar activities were treated as separate tasks, VLABench unifies such variations under a single task category. For example, “placing an apple on a plate” and “placing a pear in a box” are two tasks in most task sets, but VLABench regards them as the same task because they share the same asset theme and involve nearly identical skills. This approach focuses on the underlying theme of the asset and the agent’s actions, providing a more generalized and flexible task definition. In Figure [8](#figure-8), we present three different rollouts from three distinct tasks. These demonstrations feature diverse language instructions, entirely different object combinations, and target objects, significantly varied visual information, and other domain randomizations, which will be elaborated further in the section [6.3](#section-6-3).

<a id="table-3"></a>

> Table 3: Assets statics. N-cate denotes the total number of object categories, while N-obj represents the total count of object instances. This table lists most of the assets.

| Grasp Obj          | N-Cate | N-Obj | Recep           | N-Cate | N-Obj |
| ------------------ | ------ | ----- | --------------- | ------ | ----- |
| Billiards          | 2      | 24    | Billiards Table | 1      | 1     |
| Books              | 8      | 52    | Shelf           | 1      | 10    |
| Baked Goods        | 5      | 60    | Microwave       | 1      | 5     |
| Condiment          | 5      | 50    | Cabinet         | 1      | 5     |
| Dessert            | 4      | 58    | Tray            | 1      | 15    |
| Drink              | 9      | 130   | Fridge          | 1      | 5     |
| Flower             | 9      | 25    | Vase            | 1      | 10    |
| Fruit              | 11     | 227   | Box Container   | 2      | 10    |
| Ingredient         | 16     | 181   | Cutting Board   | 1      | 15    |
| Mahjong            | 1      | 38    | Counter         | 1      | 20    |
| Number Cube        | 1      | 10    | Safe            | 1      | 5     |
| Painting           | 1      | 286   | Stove           | 1      | 5     |
| Poker              | 1      | 54    | Table           | 1      | 20    |
| Snack              | 8      | 97    | Juicer          | 1      | 3     |
| Flatware           | 4      | 80    | Crockery        | 7      | 136   |
| Tool               | 9      | 49    | Coffee Machine  | 1      | 5     |
| Toy                | 35     | 140   | Placemat        | 1      | 10    |
| Chemistry Solution | 1      | 30    | Tube Container  | 1      | 1     |
| Name Tag           | 1      | 30    | Flask           | 1      | 5     |

<a id="section-6-2"></a>

### 6.2 Task Observation

Each task in VLABench supports multi-view RGB-D images, semantic segmentation images, and point cloud inputs. Figure [7](#figure-7) illustrates an example, showcasing the visualized point cloud data along with images from multiple viewpoints. Similar to general standard RLDS format datasets [46], each demonstration in VLABench not only includes the aforementioned multi-view RGB-D images and point clouds but also comprises: a list of language instructions, episode terminal, sparse reward, actions, full observations including joint positions, joint velocities, end effector position and orientation, grasping state, etc.

<a id="section-6-3"></a>

### 6.3 Domain Randomization

To ensure task diversity and broad data distribution, each task in VLABench incorporates multiple domain randomization techniques. These diversifications include:

- •
  Mesh&Texture Randomization. This refers to the random variation of different instances within the same object category. For example, if a task scene requires an apple, the apple’s mesh is randomly selected from a pool of 20 distinct instances.
- •
  Position&Orientation Randomization. The default values for this randomized attribute are set as follows: the position offset is a random value within the range $[-0.05,0.05]$ along the x and y directions, and the orientation is randomized with the yaw angle in the range $[-\pi/10,\pi/10]$.
  In certain tasks, including SelectFruit, grid sampling is employed for the random distribution of scene objects. Objects are constrained to be distributed within a grid space based on a maximum distance limit and are further subjected to the aforementioned basic pose offset.
- •
  Mesh Scale Randomization. For the same mesh, VLABench scales the size of objects within a reasonable range, with the default scaling range set to $[0.95,1.05]$.
- •
  Visual Disturbance. VLABench employs random transformations of scenes and their relative positions, along with texture randomization of elements such as desks, floors, and walls, to achieve robust visual perturbations. In addition to the aforementioned color space transformations, the lighting intensity is randomly augmented within the range of $[0.8,1.2]$.
- •
  Random Distractors. VLABench requires different approaches to interpret scenes and extract key visual information accurately. To further enhance the robustness of task settings, we introduced the option to add irrelevant distractor objects to the tasks. For example, in the SelectToy task, 1–2 fruits can be included as visual distractors.

<a id="figure-10"></a>

![Figure_VLM_result](images/Figure_VLM_result.png)

> Figure 10: Cross embodiment. VLABench supports a wide range of different embodiments.

<a id="section-7"></a>

## 7 Simulation and Framework

<a id="section-7-1"></a>

### 7.1 Scenes

To ensure diverse task environments and rich visual inputs, we curated over 20 distinct scene types, drawing inspiration from real-life contexts and task-specific backgrounds. These scenes span everyday household settings, such as kitchens, living rooms, and dining areas, and dynamic social scenarios, including shopping malls, supermarkets, chemistry laboratories, and medical rooms. Figure [11](#figure-11) highlights a small part of these carefully designed scenes. Beyond the variety of scene types and structures, we incorporated over 20 unique material textures for floors and walls, further enriching the visual complexity and enhancing the overall data diversity.

<a id="section-7-2"></a>

### 7.2 Cross Embodiment

To enable the creation of more diverse task types and datasets, VLABench supports various embodiments, including multiple models of single-arm and dual-arm robots, humanoid robots, quadrupedal robots equipped with end-effectors, and mobile robots. Figure [10](#figure-10) illustrates the performance of these different embodiments within VLABench.

<a id="figure-11"></a>

![eval_workflow](images/eval_workflow.png)

> Figure 11: Diverse scenes. VLABench supports a wide range of different scenes.

<a id="section-7-3"></a>

### 7.3 Assets

In Section [3.3](#section-3-3), we provide a brief overview of our asset library. Assets are divided into two main categories: objects-to-grasp and receptacles. For objects-to-grasp, recommended grasping points need to be annotated and are represented in the XML file using sites with class=grasppoint. For receptacles, both bounding boxes and recommended placement points are required: the former is annotated using sites with class=keypoint, while the latter is represented with class=placepoint. In the coarse and large-scale pre-annotation process, we annotated grasp points on all objects-to-grasp with Graspnet [15, 16] and manually refined them as needed. For receptacles, we used SAM [28] to assist in annotating bounding boxes and assigning the placement point default above the bottom of the receptacles. Subsequent manual refinement and post-processing were applied after pre-annotation. Table [3](#table-3) provides an overview of the rough categories and the corresponding number of assets.

<a id="section-8"></a>

## 8 Dataset Building

<a id="section-8-1"></a>

### 8.1 Skill Library as Domain Specific Language

To facilitate task description and execution in robotic manipulation, we design a domain-specific language (DSL) tailored for our system. The DSL provides a structured and human-readable way to define manipulation skills, their parameters, and execution sequences. By abstracting low-level commands into high-level instructions, the DSL ensures clarity, modularity, and ease of interpretation for various tasks. The DSL consists of three primary components:

- •
  Skills. Atomic manipulation operations such as Pick, Place, Lift, etc.
- •
  Parameters. Arguments specify each skill’s details, such as the target object, orientation, and gripper state.
- •
  Task Execute Sequence. Sequential or hierarchical combination of skills to define a complete manipulation task.

<a id="section-8-2"></a>

### 8.2 Data Collection Progress

The automatic data collection process in VLABench is built upon the aforementioned DSL-encapsulated code. For each designed task, a corresponding task sequence is defined to represent the order of operations required to complete the task. For example, a case in Select Fruit requiring the robot to pick up an apple and place it in a basket can be expressed as a DSL sequence as follows. The parameters, such as grasp pose and target position, are dynamically generated based on the simulation environment, and prior annotation information.

All tasks involve the execution of Skills using motion planning algorithms for trajectory generation. Notably, the execution of the Pick Skill requires the robotic arm to first move to a preparation position. During this process, we compute the overlap between the gripper’s point cloud and the environment’s point cloud along the trajectory from the preparation position to the grasping position. This overlap is used as a rejection sampling condition to determine an appropriate grasping direction.

<a id="section-8-3"></a>

### 8.3 Prompt for Interactive Instruction

We have generated a diverse set of instructions for VLABench’s dataset and evaluation tasks. These linguistically rich instructions effectively assess the ability of different models to achieve a comprehensive understanding of task scenarios. All task types, including the five categories of Primitive tasks and Composite tasks, share the following system prompt. In the system prompt, {object list} and {target objects} should be replaced with the actual objects and target objects involved in each task’s scenario. For example, in the Insert Flower task, the object list might be [“rose”, “tulip”, “sunflower”], while the target objects would be [“rose”]. For each data point or evaluation task, we require the generation of ten distinct instructions, all referring to the same target object but expressed in completely different ways.

For tasks involving common sense and world knowledge, the prompt should additionally include the following description, emphasizing the unique characteristics of the target objects. An example for {Task-Specific Descriptions and Emphases} in Select Toy with Common Sense task is: “The target entity is the one that the robotic arm is supposed to grasp, move, or perform other operations on. Our task requirements are related to the characteristics of the target object. The instruction should focus on IP, rather than directly saying which toy to choose.”. While the few-shot examples are:[“target_object: Donald, instruction: ’I want a toy in the Disney series.’”, “target_object: Goku, instruction: ’Pick a toy which belongs to the dragon ball.’”].

For tasks requiring linguistically rich instructions, the prompt extends the system prompt by incorporating the following semantic prompt. The few-shot examples in Select Toy Semantic may be like: [“target_object: batman, instruction: ‘I’m a big fan of DC series, please help me choose a suitable toy.’”, “target_object: Luffy, instruction: ‘Today is my friend’s birthday, and I want to buy a Luffy figure for him. Could you help me wrap it? Thank you!’”].

Composite tasks integrate the abilities and skills involved in primitive tasks, with each composite task featuring its unique scenario and context. In this setup, while the system prompt remains shared, each task is accompanied by a specific prompt. Here, we present the specific prompt for the Cluster Book task.

<a id="section-9"></a>

## 9 Experiment Implementation

<a id="section-9-1"></a>

### 9.1 VLA Setting

To assess the generalization ability of various VLAs, we primarily fine-tune OpenVLA, Octo, and RDT-1B using our dataset. We utilize the original open-source code and adhere to the default hyperparameters set by the authors. To ensure comparability across datasets of varying sizes, we fix the number of training epochs instead of the maximum training steps. Given that OpenVLA has 7B parameters, we apply the recommended LoRA strategy in all experiments, rather than performing full parameter fine-tuning. In contrast, the other two models undergo full parameter fine-tuning. We train all models until convergence is achieved. Notably, Octo exhibits a certain reluctance to converge, which might be attributed to its relatively low level of generality. Consequently, we conduct training for over 5 epochs to obtain the optimal fit. Note that we adhere to the default configurations of these models: OpenVLA and Octo process a single-view image as input, whereas RDT-1B utilizes three different views. All experiments are conducted on 4 NVIDIA A800 with 80GB of memory.

<a id="section-9-2"></a>

### 9.2 Evaluation of Worksflows

In evaluating the foundation model-based workflow algorithms, we adopt the same evaluation process and metrics used for assessing the VLAs. The procedure for evaluating each task individually is outlined as follows:

- 1.  Run the Base Model Workflow. Execute the base workflow in the specified environment and record the corresponding outputs, with particular emphasis on data related to the model’s target entity detection information.
- 2.  Task Evaluation. Once the relevant information has been collected, the success of the task and the accuracy of target identification are assessed. Specifically, correct identification of a target contributes 20% of the total score, while task success will award full points.
- 3.  Final Score Calculation. After evaluating individual tasks several times, the scores for each time task are aggregated to yield the final score for the model under each configuration.

By applying this evaluation framework, we ensure a consistent and comprehensive assessment of the model’s performance across different tasks and settings.

<a id="section-9-3"></a>

### 9.3 Evaluation of VLMs

As discussed in Section [4.3](#section-4-3), the entire evaluation process of VLMs can be simplified to DSL generation and the score can be computed through direct graph matching. The assessment of the skill sequences output by the VLM is based on the following four metrics.

Skill Recall Rate (SR).
We use SR as the coarsest-grained metric to evaluate the model’s capability to identify and invoke the correct skills.

$$
SR=\frac{|\textit{SL}_{\text{gt}}\cap\textit{SL}_{\text{pred}}|}{|\textit{SL}_ {\text{gt}}|}(2)
$$

where $\textit{SL}_{\text{gt}}$ represents the list of skills manually labeled for completing tasks, and $\textit{SL}_{\text{pred}}$ refers to the list of skills predicted by the model. The denominator corresponds to the total number of relevant skills in the dataset, while the numerator counts the intersection of the relevant skills and those correctly identified by the model.

Parameter Recall Rate (PR).
The PR quantifies the model’s ability to correctly identify the parameters associated with each skill. In many cases, each skill is contingent upon specific parameters, which are often represented by the labels of relevant objects within an image. The PR thus measures the model’s accuracy in recognizing and interpreting these parameters, a crucial aspect for ensuring the correct execution of the task. Accurate parameter identification is fundamental not only for skill invocation but also for the model’s overall performance in real-world applications. A higher PR indicates a higher accuracy of the parameters predicted by the model, thus ensuring that the model correctly identifies the entities that need to be valued in the figure.

$$
PR=\frac{|\textit{Param}_{\text{gt}}\cap\textit{Param}_{\text{pred}}|}{| \textit{Param}_{\text{gt}}|}(3)
$$

where $\textit{Param}_{\text{gt}}$ refers to the list of parameters manually labeled for each skill, and $\textit{Param}_{\text{pred}}$ denotes the list of parameters predicted by the model.

Skill&Parameter Recall Rate (SPR).
Unlike the individual metrics SR and PR, SPR requires the model to identify both the correct skills and the exact parameters associated with each skill. It provides a more comprehensive and strict evaluation of the model’s ability of scene understanding and task planning in a real-world context. This metric is particularly useful in evaluating scenarios where both skills and their contextual parameters are critical for task execution, such as in visual recognition tasks where precise associations between actions and objects are necessary.

$$
SPR=\frac{|\textit{SP-Pair}_{\text{gt}}\cap\textit{SP-Pair}_{\text{pred}}|}{| \textit{SP-Pair}_{\text{gt}}|}(4)
$$

where $\textit{SP-Pair}_{\text{gt}}$ represents the set of all manually labeled skill-parameter combinations, and $\textit{SP-Pair}_{\text{pred}}$ refers to the corresponding combinations predicted by the model.

#### Precise Matching Rate (PM).

In addition to evaluating the correctness of skill-parameter matching, PM places greater emphasis on assessing the logical dependencies of the skill sequence, particularly for tasks with strict temporal requirements. Instead of totally strict sequential order, this metric focuses on ensuring that the necessary dependencies are satisfied for successful task execution. For example, in Make Juice task, the model must ensure that the juicer is opened before adding fruit, but the order of adding apples versus oranges is irrelevant.

We begin by aggregating the skill sequence according to predefined operational patterns and constructing a directed acyclic graph (DAG) with a designated source node to represent the logical dependencies among operations. A match is defined as a node in the model-generated graph that shares the same skill name and parameters as a corresponding node in the ground-truth graph while also satisfying the logical dependency relationships, e.g. incoming and outgoing edges.The formula for this metric is as follows:

$$
PM=\frac{|\textit{Node}_{\text{matched}}|}{|\textit{Node}_{\text{total}}|}(5)
$$

where the numerator $\textit{Node}_{\text{matched}}$ represents the number of nodes in the model-generated DAG that match the corresponding nodes in the ground-truth DAG. $\textit{Node}_{\text{total}}$ represents the total number of nodes in the ground-truth DAG.

Finally, these four scores are combined using predetermined weights to compute a total score for each model. The formula for this metric is as follows:

$$
Score=w_{1}\cdot SR+w_{2}\cdot PR+w_{3}\cdot SPR+w_{4}\cdot PM(6)
$$

where $w_{1},w_{2},w_{3},w_{4}$ are the weights of different metrics, with the constraint $w_{1}+w_{2}+w_{3}+w_{4}=1$.

<a id="section-10"></a>

## 10 Detailed Analysis and Case Study

<a id="figure-12"></a>

![pcd_multicam](images/pcd_multicam.png)

> Figure 12: Scaling trend. This result is evaluated on Put Box on Paintig Task with data scales of 100, 500, 1000, and 2000.

<a id="table-4"></a>

> Table 4: Ablation of fine-tuning from scratch and pretrain. Evaluated on primitive tasks with seen objects.

| Model   | From Scratch | From Pretrained |
| ------- | ------------ | --------------- |
| Octo    | 1.02         | 1.34            |
| OpenVLA | 3.02         | 11.74           |
| RDT-1B  | 6.26         | 15.37           |

<a id="section-10-1"></a>

### 10.1 Ablations and Analysis for VLAs

Experimental results show that the current open-source VLAs perform poorly on our tasks. On one hand, this can be attributed to the high difficulty of VLABench tasks, which impose stringent requirements on the generalization capabilities of the models. More importantly, the limitations and deficiencies in both the architecture and pretraining process of current VLAs make it challenging for them to adapt effectively to downstream tasks after large-scale pretraining, especially under fine-tuning scenarios with diverse data distributions. This stands in stark contrast to LLMs, which excel in adapting to downstream tasks with minimal fine-tuning on small datasets.

![multicase_in_onetask](images/multicase_in_onetask.png)

> (a) Select Poker

To further illustrate the aforementioned issues, we conducted several ablation experiments:

- •
  Data Scaling. More data implies a greater number of visual-language to trajectory mappings. For specific primitive tasks, we expanded the dataset to 2,000 samples and conducted separate evaluations on three models using datasets of varying scales: 100, 500, 1,000, and 2,000 samples. However, the experimental results show that under diverse data distributions, the task success rates of all three models remain consistently low across the four scales. This issue is primarily reflected in their operational accuracy. As illustrated in Figure [12](#figure-12), while the trajectory generation becomes increasingly smooth and the PS shows a slight improvement with larger datasets, the overall task success rate remains notably low.
- •
  Pretrained Effect. We also conducted an evaluation of models trained with the fine-tuning dataset from scratch, with the results summarized in Table [5](#table-5). The findings indicate that models of this scale struggle to quickly adapt to downstream tasks with limited data. It is reasonable to infer that pretraining on large-scale, domain-relevant data can significantly facilitate the faster transfer of VLAs to downstream tasks.
- •
  Closed-loop for RDT. Following the open-source RDT framework, the primary experiment employs a single-trajectory inference scheme with 64 trajectory points, implemented using open-loop control. However, open-loop control is prone to error accumulation. To address this, we conducted additional evaluations of RDT using closed-loop control. The results in Table [5](#table-5) show that closed-loop control achieves slightly better performance compared to open-loop control. This suggests that the low success rate in task execution is primarily due to the inherent limitations of the model itself.

To analyze why these models perform poorly, we base our discussion on experimental results and observations from two key perspectives.

Limitation of Model Architecture.

- •
  Incomplete Information Intake. Some shortcomings in the model architecture result in this issue. For instance, OpenVLA and Octo only process single images with a resolution of 224×224, which inherently puts them at a disadvantage when the input images contain occlusions or require finer texture details. Similarly, due to issues with perspective and low resolution, directly mapping visual information to precise spatial coordinate points becomes challenging.
- •
  Lack of Memory. Current models only accept inputs representing the current state, lacking position embeddings to capture temporal sequences or tokens to represent historical actions. This limitation can cause the model’s behavior to become stuck in certain states. This issue is particularly pronounced in long-horizon tasks, where the model may “forget” previous actions and repeatedly perform the same behavior.
- •
  Inherent Flaws of Different Architecture. The VLAs we used primarily include two forms: transformer-based next-token prediction architectures and diffusion model-based architectures. The former, leveraging VLMs, benefits from pretraining on world knowledge but inherently suffers from precision loss due to the discretization required by action tokenization. On the other hand, diffusion policies are better suited for continuous spatial distributions, yet they lack visual and language pretraining. Additionally, diffusion models rely on multiple large encoders, such as T5, making it challenging to jointly fine-tune parameters during unified training. This limitation contributes to the poor performance of diffusion policies in VLABench tasks requiring common sense.

Shortcomings of Pretrain.
VLA pretraining has been proven effective for efficient transfer to downstream tasks. However, the current pretraining approaches may have certain issues. For example, RT-2 [4] highlights a pretraining strategy that jointly trains on multiple text tasks and text-visual tasks to preserve the model’s inherent language and reasoning capabilities, resulting in impressive generalization behaviors. In contrast, OpenVLA, which is also based on VLM, is pretrained solely on trajectory datasets. This likely leads to the degradation of VLM’s original capabilities, such as commonsense knowledge and reasoning skills.

Additionally, constrained by the availability of datasets, the scale of current VLA pretraining data is far smaller than that of language models. Drawing inspiration from the scaling laws and emergent behaviors observed in language models, there is likely a critical point and correlation between model parameter size and data volume. The scaling curve for VLA pretraining, however, remains an open topic for future research.

<a id="section-10-2"></a>

### 10.2 Further Analysis for Workflows

From the experimental results, we observe that while the framework algorithm based on the foundation model demonstrates some degree of robustness in handling complex semantic settings, the overall success rate and PS score remain relatively low. A comprehensive analysis of the failure cases reveals that the underlying issues can be broadly categorized into the following groups.

Perception.
One of the primary challenges lies in the model’s image and spatial perception capabilities. As Voxposer is implemented as a purely text-based framework, its perception module relies directly on the ground-truth labels of all items, which are provided as input for selection. While this leverages the comprehension and generalization capabilities of large language models to understand tasks, it exposes significant limitations in scenarios that require spatial perception and image-based reasoning. Specifically, Voxposer demonstrates clear incompetence in handling tasks involving spatial awareness or detailed image descriptions.

To address this, we augmented our experimental setup by incorporating an image perception module into Voxposer. Although this adjustment improved success rates on spatial perception tasks, the overall performance deteriorated due to errors introduced by the visual perception module. A similar issue was observed in CoPA, where the SoM family of models exhibited high sensitivity to segmentation parameters, requiring extensive tuning to achieve accurate entity recognition. Even with optimization, a substantial number of incorrect object recognition cases persisted, highlighting fundamental challenges in the perception component.

Planning.
Another significant limitation emerges in the model’s planning capabilities. After selecting the target object, the model’s lack of spatial perception often prevents it from recognizing the need to adjust its pose, such as rotating the robotic arm when grasping certain objects. This deficiency leads to frequent task failures, particularly in scenarios involving objects like cardboard sheets or books, as illustrated in Figure [15](#figure-15). Additionally, the simple point-cloud-based center-of-mass grasping strategy employed by the model exhibits a high probability of failure when interacting with objects of complex shapes, such as toys, as shown in Figure [15](#figure-15).

For CoPA, similar challenges were encountered in the graspnet module, where planning grasping actions was hindered by its instability. In many instances, the module failed to identify a valid grasping point, resulting in task failures, as depicted in Figure [15](#figure-15). These issues underscore the model’s inability to effectively plan and execute tasks involving diverse and irregularly shaped objects.

![Figure_horizon_step](images/Figure_horizon_step.png)

> (a) Result of InternVL2-8B.

Module Connections.
As hierarchical systems, such algorithms rely on the integration of multiple independent modules, which inevitably introduces errors at the interfaces between components. For example, the large language model may generate incorrect outputs, such as failing to locate the corresponding object or the constraints generated by the system may not be successfully converted into waypoints by the solver. These errors significantly reduce the system’s robustness when handling diverse task conditions. The inability to reliably bridge constraints and waypoints highlights a critical limitation in the framework’s modular connectivity, further undermining its ability to adapt to varying operational scenarios.

<a id="section-10-3"></a>

### 10.3 Ablations and Analysis for VLMs

In our evaluation of VLMs, we conducted two key experiments to explore the impact of Chain-of-Thought (CoT) prompting and few-shot learning on model performance.

Effect of CoT Prompting.
Our investigation into the use of CoT prompting revealed a notable improvement in overall performance for the InternVL2 model, as shown in Figure [17](#figure-17). Similarly, LLaVA-NeXT and Qwen2-VL demonstrated enhanced performance in challenging tasks, particularly those requiring reasoning about complex scenarios and physics laws. However, their performance on semantically common-sense tasks remained stagnant or experienced minor degradation. In contrast, the MiniCPM model exhibited significant limitations: it failed to output answers at the conclusion of the reasoning process when CoT was applied, resulting in all scores dropping to 0.0.

Effect of Few-Shot Learning.
As shown in Figure [17](#figure-17) our exploration of few-shot learning with the Qwen2-VL model indicated that increasing the number of few-shot examples (0 to 7) enhances the model’s multimodal reasoning capabilities, particularly under CoT prompting. This enhancement was observed across both basic and complex scenarios. However, we found diminishing returns beyond two or three shots for tasks involving diverse semantic requirements or spatial reasoning. This suggests that the utility of additional examples is context-dependent and saturates relatively quickly in certain domains.

<a id="figure-18"></a>

![cross_emb](images/cross_emb.png)

> Figure 18: Part of tasks in VLABench.

<a id="table-6"></a>

> Table 6: Task List. Include the name, type, ability required, and detailed description of all the tasks.

| Task                                       | Type      | Ability<br>Dimension              | Skill<br>Involved                                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------ | --------- | --------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Select Fruit                               | Primitive | M&T                               | Pick&place                                               | Pick the specific fruit into a specific receptacle, such as “put the strawberry into the basket”.                                                                                                                                                                                                                                                                                                                                                                                                 |
| Select Drink                               | Primitive | M&T                               | Pick&place, Pull                                         | Get the specific drink from a particular receptacle, such as “pick the cola out of the fridge”.                                                                                                                                                                                                                                                                                                                                                                                                   |
| Select Toy                                 | Primitive | M&T                               | Pick&place                                               | Put the specific toy into a specific receptacle, such as “Select Ironman from the toys and wrap it in the gift box”.                                                                                                                                                                                                                                                                                                                                                                              |
| Select Book                                | Primitive | M&T                               | Pick&place, Pull                                         | Take a particular book from the receptacle, such as “Take Pride and Prejudice from the bookshelf”.                                                                                                                                                                                                                                                                                                                                                                                                |
| Select Ingredient                          | Primitive | M&T                               | Pick&place                                               | Get the specific ingredient from the particular receptacle, such as “Take the bell pepper from the fridge and place it on the tray”.                                                                                                                                                                                                                                                                                                                                                              |
| Insert Flower                              | Primitive | M&T                               | Pick&place, Insert                                       | Insert the specific flower into the container, such as “Insert the rose into a vase.”                                                                                                                                                                                                                                                                                                                                                                                                             |
| Add Condiment                              | Primitive | M&T                               | Pick&place, Pour                                         | Add the specific condiment into the dish, such as “Add some salt into the dish in the pot.”                                                                                                                                                                                                                                                                                                                                                                                                       |
| Put Box on Famous Painting                 | Primitive | M&T                               | Pick&place                                               | Place the geometric shape on the specified famous painting., such as “Press the button before the painting The Starry Night”                                                                                                                                                                                                                                                                                                                                                                      |
| Pick ChemistryTube                         | Primitive | M&T                               | Pick&place                                               | Select specific solution tube based on the nametag, such as “pick up the tube of CuCl2”.                                                                                                                                                                                                                                                                                                                                                                                                          |
| Select Poker                               | Primitive | M&T                               | Pick&place                                               | Select specific poker, such as “Pick jack of red heart”.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Select Mahjong                             | Primitive | M&T                               | Pick&place                                               | Select specific mahjong, such as “Pick mahjong: 2 of Man and put it on the placemat”.                                                                                                                                                                                                                                                                                                                                                                                                             |
| Select Billiards                           | Primitive | M&T                               | Pick&place                                               | Select specific Billiards, such as “Pick Black 8 and place it in any hole.”                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Hammer Loose Nail                          | Primitive | M&T                               | Pick&place, Tool use                                     | By comparing the lengths of different nails, use a hammer to tighten the loose nail, such as “Hammer the loose nail on the wall”.                                                                                                                                                                                                                                                                                                                                                                 |
| Select Fruit-Spatial                       | Primitive | M&T, SP                           | Pick&place                                               | Pick the fruit in a specific place or a certain spatial relationship, such as “put the nearest strawberry into the plate”.                                                                                                                                                                                                                                                                                                                                                                        |
| Select Drink-Spatial                       | Primitive | M&T, SP                           | Pick&place                                               | Get the drink from a specific place or a certain spatial relationship, such as “pick the monster outside”, while there is also a can of monster inside the fridge.                                                                                                                                                                                                                                                                                                                                |
| Select Toy-Spatial                         | Primitive | M&T, SP                           | Pick&place                                               | Take the toy from a specific place or a certain spatial relationship, such as “Place the toy on Luffy’s right-hand side into the box”.                                                                                                                                                                                                                                                                                                                                                            |
| Select Book-Spatial                        | Primitive | M&T, SP                           | Pick&place                                               | Take the book on the specific position or a certain spatial relationship, such as “Take out the book on the most left in the top layer”.                                                                                                                                                                                                                                                                                                                                                          |
| Select Ingredient-<br>Spatial              | Primitive | M&T, SP                           | Pick&place                                               | Get the ingredient on the specific position or in a certain spatial relationship, such as “Place the ingredient on the bottom layer in the fridge onto the tray”.                                                                                                                                                                                                                                                                                                                                 |
| Insert Flower-Spatial                      | Primitive | M&T, SP                           | Pick&place                                               | Insert the flower on the specific position or in a certain spatial relationship, “Place the flower on the far left into the vase.”.                                                                                                                                                                                                                                                                                                                                                               |
| Add Condiment-<br>Spatial                  | Primitive | M&T, SP                           | Pick&place                                               | Add the condiment on the specific position or in a certain spatial relationship, such as “Add the furthest spice to the dish”.                                                                                                                                                                                                                                                                                                                                                                    |
| Hang Picture                               | Primitive | M&T, SP                           | Pick&place, Hang                                         | Hang the picture on the nail in the specified location, such as “Hang the picture on the highest nail”.                                                                                                                                                                                                                                                                                                                                                                                           |
| Pick ChemistryTube-<br>Spatial.            | Primitive | M&T, SP                           | Pick&Place                                               | Take out the chemistry solution tube on the specific position or in a certain spatial relationship, such as “Take the tube in the first row, the second column before you”.                                                                                                                                                                                                                                                                                                                       |
| Select Poker-Spatial                       | Primitive | M&T, SP                           | Pick&Place                                               | Select the poker on the specific position or in a certain spatial relationship, such as “Pick the second poker from left to right”.                                                                                                                                                                                                                                                                                                                                                               |
| Select Mahjong-<br>Spatial                 | Primitive | M&T, SP                           | Pick&Place                                               | Select the mahjong on the specific position or in a certain spatial relationship, such as “Pick the mahjong on the right of six of sou”.                                                                                                                                                                                                                                                                                                                                                          |
| Put Billiards<br>in Pocket                 | Primitive | M&T, SP                           | Pick&Place                                               | Place the billiard ball into the specified pocket, such as “Place the 8-ball into the pocket in the right front”.                                                                                                                                                                                                                                                                                                                                                                                 |
| Select Fruit<br>with Common Sense          | Primitive | M&T, C&W                          | Pick&Place                                               | Select fruit with specific characteristics, including nutritional characteristics, common uses, whether they grow in clusters, easy to peel, etc. Example: “Put the fruit with the most vitamin C into the basket” from among orange, banana, and apple.                                                                                                                                                                                                                                          |
| Select Drink<br>with Common Sense          | Primitive | M&T, C&W                          | Pick&Place                                               | Select a drink with some specific characteristics including types of beverages, functions of the beverages, flavors of the beverages, etc. Example: “Get a can of energy drink from the fridge” from among cola, apple juice, and redbull.                                                                                                                                                                                                                                                        |
| Select Toy<br>with Common Sense            | Primitive | M&T, C&W                          | Pick&Place                                               | Select the toy with some specific characteristics including the associated IP, character personality, character background, etc. Example: “Put the toy from the Marvel series to the giftbox” from among Hulk, Batman, and Mickey.                                                                                                                                                                                                                                                                |
| Select Book<br>with Common Sense           | Primitive | M&T, C&W                          | Pick&Place                                               | Select the book with some specific characteristics including the type of the book, the content of the book, the main message it conveys, etc. Example: “Get the book about computer science” from among Steve Jobs, 3D Computer Vision and War and Peace.                                                                                                                                                                                                                                         |
| Select Ingredient<br>with Common Sense     | Primitive | M&T, C&W                          | Pick&Place                                               | Select the ingredient with some specific characteristics, such as “Pick an ingredient full of protein from the fridge and put it on the tray” from among egg, tomato and bell pepper.                                                                                                                                                                                                                                                                                                             |
| Insert Flower<br>with Common Sense         | Primitive | M&T, C&W                          | Pick&Place                                               | Insert the flower with some specific characteristics into vase including the flower language, the symbolic qualities of the flower, appropriate occasions for giving flowers, etc. Example:“Insert the flower suitable for Valentine’s Day into the vase” from rose, sunflower, and tulip.                                                                                                                                                                                                        |
| Insert Bloomed<br>Flower                   | Primitive | M&T, C&W                          | Pick&Place                                               | An intelligent agent should possess awareness: flowers should be arranged with blooming ones, not with those that are already withered. Example: “Insert a proper flower into the vase” from among wilted rose, wilted daisy, and sunflower.                                                                                                                                                                                                                                                      |
| Add Condiments<br>with Common Sense        | Primitive | M&T, C&W                          | Pick&Place                                               | Add the condiment with some specific characteristics including distinctive flavor, seasoning role, suitability for various dishes, and etc. Example: “Add the condiment that makes the dish taste more salty” from among salt, ketchup, and salad dressing.                                                                                                                                                                                                                                       |
| Select Painting<br>with Common Sense       | Primitive | M&T, C&W                          | Pick&Place                                               | Press the button before the painting with specific styles or contents. Example: “Choose the painting in the style of rococo” among from paintings of La Liseuse, The Stary Night, and Golden Autumn.                                                                                                                                                                                                                                                                                              |
| Pick Chemistry Tube<br>with Common Sense   | Primitive | M&T, C&W                          | Pick&Place                                               | Pick up the specific solution without the nametag and distinguish by the solution color. Example: “Pick up the solution of CuSO4” from among the solution of blue, green, and yellow.                                                                                                                                                                                                                                                                                                             |
| Select nth<br>Largest Poker                | Primitive | M&T, C&W                          | Pick&Place                                               | Choose the largest poker under the rule of the specific poker games. Example: “Pick the largest poker in single under the rule of Texas Holdem” from among Ace of Spades, Three of Hearts, and Queen of Clubs.                                                                                                                                                                                                                                                                                    |
| Select Unique<br>Mahjong                   | Primitive | M&T, C&W                          | Pick&Place                                               | Choose the mahjong with the unique type. Example:“Pick the unique type of mahjong” among from East, One of Man, Nine of Man.                                                                                                                                                                                                                                                                                                                                                                      |
| Select Billiards<br>with Common Sense      | Primitive | M&T, C&W                          | Pick&Place                                               | Select a specific billiard game under particular rules with a specific score. Example: “Place the two-point ball from a snooker match into any pocket” from among green ball, yellow ball, and red ball.                                                                                                                                                                                                                                                                                          |
| Select Fruit-<br>Semantic                  | Primitive | M&T, SEM                          | Pick&Place                                               | The user expresses implicit needs for a certain fruit during a semantically rich conversation or context, such as: “Today, I suddenly feel like doing some baking and plan to make a strawberry cake! Could you help me prepare the fruits I’ll need?”                                                                                                                                                                                                                                            |
| Select Drink-<br>Semantic                  | Primitive | M&T, SEM                          | Pick&Place,<br>Pull                                      | The user expresses implicit needs for a certain drink during a semantically rich conversation or context, such as: “I just worked out at the gym for a long time, and now I’m a bit dehydrated. Could you help me grab a bottle of electrolyte drink from the fridge?”                                                                                                                                                                                                                            |
| Select Toy-<br>Semantic                    | Primitive | M&T, SEM                          | Pick&Place                                               | The user expresses implicit needs for a specific toy during a semantically rich conversation or context, such as “I’ve loved Disney since I was a kid, especially the Toy Story series! I want to place Buzz Lightyear on the top layer of the shelf, but I can’t reach it! ”                                                                                                                                                                                                                     |
| Select Book-<br>Semantic                   | Primitive | M&T, SEM                          | Pick&Place,<br>Pull                                      | The user expresses implicit needs for a specific book during a semantically rich conversation or context, such as “I’m getting ready to review for my final Python exam. Could you help me prepare the textbook?”                                                                                                                                                                                                                                                                                 |
| Select Ingredient-<br>Semantic             | Primitive | M&T, SEM                          | Pick&Place                                               | The user expresses implicit needs for a specific ingredient during a semantically rich conversation or context, such as “I’m keeping fit so I want to eat something full of protein. I want a steak as my lunch and could you get one for me?”                                                                                                                                                                                                                                                    |
| Insert Flower-<br>Semantic                 | Primitive | M&T, SEM                          | Pick&Place                                               | The user expresses implicit needs for a specific flower during a semantically rich conversation or context, such as “Today is Teacher’s Day, and Ms. Lisa has always been kind to me. I want to give her a bouquet of carnations. Could you help me place them in the vase on her desk?”                                                                                                                                                                                                          |
| Add Condiment-<br>Semantic                 | Primitive | M&T, SEM                          | Pick&Place,<br>Pour                                      | The user expresses implicit needs for a specific condiment during a semantically rich conversation or context, such as “I’m making tomato-braised beef brisket, but the tomato flavor doesn’t seem strong enough. Could you help me add some tomato paste? Thanks!”                                                                                                                                                                                                                               |
| Select Painting-<br>Semantic               | Primitive | M&T, SEM                          | Press                                                    | The user expresses implicit needs for a specific solution during a semantically rich conversation or context, such as: “I am a student who has just started learning painting, and I’m not very good at distinguishing between different styles of painting. Could you help me identify which of these three paintings is in the realist style?”                                                                                                                                                  |
| Select ChemistryTube-<br>Semantic          | Primitive | M&T, SEM                          | Pick&Place                                               | The user expresses implicit needs for a specific solution during a semantically rich conversation or context, such as: “I’m going to demonstrate an acid-base neutralization experiment today, but I’m missing an acid-base indicator. Could you help me grab the phenolphthalein solution?”                                                                                                                                                                                                      |
| Simple Poker<br>Play                       | Primitive | M&T, SEM                          | Pick&Place                                               | The agent plays the poker that should be played on behalf of the player during the semantically rich interaction. Example: “We’re playing Landlord, and the player before me just played a 10. Now it’s our turn. Please play a 2 for me.”                                                                                                                                                                                                                                                        |
| Simple Mahjong<br>Play                     | Primitive | M&T, SEM                          | Pick&Place                                               | The agent plays the Mahjong that should be played on behalf of the player during the semantically rich interaction. Example: “It’s hard to win with the ’Wan’ character tiles left. Go ahead and discard the ’1 Wan’.”                                                                                                                                                                                                                                                                            |
| Simple Snooker<br>Play                     | Primitive | M&T, SEM                          | Pick&Place                                               | The agent picks the billiard that should be played on behalf of the player during the semantically rich interaction. “We’re playing a simple game of snooker. Now, let’s pot the yellow ball into the pocket.”                                                                                                                                                                                                                                                                                    |
| Friction QA                                | Primitive | M&T, PHY                          | Press                                                    | Using the relevant physics knowledge of sliding friction and rolling friction, determine the rolling speed of different shaped and material objects on a slope. Example: “Press the button before the object that falls the fastest down the slope.”                                                                                                                                                                                                                                              |
| Density QA                                 | Primitive | M&T, PHY                          | Press                                                    | Visually judge the material of an object and determine the relative density of objects made from different materials. Example: “Press the button before the object can float on water.”                                                                                                                                                                                                                                                                                                           |
| Magnetism QA                               | Primitive | M&T, PHY                          | Press                                                    | Visually identify the material of an object and determine whether objects made from different materials are magnetic. Example: “Press the button before the object that is not magnetic.”                                                                                                                                                                                                                                                                                                         |
| Weight QA                                  | Primitive | M&T, PHY                          | Press                                                    | Visually identify the material of an object, and combine the material density and shape (in the actual setup, this includes cubes of different shapes, along with their corresponding inscribed spheres and circumscribed spheres) to make a comprehensive judgment of the object’s mass. Example: “Press the button before the object with the smallest weight.”                                                                                                                                 |
| Thermal Expansion<br>QA                    | Primitive | M&T, PHY                          | Press                                                    | Visually identify the material of an object and determine the thermal expansion properties of objects made from different materials. Example: “Press the button before the object with a medium thermal expansion coefficient.”                                                                                                                                                                                                                                                                   |
| Speed of Sound QA                          | Primitive | M&T, PHY                          | Press                                                    | Visually identify the material of an object and determine the sound propagation speed in objects made from different materials. Example: “Press the button before the object that sound propagates fastest in.”                                                                                                                                                                                                                                                                                   |
| Specular Reflection<br>QA                  | Primitive | M&T, PHY                          | Press                                                    | Judge based on visual information whether different objects exhibit specular reflection and make a selection. Example: “Press the button before the object that can reflect the image of others.”                                                                                                                                                                                                                                                                                                 |
| Drag Force QA                              | Primitive | M&T, PHY                          | Press                                                    | Determine the object’s free fall speed based on its shape, texture, and material. This involves physical theories such as air resistance, the Kármán vortex street effect, and others. Example: “Press the button before the object falls slowest in the air” from among golf, basketball, football.                                                                                                                                                                                              |
| Basic Seesaw Usage                         | Primitive | M&T, PHY                          | Pick&place,<br>Tool use                                  | Using the principle of leverage, place a heavy object on one side of the seesaw to lift the other side. Example: “Make the other side of the seesaw lift”.                                                                                                                                                                                                                                                                                                                                        |
| Strike Billiards                           | Primitive | M&T, PHY                          | Pick&place,<br>Tool use                                  | Use the laws of collision to perform a simple strike. Example: “Use the cue stick to strike the white ball, aiming to make it hit other colored balls”.                                                                                                                                                                                                                                                                                                                                           |
| Take Chemistry<br>Experiment               | Composite | M&T, SP,<br>SEM, C&W,<br>L&R      | Pick&place, Insert,<br>Pour                              | The agent should first use the user’s request for the desired chemical product, combine it with visual observation and common knowledge for logical reasoning, and determine the chemical solutions involved in the reaction. After identifying the appropriate solutions using the name tag, the agent should select the solutions and mix them into the flask. Example: “I would like to obtain AgCl precipitation in the flask. Please carry out this experiment.”                             |
| Find Unseen Object                         | Composite | M&T, SP,<br>L&R                   | Open&close drawer, Pick&Place,<br>Explore                | The target object is not directly visible, requiring the agent to open multiple drawers and eventually find the target object. Example: “Find a snack in the drawer for me”.                                                                                                                                                                                                                                                                                                                      |
| Find Unseen Object<br>without Telling Find | Composite | M&T, SP,<br>SEM, C&W,<br>L&R      | Open&close drawer, Pick&Place,<br>Explore                | The other settings are the same as for Find Unseen Object, but the requirements are implicitly conveyed through semantically rich dialogue. The agent needs to be aware of the need for exploration and search on its own. Example: “I’m a bit hungry, could you get me something to eat?”                                                                                                                                                                                                        |
| Make Juice<br>with Juicer                  | Composite | M&T, SEM,<br>L&R                  | Pick&place,<br>Tool use,<br>Press                        | Select the appropriate fruits based on semantically rich user instructions, place them into a container, and correctly use the juicer. Example: “It’s so hot today! I feel like having a freshly squeezed kiwi and strawberry juice right now.”                                                                                                                                                                                                                                                   |
| Find Fruit to<br>Make Juice                | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place,<br>Tool use,<br>Press,<br>Explore            | The fruits are not directly available and visible to the agent because the fruits are stored in a closed fridge or a cabinet. The agents should find the proper fruit first. The example is the same as above.                                                                                                                                                                                                                                                                                    |
| Plug-in Power Cord<br>to Make Juice        | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place,<br>Tool use,<br>Press,<br>Explore,<br>Insert | The other basic settings remain the same as above. However, the juicer’s power cord is not plugged in. The agent needs to first observe this and, using common sense, plug in the power cord to supply power. The example is the same as above.                                                                                                                                                                                                                                                   |
| Take out Cool Drink                        | Composite | M&T, SP,<br>C&W, SEM,<br>L&R      | Open&close door,<br>Pick&place                           | Obtain user requirements through semantically rich interaction: the user wants a cold drink. Given the observation of the same target drink on the desk as disturbance, the agent should use common sense to determine that the drink from the fridge should be chosen. Example: “The weather is so hot! I feel like having a cold soda.”                                                                                                                                                         |
| No Drink in Fridge<br>& Refrigerate Drink  | Composite | M&T, SP,<br>C&W, SEM,<br>L&R      | Open&close door,<br>Pick&place,<br>Explore               | The task is set the same as above. However, after the agent opens the fridge door, it finds that the target object is not there. The agent needs to realize that it should first refrigerate the room-temperature target drink.                                                                                                                                                                                                                                                                   |
| Wrap Proper Toy<br>as Gift                 | Composite | M&T, C&W,<br>SEM, L&R             | Open&close door,<br>Pick&place                           | Choose a suitable toy for kids as a gift from product shelf during the semantic interaction with the user. Then wrap in as a gift. Example: “My son is a superhero fun, but I don’t know that much. Could you wrap a gift for him?”                                                                                                                                                                                                                                                               |
| Rearrange Books<br>by Year                 | Composite | M&T, C&W,<br>SP, L&R              | Pick&place                                               | Identify the book title and use world knowledge to determine the publication period. Then rearrange them. Example: “Rearrange the book by published year order in the top layer of the shelf, the far left is the earliest one.”                                                                                                                                                                                                                                                                  |
| Rearrange Books<br>by Author Name          | Composite | M&T, C&W,<br>SP, L&R              | Pick&place                                               | Identify the book title and use world knowledge to determine the author name. Then rearrange them. Example: “Rearrange the book by their author names, the far right starts with the largest word.”                                                                                                                                                                                                                                                                                               |
| Classify the Books                         | Composite | M&T, C&W,<br>SP, L&R              | Pick&place                                               | Identify the book titles and categorize the books based on their genre or content. The agent needs to infer the classification criteria on its own and correctly divide the books into two layers. Example: “Divide the books into two classes, one class on the top layer while another on the bottom.”                                                                                                                                                                                          |
| Cook Dishes<br>Following Menu              | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place                                               | Multi-turn pick and place the correct ingredients for a dish whose menu is offered by semantic instructions. Example: “I’m about to cook a dish of tomato-fried eggs, prepare ingredients in the tray.”                                                                                                                                                                                                                                                                                           |
| Store Proper Food                          | Composite | M&T, C&W,<br>SEM, L&R             | Open&close door, Pick&place                              | Store the ingredients or fruits into the fridge and do not put the disturbance including snacks into the fridge. Example: “I left some food on the table in the last meal, store them properly please.”                                                                                                                                                                                                                                                                                           |
| Heat Food with<br>Microwave                | Composite | M&T, C&W,<br>SEM, L&R             | Open&close door, Pick&place, Press                       | Extract implicit goals from semantically rich interactions: heating food. Use common sense to choose the proper food, such as a hot dog, with the microwave, while avoiding heating canned food or raw ingredients. Finally, correctly use the microwave. Example: “I just finished class, and now my stomach is growling. Could you heat up some food for me to have a quick bite?”                                                                                                              |
| Plug-in Power Cord<br>to Heat Food         | Composite | M&T, C&W,<br>SEM, L&R             | Open&close door, Pick&place, Press,<br>Insert            | The other experimental settings remain the same as above. The agent must first have the common sense to plug in the power source for the device to operate. The example is the same as above.                                                                                                                                                                                                                                                                                                     |
| Replace Wilted<br>Flower and Drop          | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place,<br>Insert                                    | Based on the semantically rich user request and using common sense, determine the target flower. Discard the wilted flower in the vase, and then insert the new flower. Example: “It’s Valentine’s Day today, replace the flower in the vase.”                                                                                                                                                                                                                                                    |
| Find Condiment<br>and Add to Dish          | Composite | M&T, C&W,<br>SEM, SP,<br>L&R      | Open&close drawer, Pick&Place, Explore,<br>Pour          | All the condiments are stored in the cabinet and the agent should proactively find them first and add proper condiment into the dish. Example: “The spiciness of this dish isn’t quite enough. Could you add some more seasoning to make it tastier?”                                                                                                                                                                                                                                             |
| Hammer Nail<br>&Hang Picture               | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place, Tool use,<br>Hang                            | The agent needs to observe and determine if the nail is loose, and then use a hammer to tighten the nail. After that, the agent should hang the appropriate picture on the wall. Example: “Hang ’the Stary Night’ on the wall steadily.”                                                                                                                                                                                                                                                          |
| Assemble Hammer<br>&Hammer Nail            | Composite | M&T, C&W,<br>L&R                  | Pick&place, Insert,<br>Tool Use                          | The agent needs to observe and reason that the task cannot be completed with the current conditions. It must first assemble the hammer handle and the hammerhead precisely before proceeding. Example: “Hammer the loose nail.”                                                                                                                                                                                                                                                                   |
| Rearrange Chemistry Tube                   | Composite | M&T, SP,<br>C&W, L&R              | Pick&place,<br>Insert                                    | Rearrange the multiple tubes by the corresponding relationships between color and name tag, the result of utilizing common sense and reasoning ability. Example: “Rearrange the solution tubes.”                                                                                                                                                                                                                                                                                                  |
| Texas Holdem Play                          | Composite | M&T, C&W,<br>SEM, L&R             | Pick&Place                                               | Deduce the strongest Texas Hold’em hand based on the common game rules and visual information. Then take multi-step pick&place. Example: “We are playing Texas Hold’em, place your strongest hand combination on the placemat.”                                                                                                                                                                                                                                                                   |
| Flip Facing-downs<br>&Play Texas Holdem    | Composite | M&T, C&W,<br>SEM, L&R             | Pick&Place,<br>Twist,<br>Explore                         | Based on the previous task, some of the cards are face down. The agent needs to have an exploration mindset and actively retrieve all the observational information, then make the correct judgment. The example is the same as above.                                                                                                                                                                                                                                                            |
| Play Mahjong                               | Composite | M&T, C&W,<br>SEM, L&R             | Pick&Place                                               | The agent makes decisions based on world knowledge of Mahjong rules combined with visual information. It discards an unnecessary tile and draws a necessary tile to win. Example: “We seem to be close to winning the game. Take the right actions to help us win this round.”                                                                                                                                                                                                                    |
| Flip Facing-downs<br>&Play Mahjong         | Composite | M&T, C&W,<br>SEM, L&R             | Pick&Place,<br>Twist,<br>Explore                         | The agent needs to have an exploration mindset and actively retrieve all the observational information, then make the correct judgment. The example is the same as above.                                                                                                                                                                                                                                                                                                                         |
| Leverage SeeSaw to<br>Grasp Target         | Composite | M&T, C&W,<br>SEM, PHY,<br>L&R     | Pick&place, Explore,<br>Tool use                         | Using the lever principle, place one or more heavy objects on one side of the seesaw to lift the target object on the other side, initially unreachable. The challenge lies in the fact that if the placed weights are insufficient, the agent will need to add additional weights. Example:“I want to eat the pear in the glass container, but I can’t get it out. Can you help me?”                                                                                                             |
| Find Weights to<br>Leverage SeeSaw         | Composite | M&T, C&W,<br>SEM, SP,<br>PHY, L&R | Open&close drawer, Pick&place,<br>Explore,<br>Tool use   | All the weights are stored in the cabinet and are not visible. The agent needs to explore multiple drawers to find enough weights before being able to properly use the seesaw. The example is the same as above.                                                                                                                                                                                                                                                                                 |
| Get Black Coffee                           | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place,<br>Tool use,<br>Press                        | The agent needs to derive the task goal from semantically rich interactions: to prepare a cup of coffee without milk and sugar. Then, it should correctly place the cup and operate the coffee machine. Example: “I’m feeling sleepy right now. Could you get me a cup of coffee? An Americano will do.”                                                                                                                                                                                          |
| Get Sweet Coffee                           | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place,<br>Tool use,<br>Press,<br>Pour               | The agent needs to additionally infer firstly: the user prefers sweet coffee -¿ the coffee needs to be prepared with sugar. Example: “Get me a cup of sweet coffee to clear my mind, thx!”                                                                                                                                                                                                                                                                                                        |
| Get Latte Coffee                           | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place,<br>Tool use,<br>Press,<br>Pour               | The agent needs to additionally infer firstly: latte coffee is composed of black coffee and milk. Example: “A cup of latte please. Nice to meet you here, thank god!”                                                                                                                                                                                                                                                                                                                             |
| Set Dining Table<br>by Menu                | Composite | M&T, C&W,<br>SEM, SP,<br>L&R      | Pick&place                                               | The agent needs to infer the appropriate utensils based on semantic interactions and the type of cuisine. For example, chopsticks for Chinese cuisine, a knife and fork for Western cuisine, and a spoon if soup is being served. Example: “Today’s main course is steak! Please help me set up the table.”                                                                                                                                                                                       |
| Set Dining Table<br>Left-Handed            | Composite | M&T, C&W,<br>SEM, SP,<br>L&R      | Pick&place                                               | The agent needs to first extract the key information from user interactions that the user is left-handed. Then, using common sense, it should adjust the placement of the utensils, such as switching from the original left-knife-right-fork arrangement to a left-fork-right-knife setup. Example: “Tonight, we’re having fried rice! Remember to get me a spoon. Oh, and don’t forget that I’m left-handed.”                                                                                   |
| Play Snooker                               | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place                                               | Put the ball into the hole by snooker order: yellow, green, brown, blue, pink, black. The agent needs to make the correct sequence of decisions based on snooker rules in world knowledge. Example: “Put the colored billiards into holes by score order in a snooker match.”                                                                                                                                                                                                                     |
| Cluster Toy                                | Composite | M&T, C&W,<br>SP, L&R              | Pick&place                                               | Based on common sense, world knowledge, and visual information, cluster the toys according to their associated IPs, character types, and other attributes. Example: “Cluster the toys into two classes.” These toys are Spiderman, Hawk Eye, Nami, Chopper.                                                                                                                                                                                                                                       |
| Classify Desserts                          | Composite | M&T, C&W,<br>SP, L&R              | Pick&place                                               | Based on common sense, world knowledge, and visual information, categorize the desserts according to their types. Example: “Classify the different desserts.” These desserts are strawberry donut, banana donut, coco cupcake, common cupcake.                                                                                                                                                                                                                                                    |
| Setup Study Table                          | Composite | M&T, C&W,<br>SP, L&R              | Pick&place<br>Open laptop                                | Determine the task goal from semantically rich interactions: the user needs a specific book and to use the computer. The agent needs to use common sense to infer the correct book and place it on the desk, while also turning on the computer. Example: “I have a Python practical exam the day after tomorrow, and I’m planning to review later. Could you help me set up my desk?”                                                                                                            |
| Organize Study Table                       | Composite | M&T, C&W,<br>SP, L&R              | Pick&place<br>Close laptop                               | Determine the task goal by observing the desk and combining user interactions: organize the desk. This requires completing subtasks in sequence, including arranging the books and closing the laptop. Example: “That’s all for today. Please help me tidy up the desk. Thanks!”                                                                                                                                                                                                                  |
| Math Game                                  | Composite | M&T, C&W,<br>SEM, L&R             | Pick&place                                               | Based on the math problem provided by the user, use logical reasoning to find the answer and display it by arranging number blocks to form the solution. Example: “Let’s play a math game, show me the answer by number blocks. The question is: ‘Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?’” This math question is from the GSM8K dataset [9]. |
| Art Game                                   | Composite | M&T, C&W,<br>SEM, SP,<br>PHY, L&R | Pick&place                                               | Place the geometric object with a specific physical property onto the painting that aligns with the user’s hinted content or style. Example: “Let’s play a game of ’Simon Says’! Place the geometric object with a specific physical property onto the painting that aligns with the user’s hinted content or style.”.                                                                                                                                                                            |
| Cluster Beverage                           | Composite | M&T, C&W,<br>SP, L&R              | Pick&place                                               | Based on common sense, world knowledge, and visual information, cluster the drinks according to their types. Example: “Cluster the beverages into two types.” These beverages are mango juice, milk, Vodka, Champagne.                                                                                                                                                                                                                                                                            |
