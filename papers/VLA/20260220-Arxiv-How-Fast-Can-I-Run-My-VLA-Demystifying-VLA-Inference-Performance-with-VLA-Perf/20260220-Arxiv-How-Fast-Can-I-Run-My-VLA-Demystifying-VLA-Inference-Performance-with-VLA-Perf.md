# Title: How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf

- ArXiv: 2602.18397
- Authors: Wenqi Jiang, NVIDIA Research, Jason Clemons, Karu Sankaralingam, Christos Kozyrakis
- Sections: 21
- Estimated tokens: 21.3k

## Contents

- 1 Introduction
- 2 Background and Motivation
  - 2.1 Vision-Language-Action Models
  - 2.2 Efficient VLA Inference
  - 2.3 Research Gap: Comprehensive Analysis of the VLA Inference Performance Landscape
- 3 Analyzing VLA Inference Performance with VLA-Perf
- 4 Evaluation and Takeaways
  - 4.1 Evaluation Setup
  - 4.2 Baseline π 0 \pi\_{0} Inference Latency Across Hardware Backends
  - 4.3 Scaling Model Sizes Under Real-Time Constraints
  - 4.4 Long-Context VLA Inference
  - 4.5 Impact of Denoising Steps and Action Chunk Size
  - 4.6 Diffusion-Based vs. Autoregressive Action Prediction
  - 4.7 On-Device vs. Server-Side Inference
  - 4.8 Device-Server Collaborative Inference
  - 4.9 Asynchronous Inference
  - 4.10 Dual-system VLA Pipelines
  - 4.11 Supporting High-Performance VLA Inference up to 100 Hz
- 5 Conclusion and Future Work
- References
- Appendix A Detailed System and Model Parameters

## Abstract

###### Abstract

Vision-Language-Action (VLA) models have recently demonstrated impressive capabilities across various embodied AI tasks.
While deploying VLA models on real-world robots imposes strict real-time inference constraints, the inference performance landscape of VLA remains poorly understood due to the large combinatorial space of model architectures and inference systems.
In this paper, we ask a fundamental research question: How should we design future VLA models and systems to support real-time inference?
To address this question, we first introduce VLA-Perf, an analytical performance model that can analyze inference performance for arbitrary combinations of VLA models and inference systems.
Using VLA-Perf, we conduct the first systematic study of the VLA inference performance landscape.
From a model-design perspective, we examine how inference performance is affected by model scaling, model architectural choices, long-context video inputs, asynchronous inference, and dual-system model pipelines.
From the deployment perspective, we analyze where VLA inference should be executed — on-device, on edge servers, or in the cloud — and how hardware capability and network performance jointly determine end-to-end latency.
By distilling 15 key takeaways from our comprehensive evaluation, we hope this work can provide practical guidance for the design of future VLA models and inference systems.

<a id="section-1"></a>

## 1 Introduction

Embodied AI is widely regarded as a promising next phase of AI, with the potential to enable physical agents that can perceive, reason, and act in the real world.
Notably, Vision–Language–Action (VLA) models have recently demonstrated strong capabilities in general-purpose manipulation tasks by integrating visual perception and language understanding into the action generation process Zitkovich et al. (2023); Black et al. (2024); Intelligence et al.; Amin et al. (2025); Team et al. (2025); Bjorck et al. (2025).

To react to real-time changes in the physical world, VLA inference must operate with low latency, motivating recent work to treat inference performance(^2^22In this paper, performance always refers to inference latency and throughput, rather than task success rate.) as a first-class concern in VLA model design.
Such efforts include adopting smaller models Wen et al. (2025a); Shukor et al. (2025); Lin et al. (2025) and quantization Kim et al. (2024); Wang et al. (2025a), skipping selected layers Yue et al. (2024); Yang et al., using fewer denoising steps in diffusion-based models Bjorck et al. (2025), enabling asynchrony between model inference and robot execution Black et al. (2025a, b); Sendai et al. (2025); Tang et al. (2025), and adopting dual-system designs comprising two models of different scales, where only the smaller model operates at high frequency Figure AI (2025); Zhang et al. (2024); Song et al. (2025).

While these efforts on efficient VLA model design are an important step forward, we still lack a comprehensive understanding of the VLA inference performance landscape, which is determined by the vast combinatorial space of possible (1) models and (2) inference systems.
Here, an inference system is a combination of (a) the inference accelerator, ranging from edge GPUs to datacenter-class GPUs; (b) the location where inference is executed — on device, on server, or hybrid; and (c) for server-side inference, the wired or wireless network connecting the robot and the server.
As we will show in our evaluation, executing the same VLA model across different inference systems can lead to performance differences of multiple orders of magnitude.

In this paper, we present the first systematic study of VLA inference performance.
This study aims to answer a simple yet fundamental question: how should we design VLA models and systems to achieve real-time inference performance?
Given that standard RGB camera frame rates typically range from 24 to 60 Hz, we define a 10 Hz inference frequency as acceptable (not too far from video ingestion rates) and 100 Hz as high-performance (exceeding common ingestion rates).
Based on this assumption, we further break down our research question to a series of concrete questions:

(1) From the perspective of VLA models, we ask: how should future VLA models be designed under real-time performance constraints?
In particular, how much can we scale up model sizes while achieving real-time inference (§[4.3](#section-4-3))?
Are long-context VLAs that possess thousands of visual frames practically feasible (§[4.4](#section-4-4))?
How does the choice between autoregressive and diffusion-based action experts affect inference performance (§[4.6](#section-4-6))?
How do denoising steps and action chunk size influence performance (§[4.5](#section-4-5))?
How much performance gain can be achieved through asynchronous or dual-system inference (§[4.9](#section-4-9) and §[4.10](#section-4-10))?

(2) From a systems perspective, we ask: how should we deploy efficient inference systems for various VLA workloads?
Given a model with verified accuracy, we decompose the deployment problem into the following considerations:
Where should inference be executed — on device, on server, or via device–server collaboration (§[4.7](#section-4-7) and §[4.8](#section-4-8))?
How to choose inference hardware given the various types of available GPUs (§[4.7](#section-4-7))?
How critical is network performance in server-side inference systems (§[4.7](#section-4-7))?
What combinations of models and systems are required to support VLA inference at rates from 10 Hz up to and beyond 100 Hz (§[4.11](#section-4-11))?

VLA-Perf.
To enable such systematic analysis across the nearly unbounded combinatorial space of VLA models and inference systems, we develop VLA-Perf, an analytical, roofline-based performance model that predicts the optimal inference latency and throughput for arbitrary model–system combinations ([Figure˜1](#figure-1)).
VLA-Perf supports a wide range of VLA configurations, including varying model sizes and architectures, stateless and long-context inference, different action chunk sizes, asynchronous inference, dual-system model pipelines.
In addition, VLA-Perf supports diverse deployment scenarios, spanning inference hardware, inference locations, and network configurations.
We open-source VLA-Perf to enable further performance analysis beyond those presented in this paper: [https://github.com/NVlabs/vla-perf](https://github.com/NVlabs/vla-perf).

Using VLA-Perf, we conduct an extensive evaluation of VLA inference performance across a broad space of model variants and system designs.
From the results, we summarize 15 key performance takeaways that provide practical guidance for the design of future VLA models and inference systems.

<a id="figure-1"></a>

![Overview-VLA-Perf](images/Overview-VLA-Perf.png)

> Figure 1: VLA-Perf enables a comprehensive performance analysis of the VLA inference landscape. Our systematic study explores the interplay between model architectures and deployment configurations, yielding 15 actionable insights for designing future VLA models and serving systems.

<a id="section-2"></a>

## 2 Background and Motivation

<a id="section-2-1"></a>

### 2.1 Vision-Language-Action Models

VLA models enable embodied agents to perceive the environment through vision, reason over language instructions, and generate physical actions.
Recent VLA models have demonstrated strong performance on general-purpose manipulation tasks using robotic arms Brohan et al. (2022); Black et al. (2024); Kim et al. (2024); Zhao et al. (2023) and humanoid robots Bjorck et al. (2025); Figure AI (2025).

Model architecture.
Existing VLA models adopt either autoregressive-based or diffusion-based (including flow matching) action generation.
Autoregressive models use a single transformer to integrate visual observations, interpret language instructions, and generate actions, producing one action dimension (or token) at a time in an iterative manner.
Representative examples of this paradigm include the RT series Brohan et al. (2022); Zitkovich et al. (2023); Gu et al. (2023), OpenVLA Kim et al. (2024), and Octo Team et al. (2024).
More recently, an alternative VLA paradigm combines a VLM backbone with a separate, typically smaller, diffusion-based action expert.
Here, the VLM backbone ingests vision and language inputs, while a diffusion-based action expert attends to the VLM’s KV cache and generates actions through an iterative refinement process, with the number of denoising steps (^3^33For brevity, we use the term _denoising steps_ to describe the iterative refinement steps in both classic diffusion models and flow-matching–based variants.) as a configurable parameter.
Representative diffusion-style VLA models include the $\pi_{0}$ series Amin et al. (2025); Intelligence et al.; Black et al. (2024), GR00T Bjorck et al. (2025), SmolVLA Shukor et al. (2025), and TinyVLA Wen et al. (2025a).

Action prediction.
Each robot action typically consists of multiple dimensions, such as joint positions, velocities, or torques for robotic arms, or whole-body joint configurations for humanoid robots.
To enable smooth and stable execution, many VLA models employ _action chunking_, where the model predicts a sequence of future actions in a single inference Zhao et al. (2023); Kim et al. (2025); Jing et al. (2025), with the sequence length referred to as the action chunk size.
Under action chunking, an execution horizon can be specified, defined as the number of actions actually executed before the next inference is performed, which is no larger than the chunk size Black et al. (2025a, b).
A larger execution horizon can improve action smoothness and reduce inference frequency, but it also reduces the model’s ability to react promptly to changes in the external environment.

<a id="figure-2"></a>

![Timeline](images/Timeline.png)

> Figure 2: An example timeline of synchronous VLA inference. An efficient inference system should aim to match camera ingest rates to provide the robot with real-time action guidance.

<a id="section-2-2"></a>

### 2.2 Efficient VLA Inference

An VLA system should aim to achieve real-time inference (10 to 100 ms of latency) to match the rate of visual signal ingestion, as visualized in [Figure˜2](#figure-2).
To meet this demand, a growing number of work has proposed techniques to improve VLA inference efficiency at both the model and system levels Yu et al. (2025).

Reduce computation.
The amount of computation can be reduced by adopting smaller models Wen et al. (2025a); Shukor et al. (2025); Lin et al. (2025); Sun et al. (2026) and quantization Kim et al. (2024); Wang et al. (2025a), skipping selected VLM layers Yue et al. (2024); Yang et al., or reducing the number of denoising steps in diffusion-based action experts Bjorck et al. (2025).
For autoregressive VLAs, parallel decoding can further accelerate inference by reusing KV cache for multi-token predictions Kim et al. (2025).
Finally, action chunking allows the model to predict a sequence of actions to execute in a single inference call, thereby reducing inference frequency Zhao et al. (2023); Black et al. (2025a).

Asynchronous inference and dual-system VLAs.
Inference performance can also be improved through various forms of asynchrony.
For example, we can allow inference to begin while the robot is still executing previous actions Black et al. (2025a, b); Sendai et al. (2025); Tang et al. (2025).
This inference-execution overlap improves GPU utilization and consequently inference throughput.
Alternatively, a dual-system VLA pipeline runs a lightweight action expert at a higher frequency (System 1) while invoking a more expensive VLM backbone at a lower frequency (System 2) Figure AI (2025); Zhang et al. (2024); Song et al. (2025), with the two systems asynchronously exchanging latent states.

Better inference systems.
While higher inference frequencies can be attained through more powerful hardware, software-level optimizations are also critical for VLA inference efficiency.
Careful CUDA-level optimizations, including CUDA graph and operator fusion, can reduce inference latency by up to $5\times$ compared to a naive PyTorch implementation Ma et al. (2025).
For server-side inference with action chunking, network latency and robot execution latency can be overlapped to reduce end-to-end execution time Huang et al. (2025).

<a id="section-2-3"></a>

### 2.3 Research Gap: Comprehensive Analysis of the VLA Inference Performance Landscape

Despite the advances in efficient VLA designs introduced above, we still lack a comprehensive understanding of the VLA inference performance landscape, largely due to (1) the wide diversity of inference system configurations across prior studies and (2) the limited exploration of model architectures driven by inference performance concerns.
From a systems perspective, VLA inference can be performed on edge GPUs integrated within the robot (on-device) Figure AI (2025), on GPU servers near the robot (edge-server) Black et al. (2024); Amin et al. (2025), or offloaded to powerful accelerators in the cloud (cloud-server) Brohan et al. (2022); Zitkovich et al. (2023) — the performance across these configurations can vary significantly, as we will show in the evaluation.
From a model-design perspective, while existing VLA models are often designed with inference efficiency in mind Shukor et al. (2025); Wen et al. (2025a), they often target specific application-system pairings.
Given the rapid evolution of both workloads and inference hardware, this approach can be myopic and can limit the exploration of alternative designs involving larger models or longer context.

<a id="section-3"></a>

## 3 Analyzing VLA Inference Performance with VLA-Perf

In this paper, we aim to provide a comprehensive analysis of VLA inference performance across both existing and future, potentially hypothetical, combinations of VLA models and inference systems.
Our evaluation focuses exclusively on performance characteristics — including latency and throughput — under the assumption that the underlying models meet the necessary accuracy thresholds for deployment.

However, conducting such a comprehensive analysis is much more challenging than profiling inference performance on a small set of existing models and system implementations.
This is because it requires (1) setting up systems with various accelerator capabilities, inference locations, and network configurations, as discussed in §[2.3](#section-2-3), and (2) evaluating not only existing models but also plausible future model variants — the resulting combinatorial explosion renders exhaustive empirical evaluation both cost- and time-prohibitive.

To address this challenge, we adopt a modeling-based approach to performance analysis, which has shown strong effectiveness in prior work on LLM inference and training systems Davies et al. (2025); Bambhaniya et al. (2024); Agrawal et al. (2024); Cho et al. (2024); Yuan et al. (2024); Jiang et al. (2025).
Analytical performance models focus on capturing the dominant performance characteristics of both the model (e.g., FLOPs and memory accesses) and the hardware (e.g., peak FLOP/s, memory bandwidth, and network bandwidth), and estimate achievable performance based on these attributes.
This approach enables fast, low-cost performance analysis across arbitrary combinations of models and hardware, without requiring the deployment of real systems.
On the downside, analytical models are not perfectly accurate, as they typically assume optimistic software implementations and therefore estimate the upper bound of achievable performance.
For example, a recent study on optimizing VLA inference performance reports that 68$\sim$75% of roofline-model-predicted performance can be achieved on real systems Ma et al. (2025).
While such predictions are not exact, we are still at an early stage in understanding VLA inference performance, and thus even coarse-grained estimates can provide valuable guidance for future model and system designs.

<a id="figure-3"></a>

![transformer-ops](images/transformer-ops.png)

> Figure 3: VLA-Perf abstracts VLA inference as model components interleaved with data transfers.

VLA-Perf overview.
We build VLA-Perf, a roofline-based analytical performance model for VLA inference.
Figure [3](#figure-3) illustrates an example VLA inference workflow, which consists of a robot and multiple model components.
Depending on the placement of each model component (either on the robot or on a server), these components exchange data either locally or over a network.
Such data transfers may include raw images, vision tokens, KV caches, or action predictions.
Each model component is abstracted as a sequence of operators, such as fully connected layers, linear projections, and attention blocks.
VLA-Perf assumes that inference for each individual model component (e.g., the VLM backbone) is executed on a single accelerator, because modern GPUs, including recent edge accelerators, already provide sufficient memory capacity to host complete VLA models, for example up to 128 GB on NVIDIA Jetson Thor.
On the contrary, different model components can be executed either on the same accelerator or on different accelerators.

Input parameters.
VLA-Perf enables the analysis of arbitrary model-system combinations by parameterizing both model and system parameters, with an example provided in LABEL:{fig:methodology:input_params}.
On the model side, these parameters include the choice of vision encoder, VLM backbone, and action expert; the input and output sequence lengths of each model; the number of denoising steps for diffusion-based action experts; action chunk size; and the dimensionality of each action.
On the system side, VLA-Perf supports various inference accelerators of configurable peak FLOP/s and memory bandwidth, as well as network systems characterized by upload/download bandwidth and latency.

Latency calculation.
Given the inputs above, the end-to-end inference latency of a VLA system is modeled as the sum of model inference latency and data movement latency across all components:

$$
T_{\text{total}}=\sum_{m\in\mathcal{M}}T_{m}+\sum_{d\in\mathcal{D}}T_{d},(1)
$$

where $\mathcal{M}$ denotes the set of model inference components and $\mathcal{D}$ denotes the set of data movement stages.

For a single model component $m$, the inference latency $T_{m}$ is modeled as the sum of latency of each of its constituent operators:

$$
T_{m}=\sum_{o\in\mathcal{O}_{m}}T_{o},(2)
$$

where $\mathcal{O}_{m}$ denotes the sequence of operators in model $m$.
For each operator $o$, VLA-Perf models its execution latency using a roofline model that accounts for both compute and memory access latency:

$$
T_{o}=\max\left(\frac{\text{FLOPs}_{o}}{\text{FLOP/s}_{h}},\frac{\text{Bytes}_{o}}{\text{MemBW}_{h}}\right),(3)
$$

where $\text{FLOPs}_{o}$ and $\text{Bytes}_{o}$ denote the total floating-point operations and memory bytes accessed by operator $o$, while $\text{FLOP/s}_{a}$ and $\text{MemBW}_{a}$ denote the peak compute throughput and memory bandwidth of the inference hardware $h$, respectively.

We assume that local data movement on the same accelerator is sufficiently fast to be treated as negligible, while network-based data movement between devices is modeled as:

$$
T_{d}^{\text{net}}=\text{NetLat}+\frac{\text{Bytes}_{d}}{\text{NetBW}},(4)
$$

where $\text{Bytes}_{d}$ denotes the amount of transferred data, and NetBW and NetLat denote the single-directional network bandwidth and latency, respectively.

<a id="table-1"></a>

> Table 1: Roofline model validation against real $\pi_{0}$ Triton inference latencies on an RTX 4090 Ma et al. (2025). This evaluation uses 10 flow-matching steps, action chunk size of 63, and an empty language prompt.

| Metric                   | 1 camera | 2 cameras | 3 cameras |
| ------------------------ | -------- | --------- | --------- |
| Roofline (VLA-Perf)      | 14.7 ms  | 22.5 ms   | 30.4 ms   |
| Real Perf. (Triton)      | 20.0 ms  | 27.3 ms   | 36.8 ms   |
| Fidelity (Real/Roofline) | 73.3%    | 82.3%     | 82.6%     |

Modeling fidelity.
Due to the scarcity of well-optimized frameworks for VLA inference, we mainly validate the fidelity of VLA-Perf using the $\pi_{0}$ implementation by Ma et al. Ma et al. (2025), a Triton-based implementation specifically tuned for RTX 4090.
[Table˜1](#table-1) compares the performance predicted by VLA-Perf to the empirical measurements conducted by Ma et al. Ma et al. (2025).
The results demonstrate that an optimized system can achieve $73.3\sim 82.6\%$ of the theoretical roofline reported by VLA-Perf, with the gap narrowing as the workload increases (e.g., when processing three camera frames).

The performance differences between a real inference system and the roofline limits reported by VLA-Perf are due to both hardware and software factors.
First, VLA-Perf abstracts away hardware-specific details, including microarchitectural design, instruction scheduling, and memory-access behavior.
Instead, VLA-Perf assumes that the maximum theoretical compute capability and memory bandwidth are attainable for every operator executed.
Second, real-world systems incur software overheads, such as kernel launch latencies, operating system interference, and runtime library overhead, which are not explicitly modeled by VLA-Perf.
Nevertheless, we believe that a modeling fidelity exceeding 80% is sufficient to provide meaningful insights into the VLA performance landscape, and thus leave the tuning of VLA-Perf for specific hardware and software platforms, such as in Davies et al. (2025); Jiang et al. (2025), to future work.

<a id="section-4"></a>

## 4 Evaluation and Takeaways

In this section, we use VLA-Perf to conduct a comprehensive analysis of VLA inference performance.
Our evaluation is structured to address two sets of research questions as below.

Question 1: How should we design future VLA models to meet real-time latency constraints?

- •
  How far can model sizes be scaled while still enabling real-time inference (§[4.3](#section-4-3))?
- •
  Are long-context VLAs that process thousands of visual frames practically feasible (§[4.4](#section-4-4))?
- •
  How do autoregressive and diffusion-based action experts compare in performance (§[4.6](#section-4-6))?
- •
  How do denoising steps and action chunk size influence performance (§[4.5](#section-4-5))?
- •
  Are asynchronous or dual-system inference much faster than synchronous inference (§[4.9](#section-4-9) and §[4.10](#section-4-10))?

Question 2: How should inference systems be deployed for different VLA workloads?

- •
  Should inference be executed on device, on server, or via device-server collaboration (§[4.7](#section-4-7) and §[4.8](#section-4-8))?
- •
  How capable must inference hardware be to meet real-time performance requirements (§[4.7](#section-4-7))?
- •
  How critical is network performance for server-side inference systems (§[4.7](#section-4-7))?
- •
  What model-system combinations can achieve inference rates from 10 Hz to 100 Hz (§[4.11](#section-4-11))?

Our experiments are organized as follows.
§[4.2](#section-4-2) presents a baseline analysis of the $\pi_{0}$ model.
§[4.3](#section-4-3)$\sim$[4.6](#section-4-6) explore various model configuration to examine their impact on inference performance.
§[4.7](#section-4-7)$\sim$[4.11](#section-4-11) additionally considers inference placement and network latency, closely reflecting real-world deployments.

<a id="section-4-1"></a>

### 4.1 Evaluation Setup

We describe the main model and system settings here, with additional details provided in Appendix [A](#appendix-a).

Models and robot.
We evaluate a set of model variants derived from the $\pi_{0}$ architecture Black et al. (2024), which we choose due to its strong robotic task performance and widespread adoption in recent VLA systems.
The original $\pi_{0}$ model consists of a 400M SigLIP vision encoder, a 2B Gemma language model, and a 300M diffusion-based action expert.
Throughout the experiments, we consider a bimanual robotic manipulation setting, which is common for both stationary robots and mobile platforms such as wheeled or humanoid robots.
With the UR5e robot arm, this setup is equipped with three cameras and an action space of 14 degrees of freedom (DoF).
Each camera image has a resolution of 224$\times$224 and is tokenized into 256 visual tokens, yielding 768 visual tokens across three cameras.
Assuming 32 language tokens per task, the total input sequence length per inference is 800 tokens.
Unless otherwise specified, we use an action chunk size of 50 and 10 denoising steps for action generation, same as the original $\pi_{0}$ configuration.

<a id="table-2"></a>

> Table 2: We consider systems with various (1) GPU capabilities (rows) and (2) inference location (columns).

| Capacity and Placement | On-Device | Edge Server | Cloud Server |
| ---------------------- | --------- | ----------- | ------------ |
| Mobile (Thor)          | ✓         |             |              |
| Consumer (RTX 4090)    |           | ✓           |              |
| Datacenter (B100)      |           | ✓           | ✓            |

Inference systems.
We evaluate a range of accelerators spanning high-end edge GPUs (e.g., NVIDIA Jetson Thor), consumer-grade GPUs commonly used in research experiments (e.g., RTX 4090), and high-end datacenter GPUs, including A100, H100, and B100.
The GPUs can be mapped to various inference location as shown in [Table˜2](#table-2).
For server-side inference, we evaluate both wired and wireless network configurations, including Ethernet, WiFi, and cellular (4G/5G) networks.
All experiments assume BF16 for inference, or FP16 when BF16 is not supported by the hardware.

Performance metrics.
We report VLA system latency, defined as the elapsed time from when the robot perceives visual observations to when the robot receives the corresponding action prediction.
We also report throughput in Hertz (Hz), defined as the number of inferences that can be executed per second at a batch size of one (i.e., a single robot).
For synchronous inference, throughput is the inverse of inference latency, whereas for asynchronous inference, throughput can exceed the inverse latency.
We report inference performance independent of robot execution latency, as the latter is highly robot-dependent.
Furthermore, the effective action execution frequency may exceed the inference frequency due to action chunking — with a chunk size of five, the robot can execute actions at up to five actions given a single inference.

<a id="section-4-2"></a>

### 4.2 Baseline π 0 \pi\_{0} Inference Latency Across Hardware Backends

Before evaluating model and system variants, we first establish a baseline by measuring the inference performance of the $\pi_{0}$ model across a range of GPUs, without considering network latency.

<a id="table-3"></a>

> Table 3: Inference performance of $\pi_{0}$ on various GPUs without considering network latency.

| Hardware    | Vision Lat. | VLM Lat. | Action Lat. | E2E Lat. | E2E Freq. |
| ----------- | ----------- | -------- | ----------- | -------- | --------- |
| Jetson Thor | 6.06 ms     | 20.30 ms | 26.20 ms    | 52.57 ms | 19.0 Hz   |
| RTX 4090    | 4.02 ms     | 19.79 ms | 7.25 ms     | 31.06 ms | 32.2 Hz   |
| A100        | 2.13 ms     | 10.47 ms | 3.60 ms     | 16.20 ms | 61.7 Hz   |
| H100        | 0.71 ms     | 3.30 ms  | 2.14 ms     | 6.15 ms  | 162.5 Hz  |
| B100        | 0.40 ms     | 1.87 ms  | 0.91 ms     | 3.18 ms  | 314.4 Hz  |

<a id="table-4"></a>

> Table 4: Compute- vs. memory-bound analysis of $\pi_{0}$ across different hardware. Operator intensity (OI) denotes the ratio between compute operations and memory accesses (FLOPs/Bytes). The balance OI denotes the hardware balance point at which compute throughput and memory bandwidth are equally limiting.

| Hardware    | Balance OI | Vision (OI=321.4) | VLM (OI=542.8) | Action (OI=54.0) |
| ----------- | ---------- | ----------------- | -------------- | ---------------- |
| Jetson Thor | 1481.5     | Memory            | Memory         | Memory           |
| RTX 4090    | 163.7      | Compute           | Compute        | Memory           |
| A100        | 153.0      | Compute           | Compute        | Memory           |
| H100        | 295.2      | Compute           | Compute        | Memory           |
| B100        | 218.8      | Compute           | Compute        | Memory           |

Takeaway 1: Existing datacenter GPUs can already achieve inference frequencies comparable to camera frame rates for small VLA models such as $\pi_{0}$, while edge GPUs remain performance-limited.

[Table˜3](#table-3) shows that A100, H100, and B100 achieve inference frequencies ranging from 61.7 Hz to 314.4 Hz, which are at least on par with the frame rates of common RGB cameras (24$\sim$60 Hz).
In contrast, Jetson Thor achieve substantially lower inference frequency (19.0 Hz), falling below the frame rates of most cameras.

Takeaway 2: Action prediction is memory-bound across hardware, while vision and VLM inference are compute-bound on most GPUs except from Jetson Thor.

[Table˜4](#table-4) summarizes the workload characteristics of each VLA model component.
The vision encoder and the VLM backbone exhibit significantly higher operator intensity (321.4 and 542.8 FLOPs/Byte, respectively) compared to the action expert (54.0 FLOPs/Byte).
This is because the vision encoder and the VLM backbone process many input tokens (e.g., 768 for SigLIP and 800 for Gemma), inherently batching computation across tokens, whereas the diffusion-based action expert operates on far fewer tokens (e.g., the same as the action chunk size of 50).
This behavior closely mirrors LLM inference, where the prefill phase (prompt processing) is compute-intensive, while the decode phase (token generation) is memory-bound Patel et al. (2024).
Jetson Thor, in contrast to the other evaluated GPUs, relies on LPDDR memory, which prioritizes low power consumption for embedded devices but provides substantially lower bandwidth (270 GB/s) than GDDR on RTX 4090 (1 TB/s) and HBM on B100 (8 TB/s).
As a result, even the vision encoder and VLM backbone become memory-bound on Jetson Thor.

<a id="section-4-3"></a>

### 4.3 Scaling Model Sizes Under Real-Time Constraints

We next study how inference latency scales with increasing VLA model sizes, which are positively correlated with task accuracy Team (2025).
Specifically, we scale each component of the $\pi_{0}$ model and construct a family of larger VLA models.
For the vision encoder, we replace the original SigLIP-So400m used in $\pi_{0}$ with the larger SigLIP-Giant model with 1.1B parameters.
For the VLM, we replace Gemma with the Llama2 family (7B, 13B, and 70B), which provides a wider range of model scales.
For the action expert, we follow the $\pi_{0}$ design principle and instantiate it as a scaled-down version of the corresponding VLM, with approximately $4\sim 8\times$ fewer parameters by reducing the transformer hidden dimension and intermediate dimension by $2\times$ and $4\times$, respectively.
By combining these components, we construct a set of hypothetical larger VLA models, denoted as $\pi_{0}$-L, $\pi_{0}$-XL, and $\pi_{0}$-XXL, whose configurations are summarized in [Table˜5](#table-5).

<a id="table-5"></a>

> Table 5: Inference performance of scaled-up VLA models across different hardware platforms.

| Model                 | Vision Encoder      | VLM                | Action Expert   | Jetson Thor | RTX 4090 | B100     |
| --------------------- | ------------------- | ------------------ | --------------- | ----------- | -------- | -------- |
| $\pi_{0}$ (2.7B)      | SigLIP-So (0.4B)    | Gemma-2B (2.0B)    | Act-M (0.3B)    | 19.0 Hz     | 32.2 Hz  | 314.4 Hz |
| $\pi_{0}$-L (9.1B)    | SigLIP-Giant (1.1B) | Llama2-7B (6.5B)   | Act-L (1.5B)    | 3.9 Hz      | 8.0 Hz   | 73.6 Hz  |
| $\pi_{0}$-XL (16.7B)  | SigLIP-Giant (1.1B) | Llama2-13B (12.7B) | Act-XL (2.9B)   | 2.1 Hz      | N/A      | 39.7 Hz  |
| $\pi_{0}$-XXL (81.3B) | SigLIP-Giant (1.1B) | Llama2-70B (68.5B) | Act-XXL (11.7B) | N/A         | N/A      | 9.6 Hz   |

<a id="figure-5"></a>

![pi0_model_scaling_components](images/pi0_model_scaling_components.png)

> Figure 5: Increased model sizes lead to proportional inference latency increases.

Takeaway 3: Latency of each VLA component scales approximately linearly with increasing model sizes.

[Figure˜5](#figure-5) breaks down the inference latency of individual VLA components as model size increases, where both axes are shown on a logarithmic scale.
Across all components, larger models impose proportionally higher computational costs, and thus inference latency grows approximately linearly with model size.

Takeaway 4: While edge and consumer GPUs struggle with larger models, datacenter GPUs can still support real-time inference for VLA models that are more than one order of magnitude larger.

[Table˜5](#table-5) summarizes inference performance across different model scales.
B100 sustains 9.6 Hz inference even for the largest 81B model variant (30$\times$ larger than $\pi_{0}$), demonstrating that modern datacenter GPUs can accommodate substantially larger VLA models under real-time constraints.
In contrast, RTX 4090 runs out of memory for $\pi_{0}$-XL (16.7B), and Jetson Thor struggles to deliver real-time performance even with sufficient memory capacity, achieving only 2.1 Hz inference frequency on $\pi_{0}$-XL.

<a id="section-4-4"></a>

### 4.4 Long-Context VLA Inference

While $\pi_{0}$ model ipredicts actions solely based on the current observation, this a memory-less design is insufficient for long-horizon tasks that require reasoning over temporal context Jang et al. (2025); Shi et al. (2025); Team et al. (2024); Wang et al. (2025b).
In this section, we adapt $\pi_{0}$ to a stateful setting by enabling it to incorporate past visual states into the VLM KV cache.
At each new timestep, the latest visual inputs (three camera images, corresponding to 768 vision tokens) attend over the accumulated KV cache of the VLM, and the action prediction is conditioned on this long context.

Takeaway 5: Datacenter GPUs can support real-time long-context VLA inference with up to 1K past timesteps, while edge and consumer GPUs are limited to roughly 100 steps.

[Table˜6](#table-6) reports inference performance and memory consumption of long-context VLA with up to 10K past timesteps.
B100 sustains 11.7 Hz inference with 1K past timesteps, whereas performance drops to 1.2 Hz at 10K steps, which no longer meets real-time requirements.
For Jetson Thor and RTX 4090, real-time performance is only achievable when the context length is limited to roughly 100 timesteps (around 8 Hz).

<a id="table-6"></a>

> Table 6: Inference performance and memory consumption of long-context VLA models.

| Timesteps | Total Memory | KV Cache Size | Jetson Thor       | RTX 4090          | B100              |
| --------- | ------------ | ------------- | ----------------- | ----------------- | ----------------- |
| 1         | 5.1 GB       | 0.01 GB       | 52.6 ms (19.0 Hz) | 31.1 ms (32.2 Hz) | 3.2 ms (314.4 Hz) |
| 10        | 5.3 GB       | 0.13 GB       | 58.4 ms (17.1 Hz) | 39.0 ms (25.7 Hz) | 3.9 ms (254.6 Hz) |
| 100       | 6.4 GB       | 1.3 GB        | 122.9 ms (8.1 Hz) | 117.3 ms (8.5 Hz) | 11.3 ms (88.4 Hz) |
| 1000      | 18.3 GB      | 13.2 GB       | 768.3 ms (1.3 Hz) | 900.6 ms (1.1 Hz) | 85.2 ms (11.7 Hz) |
| 10000     | 137.0 GB     | 131.8 GB      | N/A               | N/A               | 823.7 ms (1.2 Hz) |

<a id="section-4-5"></a>

### 4.5 Impact of Denoising Steps and Action Chunk Size

Given a diffusion-based action expert model, two key parameters influence inference performance: (1) the number of denoising steps, where each step incurs a forward pass, and (2) the action chunk size, i.e., the number of predicted actions.
To this end, we vary the number of diffusion steps of $\pi_{0}$ from 1 to 50 (default: 10) and the action chunk size from 5 to 250 (default: 50), each spanning a $50\times$ range.
For brevity, we present results on B100, but the observed trends below are consistent across all evaluated GPUs.

Takeaway 6: Denoising steps have a significant impact on both action expert latency and end-to-end VLA latency, whereas action chunk size has a negligible effect.

[Figure˜6(a)](https://arxiv.org/html/2602.18397v1#S4.F6.sf1) and [Figure˜6(b)](https://arxiv.org/html/2602.18397v1#S4.F6.sf2) report the action expert latency and end-to-end VLA inference latency, respectively.
On the one hand, action prediction latency scales linearly with the number of diffusion steps and thus has a substantial impact on overall VLA latency.
For example, with the default action chunk size of 50, increasing the number of diffusion steps from 10 to 50 leads to a proportional increase in action prediction latency ($5\times$) and a $2.15\times$ increase in overall VLA latency.
On the other hand, action chunk size has only a marginal effect on both action-expert latency and end-to-end VLA inference latency.
With the default setting of 10 denoising steps, increasing the action chunk size from 50 to 250 ($5\times$) increases action prediction latency by only 40%, resulting in just an 11% increase in end-to-end VLA latency.
This is because action prediction is typically memory-bound ([Figure˜6(c)](https://arxiv.org/html/2602.18397v1#S4.F6.sf3)): performance is limited by loading model parameters and KV cache from memory, and the additional computation given more action tokens has little effect on overall latency.

![pi0_device_vs_server_e2e_total_ms](images/pi0_device_vs_server_e2e_total_ms.png)

> (a) Action Expert Latency.

<a id="section-4-6"></a>

### 4.6 Diffusion-Based vs. Autoregressive Action Prediction

Diffusion-based and autoregressive action prediction are two dominant paradigms in recent VLA models.
Autoregressive action decoder typically uses the same transformer to process vision and language inputs and to generate actions Kim et al. (2024); Team et al. (2024).
Accordingly, we adapt $\pi_{0}$ to an autoregressive variant that uses the VLM backbone directly for action prediction.
In contrast, diffusion-based VLAs usually employ a separate action expert that is significantly smaller than the VLM (e.g., $6.7\times$ smaller in $\pi_{0}$) Black et al. (2024); Bjorck et al. (2025).
For fairness, we additionally evaluate a diffusion-based variant with an action expert that matches the VLM size, referred to as Diffusion-Large.
Classic autoregressive VLAs generate one action dimension at a time, which results in high inference latency as it requires many sequential prediction steps (e.g., 700 steps in our case with a 14-DoF action space and an action chunk size of 50).
Thus, we also evaluate a faster autoregressive variant with parallel decoding Kim et al. (2025), which predicts all actions in a single inference, denoted as Autoregressive-Parallel.

Takeaway 7: With action chunking, diffusion-based VLA inference is one to two orders of magnitude faster than the vanilla autoregressive VLA.

[Figure˜7(a)](https://arxiv.org/html/2602.18397v1#S4.F7.sf1) compares inference latency across different architectures on the B100 GPU.
Across all action chunk sizes, diffusion-based models (both the standard and large variants) consistently outperform the vanilla autoregressive VLA.
With the default chunk size of 50, the standard diffusion model achieves an inference latency of 3.2 ms, which is $102.4\times$ faster than the classic autoregressive model (327.6 ms).

Takeaway 8: Autoregressive VLAs are competitive only when generating a small number of action tokens or when parallel decoding is enabled.

To further analyze scenarios where autoregressive VLAs can be efficient, we evaluate inference performance without action chunking across common action dimensionalities, ranging from 7 DoF for a single robot arm Black et al. (2024); Zitkovich et al. (2023) to over 40 DoF for two dexterous hands Wen et al. (2025b); Christoph et al. (2025).
As shown in [Figure˜7(b)](https://arxiv.org/html/2602.18397v1#S4.F7.sf2), the autoregressive model can slightly outperform the large diffusion-based model when the number of generated action tokens is small (e.g., 7), although the standard-size diffusion model remains faster.
Another scenario in which autoregressive inference becomes competitive is when parallel decoding is employed.
As shown in [Figure˜7(a)](https://arxiv.org/html/2602.18397v1#S4.F7.sf1), parallel decoding outperforms the standard diffusion model for action chunk sizes up to 10.
However, for larger chunk sizes such as 50, the latency of parallel decoding increases substantially as the workload transitions from memory-bound to compute-bound (OI increases from 135.9 at chunk size 10 to 477.7 at chunk size 50, exceeding the B100 balance OI of 218.8).
In contrast, the diffusion-based action expert remains memory-bound (OI = 54.0 at chunk size of 50), leading to more stable inference performance across chunk sizes.

![pi0_device_server_collab](images/pi0_device_server_collab.png)

> (a) Varying action chunk sizes.

<a id="section-4-7"></a>

### 4.7 On-Device vs. Server-Side Inference

We evaluate three classes of VLA inference systems with different GPU and network configurations.
First, _on-device inference_, where inference is executed directly on an edge GPU integrated into the robot (e.g., Jetson Thor), as demonstrated by systems such as Figure AI’s Helix Figure AI (2025).
Second, _edge-server inference_, where inference is performed on a server located close to the robot Black et al. (2024); Bjorck et al. (2025); Huang et al. (2025).
In this setting, communication between the robot and the server may use wired networks (Ethernet) for fixed-base robots or wireless networks (WiFi or cellular networks) for mobile robots (e.g., wheeled or humanoid robots), while the server-side accelerator may range from consumer-grade GPUs (e.g., RTX 4090) to datacenter-class GPUs (e.g., B100).
Third, _cloud-server inference_, where inference runs on high-end datacenter GPUs.
In this case, the robot first communicates with a nearby gateway server via a wired or wireless connection, which then forwards inference requests to the cloud, incurring two stages of communication latency.
Note that network latency to cloud servers can vary substantially, depending on factors such as physical distance and routing topology Mok et al. (2021); Sfiligoi et al. (2020); thus, we consider two cloud network configurations with different performance.
We summarize detailed network performance parameters in [Table˜7](#table-7).

Takeaway 9: Server-side inference, even with only consumer GPUs, significantly outperforms on-device inference in most scenarios, except under extremely poor network conditions.

As shown in [Figure˜8](#figure-8), even when using a consumer GPU (RTX 4090) connected via WiFi, server-side inference achieves lower end-to-end latency than on-device inference on Jetson Thor.
With a more powerful B100 GPU, inference remains faster than on-device execution even when deployed on an edge server with only cellular (5G) connectivity or in a cloud instance with fast network.
On-device inference becomes preferable only when network conditions are extremely constrained, such as (i) slow cellular connections (4G or below), or (ii) cloud deployments where the datacenter is distant from the robot.

<a id="section-4-8"></a>

### 4.8 Device-Server Collaborative Inference

Some robots already have an on-board GPU — so a natural idea is to split the VLA workload between server and device to (1) reduce server workloads and to (2) improve performance over device-only deployments.
Since the action expert model is usually several times smaller than the VLM backbone Black et al. (2024); Shukor et al. (2025); Wen et al. (2025a), a natural idea is to run VLM inference (including the vision encoder) on server (B100) and run action expert inference on device (Jetson Thor).
In comparison to either device-only or server-only solutions, here, device-server collaboration involves an extra communication step, where the KV cache of the VLM has to be downloaded to the device GPU before action prediction begins.

Takeaway 10. Device–server collaboration is often slower than device-only inference and always slower than server-side inference, making this solution generally unattractive in practice.

As shown in [Figure˜9](#figure-9), collaborative inference is always slower compared to server-only inference — which is not surprising as now the action expert runs on a less powerful device.
What we found interesting is that it is even slower than on-device inference in most cases, except with a fast wired network (Ethernet 10G) — this is because of the KV cache download process from the server to the device, which can be very slow without a fast network (12.4, 43.7, and 257.7 ms for Ethernet 10G, WiFI 7, and 5G networks, respectively).
However, we argue that such scenarios are rare in practice: robots equipped with on-device GPUs are typically mobile platforms that relies on wireless connectivity, in which case using the on-device GPU alone (rather than device-server collaboration) is the more performant choice.

<a id="table-7"></a>

> Table 7: Network configuration specifications.

| Metric       | Ethernet 1G | Ethernet 10G | WiFi 6   | WiFi 7  | 4G       | 5G       | Slow Cloud | Fast Cloud |
| ------------ | ----------- | ------------ | -------- | ------- | -------- | -------- | ---------- | ---------- |
| Upload BW    | 1 Gbps      | 10 Gbps      | 560 Mbps | 2 Gbps  | 19 Mbps  | 80 Mbps  | 1 Gbps     | 10 Gbps    |
| Download BW  | 1 Gbps      | 10 Gbps      | 800 Mbps | 3 Gbps  | 75 Mbps  | 500 Mbps | 1 Gbps     | 10 Gbps    |
| Base Latency | 0.10 ms     | 0.05 ms      | 3.50 ms  | 2.50 ms | 25.00 ms | 10.00 ms | 100.00 ms  | 10.00 ms   |

Figure: Figure 8: Inference performance on device, on edge servers, and on cloud servers.
Refer to caption: x10.png

Figure: Figure 9: Device-server collaborative inference versus server-only and device-only solutions.
Refer to caption: x11.png

<a id="section-4-9"></a>

### 4.9 Asynchronous Inference

With asynchronous inference, the model predicts actions based on stale observations rather than the latest state, allowing model inference and robot action execution to be partially overlapped.
While this form of asynchrony does not increase the maximum inference throughput for on-device inference without network latency, it can benefit server-side inference by allowing network transmission and GPU computation to proceed concurrently.
Thus, the asynchronous inference throughput is bounded by the minimum of GPU inference throughput and network transmission throughput.

Takeaway 11: Asynchrony between robot execution and inference can significantly improve system throughput for server-side inference, especially under slow wireless network connections.

[Table˜8](#table-8) reports the server-side inference throughput under different network configurations.
With fast wired networks (e.g., 1 GbE and 10 GbE Ethernet), synchronous and asynchronous inference achieve similar throughput.
In contrast, under slower wireless networks (WiFi 7, 5G, and 4G), asynchronous inference improves throughput by 2.63$\sim$5.99$\times$.
With WiFi 7, inference remains GPU-bound, and thus achieves the same throughput to wired networks (314.4Hz).
For 5G and 4G, the bottleneck shifts to network transmission, resulting in lower asynchronous throughput.
Note that while asynchronous inference improves throughput, it does not reduce end-to-end latency; increased staleness may degrade action quality, which warrants further investigation from the perspective of control stability and task success rate.

<a id="table-8"></a>

> Table 8: Inference frequency of synchronous and asynchronous systems.

| Hardware | Network            | Latency  | Freq. (Sync) | Freq. (Async) | Speedup       |
| -------- | ------------------ | -------- | ------------ | ------------- | ------------- |
| B100     | Ethernet 10G       | 3.3 ms   | 301.4 Hz     | 314.4 Hz      | 1.04$\times$  |
| B100     | Ethernet 1G        | 3.8 ms   | 266.5 Hz     | 314.4 Hz      | 1.18$\times$  |
| B100     | WiFi 7             | 8.4 ms   | 119.7 Hz     | 314.4 Hz      | 2.63$\times$  |
| B100     | 5G                 | 27.8 ms  | 35.9 Hz      | 215.3 Hz      | 5.99$\times$  |
| B100     | 4G                 | 73.0 ms  | 13.7 Hz      | 50.5 Hz       | 3.68$\times$  |
| B100     | Wired + Fast Cloud | 23.4 ms  | 42.8 Hz      | 314.4 Hz      | 7.34$\times$  |
| B100     | 4G + Slow Cloud    | 273.4 ms | 3.7 Hz       | 50.5 Hz       | 13.79$\times$ |

<a id="section-4-10"></a>

### 4.10 Dual-system VLA Pipelines

Recent work proposes a System 1 + System 2 paradigm for action generation, where a slower System 2 (the VLM) responsible for high-level reasoning operates at a lower frequency (e.g., 5–10 Hz), while a faster System 1 (the action model) reacts to the environment at a higher frequency using the most recent visual inputs Figure AI (2025); Zhang et al. (2024); Song et al. (2025).
The two systems run asynchronously: the action expert conditions its predictions on the VLM’s KV cache, which is updated at a lower frequency by System 2.
While this design is conceptually appealing, we are not aware of a widely adopted, open-source diffusion-style implementation of a dual-system VLA.
Therefore, we make the following approximations in our evaluation:
(1) System 1 latency consists of image upload, vision encoding, diffusion-based action prediction, and action download, where the cost of integrating vision features into the action expert is assumed to be negligible; and
(2) System 2 latency equals VLM inference, which incorporates the visual encoding of the most recently uploaded image.

Takeaway 12: Asynchronous inference between System 1 and System 2 can improve action prediction performance, with performance gains strongly dependent on hardware capability and network latency.

[Table˜9](#table-9) reports performance across different system configurations and System 2 frequency caps (5 Hz and 10 Hz).
On Jetson Thor, the improvement is moderate (1.46$\times$ at a 5 Hz cap and 1.30$\times$ at a 10 Hz cap), since the asynchronous frequency cap is comparable to the synchronous VLM frequency of 19 Hz.
In contrast, on B100 with a fast 10G Ethernet connection, the speedup is substantial (2.24$\times$ at a 10 Hz cap).
In this case, asynchronous execution significantly reduces the effective VLM invocation rate from 301.4 Hz to 10 Hz, freeing compute resources that can be reallocated to action prediction.
However, under slower network conditions (e.g., 5G), the benefit diminishes, yielding only a 1.05$\times$ speedup at a 10 Hz cap.
This is because network latency substantially increases System 1 latency — from 1.5 ms with Ethernet to 26.0 ms with 5G — thereby limiting the achievable performance regardless of the inference hardware capability.

<a id="table-9"></a>

> Table 9: Performance gains by using dual-system inference.

| Hardware    | Network      | S1 Lat. | S2 Lat. | Freq. (Sync) |     | S2 Cap = 5 Hz |              | S2 Cap = 10 Hz |               |              |
| ----------- | ------------ | ------- | ------- | ------------ | --- | ------------- | ------------ | -------------- | ------------- | ------------ |
|             |              |         |         |              |     | Freq. (Async) | Speedup      |                | Freq. (Async) | Speedup      |
| Jetson Thor | On-device    | 32.3 ms | 20.3 ms | 19.0 Hz      |     | 27.8 Hz       | 1.46$\times$ |                | 24.7 Hz       | 1.30$\times$ |
| B100        | Ethernet 10G | 1.5 ms  | 1.9 ms  | 301.4 Hz     |     | 682.4 Hz      | 2.26$\times$ |                | 676.0 Hz      | 2.24$\times$ |
| B100        | WiFi 7       | 6.5 ms  | 1.9 ms  | 119.7 Hz     |     | 152.6 Hz      | 1.28$\times$ |                | 151.2 Hz      | 1.26$\times$ |
| B100        | 5G           | 26.0 ms | 1.9 ms  | 35.9 Hz      |     | 38.2 Hz       | 1.06$\times$ |                | 37.8 Hz       | 1.05$\times$ |

<a id="section-4-11"></a>

### 4.11 Supporting High-Performance VLA Inference up to 100 Hz

In this section, we analyze how 10 Hz and 100 Hz performance targets (§[2](#section-2)) can be achieved with the $\pi_{0}$ model across on-device, edge-server, and cloud-server inference systems.
We also discuss what algorithm-level adjustments may be required when the target performance cannot be met by those systems.

Takeaway 13: For on-device inference, the most advanced edge GPUs (Jetson Thor) can already achieve 10 Hz inference for $\pi_{0}$, but reaching 100 Hz requires model-level adjustments.

[Table˜3](#table-3) shows that Jetson Thor already achieves 19 Hz inference throughput for $\pi_{0}$, exceeding the 10 Hz target.
However, achieving 100 Hz would require roughly a $5\times$ improvement.
This gap have to be closed through model-level optimizations, such as reducing model size (§[4.3](#section-4-3)), decreasing the number of diffusion steps (§[4.5](#section-4-5)), or using lower-precision quantization Kim et al. (2024); Wang et al. (2025a).

Takeaway 14: For edge-server inference, 10 Hz is achievable with consumer GPUs and wireless networks, while 100 Hz requires datacenter GPUs and faster networks.

[Figure˜8](#figure-8) shows that an RTX 4090 can achieve 10 Hz inference even with a slow 4G network.
However, achieving sub-10 ms latency (100 Hz) requires either a more powerful accelerator such as B100 or the aforementioned model-level optimizations.
For B100, reaching 100 Hz further depends on network performance, requiring either wired Ethernet or high-quality wireless connectivity (e.g., WiFi 7).

Takeaway 15: For cloud-server inference, 10 Hz is feasible with good networking, while achieving 100 Hz generally requires asynchronous inference.

As shown in [Table˜8](#table-8), B100 achieves only 42.8 Hz under synchronous cloud inference even with a fast network.
In this regime, network latency alone (exceeding 10 ms per upload or download) prevents achieving 100 Hz, making computation reduction insufficient and rendering asynchronous inference necessary for high-frequency operation.
With a fast network (WiFi 7 or better), asynchronous execution cab achieve a throughput of 314.4 Hz.
Even under poor network conditions where synchronous inference becomes unacceptable (3.7 Hz), asynchronous inference can still restore acceptable performance (50.5 Hz).

<a id="section-5"></a>

## 5 Conclusion and Future Work

We present the first comprehensive study of VLA inference performance.
Using VLA-Perf, an analytical performance modeling tool that we develop, we systematically explore a wide range of (1) model configurations, including model size, context length, architectural choices, and synchronous versus asynchronous execution, and (2) system configurations spanning different hardware platforms, inference placements, and network conditions.
From the performance study, we distill 15 key takeaways that provide practical guidance for the design of future VLA models and inference systems.

While this work represents an important step toward understanding and building next-generation VLA systems, we view it as only a starting point.
First, our study focuses primarily on VLA models for manipulation tasks and does not consider other embodied AI domains such as autonomous driving, quadrupeds, or drones.
These settings often involve different system constraints (e.g., stronger emphasis on on-device execution) and additional model components (e.g., SLAM and specialized control modules), which are beyond the scope of this work.
Second, robotic systems are complex end-to-end pipelines that go beyond model inference alone.
In this work, we do not account for robot execution latency or sensor latency (e.g., cameras), as these factors vary widely across platforms.
A more comprehensive performance analysis that integrates inference, sensing, and actuation would provide deeper insights into end-to-end robotic system behavior.
We leave these directions to future work.

## References

- [1]
  A. Agrawal, N. Kedia, J. Mohan, A. Panwar, N. Kwatra, B. S. Gulavani, R. Ramjee, and A. Tumanov (2024)
  Vidur: a large-scale simulation framework for llm inference.
  Proceedings of Machine Learning and Systems 6, pp. 351–366.
  Cited by: [§3](https://arxiv.org/html/2602.18397v1#S3.p3.1).
- [2]
  A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, D. Driess, et al. (2025)
  $\pi$0.6: a vla that learns from experience.
  arXiv preprint arXiv:2511.14759.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.3](https://arxiv.org/html/2602.18397v1#S2.SS3.p1.1).
- [3]
  A. Bambhaniya, R. Raj, G. Jeong, S. Kundu, S. Srinivasan, S. Subramanian, M. Elavazhagan, M. Kumar, and T. Krishna (2024)
  Demystifying ai platform design for distributed inference of next-generation llm models.
  arXiv preprint arXiv:2406.01698.
  Cited by: [§3](https://arxiv.org/html/2602.18397v1#S3.p3.1).
- [4]
  J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025)
  Gr00t n1: an open foundation model for generalist humanoid robots.
  arXiv preprint arXiv:2503.14734.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p1.1),
  [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1),
  [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p1.3),
  [§4.7](https://arxiv.org/html/2602.18397v1#S4.SS7.p1.1).
- [5]
  K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)
  $\pi$0: A visionlanguage-action flow model for general robot control, 2024a.
  URL https://arxiv. org/abs/2410.24164.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.3](https://arxiv.org/html/2602.18397v1#S2.SS3.p1.1),
  [§4.1](https://arxiv.org/html/2602.18397v1#S4.SS1.p2.4),
  [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p1.3),
  [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p5.1),
  [§4.7](https://arxiv.org/html/2602.18397v1#S4.SS7.p1.1),
  [§4.8](https://arxiv.org/html/2602.18397v1#S4.SS8.p1.1).
- [6]
  K. Black, M. Y. Galliker, and S. Levine (2025)
  Real-time execution of action chunking flow policies.
  arXiv preprint arXiv:2506.07339.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p3.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p3.1).
- [7]
  K. Black, A. Z. Ren, M. Equi, and S. Levine (2025)
  Training-time action conditioning for efficient real-time chunking.
  arXiv preprint arXiv:2512.05964.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p3.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p3.1).
- [8]
  A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. (2022)
  Rt-1: robotics transformer for real-world control at scale.
  arXiv preprint arXiv:2212.06817.
  Cited by: [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.3](https://arxiv.org/html/2602.18397v1#S2.SS3.p1.1).
- [9]
  J. Cho, M. Kim, H. Choi, G. Heo, and J. Park (2024)
  Llmservingsim: a hw/sw co-simulation infrastructure for llm inference serving at scale.
  In 2024 IEEE International Symposium on Workload Characterization (IISWC),
  pp. 15–29.
  Cited by: [§3](https://arxiv.org/html/2602.18397v1#S3.p3.1).
- [10]
  C. C. Christoph, M. Eberlein, F. Katsimalis, A. Roberti, A. Sympetheros, M. R. Vogt, D. Liconti, C. Yang, B. G. Cangan, R. J. Hinchet, et al. (2025)
  ORCA: an open-source, reliable, cost-effective, anthropomorphic robotic hand for uninterrupted dexterous task learning.
  arXiv preprint arXiv:2504.04259.
  Cited by: [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p5.1).
- [11]
  M. Davies, N. Crago, K. Sankaralingam, and C. Kozyrakis (2025)
  LIMINAL: exploring the frontiers of llm decode performance.
  arXiv preprint arXiv:2507.14397.
  Cited by: [§3](https://arxiv.org/html/2602.18397v1#S3.p10.1),
  [§3](https://arxiv.org/html/2602.18397v1#S3.p3.1).
- [12]
  Figure AI (2025)
  Helix: a vision-language-action model for generalist humanoid control.
  Note: [https://www.figure.ai/news/helix](https://www.figure.ai/news/helix)
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p1.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p3.1),
  [§2.3](https://arxiv.org/html/2602.18397v1#S2.SS3.p1.1),
  [§4.10](https://arxiv.org/html/2602.18397v1#S4.SS10.p1.1),
  [§4.7](https://arxiv.org/html/2602.18397v1#S4.SS7.p1.1).
- [13]
  J. Gu, S. Kirmani, P. Wohlhart, Y. Lu, M. G. Arenas, K. Rao, W. Yu, C. Fu, K. Gopalakrishnan, Z. Xu, et al. (2023)
  Rt-trajectory: robotic task generalization via hindsight trajectory sketches.
  arXiv preprint arXiv:2311.01977.
  Cited by: [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1).
- [14]
  Y. Huang, Y. Hao, B. Yu, F. Yan, Y. Yang, F. Min, Y. Han, L. Ma, S. Liu, Q. Liu, et al. (2025)
  Dadu-corki: algorithm-architecture co-design for embodied ai-powered robotic manipulation.
  In Proceedings of the 52nd Annual International Symposium on Computer Architecture,
  pp. 327–343.
  Cited by: [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p4.1),
  [§4.7](https://arxiv.org/html/2602.18397v1#S4.SS7.p1.1).
- [15]
  P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al.
  $\pi$0.5: a vision-language-action model with open-world generalization. arxiv 2025.
  arXiv preprint arXiv:2504.16054.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1).
- [16]
  H. Jang, S. Yu, H. Kwon, H. Jeon, Y. Seo, and J. Shin (2025)
  ContextVLA: vision-language-action model with amortized multi-frame context.
  arXiv preprint arXiv:2510.04246.
  Cited by: [§4.4](https://arxiv.org/html/2602.18397v1#S4.SS4.p1.2).
- [17]
  W. Jiang, S. Subramanian, C. Graves, G. Alonso, A. Yazdanbakhsh, and V. Dadu (2025)
  Rago: systematic performance optimization for retrieval-augmented generation serving.
  In Proceedings of the 52nd Annual International Symposium on Computer Architecture,
  pp. 974–989.
  Cited by: [§3](https://arxiv.org/html/2602.18397v1#S3.p10.1),
  [§3](https://arxiv.org/html/2602.18397v1#S3.p3.1).
- [18]
  D. Jing, G. Wang, J. Liu, W. Tang, Z. Sun, Y. Yao, Z. Wei, Y. Liu, Z. Lu, and M. Ding (2025)
  Mixture of horizons in action chunking.
  arXiv preprint arXiv:2511.19433.
  Cited by: [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p3.1).
- [19]
  M. J. Kim, C. Finn, and P. Liang (2025)
  Fine-tuning vision-language-action models: optimizing speed and success.
  arXiv preprint arXiv:2502.19645.
  Cited by: [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p3.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1),
  [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p1.3).
- [20]
  M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024)
  Openvla: an open-source vision-language-action model.
  arXiv preprint arXiv:2406.09246.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1),
  [§4.11](https://arxiv.org/html/2602.18397v1#S4.SS11.p3.2),
  [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p1.3).
- [21]
  T. Lin, Y. Zhong, Y. Du, J. Zhang, J. Liu, Y. Chen, E. Gu, Z. Liu, H. Cai, Y. Zou, et al. (2025)
  Evo-1: lightweight vision-language-action model with preserved semantic alignment.
  arXiv preprint arXiv:2511.04555.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1).
- [22]
  Y. Ma, Y. Zhou, Y. Yang, T. Wang, and H. Fan (2025)
  Running vlas at real-time speed.
  arXiv preprint arXiv:2510.26742.
  Cited by: [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p4.1),
  [Table 1](#table-1),
  [Table 1](https://arxiv.org/html/2602.18397v1#S3.T1.2.1),
  [§3](https://arxiv.org/html/2602.18397v1#S3.p3.1),
  [§3](https://arxiv.org/html/2602.18397v1#S3.p9.2).
- [23]
  R. K. Mok, H. Zou, R. Yang, T. Koch, E. Katz-Bassett, and K. C. Claffy (2021)
  Measuring the network performance of google cloud platform.
  In Proceedings of the 21st ACM internet measurement conference,
  pp. 54–61.
  Cited by: [§4.7](https://arxiv.org/html/2602.18397v1#S4.SS7.p1.1).
- [24]
  P. Patel, E. Choukse, C. Zhang, A. Shah, Í. Goiri, S. Maleki, and R. Bianchini (2024)
  Splitwise: efficient generative llm inference using phase splitting.
  In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA),
  pp. 118–132.
  Cited by: [§4.2](https://arxiv.org/html/2602.18397v1#S4.SS2.p5.1).
- [25]
  K. Sendai, M. Alvarez, T. Matsushima, Y. Matsuo, and Y. Iwasawa (2025)
  Leave no observation behind: real-time correction for vla action chunks.
  arXiv preprint arXiv:2509.23224.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p3.1).
- [26]
  I. Sfiligoi, J. Graham, and F. Wuerthwein (2020)
  Characterizing network paths in and out of the clouds.
  In EPJ Web of Conferences,
  Vol. 245, pp. 07059.
  Cited by: [§4.7](https://arxiv.org/html/2602.18397v1#S4.SS7.p1.1).
- [27]
  H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang (2025)
  Memoryvla: perceptual-cognitive memory in vision-language-action models for robotic manipulation.
  arXiv preprint arXiv:2508.19236.
  Cited by: [§4.4](https://arxiv.org/html/2602.18397v1#S4.SS4.p1.2).
- [28]
  M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, et al. (2025)
  Smolvla: a vision-language-action model for affordable and efficient robotics.
  arXiv preprint arXiv:2506.01844.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1),
  [§2.3](https://arxiv.org/html/2602.18397v1#S2.SS3.p1.1),
  [§4.8](https://arxiv.org/html/2602.18397v1#S4.SS8.p1.1).
- [29]
  H. Song, D. Qu, Y. Yao, Q. Chen, Q. Lv, Y. Tang, M. Shi, G. Ren, M. Yao, B. Zhao, et al. (2025)
  Hume: introducing system-2 thinking in visual-language-action model.
  arXiv preprint arXiv:2505.21432.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p3.1),
  [§4.10](https://arxiv.org/html/2602.18397v1#S4.SS10.p1.1).
- [30]
  W. Sun, S. Hou, Z. Wang, B. Yu, S. Liu, X. Yang, S. Liang, Y. Gan, and Y. Han (2026)
  Dadu-e: rethinking the role of large language model in robotic computing pipelines.
  Journal of Field Robotics.
  Cited by: [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1).
- [31]
  J. Tang, Y. Sun, Y. Zhao, S. Yang, Y. Lin, Z. Zhang, J. Hou, Y. Lu, Z. Liu, and S. Han (2025)
  VLASH: real-time vlas via future-state-aware asynchronous inference.
  arXiv preprint arXiv:2512.01031.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p3.1).
- [32]
  G. R. Team, S. Abeyruwan, J. Ainslie, J. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, et al. (2025)
  Gemini robotics: bringing ai into the physical world.
  arXiv preprint arXiv:2503.20020.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p1.1).
- [33]
  G. A. Team (2025)
  GEN-0: embodied foundation models that scale with physical interaction.
  Generalist AI Blog.
  Note: https://generalistai.com/blog/preview-uqlxvb-bb.html
  Cited by: [§4.3](https://arxiv.org/html/2602.18397v1#S4.SS3.p1.9).
- [34]
  O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. (2024)
  Octo: an open-source generalist robot policy.
  arXiv preprint arXiv:2405.12213.
  Cited by: [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§4.4](https://arxiv.org/html/2602.18397v1#S4.SS4.p1.2),
  [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p1.3).
- [35]
  H. Wang, C. Xiong, R. Wang, and X. Chen (2025)
  BitVLA: 1-bit vision-language-action models for robotics manipulation.
  arXiv preprint arXiv:2506.07530.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1),
  [§4.11](https://arxiv.org/html/2602.18397v1#S4.SS11.p3.2).
- [36]
  Z. Wang, B. Yu, J. Zhao, W. Sun, S. Hou, S. Liang, X. Hu, Y. Han, and Y. Gan (2025)
  Karma: augmenting embodied ai agents with long-and-short term memory systems.
  In 2025 IEEE International Conference on Robotics and Automation (ICRA),
  pp. 1–8.
  Cited by: [§4.4](https://arxiv.org/html/2602.18397v1#S4.SS4.p1.2).
- [37]
  J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, et al. (2025)
  Tinyvla: towards fast, data-efficient vision-language-action models for robotic manipulation.
  IEEE Robotics and Automation Letters.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1),
  [§2.3](https://arxiv.org/html/2602.18397v1#S2.SS3.p1.1),
  [§4.8](https://arxiv.org/html/2602.18397v1#S4.SS8.p1.1).
- [38]
  R. Wen, J. Zhang, G. Chen, Z. Cui, M. Du, Y. Gou, Z. Han, J. Hu, L. Huang, H. Niu, et al. (2025)
  Dexterous teleoperation of 20-dof bytedexter hand via human motion retargeting.
  arXiv preprint arXiv:2507.03227.
  Cited by: [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p5.1).
- [39]
  Z. Yang, Y. Qi, T. Xie, B. Yu, S. Liu, and M. Li
  DySL-vla: efficient vision-language-action model inference via dynamic-static layer-skipping for robot manipulation.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1).
- [40]
  Z. Yu, B. Wang, P. Zeng, H. Zhang, J. Zhang, L. Gao, J. Song, N. Sebe, and H. T. Shen (2025)
  A survey on efficient vision-language-action models.
  arXiv preprint arXiv:2510.24795.
  Cited by: [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p1.1).
- [41]
  Z. Yuan, Y. Shang, Y. Zhou, Z. Dong, Z. Zhou, C. Xue, B. Wu, Z. Li, Q. Gu, Y. J. Lee, et al. (2024)
  Llm inference unveiled: survey and roofline model insights.
  arXiv preprint arXiv:2402.16363.
  Cited by: [§3](https://arxiv.org/html/2602.18397v1#S3.p3.1).
- [42]
  Y. Yue, Y. Wang, B. Kang, Y. Han, S. Wang, S. Song, J. Feng, and G. Huang (2024)
  Deer-vla: dynamic inference of multimodal large language models for efficient robot execution.
  Advances in Neural Information Processing Systems 37, pp. 56619–56643.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1).
- [43]
  J. Zhang, Y. Guo, X. Chen, Y. Wang, Y. Hu, C. Shi, and J. Chen (2024)
  Hirt: enhancing robotic control with hierarchical robot transformers.
  arXiv preprint arXiv:2410.05273.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p2.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p3.1),
  [§4.10](https://arxiv.org/html/2602.18397v1#S4.SS10.p1.1).
- [44]
  T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023)
  Learning fine-grained bimanual manipulation with low-cost hardware.
  arXiv preprint arXiv:2304.13705.
  Cited by: [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p3.1),
  [§2.2](https://arxiv.org/html/2602.18397v1#S2.SS2.p2.1).
- [45]
  B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023)
  Rt-2: vision-language-action models transfer web knowledge to robotic control.
  In Conference on Robot Learning,
  pp. 2165–2183.
  Cited by: [§1](https://arxiv.org/html/2602.18397v1#S1.p1.1),
  [§2.1](https://arxiv.org/html/2602.18397v1#S2.SS1.p2.1),
  [§2.3](https://arxiv.org/html/2602.18397v1#S2.SS3.p1.1),
  [§4.6](https://arxiv.org/html/2602.18397v1#S4.SS6.p5.1).

<a id="appendix-a"></a>

## Appendix A Detailed System and Model Parameters

In this section, we show the detailed hardware performance configuration used in our evaluation and the model parameters of $\pi_{0}$ in LABEL:{tab:pi0_model_specs}.

<a id="table-10"></a>

> Table 10: Hardware specifications for the GPUs used in our evaluation.

| Hardware    | FP32        | BF16/FP16    | INT8       | Memory | Memory BW |
| ----------- | ----------- | ------------ | ---------- | ------ | --------- |
| Jetson Thor | 100 TFLOP/s | 400 TFLOP/s  | 800 TOP/s  | 128 GB | 270 GB/s  |
| RTX 4090    | 83 TFLOP/s  | 165 TFLOP/s  | 330 TOP/s  | 24 GB  | 1008 GB/s |
| A100        | 20 TFLOP/s  | 312 TFLOP/s  | 624 TOP/s  | 80 GB  | 2039 GB/s |
| H100        | 67 TFLOP/s  | 989 TFLOP/s  | 1979 TOP/s | 80 GB  | 3350 GB/s |
| B100        | 60 TFLOP/s  | 1750 TFLOP/s | 3500 TOP/s | 192 GB | 8000 GB/s |

<a id="table-11"></a>

> Table 11: Parameter specifications for $\pi_{0}$ model components (without vocabulary table).

| Component      | Layers | Hidden Dim | Interm. Dim | Q Heads | KV Heads | Params  |
| -------------- | ------ | ---------- | ----------- | ------- | -------- | ------- |
| Vision Encoder | 27     | 1,152      | 4,304       | 16      | 16       | 411.19M |
| VLM Backbone   | 18     | 2,048      | 16,384      | 8       | 1        | 1.98B   |
| Action Expert  | 18     | 1,024      | 4,096       | 8       | 1        | 292.63M |
