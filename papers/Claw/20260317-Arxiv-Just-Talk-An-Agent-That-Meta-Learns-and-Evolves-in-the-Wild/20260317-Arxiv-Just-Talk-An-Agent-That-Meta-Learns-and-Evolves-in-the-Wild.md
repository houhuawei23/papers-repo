# Title: : Just Talk – An Agent That Meta-Learns and Evolves in the Wild

- ArXiv: 2603.17187
- Authors: Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, Huaxiu Yao
- Sections: 24
- Estimated tokens: 11.6k

## Contents

- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Problem Setup](#2-problem-setup)
- [3 MetaClaw](#3-metaclaw)
  - [3.1 Overview](#31-overview)
  - [3.2 Skill-Driven Fast Adaptation](#32-skill-driven-fast-adaptation)
  - [3.3 Opportunistic Policy Optimization](#33-opportunistic-policy-optimization)
  - [3.4 Skill Generation Versioning](#34-skill-generation-versioning)
  - [3.5 Opportunistic Meta-Learning Scheduler](#35-opportunistic-meta-learning-scheduler)
- [4 Experiments](#4-experiments)
  - [4.1 Experimental Setup](#41-experimental-setup)
    - [4.1.1 Benchmark and Evaluation Platform](#411-benchmark-and-evaluation-platform)
  - [4.2 Main Results](#42-main-results)
  - [4.3 Analysis](#43-analysis)
- [5 Related Work](#5-related-work)
- [6 Conclusion](#6-conclusion)

## Abstract

Large language model (LLM) agents have rapidly emerged as powerful assistants for complex, multi-step tasks, yet agents deployed in the wild remain largely static, trained once and served unchanged regardless of how user needs evolve. This creates a fundamental tension: they must serve users continuously without interruption, yet their capabilities grow stale as the task distribution drifts with real-world usage. On platforms such as OpenClaw, where a single agent connects to 20+ messaging channels and handles diverse, evolving workloads, existing approaches either store raw trajectories without distilling transferable behavioral knowledge, maintain static skill libraries disconnected from weight optimization, or incur service downtime during retraining. We present MetaClaw, a continual meta-learning framework that jointly maintains a base LLM policy and an evolving skill library of reusable behavioral instructions, improving both through two complementary mechanisms. _Skill-driven fast adaptation_ analyzes failure trajectories and synthesizes new skills via an LLM evolver, taking effect immediately with zero service downtime. _Opportunistic policy optimization_ performs gradient-based weight updates via cloud LoRA fine-tuning using RL with a process reward model, triggered only during user-inactive windows by the Opportunistic Meta-Learning Scheduler (OMLS), which monitors configurable sleep hours, system keyboard inactivity, and Google Calendar occupancy. The two mechanisms are mutually reinforcing: a better policy produces more informative failures for skill synthesis, and richer skills yield higher-reward trajectories for policy optimization. To prevent stale reward contamination, a skill generation versioning mechanism strictly separates support data (failure trajectories consumed by skill evolution) from query data (post-adaptation trajectories used for RL updates). Built on a proxy-based architecture, MetaClaw scales to production-size LLMs without a local GPU. Experiments on MetaClaw-Bench (934 questions, 44 simulated workdays) and AutoResearchClaw (23-stage autonomous research pipeline) demonstrate consistent improvements: skill-driven adaptation improves accuracy by up to 32% relative; the full pipeline advances Kimi-K2.5 from 21.4% to 40.6% accuracy (vs. GPT-5.2 baseline 41.1%) with an 8.25$\times$ gain in end-to-end task completion; and skill injection alone improves AutoResearchClaw composite robustness by 18.3%.

[Github][https://github.com/aiming-lab/MetaClaw](https://github.com/aiming-lab/MetaClaw)

<a id="section-1"></a>

## 1 Introduction

<a id="figure-1"></a>

![unc_logo](images/unc_logo.png)

> Figure 1: Overview of . The framework improves the meta-model $\mathcal{M}=(\theta,\mathcal{S})$ via two complementary loops operating at different timescales. Skill-driven fast adaptation (left) analyzes failed trajectories and instantly expands the skill library $\mathcal{S}$ without parameter updates, taking effect immediately for subsequent tasks. Opportunistic policy optimization (right) accumulates post-adaptation trajectories and, once sufficient data is available, leverages idle signals (sleep, inactivity, calendar) detected by the Opportunistic Meta-Learning Scheduler to trigger RL-based weight updates on $\theta$ via Cloud LoRA fine-tuning.

Large language model (LLM) agents have demonstrated remarkable capabilities across complex tasks (yao2022react; shinn2023reflexion), yet agents deployed in the wild remain largely static, trained once and served unchanged regardless of how the user’s needs evolve (zhang2025agentracer; naihin2023testing; song2026agents). Consider OpenClaw (openclaw), an open-source CLI agent platform connecting to 20+ messaging channels, where a single user’s workload may shift from multi-step file system operations one week to multi-agent messaging workflows the next. As the task distribution drifts, a frozen model becomes increasingly misaligned with actual usage patterns, repeatedly failing on task types underrepresented during pretraining.

Existing approaches to agent adaptation fall into three broad categories, each with notable limitations. Memory-based methods (shinn2023reflexion; zhao2024expel; fang2025memp; tang2025agent; ouyang2025reasoningbank; chhikara2025mem0; liu2026simplemem) store raw conversation trajectories for future retrieval, but such trajectories are verbose and redundant, preventing the agent from extracting transferable behavioral patterns. Skill-based methods (xia2026skillrl; zhang2025memevolve; zhang2026memrl; wu2025evolver; zhang2026memskill) compress experience into reusable behavioral instructions, yet treat the resulting skill library as a static database never coordinated with weight optimization. RL-based methods (schulman2017proximal; ahmadian2024back; shao2024deepseekmath; feng2025group; zheng2025group) update model weights, but operate in small-scale or offline settings and ignore a critical data validity problem: once skills have evolved, trajectories collected under the old skill context carry stale rewards that contaminate gradient updates if reused without filtration. A common thread across all three categories is that each addresses only one aspect of adaptation in isolation, leaving the complementary dimensions unexploited.

Our key observation is that two fundamentally different timescales of adaptation are in fact naturally complementary. Behavioral heuristics (e.g., “always verify a file path before reading,” “confirm before destructive commands”) can be distilled within seconds from a single failed conversation and injected immediately as skill instructions. Improving the model’s underlying policy across diverse task types requires gradient-based optimization over many trajectories, on a timescale of minutes to hours. The two mechanisms are also mutually reinforcing: a better policy produces more informative failures for skill synthesis, and richer skills yield higher-reward trajectories for policy optimization. No existing system unifies these two forms of adaptation into a coherent framework that exploits this virtuous cycle.

We present MetaClaw, a continual meta-learning (finn2017model; yao2021meta) framework that jointly maintains a base LLM policy and an evolving skill library of reusable behavioral instructions. The skill library serves a dual role: as a _meta-parameter_ that accumulates behavioral knowledge across the task stream, and as an _adaptation basis_ from which task-specific skills are retrieved at inference time. MetaClaw improves both components through two mechanisms. _Skill-driven fast adaptation_ performs gradient-free skill evolution: an LLM analyzes failure trajectories and synthesizes new behavioral instructions (xia2026skillrl) that take effect immediately with zero service downtime. _Opportunistic policy optimization_ uses RL with a process reward model (PRM) (zhang2025lessons) to update model weights via cloud (tinker) LoRA fine-tuning (hu2021lora), optimizing post-adaptation performance. Two design principles govern their coordination. First, _when_ to run policy optimization: our Opportunistic Meta-Learning Scheduler (OMLS) monitors three idle signals, i.e., configurable sleep hours, system keyboard inactivity, and Google Calendar event occupancy, and triggers weight updates only during user-inactive windows, eliminating downtime. Second, _which data_ to use: we distinguish _support data_ (failure trajectories consumed by skill evolution) from _query data_ (trajectories collected after new skills take effect). Only query data, reflecting the agent’s post-adaptation behavior, is valid for RL; support data carries rewards conditioned on the old skill context and is excluded. Our skill generation versioning mechanism enforces this separation by stamping each trajectory with its skill generation index and flushing stale samples from the training buffer whenever skills evolve.

In summary, our primary contribution is MetaClaw, a continual meta-learning framework that unifies skill-driven fast adaptation with opportunistic policy optimization, enabling deployed LLM agents to evolve continuously through a proxy-based architecture without requiring a local GPU. We evaluate on MetaClaw-Bench, a new benchmark of 934 questions over 44 simulated workdays, where each day forms a sequential, feedback-driven multi-round session of real CLI tasks (file editing, JSON structuring, shell scripting). Experiments with GPT-5.2 and Kimi-K2.5 show that skill-driven fast adaptation alone improves overall accuracy by up to 32.2% in relative terms; MetaClaw (Full) further advances Kimi-K2.5 from 21.4% to 40.6%, improves end-to-end task completion by 8.25$\times$ on Part I and file-check completion by 185% on Part II, and nearly closes the gap with GPT-5.2’s baseline. We further validate on AutoResearchClaw, a 23-stage autonomous research pipeline, where skill injection alone improves the composite robustness score by 18.3%, demonstrating cross-domain generalization of MetaClaw’s adaptation mechanisms.

<a id="section-2"></a>

## 2 Problem Setup

We consider a deployed CLI agent that serves a user over a stream of tasks $\tau_{1},\tau_{2},\ldots$ drawn from a non-stationary distribution $p_{t}(\tau)$. Each task $\tau_{i}$ consists of a user instruction and environmental context (file system state, shell history, etc.), and the agent must produce a sequence of actions $a_{1:T}$ to accomplish the task. The agent’s behavior at any point in time is fully determined by a _meta-model_:

$$
\mathcal{M}=(\theta,\mathcal{S}),(1)
$$

where $\theta$ denotes the parameters of the base LLM policy and $\mathcal{S}=\{s_{1},s_{2},\ldots,s_{K}\}$ is a library of _skill instructions_, i.e., concise, reusable behavioral directives injected into the agent’s system prompt at inference time. Given a task $\tau$, the agent generates actions according to:

$$
a\sim\pi_{\theta}\!\left(\cdot\mid\tau,\;\textsc{Retrieve}(\mathcal{S},\tau)\right),(2)
$$

where $\textsc{Retrieve}(\mathcal{S},\tau)\subseteq\mathcal{S}$ selects the most relevant skills for the current task via embedding-based retrieval.

The meta-model $\mathcal{M}$ evolves over the task stream as the agent accumulates experience. We distinguish two types of trajectory data based on their role in this evolution. _Support data_ $\mathcal{D}^{\text{sup}}$ consists of trajectories whose failures drive adaptation of the skill library $\mathcal{S}$; these trajectories are consumed by the adaptation process and reflect pre-adaptation behavior. _Query data_ $\mathcal{D}^{\text{qry}}$ consists of trajectories collected after adaptation has taken effect; these reflect the agent’s post-adaptation behavior and are used to optimize the policy parameters $\theta$. Maintaining a strict separation between support and query data is essential: mixing them would cause $\theta$ to be optimized against stale reward signals that no longer reflect the agent’s current capabilities.

The goal of MetaClaw is to continuously improve $\mathcal{M}$ over the task stream, not merely to solve each task in isolation, but to _become progressively better at adapting_ to new tasks as they arrive. This positions MetaClaw as a continual meta-learning system: the agent learns from a non-stationary task stream while simultaneously improving its own adaptation capability.

<a id="section-3"></a>

## 3 MetaClaw

- [3.1 Overview](#31-overview)
- [3.2 Skill-Driven Fast Adaptation](#32-skill-driven-fast-adaptation)
- [3.3 Opportunistic Policy Optimization](#33-opportunistic-policy-optimization)
- [3.4 Skill Generation Versioning](#34-skill-generation-versioning)
- [3.5 Opportunistic Meta-Learning Scheduler](#35-opportunistic-meta-learning-scheduler)

<a id="section-3-1"></a>

### 3.1 Overview

MetaClaw improves the meta-model $\mathcal{M}=(\theta,\mathcal{S})$ through two complementary mechanisms operating at different timescales (Figure [1](#figure-1)). _Skill-driven fast adaptation_ analyzes failure trajectories and synthesizes new skill instructions that are immediately injected into the agent’s prompt, evolving $\mathcal{S}$ without touching model weights. _Opportunistic policy optimization_ uses post-adaptation trajectories to update $\theta$ via reinforcement learning, deferred to user-inactive windows by the Opportunistic Meta-Learning Scheduler (OMLS). A skill generation versioning mechanism ensures that policy optimization always trains on query data collected under the current skill library, preventing stale reward contamination from support data. The two mechanisms are mutually reinforcing: a better $\theta$ produces more informative failures for skill synthesis, and richer skills produce higher-reward trajectories for policy optimization. This virtuous cycle enables the system to _learn to become better at adapting_. The complete procedure is summarized in Algorithm [1](#algorithm-1).

<a id="section-3-2"></a>

### 3.2 Skill-Driven Fast Adaptation

Given the current meta-model $(\theta,\mathcal{S}_{g})$, the agent executes tasks and collects trajectories. Trajectories that reveal failure modes form the support set $\mathcal{D}^{\text{sup}}_{g}$. Skill-driven adaptation evolves the skill library via a gradient-free experience distillation process:

$$
\mathcal{S}_{g+1}=\mathcal{S}_{g}\cup\mathcal{E}(\mathcal{S}_{g},\mathcal{D}^{\text{sup}}_{g}),(3)
$$

where $\mathcal{E}$ is a _skill evolver_, an LLM that analyzes failure trajectories and synthesizes new behavioral instructions. The index $g$ denotes the _skill generation_, incremented each time the library changes. This step modifies only $\mathcal{S}$, leaving $\theta$ fixed, and takes effect immediately for all subsequent tasks. Because skill injection operates through the prompt rather than model parameters, fast adaptation incurs zero service downtime.

This mechanism is gradient-free by design, not by approximation. The skill library $\mathcal{S}$ lives in a discrete natural-language space where gradient descent is ill-defined; LLM-based failure analysis is the natural adaptation mechanism for this space.

The skill library $\mathcal{S}$ plays a dual role in the learning structure. As a _meta-parameter_, $\mathcal{S}$ accumulates behavioral knowledge across the entire task stream, with each skill generation $\mathcal{S}_{g+1}\supseteq\mathcal{S}_{g}$ representing the system’s growing operational knowledge. As an _adaptation basis_, $\textsc{Retrieve}(\mathcal{S},\tau)$ extracts a task-specific subset at inference time, providing instant specialization without any parameter update. This dual character arises because natural-language instructions are inherently cross-task transferable: a skill distilled from one failure (e.g., “verify file path before reading”) generalizes to all tasks involving file operations. Unlike systems where task-specific adaptations are ephemeral and discarded after each task, each adaptation episode in MetaClaw contributes lasting knowledge to the meta-model, making knowledge accumulation a feature rather than a side effect.

<a id="section-3-3"></a>

### 3.3 Opportunistic Policy Optimization

After each skill-driven adaptation step, the agent continues serving tasks under the latest skill library. Because policy optimization is deferred to idle windows, the skill library may have advanced through several generations by the time training begins. Let $g^{*}$ denote the current skill generation when a training window opens. The RL buffer $\mathcal{B}$ accumulates query trajectories across all post-adaptation generations, and policy optimization updates $\theta$ over this buffer:

$$
\theta_{t+1}=\theta_{t}+\alpha\nabla_{\theta}\mathbb{E}_{(\tau,\xi,g^{\prime})\sim\mathcal{B}}\!\left[R(\pi_{\theta}(\cdot\mid\tau,\mathcal{S}_{g^{\prime}}))\right],(4)
$$

where $g^{\prime}\leq g^{*}$ is the skill generation under which each trajectory was collected, and $R$ is a process reward model (PRM) score. The versioning mechanism (Section [3.4](#section-3-4)) guarantees that $\mathcal{B}$ contains only query data, i.e., every sample reflects post-adaptation behavior under its respective skill generation. Crucially, policy optimization does not optimize $\theta$ for raw task performance, but for how well the agent performs _after skill adaptation_. A better $\theta$ yields a meta-model from which skill-driven adaptation produces stronger post-adaptation behavior, resulting in an improved meta-model $\mathcal{M}^{\prime}=(\theta_{t+1},\mathcal{S}_{g^{*}})$.

In practice, policy optimization is realized via cloud LoRA fine-tuning using GRPO, deferred to idle windows by the Opportunistic Meta-Learning Scheduler (Section [3.5](#section-3-5)). Importantly, training is initiated only after the query buffer $\mathcal{B}$ has accumulated a sufficient number of trajectories; launching RL with too few samples leads to high-variance gradient estimates and unstable policy updates. This means policy optimization naturally lags behind skill-driven adaptation by days or longer, reinforcing the asymmetry between the two timescales: skills evolve continuously, while the policy improves in discrete, data-gated steps.

<a id="section-3-4"></a>

### 3.4 Skill Generation Versioning

The support-query separation defined in Section [2](#section-2) must be enforced in MetaClaw’s online setting, where tasks arrive sequentially and skill evolution is triggered asynchronously. Without a dedicated mechanism, support data can leak into the policy optimization buffer.

The problem is concrete: a trajectory $(\tau_{i},\xi_{i})$ that triggers skill evolution from $\mathcal{S}_{g}$ to $\mathcal{S}_{g+1}$ carries a reward $r_{i}$ reflecting performance under $\mathcal{S}_{g}$, _before_ the new skill existed. If this trajectory enters the RL buffer, policy optimization receives a gradient that penalizes $\theta$ for a failure that skill-driven adaptation has already corrected, optimizing for pre-adaptation rather than post-adaptation performance and violating the meta-learning objective in Eq. [4](#S3.E4).

We enforce separation via a _skill generation version_ $g_{i}$ stamped on each collected sample:

- Support set $\mathcal{D}^{\text{sup}}_{g}$: trajectories collected under $\mathcal{S}_{g}$ whose failures trigger skill evolution $\mathcal{S}_{g}\to\mathcal{S}_{g+1}$. These are consumed by the skill evolver and _discarded from the RL buffer_.
- Query set $\mathcal{D}^{\text{qry}}_{g+1}$: trajectories collected after $\mathcal{S}_{g+1}$ takes effect. Only these, reflecting the agent’s post-adaptation behavior, are eligible for policy optimization gradient updates.

When the skill generation counter advances from $g$ to $g+1$, the trainer flushes all samples with version $\leq g$ from its buffer. This ensures policy optimization always updates $\theta$ with respect to the agent’s adapted behavior, preserving the integrity of the meta-learning structure.

<a id="section-3-5"></a>

### 3.5 Opportunistic Meta-Learning Scheduler

Policy optimization requires a model weight hot-swap upon completion, which briefly interrupts inference. In a deployed interactive system, this creates a tension: policy optimization must run periodically to improve $\theta$, but it must not degrade the user’s experience.

We introduce the _Opportunistic Meta-Learning Scheduler_ (OMLS), a background daemon that defers policy optimization to periods when the user is not actively interacting with the agent. OMLS monitors three complementary idle signals:

(1) Sleep window. The user configures a sleep schedule (e.g., 23:00–07:00). During this window, the system is guaranteed to be idle, providing the largest contiguous training block.

(2) System inactivity. OMLS polls the operating system’s input device idle timer (e.g., ioreg HIDIdleTime on macOS). If no keyboard or mouse activity is detected for $\delta$ minutes (default: 30), a training window opens. Upon renewed input, the trainer pauses gracefully via mid-batch checkpointing.

(3) Calendar-aware scheduling. OMLS queries the user’s Google Calendar API. When the current time falls within a scheduled meeting, the user is presumed unavailable, opening an opportunistic training window. This is the most anticipatory of the three signals: it leverages the user’s own schedule to predict idle periods proactively.

A training window opens when _any_ signal indicates user absence and closes when _any_ signal indicates the user has returned. The RL trainer supports pause/resume across fragmented idle windows, accumulating gradient steps opportunistically without requiring a single long contiguous block.

<a id="algorithm-1"></a>

**Algorithm 1 MetaClaw: Continual Meta-Learning for Deployed LLM Agents**

```text

0:  Meta-model $\mathcal{M}=(\theta_{0},\mathcal{S}_{0})$, skill evolver $\mathcal{E}$, task stream $\{\tau_{i}\}$, PRM $R$, OMLS idle detector

0:  Continuously improved meta-model $\mathcal{M}$

1:  Initialize skill generation $g\leftarrow 0$, RL buffer $\mathcal{B}\leftarrow\varnothing$

2:  for each task $\tau_{i}$ in stream do

3:   $\triangleright$ Serve task with current meta-model

4:   $\mathcal{S}_{\tau_{i}}\leftarrow\textsc{Retrieve}(\mathcal{S}_{g},\tau_{i})$ // retrieve relevant skills

5:   $\xi_{i}\leftarrow\textsc{Execute}(\pi_{\theta}(\cdot\mid\tau_{i},\mathcal{S}_{\tau_{i}}))$ // collect trajectory

6:   $r_{i}\leftarrow R(\xi_{i})$; stamp $(\tau_{i},\xi_{i},r_{i})$ with generation $g$

7:   if $\xi_{i}$ reveals failure then

8:    Add $(\tau_{i},\xi_{i})$ to support set $\mathcal{D}^{\text{sup}}_{g}$

9:   else

10:    Add $(\tau_{i},\xi_{i},r_{i},g)$ to RL buffer $\mathcal{B}$

11:   end if

12:   $\triangleright$ Skill-driven fast adaptation (when failures accumulate)

13:   if $|\mathcal{D}^{\text{sup}}_{g}|\geq$ threshold then

14:    $\Delta\mathcal{S}\leftarrow\mathcal{E}(\mathcal{S}_{g},\mathcal{D}^{\text{sup}}_{g})$ // synthesize new skills from failures

15:    $\mathcal{S}_{g+1}\leftarrow\mathcal{S}_{g}\cup\Delta\mathcal{S}$ // evolve skill library

16:    Flush all samples with version $\leq g$ from $\mathcal{B}$ // support-query separation

17:    $g\leftarrow g+1$

18:   end if

19:   $\triangleright$ Opportunistic policy optimization (when user is idle)

20:   if OMLS detects idle window  and $|\mathcal{B}|\geq$ batch size then

21:    $\theta\leftarrow\theta+\alpha\nabla_{\theta}\mathbb{E}_{(\tau,\xi,r,g^{\prime})\sim\mathcal{B}}[R(\pi_{\theta}(\cdot\mid\tau,\mathcal{S}_{g^{\prime}}))]$ // RL update

22:    Hot-swap model weights // deploy updated $\theta$

23:   end if

24:  end for
```

<a id="section-4"></a>

## 4 Experiments

- [4.1 Experimental Setup](#41-experimental-setup)
- [4.2 Main Results](#42-main-results)
- [4.3 Analysis](#43-analysis)

<a id="section-4-1"></a>

### 4.1 Experimental Setup

- [4.1.1 Benchmark and Evaluation Platform](#411-benchmark-and-evaluation-platform)

#### 4.1.1 Benchmark and Evaluation Platform

MetaClaw-Bench. We construct MetaClaw-Bench, a continual agentic benchmark comprising two complementary evaluation parts (934 questions total across 44 simulated workdays) for evaluating an agent’s ability to adapt across a sequential stream of real-world CLI tasks. Existing agent benchmarks present tasks as independent episodes, providing no mechanism to assess whether an agent improves from accumulated experience. MetaClaw-Bench addresses this gap by structuring evaluation as multi-workday simulations in which the agent operates under consistent workspace and policy rulesets that evolve through user feedback.

1. Part I structures evaluation as a 30-workday simulation (346 questions, days 01–30, 10–15 per day). The workspace state (files, configs, project records) persists across rounds within each day, and each question includes the evaluation outcome of the previous round as corrective feedback context. Questions fall into two types: _file-check_ tasks (structured edits or transformations producing output files validated by automated checkers) and _multi-choice_ tasks (conceptual procedural questions on domain-specific rules). Task difficulty increases monotonically with day index, with days 25–30 requiring sophisticated multi-step reasoning. Part I’s file-check tasks are heavily execution-oriented, with many interdependent side effects, providing a conservative measure of end-to-end completion.

2. Part II extends the evaluation to a 14-workday simulation (588 questions, 42 per day: 434 multi-choice and 154 file-check). Part II’s file-check tasks are rule-based transformations where compliance with behavioral heuristics (e.g., schema conventions, timestamp formats) is the primary bottleneck, making them more amenable to skill distillation. This design provides a complementary signal: while Part I stress-tests execution reliability, Part II directly measures how quickly the RL-trained policy internalizes procedural rules across a higher-density task stream.

We report two primary metrics across both parts: overall accuracy (mean per-question score) and file-check completion rate (fraction of file-check outputs passing all automated checker assertions simultaneously). Because the benchmark tasks are authored to simulate realistic deployment rather than collected from actual user sessions, we view both parts as controlled stress tests of continual adaptation under increasing difficulty.

Downstream evaluation: AutoResearchClaw. To test whether MetaClaw’s adaptation mechanisms generalize beyond CLI-task benchmarks, we additionally evaluate on AutoResearchClaw (liu2026autoresearchclaw), a fully autonomous 23-stage research pipeline that transforms a single research idea into a conference-ready paper, covering literature search, hypothesis generation, experiment design, code synthesis, sandbox execution, result analysis, paper drafting, and multi-agent peer review. Unlike MetaClaw-Bench’s structured file-check and multi-choice tasks, AutoResearchClaw presents an open-ended, long-horizon agentic workload where failures manifest as stage retries, excessive refinement cycles, and incomplete pipeline runs. We report four pipeline-level metrics: _stage retry rate_, _refine cycle count_, _pipeline stage completion_ (out of 19 scorable stages), and a _composite robustness score_ (weighted average of stage completion rate at 40%, retry reduction at 30%, and refine cycle efficiency at 30%).

Baselines and Implementation Details. We evaluate two frontier LLMs as backbone policies: GPT-5.2 (openai2025gpt52) and Kimi-K2.5 (team2026kimi). We compare three conditions: 1) Baseline: the base model served without any adaptation mechanism. 2) MetaClaw (Skills): the base model augmented with skill-driven fast adaptation; after each failed trajectory, the skill evolver synthesizes behavioral instructions immediately injected into the system prompt, with top-$k$ retrieval via cosine similarity over sentence embeddings. 3) MetaClaw (Full): the full pipeline combining skill-driven fast adaptation with opportunistic policy optimization via RL (5-day training run), evaluated for Kimi-K2.5 only, as it requires a cloud LoRA training endpoint configured for the target backbone. All conditions use identical prompts and tool sets. This design isolates the individual contributions of the two MetaClaw components as defined in Section [3](#section-3).

For the AutoResearchClaw evaluation, we deploy MetaClaw’s _skill-driven fast adaptation_ within AutoResearchClaw’s pipeline executor. After each pipeline run, failures and warnings from all 23 stages are captured as structured lessons and converted into reusable skill files via MetaClaw’s lesson-to-skill evolver. On subsequent runs, accumulated skills are injected into the system prompt of all 18 LLM-driven stages. We run controlled A/B experiments with the same research topic, backbone LLM, and pipeline configuration, differing only in whether MetaClaw’s skill injection is active.

<a id="section-4-2"></a>

### 4.2 Main Results

Table [1](#table-1) reports performance on both parts of MetaClaw-Bench for all five model–condition pairs. MetaClaw consistently improves over the respective baselines across both models, both adaptation modes, and both benchmark parts.

<a id="table-1"></a>

> Table 1: Main results on MetaClaw-Bench Parts I and II. Acc.: mean per-question accuracy. Compl.: file-check completion rate. MetaClaw (Full) is evaluated for Kimi-K2.5 only. Best result per model per part is bolded.

|           |                   | Part I (30 days, 346 Q) | Part II (14 days, 588 Q) |          |            |
| --------- | ----------------- | ----------------------- | ------------------------ | -------- | ---------- |
| Model     | Condition         | Acc. (%)                | Compl. (%)               | Acc. (%) | Compl. (%) |
| GPT-5.2   | Baseline          | 41.1                    | 14.7                     | 44.9     | 58.4       |
| GPT-5.2   | MetaClaw (Skills) | 44.0                    | 17.1                     | 49.1     | 67.5       |
| Kimi-K2.5 | Baseline          | 21.4                    | 2.0                      | 21.1     | 18.2       |
| Kimi-K2.5 | MetaClaw (Skills) | 28.3                    | 2.0                      | 26.9     | 33.8       |
| Kimi-K2.5 | MetaClaw (Full)   | 40.6                    | 16.5                     | 39.6     | 51.9       |

MetaClaw improves both models and the full pipeline yields the largest gains. Results are consistent across both benchmark parts. For GPT-5.2, MetaClaw (Skills) raises overall accuracy from 41.1% to 44.0% on Part I (+7.1% relative) and from 44.9% to 49.1% on Part II (+9.4% relative), with file-check completion rising from 14.7% to 17.1% on Part I and from 58.4% to 67.5% on Part II. For Kimi-K2.5, MetaClaw (Skills) improves accuracy from 21.4% to 28.3% on Part I (+32.2%) and from 21.1% to 26.9% on Part II (+27.5%). MetaClaw (Full) yields substantially larger gains: on Part I, accuracy reaches 40.6% and task completion rises 8.25$\times$ (from 2.0% to 16.5%); on Part II, accuracy reaches 39.6% and file-check completion jumps from 18.2% to 51.9% (+185% relative).

Stronger models benefit less and weaker models benefit more. GPT-5.2 starts from a higher baseline (41.1% vs. 21.4% on Part I), leaving less headroom for skill-driven gains. Kimi-K2.5, by contrast, lacks implicit procedural knowledge that the skill library provides explicitly, so skill injection yields larger returns. Notably, MetaClaw (Full) with Kimi-K2.5 (40.6%) nearly closes the gap with GPT-5.2’s baseline (41.1%), demonstrating that the combination of skill injection and gradient-based policy optimization can largely compensate for model capability differences. This pattern suggests MetaClaw is particularly valuable for deploying capable but not state-of-the-art models at production scale.

The full pipeline unlocks end-to-end task completion and skills alone do not. On Part I, MetaClaw (Skills) leaves task completion rates unchanged for both models, confirming that skill injection sharpens partial execution quality without reliably enabling zero-defect outputs under heavy execution demands. MetaClaw (Full) closes this gap: Kimi-K2.5’s completion rate jumps from 2.0% to 16.5% (8.25$\times$). On Part II, where file-check tasks are rule-based, skills already drive a substantial completion gain (18.2%→33.8%), and the full pipeline pushes this further to 51.9%, confirming that weight-level optimization provides an additive benefit on top of skill injection regardless of task type.

Since MetaClaw-Bench is an authored simulation rather than a collection of real user sessions (see Section [4](#section-4)), the absolute magnitudes of these gains are specific to this benchmark and may not transfer directly to production workloads. The primary value of these results lies in the consistent directional trends: skill-driven adaptation reliably improves partial execution quality across both models, while weight-level optimization is necessary to unlock end-to-end task completion.

MetaClaw generalizes to open-ended multi-stage pipelines. Table [2](#table-2) reports MetaClaw’s impact on AutoResearchClaw, an evaluation setting structurally different from MetaClaw-Bench. Using skills-only adaptation (no RL), MetaClaw reduces the stage retry rate by 24.8% (from 10.5% to 7.9%) and cuts refine cycles by 40.0% (from 2.0 to 1.2 per stage). Pipeline completion improves from 18/19 to 19/19 stages (+5.3%), and the composite robustness score rises from 0.714 to 0.845, an 18.3% improvement. These gains are achieved _without any gradient-based policy updates_, demonstrating that MetaClaw’s lightweight, zero-downtime skill injection transfers effectively to complex, long-horizon agentic workflows beyond structured CLI tasks.

<a id="table-2"></a>

> Table 2: MetaClaw (Skills-Only) on AutoResearchClaw, a 23-stage autonomous research pipeline. Skill injection alone yields consistent improvements across all robustness metrics without requiring RL weight updates.

| Metric                                  | Baseline | + MetaClaw (Skills) | Relative Change    |
| --------------------------------------- | -------- | ------------------- | ------------------ |
| Stage retry rate ($\downarrow$)         | 10.5%    | 7.9%                | $\downarrow$ 24.8% |
| Refine cycle count ($\downarrow$)       | 2.0      | 1.2                 | $\downarrow$ 40.0% |
| Pipeline stage completion ($\uparrow$)  | 18 / 19  | 19 / 19             | $\uparrow$ 5.3%    |
| Composite robustness score ($\uparrow$) | 0.714    | 0.845               | $\uparrow$ 18.3%   |

<a id="section-4-3"></a>

### 4.3 Analysis

Per-day accuracy trends. Figure [3](#figure-3) visualizes per-day accuracy (3-day rolling average) for all five conditions. Both models and all conditions show a consistent accuracy decline from day01–10 (where accuracies routinely exceed 50%) to day25–30 (where most models fall below 30%), confirming that MetaClaw-Bench exhibits increasing difficulty. MetaClaw’s advantage over the baseline is most pronounced in the _mid-range_ days (day11–22), where tasks require multi-step procedural compliance that is learnable through failure distillation, and MetaClaw (Full) reaches its peak advantage of nearly 0.8 accuracy around day 19–20. The _early_ days (day01–10) involve simpler manipulations where both conditions perform reasonably, and the _late_ days (day23–30) are sufficiently complex that accumulated skills are insufficient without stronger model weights, leading all five conditions to converge toward similarly low performance.

Task-type breakdown. Figure [3](#figure-3) decomposes performance by task type, revealing that the two MetaClaw components address fundamentally different bottlenecks. Skills-only adaptation lifts multi-choice pass rates for both models while leaving file-check completion flat, as procedural knowledge helps reasoning but not execution. MetaClaw (Full) reverses this: Kimi-K2.5’s file-check completion rate jumps to match GPT-5.2’s baseline, while multi-choice accuracy slightly decreases as the policy shifts toward file-execution behavior during training.

RL training dynamics. Part II provides a fine-grained view of how policy optimization evolves over time. The file-check completion curve for MetaClaw (Full)–Kimi-K2.5 shows a clear inflection at day 8, after which per-day pass rates escalate rapidly: from $\sim$9% on days 1–4, through 27–36% on days 5–8, to 55–64% on days 9–10, and ultimately reaching 100% on days 12 and 14. This learning trajectory mirrors the MAML inner-loop update structure: the first several days accumulate support trajectories for skill synthesis and weight updates, the inflection marks when sufficient gradient signal has been collected for the LoRA fine-tune to shift the policy’s execution strategy, and the late-phase convergence indicates that the policy has internalized the procedural rules surfaced by the skill library. The two-phase pattern (skill-driven gains first, RL-driven gains after day 8) directly validates the complementary timescale hypothesis underlying MetaClaw’s design.

Skill library analysis. Across the 30-day session, MetaClaw’s skill evolver synthesizes skills clustered around three recurring failure categories: (1) _temporal format compliance_, normalizing natural-language time expressions to ISO 8601 format with timezone offsets; (2) _backup-before-modify protocol_, creating .bak files before any destructive file operation; and (3) _naming convention adherence_, following date-prefixed file naming patterns (e.g., 20260408\_\*.json). These cross-cutting behavioral heuristics generalize across tasks, explaining why a single failure can yield skills that improve performance on subsequent, structurally different questions.

Cross-domain skill transfer to AutoResearchClaw. The AutoResearchClaw results (Table [2](#table-2)) provide complementary evidence for skill generalization. In this setting, the skill evolver, designed for CLI-task adaptation, synthesizes actionable skills for a fundamentally different workload (multi-stage research automation) without any domain-specific tuning. The 40% reduction in refine cycles indicates that skills distilled from earlier pipeline failures (e.g., citation formatting errors, experiment code validation failures) directly prevent repeated mistakes in subsequent runs. This cross-domain transferability, combined with the zero-downtime deployment model (skill injection operates entirely at the prompt level), confirms that MetaClaw functions as a general-purpose continual learning layer applicable to diverse agentic systems.

<a id="figure-2"></a>

![unc_logo](images/unc_logo.png)

> Figure 2: Per-day accuracy over 30 simulated workdays (3-day rolling average). Solid lines: GPT-5.2; dashed lines: Kimi-K2.5. MetaClaw (Full) dominates in the mid phase (day 11–22) before difficulty outpaces accumulated knowledge in late days.

Case studies. Table [3](#table-3) contrasts two representative cases that illustrate the distinct contributions of the two MetaClaw mechanisms. In Case 1, a single distilled skill resolves a compliance error with zero weight update. In Case 2, skill injection provides necessary format context but is insufficient alone; weight-level RL updates are required to reliably execute a structurally complex file operation.

<a id="table-3"></a>

> Table 3: Representative case studies. Case 1 shows skill-driven fast adaptation (MetaClaw Skills, GPT-5.2); Case 2 shows the full pipeline (MetaClaw (Full), Kimi-K2.5). Both recover from score 0 to score 1.0; the mechanisms differ fundamentally.

|                                | Case 1 MetaClaw (Skills)                                                                                                                                         | Case 2 MetaClaw (Full)                                                                                                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model                          | GPT-5.2                                                                                                                                                          | Kimi-K2.5                                                                                                                                                                           |
| Day / Round                    | Day 19 / Round 4                                                                                                                                                 | Day 18 / Round 6                                                                                                                                                                    |
| Task type                      | File-check                                                                                                                                                       | File-check                                                                                                                                                                          |
| Task instruction               |                                                                                                                                                                  |                                                                                                                                                                                     |
|                                | Update sprint8_board.json: set T-405/T-406 to "done", add completed_at fields.                                                                                   | Append a deployment record to deploy_log.json with fields timestamp (ISO 8601+TZ), env, status, and changes.                                                                        |
| Baseline response (score: 0)   |                                                                                                                                                                  |                                                                                                                                                                                     |
|                                | Reads file; directly overwrites it. Checker detects missing sprint8_board.json.bak $\to$ 0.                                                                      | Uses field name date instead of timestamp; omits changes. Checker rejects schema $\to$ 0.                                                                                           |
| MetaClaw response (score: 1.0) |                                                                                                                                                                  |                                                                                                                                                                                     |
|                                | Skill distilled from Day 2: _“Always create .bak before modifying (P4).”_ Agent writes sprint8_board.json.bak, applies targeted patch. Checker passes $\to$ 1.0. | Skills inject _“use ISO 8601 with timezone offset”_; skills-only Kimi still omits changes array $\to$ 0. After RL: all four fields present, schema valid, backup created $\to$ 1.0. |
| Day accuracy (all rounds)      |                                                                                                                                                                  |                                                                                                                                                                                     |
|                                | Baseline: 43.9% MetaClaw (Skills): 62.1% $\Delta$ +18.2 pp                                                                                                       | Baseline: 8.3% Skills-only: 25.0% MetaClaw (Full): 80.6%                                                                                                                            |
| Key mechanism                  |                                                                                                                                                                  |                                                                                                                                                                                     |
|                                | Skills: one distilled rule generalizes across file types and subsequent days with zero weight update.                                                            | RL: skills supply declarative format context; weight updates internalize the execution reliability that skill injection alone cannot enforce.                                       |

<a id="section-5"></a>

## 5 Related Work

Skill-based and memory-augmented agents. A line of work augments LLM agents with external memory or reusable skill libraries to improve performance without modifying model weights (shinn2023reflexion; zhao2024expel; fang2025memp; tang2025agent; ouyang2025reasoningbank; chhikara2025mem0; liu2026simplemem; wang2024agent). Reflexion (shinn2023reflexion) stores verbal self-reflections in an episodic buffer, allowing the agent to avoid repeating past mistakes. Mem0 (chhikara2025mem0) and SimpleMem (liu2026simplemem) maintain longer-horizon memory through hierarchical retrieval. On the skill side, Voyager (wangvoyager) incrementally builds a library of executable code skills from successful episodes, while ExpeL (zhao2024expel) and Agent-KB (tang2025agent) distills cross-task experience into natural-language rules. A key limitation shared by these methods is that the skill library (or memory) is treated as a static artifact (xia2026skillrl): it is never coordinated with weight-level optimization, and successful trajectories are reused indiscriminately without regard for whether the agent’s behavior has changed since they were collected. MetaClaw addresses both gaps by coupling skill evolution with policy optimization through explicit support-query separation.

Reinforcement learning for LLM agents. RLHF (ouyang2022training) and its variants establish the use of reward signals to fine-tune LLM behavior, and subsequent work applies RL to tool-using and agentic settings (nakano2021webgpt; yao2022react). More recently, GRPO (shao2024deepseekmath) and DAPO (yu2025dapo) demonstrate stable online policy gradient training for reasoning tasks (schulman2017proximal; ahmadian2024back; shao2024deepseekmath; feng2025group; zheng2025group; team2025tongyi; dong2025agentic). However, these approaches optimize a fixed policy against a fixed reward signal, with no mechanism for the agent to update its behavioral context between rollouts. In deployed interactive settings, they also do not address _when_ to run training or _which_ data remains valid for gradient updates after behavioral changes. MetaClaw targets exactly these practical constraints via opportunistic scheduling and skill generation versioning.

Continual and meta-learning. Meta-learning (finn2017model; nichol2018first; hospedales2021meta) frames learning as optimizing for fast adaptation to new tasks, typically in an offline episode-based setting. Meta-reinforcement learning extends this idea to sequential decision-making: RL^2 (duan2016rl2) trains a recurrent policy whose hidden state implicitly encodes task context, PEARL (rakelly2019efficient) infers a probabilistic context variable for off-policy adaptation, and ProMP (rothfuss2019promp) applies trust-region constraints at the meta-level. These methods demonstrate effective fast adaptation in robotic control and navigation, but operate on simple network architectures with low-dimensional action spaces and assume fixed offline task distributions. Continual learning (kirkpatrick2017ewc; lopez2017gem; chaudhry2019agem; zenke2017continual; wang2024comprehensive; wang2022learning) studies sequential task adaptation without forgetting through regularization, replay, or architectural strategies, yet does not incorporate fast adaptation mechanisms at inference time. Online meta-learning approaches (finn2019online; nagabandi2018deep; harrison2020continuous; yao2020online) relax the offline assumption and even handle task heterogeneity, but remain grounded in representation learning over simple networks. MetaClaw extends the meta-learning objective to a non-stationary stream of LLM agent tasks where fast adaptation is gradient-free (skill synthesis in discrete natural-language space) and slow adaptation is gradient-based (policy optimization via RL), with a versioning protocol that preserves the support-query structure in an online, asynchronous setting.

<a id="section-6"></a>

## 6 Conclusion

We presented MetaClaw, a continual meta-learning framework that enables deployed LLM agents to improve autonomously through normal usage. MetaClaw combines two complementary adaptation mechanisms operating at different timescales: fast, inference-time skill injection that distills reusable behavioral knowledge from failures, and slow, gradient-based policy optimization that refines the model during idle windows. Built on a lightweight proxy architecture, the system requires no local GPUs and integrates transparently with existing personal agents and LLM providers. Experiments on MetaClaw-Bench demonstrate consistent improvements across models and adaptation modes, with the full pipeline yielding the largest gains on both partial execution quality and end-to-end task completion. Evaluation on AutoResearchClaw further shows that skill injection generalizes to open-ended research pipelines without any gradient updates. A current limitation is that idle-window detection depends on user configuration, which may not generalize to all deployment environments. We believe MetaClaw establishes a principled foundation for agents that genuinely learn and evolve in the wild, simply by being used.
