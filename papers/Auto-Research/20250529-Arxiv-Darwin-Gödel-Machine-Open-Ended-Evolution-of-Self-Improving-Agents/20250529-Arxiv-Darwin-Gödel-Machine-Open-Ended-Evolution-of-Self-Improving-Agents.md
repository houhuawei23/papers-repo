# Title: Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents

- ArXiv: 2505.22954
- Authors: Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune, University of British Columbia, Vector Institute, Sakana AI, Canada CIFAR AI Chair, ,
- Sections: 42
- Estimated tokens: 36.6k

## Contents

- 1 Introduction
- 2 Related Work
- 3 Darwin Gödel Machine
- 4 Experiments
  - 4.1 Experiment Setup
  - 4.2 Benchmarks
  - 4.3 Baselines
  - 4.4 Results
- 5 Safety Discussion
- 6 Conclusion and Limitations
- Ethics Statement
- Reproducibility Statement
  - Acknowledgments
- References
- Appendix
- Table of Contents
- Appendix A Additional Results
  - A.1 Baselines on SWE-bench
  - A.2 Generality across models on Polyglot
  - A.3 Ablation of Parent Selection
  - A.4 Additional Statistics of DGM runs
- Appendix B Additional Related Work
- Appendix C Algorithmic Details
  - C.1 Initial Coding Agent
  - C.2 Parent Selection
  - C.3 Self-Improve Prompts
  - C.4 Pseudocode
- Appendix D Experiment Details
  - D.1 Hyperparameters for Foundation Models
- Appendix E Benchmark Details
  - E.1 Cost Estimate
  - E.2 SWE-bench Tasks
  - E.3 Polyglot Tasks
  - E.4 SWE-bench State-of-The-Art
  - E.5 Polyglot Representative Agent
- Appendix F Best-Discovered Agents
  - F.1 DGM on SWE-bench
  - F.2 DGM on Polyglot
- Appendix G Similar Target Functionality, Different Implementations
- Appendix H Case Study: Solving Hallucination
- Appendix I Additional Safety Discussion
- Appendix J Additional Future Work Directions

## Abstract

###### Abstract

Most of today’s AI systems are constrained by human-designed, fixed architectures and cannot autonomously and continuously improve themselves. The scientific method, on the other hand, is a cumulative and open-ended system, where each innovation builds upon previous artifacts, enabling future discoveries. There is growing hope that the current manual process of advancing AI could itself be automated. If done safely, such automation would accelerate AI development and allow us to reap its benefits much sooner. This prospect raises the question of how AI systems can endlessly improve themselves while getting better at solving relevant problems. Meta-learning can automate the discovery of novel algorithms, but is limited by first-order improvements and the human design of a suitable search space. The Gödel machine (Schmidhuber, 2007) proposed a theoretical alternative: a self-improving AI that repeatedly modifies itself in a provably beneficial manner. Unfortunately, proving that most changes are net beneficial is impossible in practice. We introduce the Darwin Gödel Machine (DGM), a novel self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks. Inspired by Darwinian evolution and open-endedness research, the DGM grows an archive of generated coding agents. It samples agents from this archive, which self-modify to create new, interesting versions of themselves. This open-ended exploration forms a growing tree of diverse, high-quality agents and allows the parallel exploration of many different paths through the search space. Empirically, the DGM automatically improves its coding capabilities (e.g., better code editing tools, long-context window management, peer-review mechanisms), increasing performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety precautions (e.g., sandboxing, human oversight). Overall, the DGM represents a significant step toward self-improving AI, capable of gathering its own stepping stones along a path that unfolds into endless innovation.
All code is open-sourced at [https://github.com/jennyzzt/dgm](https://github.com/jennyzzt/dgm).

<a id="section-1"></a>

## 1 Introduction

Scientific progress is cumulative and open-ended, with each breakthrough standing on the shoulders of countless prior insights. In the same way, our most advanced AI systems are built upon a long lineage of innovations. For instance, transformers (Vaswani et al., 2017), the backbone of current large language models (LLMs) (Brown et al., 2020), did not emerge in isolation but were built upon years of past innovations, such as recurrent neural networks (Linnainmaa, 1970; Amari, 1972; Hopfield, 1982; Rumelhart et al., 1985) and attention mechanisms (Schmidhuber and Huber, 1990; Bahdanau et al., 2015; Kim et al., 2017; Parikh et al., 2016). However, most of today’s AI systems remain bound by fixed, human-designed architectures that learn within predefined boundaries, without the capacity to autonomously rewrite their own source code to self-improve. As a result, each advancement in AI development still leans heavily on human interventions, tethering the pace of progress. This paper investigates the intriguing possibility of safely automating the search for ever-better AI. One can imagine an AI system that, like scientific discovery itself, becomes an engine of its own advancement: building upon its past, recursively improving, and propelling itself toward more advanced capabilities.

<a id="figure-1"></a>

![conceptual](images/conceptual.png)

> Figure 1: Darwin Gödel Machine. The DGM iteratively builds a growing archive of agents by interleaving self-modification with downstream task evaluation. Agents in the archive are selected for self-modification through open-ended exploration.

Schmidhuber (2007) presented a class of mathematically rigorous, self-referential, self-improving problem solvers. It relies on formal proofs to justify code rewrites, ensuring that any self-modification is provably beneficial. However, in practice and without restrictive assumptions about the system, it is impossible to formally prove whether a modification to an AI system will be beneficial. For example, while it may seem that an LLM-based coding agent would benefit from access to more tools (e.g., code search, test runners), the actual impact depends heavily on the model’s training and task context (e.g., a testing tool that is optimized for one setup may confuse the agent when working with others). Instead of requiring formal proofs, we empirically validate self-modifications against a benchmark, allowing the system to improve and explore based on observed results. This approach mirrors biological evolution, where mutations and adaptations are not verified in advance but are produced, trialed, and then selected via natural selection. We also take inspiration from Darwinian evolution (Darwin, 2023) and investigate the effectiveness of maintaining a library of previously discovered agents to serve as stepping stones for future generations.

We propose the Darwin Gödel Machine (DGM), a self-referential, self-improving system that writes and modifies its own code to become a better coding agent. Each self-modification requires the DGM to edit its own codebase. We use Python, which is Turing-complete, giving the DGM the potential to build any computable machine. Our framework envisions agents that can rewrite their own training scripts (including training a new foundation model (FM)). However, we do not show that in this paper, as training FMs is computationally intensive and would introduce substantial additional complexity, which we leave as future work. Instead, this paper focuses on improving the design of coding agents with frozen pretrained FMs (e.g., tool use, workflows). The DGM alternates between self-modification and evaluation phases. During the self-modification phase, selected coding agents from the archive generate modified versions of themselves. During the evaluation phase, each modified agent is tested on a coding benchmark, estimating the agent’s coding capabilities, and then added to the archive. By improving its own capabilities through this loop, the DGM becomes better at both solving coding tasks and making future self-improvements. A key assumption is that an increase in performance on coding benchmarks indicates better coding capabilities, and hence better ability to self-modify and self-improve. Furthermore, the DGM maintains an archive of generated coding agents, initialized with only one agent, and continuously accumulates all generated variants over time. To support continual self-improvement, the DGM draws inspiration from open-endedness research (Wang et al., 2019; Fernando et al., 2024; Faldor et al., 2025), accumulating diverse stepping stones (i.e., interesting yet suboptimal solutions or features that may enable future breakthroughs). This open-ended exploration encourages the discovery of novel and potentially useful self-modifications beyond immediate performance gains.

We present results on two coding benchmarks: SWE-bench (Jimenez et al., 2024) and Polyglot (Paul Gauthier, 2024). The DGM automatically improves itself from 20.0% to 50.0% on SWE-bench, and from 14.2% to 30.7% on Polyglot. We show that self-improvement enables continued progress, as the DGM outperforms the baseline where the same base agent is repeatedly used to modify and generate new agents without self-improvement. We also show that open-ended exploration and keeping an archive of all previously generated agents lead to the discovery of better coding agents. The DGM outperforms the baseline of not having open-ended exploration (i.e., a baseline without the accumulation of an archive of interestingly different stepping stones), where the coding agent always builds off the most recent version of itself. Overall, the DGM represents a step toward AI systems that can build upon their own prior innovations and improve recursively. We consider and discuss safety aspects extensively, including sandboxing and traceability of self-modifications, to ensure responsible experimentation ([Section˜5](#S5)). By advancing the possibility of safe, self-referential, self-improving models, the DGM moves us closer to AI that not only learns but evolves in an open-ended, self-accelerating trajectory, much like science itself.

<a id="section-2"></a>

## 2 Related Work

Open-Endedness.
A grand challenge for driving unbounded innovation is designing open-ended AI systems that continuously generate novel and learnable artifacts (Stanley et al., 2017). Hughes et al. (2024) characterized open-endedness as a system’s capacity to generate sequences of artifacts that are both novel and learnable from an observer’s perspective. A central difficulty lies in structuring and exploring vast search spaces to consistently produce artifacts that are interesting to humans (Clune, 2019; Jiang et al., 2023). Early progress drew on quality-diversity algorithms, goal-directed exploration, intrinsic motivation, and learning-progress frameworks (Pugh et al., 2016; Ecoffet et al., 2019; Lehman and Stanley, 2011; Oudeyer et al., 2007), while recent advances leverage large-scale foundation models (FMs) as proxies for human interestingness and versatile engines for generating and evaluating novel behaviors across diverse domains (Brown et al., 2020; Hu et al., 2025; Zhang et al., 2024b). However, these approaches have yet to close the self-referential self-improvement loop, meaning improvements on downstream tasks do not translate into enhanced capabilities for self-modification or the acceleration of further innovations. We aim to mimic the acceleration of science and technology, where new tools and discoveries catalyze the creation of even more discoveries. How can we emulate nature’s arc of evolution, which bends not only toward complexity but also an ever greater capacity to evolve (Dawkins, 2019; Gerhart and Kirschner, 2007; Hendrikse et al., 2007)?

Meta-Learning FM Agents.
Many FM-based agents are handcrafted. Some building blocks include prompt engineering (Chen et al., 2023; Schulhoff et al., 2024), chain-of-thought (Wei et al., 2022; Yao et al., 2023; Hu and Clune, 2024; Guo et al., 2025; Lightman et al., 2023; Muennighoff et al., 2025; Zelikman et al., 2024a), self-reflection (Shinn et al., 2023; Yao et al., 2023; Madaan et al., 2023), multi-agent debate (Zhuge et al., 2023; Liang et al., 2023; Khan et al., 2024), memory (Liu et al., 2023; Zhong et al., 2024; Modarressi et al., 2023), temperature sampling (Zhu et al., 2024), and retrieval augmented generation (Lewis et al., 2020). The manual composition of these components limits the system’s abilities to the ingenuity of its human designer. More recently, several meta-learning approaches have emerged that leverage FM to automatically optimize prompts (Fernando et al., 2024; (FAIR)† et al., 2022; Khattab et al., 2023; Cheng et al., 2024; Yuksekgonul et al., 2024; Yuan et al., 2024) and design agentic modules (Zhang et al., 2024c; Zhou et al., 2024; Yin et al., 2024; Zhuge et al., 2024; Rosser and Foerster, 2025; Zhang et al., 2025a; Ye et al., 2025; Gao et al., 2025; Nie et al., 2025; Su et al., 2025; Zhang et al., 2025b; Niu et al., 2025). The Automated Design of Agentic Systems (ADAS, Hu et al., 2025) iteratively generates downstream agents with a fixed meta-agent, evaluates them against a target benchmark, and incorporates feedback to refine subsequent generations. In contrast, the DGM is a single system that both solves downstream tasks (i.e., coding problems) and refines its own implementation (i.e., its codebase), removing the need for a fixed, handcrafted meta-agent and enabling self-referential improvements.

Self-Improving AI.
Early on, various researchers outlined theoretical and conceptual approaches to self-improvement (Good, 1966; Schmidhuber, 1987; 2007). Some practical approaches to automated self-improvement include systems defined by neural network weight parameterizations (Schmidhuber, 1993; Hall, 2007; Hobbhahn, 2025; Kirsch and Schmidhuber, 2022; Irie et al., 2022; 2025; Lu et al., 2023; Havrilla et al., 2024b). Metz et al. (2021) developed a gradient-based optimizer that is self-referentially meta-trained using a variant of population-based training (Jaderberg et al., 2017). Lange et al. (2023) extended this approach to gradient-free learning. Silver et al. (2017) used self-play to continuously evolve agents, achieving superhuman performance in challenging domains such as chess and Go. More closely related to the DGM are recent approaches that leverage FM-based agents for self-improvement (Yin et al., 2024; Robeyns et al., 2025; Hu et al., 2024; Zelikman et al., 2024b; Huang et al., 2022; Singh et al., 2023). Zelikman et al. (2024b) use a meta-agent to generate downstream agents, updating the meta-agent based on the meta-utility derived from the generated solutions. Yin et al. (2024) use a single system to both solve downstream tasks and recursively modify itself. However, the downstream tasks or the meta-utility do not always align with the capabilities required for self-improvement. In the DGM, improvement in downstream tasks directly reflects an increase in self-improvement ability, enabling the potential for self-accelerating progress. Most similar is concurrent work by Robeyns et al. (2025), which also has a single agent recursively solving coding problems and modifying its own codebase. The main difference from Robeyns et al. (2025) (and also Zelikman et al. (2024b); Yin et al. (2024)) is that the DGM has an open-ended exploration loop, encouraging self-modifications beyond immediate performance gains and thus avoiding stagnation in suboptimal states. [Appendix˜B](#A2) also discusses additional related work on program synthesis and Darwinian evolution.

<a id="section-3"></a>

## 3 Darwin Gödel Machine

A Gödel Machine is a theoretical idea of an AI that searches for ways that _provably_ improve itself (Schmidhuber, 2007). In this paper, we propose Darwin Gödel Machine (DGM), an attempt to realize the long-held dream of creating a Gödel Machine. The DGM relaxes the Gödel Machine’s impractical requirement of theoretically _proving_ that a change will improve the system, instead requiring _empirical evidence_ from experiments to demonstrate that a proposed new version enhances performance. Additionally, since the DGM relies on empirical evidence of improvement, it may get stuck in a local optimum within the vast search space of possible systems (i.e., all computable algorithms). To address this, the DGM maintains an archive of discovered solutions during the search, facilitating open-ended exploration rather than relying on evolving a single solution. Since the principles echo Darwinian evolution (Darwin, 2023) ([Appendix˜B](#A2)), where new innovations emerge by selecting an entity from an archive of previously discovered solutions, modifying it, and keeping it if it is interestingly new (Zhang et al., 2024b; Faldor et al., 2025; Stanley and Lehman, 2015), we call our algorithm a Darwin Gödel Machine ([Figure˜1](#S1.F1)).

Self-referential Self-improvement of Coding Agents.
The DGM is initialized with only one coding agent, and its progression is evaluated on coding benchmarks. A coding agent is defined as a single system, implemented with a code repository and powered by frozen pretrained foundation models (FMs), capable of reading, writing, and executing code. Code, when expressed in a general-purpose Turing-complete language (e.g., Python), is a powerful medium for building and improving intelligent systems because it can represent any computable process. Recent works (Hu et al., 2025; Zhang et al., 2024c) demonstrate that such agents can be improved through meta-learning of their designs (e.g., prompts, workflows, and tools), which are implemented in their code repository. Therefore, we define self-improvement as a coding task that involves modifying the design of an agent’s own components (i.e., its own code, which does not include the open-ended exploration process described in the next paragraph). The key motivation is that the empirical evidence must reflect the system’s ability to both self-improve and solve downstream tasks. By configuring the DGM as a coding agent and testing its coding capabilities, the observed improvements demonstrate not only enhanced performance in downstream tasks but also the capacity for further self-improvement, as self-improvement is fundamentally a coding task that modifies the coding agent’s own code repository.

Population-based Open-ended Exploration.
Starting from a single initial coding agent, the DGM builds an archive of all discovered agents. In each iteration, the DGM selects parent agents to self-modify and branch off to produce new agents. Parent selection is roughly proportional to each agent’s performance score and inversely proportional to the number of its children with codebase-editing functionality ([Section˜C.2](#A3.SS2)). This favors high-performing agents that have been underexplored (i.e., have fewer existing children), thereby promoting both exploitation of strong performers and exploration of promising but less-sampled lineages. All agents retain a non-zero selection probability, ensuring that any path to improvement remains feasible given sufficient compute.
Each selected parent analyzes its own benchmark evaluation logs, proposes the next feature to implement, and receives this proposal as a problem statement to execute ([Section˜C.3](#A3.SS3)). The parent then implements the suggested feature into its own codebase, generating a new coding agent. Each newly generated agent is quantitatively evaluated on a chosen coding benchmark to estimate its coding abilities. Only agents that compile successfully and retain the ability to edit a given codebase are added to the DGM archive, as only they can continue self-modification. All others are discarded. The cycle of parent selection, self-modification, and evaluation continues, progressively growing the archive of solutions. Importantly, we note that archived solutions can serve as stepping stones that result in improvements much later than their original discovery, making our approach substantially different from hill-climbing agentic design approaches (Robeyns et al., 2025). Currently, the open-ended exploration process (i.e., archive maintenance, parent selection) is fixed and not modifiable by the DGM, which we leave as an avenue for future work. [Section˜C.4](#A3.SS4) shows the pseudocode for the DGM algorithm.

<a id="section-4"></a>

## 4 Experiments

[Section˜4.1](#S4.SS1) describes the experimental setup, including the initial coding agent that initializes the DGM. We conduct experiments on two coding benchmarks: SWE-bench (Jimenez et al., 2024) and Polyglot (Paul Gauthier, 2024) ([Section˜4.2](#S4.SS2)). For each benchmark, we compare the DGM against two baselines: DGM without self-improving agents and DGM without open-ended exploration ([Section˜4.3](#S4.SS3)). Across all experiments, we find that the DGM outperforms both baselines, showing that the self-improvement mechanism and open-ended exploration are essential for sustained performance gains ([Section˜4.4](#S4.SS4)). Furthermore, we show that the features discovered by the DGM transfer across models, benchmarks, and tasks ([Section˜4.4](#S4.SS4)).

<a id="section-4-1"></a>

### 4.1 Experiment Setup

The DGM is initialized with a single coding agent. This base agent is built around a frozen FM and augmented with tool use capabilities (Schick et al., 2023; Anthropic, 2024b). The FM autonomously determines its action at each step, using whichever tool it deems appropriate. The base agent has access to two tools: a Bash tool for executing bash commands, and an edit tool for viewing and editing directories and entire files ([Section˜C.1](#A3.SS1)). The agent receives a single input prompt that specifies the location of the target repository, the problem statement, and the repository’s testing framework ([Section˜C.1](#A3.SS1)). The base agent is intentionally lightweight to provide a simple starting point from which self-improvement can be clearly demonstrated. We run the DGM for 80 iterations (generating one new agent per iteration), with two iterations running in parallel for SWE-bench and four for Polyglot ([Section˜C.2](#A3.SS2)). During self-modifications, coding agents are powered by Claude 3.5 Sonnet (New) (Anthropic, 2024a) in both SWE-bench and Polyglot experiments. During benchmark evaluation, coding agents are powered by Claude 3.5 Sonnet (New) for SWE-bench and o3-mini (OpenAI, 2025) for Polyglot ([Section˜D.1](#A4.SS1)).

<a id="section-4-2"></a>

### 4.2 Benchmarks

We evaluate the DGM on two popular benchmarks that assess different aspects of coding tasks to validate the algorithm’s effectiveness across various use cases. Both SWE-bench and Polyglot are benchmarks that evaluate how well AI agents solve coding tasks automatically. Given a code repository and a task instruction, the agent is expected to make changes to the repository in order to fulfill the task. Both SWE-bench and Polyglot are widely used benchmarks (Zhang et al., 2024d; a; Xia et al., 2024; Cao et al., 2024; Google DeepMind, 2025; Gauthier, 2024) that require the AI agent to navigate a code repository, understand the interplay between functions in different files, and spot small errors in convoluted code. SWE-bench only has Python tasks, while Polyglot has tasks in multiple programming languages. Another difference is that each SWE-bench task may require edits to multiple files, whereas each Polyglot task primarily involves implementing a solution from scratch in a single file (although the agent still needs to examine other files to understand what changes are necessary), resulting in fewer file edits overall.

SWE-bench. To avoid wasting compute on unsolvable tasks, we use SWE-bench Verified (OpenAI, 2024), a human-filtered subset of SWE-bench (Jimenez et al., 2024) where all tasks are solvable. Throughout this paper, the term SWE-bench refers by default to to the SWE-bench Verified subset.

Polyglot. Polyglot includes tasks in multiple programming languages (C++, Rust, Python, etc.) (Paul Gauthier, 2024). Compared to SWE-bench, one of the most widely used coding benchmarks and likely included in the training sets of FMs, Polyglot is more niche and less likely to be included in FMs’ post-training data. Additionally, Polyglot is primarily used by its developer to evaluate Aider (Gauthier, 2024). This provides an opportunity to compare automatically designed agents with a representative coding agent in the community, which has been continuously developed and tested against the Polyglot benchmark over a year by human developers. Finally, Polyglot enables testing of the DGM’s ability to generalize across multiple programming languages beyond just Python. We follow the Polyglot leaderboard setup (Paul Gauthier, 2024), with one key difference: the leaderboard reports pass@2, where the agent can view feedback from ground-truth tests once, whereas we report pass@1, where the agent never sees ground-truth test results.

Since it is expensive to evaluate on the full benchmarks ([Section˜E.1](#A5.SS1)), we use a staged evaluation strategy to estimate the coding agent’s performance. In general, we evaluate the coding agent on more tasks when we are more confident that it is a relatively strong performer. We first evaluate each coding agent on a set of 10 tasks to verify basic codebase-editing functionality. Only agents that retain the ability to edit code can solve downstream coding tasks or perform further self-modifications. Agents that pass this initial stage are then evaluated on an expanded set of 50 tasks. For SWE-bench, tasks are selected based on SWE-bench-verified-mini (Hobbhahn, 2025), which is designed to be a representative sample of the full benchmark ([Section˜E.2](#A5.SS2)). For Polyglot, tasks are chosen as a random subset of the full benchmark ([Section˜E.3](#A5.SS3)). Because the LLMs we use are inherently stochastic, performance can be noisy. We use smaller subset evaluations to estimate general effectiveness and identify top candidates. For SWE-bench, if a coding agent achieves over 40% success (a heuristic chosen based on the noise observed in preliminary runs) on the 60-task subset and ranks among the top two performers in the archive, it is then evaluated on 200 tasks (140 plus the previous 60) to more accurately assess its true coding capability ([Section˜E.2](#A5.SS2)). For Polyglot, if a coding agent achieves over 40% success on the 10-task subset, it is evaluated on an expanded 50-task subset ([Section˜E.3](#A5.SS3)).

<a id="section-4-3"></a>

### 4.3 Baselines

To isolate and evaluate the contribution of the DGM’s core components, we compare the DGM against two baselines: DGM without self-improving agents (DGM w/o self-improve) and DGM without open-ended exploration (DGM w/o open-ended exploration). DGM w/o self-improve replicates the approach of ADAS (Hu et al., 2025) in this setting, wherein the meta agent responsible for modifying the coding agents remains fixed as the base agent throughout the experiment ([Algorithm˜2](#algorithm2)). This baseline allows us to assess the role of iterative self-improvement in accelerating the evolution of better coding agents. DGM w/o open-ended exploration eliminates the use of an archive and always self-modifies the latest stored version of itself ([Algorithm˜3](#algorithm3)). If a coding agent self-modifies to the point where it loses the basic functionality required to edit a codebase, it can no longer modify itself or solve any coding task. Therefore, DGM w/o open-ended exploration retains the latest version of itself that still maintains the basic functionality for codebase editing. This baseline allows us to evaluate the impact of having an archive and the well-documented beneficial principles of open-ended exploration (Clune, 2019; Stanley and Lehman, 2015; Zhang et al., 2024b; Fernando et al., 2024; Lee et al., 2020; Samvelyan et al., 2024; Colas et al., 2022b) in guiding the agent’s evolution.

In addition to the learned baselines, we compare the DGM against handcrafted, open-source solutions. For SWE-bench, we take the state-of-the-art (SoTA) open-source solution that has been checked (i.e., the SWE-bench team was able to reproduce the results) ([Section˜E.4](#A5.SS4)). For Polyglot, we take the representative agent (Aider) (Gauthier, 2024), which is open-sourced and designed to support multiple programming languages and large codebase editing ([Section˜E.5](#A5.SS5)). For a fair comparison, we measure the percentage of solved tasks on the same benchmark subsets used to evaluate the DGM ([Section˜E.2](#A5.SS2), [Section˜E.3](#A5.SS3)). These values are shown as dotted horizontal lines in [Figure˜2](#S4.F2).

<a id="section-4-4"></a>

### 4.4 Results

<a id="figure-2"></a>

![dgm_comparisons](images/dgm_comparisons.png)

> Figure 2: Self-improvement and open-ended exploration enable the DGM to continue making progress and improve its performance. The DGM automatically discovers increasingly better coding agents and performs better on both (Left) SWE-bench and (Right) Polyglot. It outperforms baselines that lack either self-improvement or open-ended exploration, showing that both components are essential for continual self-improvement. These scores are obtained from evaluating on the benchmark subsets detailed in [Section˜4.2](#S4.SS2).

<a id="figure-3"></a>

![dgm_comparisons_polyglot](images/dgm_comparisons_polyglot.png)

> Figure 3: The DGM automatically self-improves to become a better coding agent. (Left) Archive of coding agents generated during the DGM run on SWE-bench. Each node represents a coding agent, with node 0 corresponding to the base agent. Node color indicates performance on SWE-bench (percentage of solved tasks), while border color reflects the number of tasks for which the agent was evaluated. Edges show which agents self-modified to produce the offsprings. Many paths to innovation traverse lower-performing nodes, and key innovations (like node 24) lead to an explosion of innovations built on top of them. Both properties underscore the benefits of open-ended search. (Right) Progress plot of the DGM on SWE-bench. The light blue line shows the average score of all agents possessing basic codebase-editing functionality. The blue line tracks the best score achieved by any agent in the archive at each iteration. The dark line shows the lineage of the final best-discovered agent and its precursor nodes, which includes two performance dips. This illustrates the benefits of open-ended search, which explores a diverse set of interesting stepping stones instead of focusing only on branching off the best solution found so far.

After 80 iterations of the DGM, the coding agent’s performance increases from 20.0% to 50.0% on SWE-bench, and from 14.0% to 38.0% on Polyglot ([Figure˜2](#S4.F2)). Since the DGM is evaluated on only 50 tasks in the Polyglot experiment setup ([Section˜4.2](#S4.SS2)), we additionally evaluate both the base agent and the best DGM-discovered agent on the full Polyglot benchmark to more accurately estimate the improvement. On the full Polyglot benchmark, the DGM improves the coding agent from 14.2% to 30.7%. This shows that the DGM can automatically self-improve to create a better coding agent. Moreover, the performance of the best DGM-discovered agent is comparable to that of the checked, open-source, human-designed SoTA on SWE-bench ([Figure˜2](#S4.F2)). On Polyglot, although the DGM starts with a base agent whose performance is lower than that of Aider, it discovers an agent that far surpasses Aider ([Figure˜2](#S4.F2)). The DGM-discovered agents are comparable to or outperform handcrafted agents on both benchmarks. While the SoTA SWE-bench agent and Aider were painstakingly shaped by human efforts, the DGM hints at a future in which such ingenuity is automated, evolving through self-referential cycles of continuous self-improvements.

The DGM automatically improves both the tools and the workflow of how FMs are utilized ([Figure˜3](#S4.F3)). For example, the DGM enhanced the edit tool to allow more granular file viewing (by lines) and more precise file editing (by string replacement), instead of always viewing or replacing the entire file. Workflow improvements include making multiple attempts to solve a task and using another FM to evaluate and select the best solution. Other workflow improvements include considering previous attempts when generating subsequent ones. [Section˜F.1](#A6.SS1) and [Section˜F.2](#A6.SS2) show all modifications leading up to the final best-discovered agents on SWE-bench and Polyglot respectively.

Because open-ended exploration allows branching from any agent in the archive with non-zero probability, the DGM can get out of deceptive dips or peaks in performance. For example, at iterations 4 and 56 of the experiment on SWE-bench, although the agent’s score temporarily fell below that of its parent, the DGM was still able to explore innovations along that path and create a new agent that outperformed all of its predecessors ([Figure˜3](#S4.F3)). Furthermore, open-ended exploration allows different implementations of the same target functionality to be attempted. For example, while the goal is to provide finer‑grained editing tools, the specific implementation of this feature can vary greatly and hence lead to very different performance ([Appendix˜G](#A7)). The DGM can explore multiple implementations to find the most suitable one and avoid getting trapped in a suboptimal one.

The DGM outperforms the baselines of DGM w/o self-improve and DGM w/o open-ended exploration on both benchmarks ([Figure˜2](#S4.F2)). Without updating the meta agent that modifies coding agents, DGM w/o self-improve improves the agents in early iterations, but its gains taper off quickly ([Section˜A.1](#A1.SS1)). In DGM w/o open-ended exploration, only the most recent agent is retained, so a poorly performing self-modification makes subsequent improvements harder to achieve ([Section˜A.1](#A1.SS1)).

<a id="figure-4"></a>

![dgm_archive](images/dgm_archive.png)

> Figure 4: Transfer between Models, Benchmarks, and Tasks. The superior performance of DGM-discovered agents can be transferred across (Left) different models, (Middle) benchmarks, and (Right) different programming language tasks in Polyglot, such as from Python tasks to C++ tasks.

To evaluate the generality of the improvements from the DGM, we tested the base agent ([Section˜4.1](#S4.SS1)) and the best agent discovered during the DGM run ([Figure˜2](#S4.F2)) with different FMs than those used during optimization. For SWE-bench, where the DGM was run using Claude 3.5 Sonnet (New), we replaced the FM with Claude 3.7 Sonnet (Anthropic, 2025) or o3-mini, and evaluated on 200 tasks ([Figure˜4](#S4.F4), Left). With o3-mini, the base agent achieved 23.0% and the DGM-discovered agent 33.0%. With Claude 3.7 Sonnet, the base agent achieved 19.0% and the DGM-discovered agent 59.5%. These results suggest that the DGM yields improvements that generalize across FMs, rather than being tightly coupled to the specific FM used during its run ([Figure˜4](#S4.F4)). Model transfer results on Polyglot are presented in [Section˜A.2](#A1.SS2).

Furthermore, we investigate the transferability of the DGM-discovered agent across different benchmarks and programming languages. First, we evaluate the best DGM-discovered agent from one benchmark (e.g., SWE-bench) on a completely held-out benchmark (e.g., Polyglot), and vice versa ([Figure˜4](#S4.F4), Middle). The best agent evolved on SWE-bench achieves 28.9% on Polyglot, compared to the initial agent’s baseline of 14.2%. Conversely, the best agent evolved on Polyglot achieves 24.5% on SWE-bench, outperforming the original baseline of 20.0%. Since each agent was optimized without ever accessing the alternate benchmark, these evaluations represent truly held-out tests. The consistent performance gains across benchmarks support our claim that DGM’s improvements reflect general skill acquisition rather than overfitting or exploitation of benchmark-specific artifacts. Second, we experiment with a version of the DGM trained exclusively on Python tasks from Polyglot and then transfer the discovered agent to tasks in other languages. Focusing primarily on Python tasks slightly improves performance on Python tasks but reduces performance on non-Python tasks compared to the DGM trained on all languages ([Figure˜4](#S4.F4), Right). However, after being transferred from Python to other unseen languages during the search, the agent still achieves performance comparable to that of the DGM trained on all languages and substantially outperforms both the base agent and Aider. These results demonstrate the robustness of the discovered improvements, showing that they do not overfit to a specific programming language. We also present additional results in [Appendix˜A](#A1).

<a id="section-5"></a>

## 5 Safety Discussion

Systems capable of self-improvement, such as the DGM, represent a step toward more autonomous AI development, aligning with long-standing goals in the field of making capable AI that can benefit humanity (Schmidhuber, 1987; Clune, 2019; Markoff, 2016; Lehman, 2023). However, this capability introduces unique safety considerations stemming from the system’s ability to autonomously modify its own code. Modifications optimized solely for benchmark performance might inadvertently introduce vulnerabilities or behaviors misaligned with human intentions, even if they improve the target metric (Bostrom, 2020). In particular, if evaluation benchmarks do not fully capture all desired agent properties (e.g., safety and robustness), the self-improvement loop could amplify misalignment over successive generations. Iterative self-modification could also lead to increasingly complex and uninterpretable internal logic, hindering human understanding, oversight, and control (Sheth et al., 2025; Anwar et al., 2024; Greenblatt et al., 2024; Ganguli et al., 2022).

Recognizing these challenges, the current implementation and experimental setup of the DGM incorporates several safeguards. All agent execution and self-modification processes are conducted within isolated sandboxed environments, limiting their ability to affect the host system, and thereby mitigating the risk of unintended actions. Each execution within the sandbox is subjected to a strict time limit, reducing the risk of resource exhaustion or unbounded behavior. The self-improvement process is currently confined to the well-defined domain of enhancing performance on specific coding benchmarks by modifying the agent’s own Python codebase, thus limiting the scope of potential modifications. Additionally, we actively monitor agent performance and code changes, with the DGM archive providing a traceable lineage of modifications for review. At this stage, we have found no evidence of harmful or malicious behavior in the generated agents, and the self-modifications have been primarily focused on improving coding capabilities.

Conversely, a significant potential benefit of the self-improvement paradigm is that it could, in principle, be directed toward enhancing safety and interpretability themselves. We conduct a preliminary investigation into how the DGM can be deployed in AI safety settings to develop countermeasures for FM hallucination ([Appendix˜H](#A8)). Just as the DGM learns to improve its coding capabilities, it could potentially discover and integrate better internal safeguards or modify itself for greater transparency (e.g., incorporating principles akin to Constitutional AI (Bai et al., 2022)), if such properties were included in its evaluation criteria (Rosser and Foerster, 2025). This suggests a promising, albeit challenging, pathway in which self-improvement becomes a tool for building more trustworthy AI systems. Additional research could also explore weaving Constitutional AI in from the start, though the challenge would be incentivizing the system to retain these directives (an option worth exploring is to create an unmodifiable part of the system to be able to evaluate at halt the rest).

The DGM demonstrates the potential of self-improving AI while still operating within safe research boundaries due to the current limitations of frontier FMs and effective mitigations like sandboxing. [Appendix˜I](#A9) presents additional discussion on broader safety uncertainties. We include this safety discussion proactively to raise awareness about the emerging prospect of self-improving AI systems and their associated safety implications, particularly as these systems inevitably become more capable (Yudkowsky and others, 2008; Bostrom, 2002; Ecoffet et al., 2020; Bengio et al., 2024; Clune, 2019). Accordingly, we advocate for continued investigation into the safe and beneficial evolution of AI-Generating Algorithms (Clune, 2019) and self-improving systems.

<a id="section-6"></a>

## 6 Conclusion and Limitations

We introduce the Darwin Gödel Machine (DGM), the first self-improving system powered by FMs with open-ended exploration, where progress on its evaluation benchmarks can directly translate into better self-improvement capabilities. We demonstrate the automatic discovery of better tools and FM systems, resulting in better performance on two benchmarks: SWE-bench and Polyglot. Through self-improvement and open-ended exploration, the DGM shows a continuous increase in performance, bringing us one step closer to self-accelerating, self-improving AI systems.

We demonstrate that the DGM can autonomously achieve performance on par with openly available solutions. However, it still falls short of closed-source SoTA SWE-bench solutions. An open question is whether running the DGM for longer would continue to yield performance gains and eventually surpass closed-source solutions. These closed-source solutions often rely on elaborately handcrafted techniques developed by teams of highly skilled experts. Since FMs have yet to match the capabilities of such experts (e.g., in reasoning), the DGM currently requires extensive compute to discover improvements. A single run of the DGM on SWE-bench, as presented in [Section˜4](#S4), takes about 2 weeks and incurs significant API costs ([Section˜E.1](#A5.SS1)). We hypothesize that further progress will require more efficient use of computational resources and the development of better reasoning skills.

Since this version of the DGM is mainly powered by FMs, it is inherently limited by the capabilities of the underlying FM. Hence, an exciting future direction is to extend self-modification beyond just prompts or FM workflows, to include more computationally intensive methods, such as rewriting its own training script to update the FM itself. While this version of the DGM focuses on coding, AI systems are increasingly applied across a wide range of domains (e.g., computer vision, creative writing). Another promising extension is to develop self-improving AI systems capable of enhancing themselves beyond just the coding domain. A key assumption in this work is that coding benchmarks are a good reflection of the agent’s ability to self-improve, since the self-modification task requires the agent to modify its own codebase. However, one could envision an alternative approach that co-evolves the target task distribution (Faldor et al., 2025; Wang et al., 2023c), thereby removing the constraint of self-improvement being tied to a single objective, as in true open-ended processes. [Appendix˜J](#A10) presents additional potential directions for future work. As we continue to explore this powerful technology, we must also keep safety front and center, as discussed in [Section˜5](#S5).

In conclusion, the DGM represents a significant step toward the automation of AI development through self-improving systems capable of editing their own codebase. While current limitations in compute and reasoning constrain its full potential, continued advances in FMs and infrastructure may unlock more powerful and general-purpose self-improvements. Provided that the safety concerns are carefully navigated ([Section˜5](#S5)), the future of self-improving AI systems and AI-Generating Algorithms (Clune, 2019) holds immense promise to open-endedly evolve AI, continually rewriting or retraining itself in pursuit of greater capabilities aligned with human values.

## Ethics Statement

We affirm compliance with the ICLR Code of Ethics. This work studies self-improving AI systems in the limited context of code-editing agents evaluated on standard programming benchmarks. No human subjects were involved and no personally identifiable information (PII) was collected or processed; IRB approval was therefore not required.

Safety and misuse.
Self-modifying systems can pose safety risks if allowed to act without constraints or if optimizations inadvertently introduce unsafe behaviors. To mitigate this, all agents in our experiments ran inside isolated sandboxes with strict resource and time limits; agents had limited network access and no ability to modify the host environment. The self-improvement scope was restricted to the agent’s own Python codebase and evaluation harnesses. We maintained a complete, auditable lineage (archive) of code changes and evaluations, enabling rollback and post-hoc analysis. We did not deploy discovered agents in real development environments. Our release plan (code, prompts, and evaluation artifacts) will exclude any components that grant elevated system access and will include default sandboxing, guardrails, and clear documentation of intended use.

Dual-use, downstream impact, and limitations.
Stronger autonomous coding agents could be dual-use (e.g., aiding software maintenance, but also potentially facilitating creation of harmful code if misapplied). We believe the research benefits (e.g., advancing methods for controlled, auditable self-improvement and demonstrating practical safeguards) outweigh the risks. Nevertheless, we explicitly discourage security-sensitive or unsandboxed deployment and provide concrete safety recommendations ([Section˜5](#S5)). Our empirical focus on benchmark optimization may not capture all desirable properties (robustness, interpretability, or broader social values). We therefore treat benchmark gains as necessary but insufficient indicators of general AI development, and discuss avenues to integrate other objectives (e.g., safety, reasoning) into the optimization loop.

Data governance, IP, and licensing.
We evaluate on SWE-bench Verified and Polyglot, which are composed of open-source repositories and tasks. We complied with dataset licenses and usage terms to the best of our knowledge. We did not introduce or distribute proprietary code. Foundation models (FMs) were accessed via provider APIs under their terms of service; we did not submit sensitive data, nor attempt to circumvent usage policies. Logs released with this work will be scrubbed of API keys and any incidental sensitive strings.

Bias, fairness, and equity.
Although our domain is software code rather than human-centered text, FM behavior can still reflect biases (e.g., language or ecosystem preferences) and may unevenly benefit communities whose tooling is better represented in training data. We partially address this by evaluating across multiple languages (Polyglot) and reporting cross-benchmark transfer. Future work should add diagnostics for biased failure modes and include broader, community-driven task sets.

Conflicts of interest and funding.
No author has a financial interest in products whose performance is evaluated here. Sponsors and employers did not influence experimental design, analysis, or the decision to publish, beyond providing salary or standard research support. Any external compute or API credits are acknowledged in the appendix.

## Reproducibility Statement

We will open-source all code and full agent logs, including the complete archive lineage of self-modifications (diffs, prompts, and configs) as well as the evaluation harness. To support exact replication, we reference the following: algorithmic details and pseudocode ([Section˜3](#S3), [Section˜C.4](#A3.SS4)); parent selection and open-ended exploration settings ([Section˜C.2](#A3.SS2)); foundation model choices and hyperparameters ([Section˜D.1](#A4.SS1)); benchmark task subsets for SWE-bench and Polyglot ([Section˜E.2](#A5.SS2), [Section˜E.3](#A5.SS3)); staged evaluation protocols and scripts ([Section˜4.2](#S4.SS2)); implementations and diffs for the best discovered agents ([Section˜F.1](#A6.SS1), [Section˜F.2](#A6.SS2)); and compute and cost estimates ([Section˜E.1](#A5.SS1)). The released code repository will include environment specifications and scripts to reproduce all results, figures, and tables.

#### Acknowledgments

This research was supported by the Vector Institute, the Canada CIFAR AI Chairs program, a grant from Schmidt Futures, an NSERC Discovery Grant, and a generous donation from Rafael Cosman. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute (https://vectorinstitute.ai/partnerships/current-partners/). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the sponsors. We also thank Aaron Dharna, Ben Norman, Cédric Colas, Sam Devlin, and Shyam Sudhakaran for insightful discussions and feedback.

## References

- M. F. A. R. D. T. (FAIR)†, A. Bakhtin, N. Brown, E. Dinan, G. Farina, C. Flaherty, D. Fried, A. Goff, J. Gray, H. Hu, et al. (2022)
  Human-level play in the game of Diplomacy by combining language models with strategic reasoning.
  Science 378 (6624), pp. 1067–1074.
  Cited by: [§2](#S2.p2.1).
- F. Aki, R. Ikeda, T. Saito, C. Regan, and M. Oka (2024)
  Llm-poet: Evolving complex environments using large language models.
  In Proceedings of the Genetic and Evolutionary Computation Conference Companion,
  pp. 243–246.
  Cited by: [Appendix B](#A2.p1.1).
- R. Alur, R. Singh, D. Fisman, and A. Solar-Lezama (2018)
  Search-based program synthesis.
  Communications of the ACM 61 (12), pp. 84–93.
  Cited by: [Appendix B](#A2.p2.1).
- S. Amari (1972)
  Learning patterns and pattern sequences by self-organizing nets of threshold elements.
  IEEE Transactions on computers 100 (11), pp. 1197–1206.
  Cited by: [§1](#S1.p1.1).
- M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, O. Pieter Abbeel, and W. Zaremba (2017)
  Hindsight experience replay.
  Advances in neural information processing systems 30.
  Cited by: [Appendix B](#A2.p1.1).
- Anthropic (2024a)
  Claude 3.5 Sonnet.
  Anthropic.
  Note: [https://www.anthropic.com/news/claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)[Accessed 17 April 2025]
  Cited by: [§4.1](#S4.SS1.p1.1).
- Anthropic (2024b)
  Claude can now use tools.
  Note: Accessed: 2025-05-03
  External Links: [Link](https://www.anthropic.com/news/tool-use-ga)
  Cited by: [§4.1](#S4.SS1.p1.1).
- Anthropic (2025)
  Claude 3.7 sonnet and claude code.
  Note: Accessed: 2025-05-06
  External Links: [Link](https://www.anthropic.com/news/claude-3-7-sonnet)
  Cited by: [§4.4](#S4.SS4.p5.1).
- U. Anwar, A. Saparov, J. Rando, D. Paleka, M. Turpin, P. Hase, E. S. Lubana, E. Jenner, S. Casper, O. Sourbut, et al. (2024)
  Foundational challenges in assuring alignment and safety of large language models.
  arXiv preprint arXiv:2404.09932.
  Cited by: [§5](#S5.p1.1).
- D. Bahdanau, K. H. Cho, and Y. Bengio (2015)
  Neural machine translation by jointly learning to align and translate.
  In International Conference on Learning Representations,
  Cited by: [§1](#S1.p1.1).
- Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, et al. (2022)
  Constitutional AI: Harmlessness from AI feedback.
  arXiv preprint arXiv:2212.08073.
  Cited by: [Appendix J](#A10.p3.1),
  [§5](#S5.p3.1).
- A. Baranes and P. Oudeyer (2013)
  Active learning of inverse models with intrinsically motivated goal exploration in robots.
  Robotics and Autonomous Systems 61 (1), pp. 49–73.
  Cited by: [Appendix B](#A2.p1.1).
- S. Barke, E. Anaya Gonzalez, S. R. Kasibatla, T. Berg-Kirkpatrick, and N. Polikarpova (2024)
  Hysynth: context-free llm approximation for guiding program synthesis.
  Advances in Neural Information Processing Systems 37, pp. 15612–15645.
  Cited by: [Appendix B](#A2.p2.1).
- Y. Bengio, G. Hinton, A. Yao, D. Song, P. Abbeel, T. Darrell, Y. N. Harari, Y. Zhang, L. Xue, S. Shalev-Shwartz, et al. (2024)
  Managing extreme AI risks amid rapid progress.
  Science 384 (6698), pp. 842–845.
  Cited by: [Appendix I](#A9.p1.1),
  [§5](#S5.p4.1).
- N. Bostrom (2002)
  Existential Risks: analyzing human extinction scenarios and related hazards.
  Journal of Evolution and Technology 9 (), pp. .
  Cited by: [§5](#S5.p4.1).
- N. Bostrom (2020)
  Ethical issues in advanced artificial intelligence.
  Machine Ethics and Robot Ethics, pp. 69–75.
  Cited by: [§5](#S5.p1.1).
- H. Bradley, A. Dai, H. B. Teufel, J. Zhang, K. Oostermeijer, M. Bellagente, J. Clune, K. Stanley, G. Schott, and J. Lehman (2024)
  Quality-diversity through ai feedback.
  In The Twelfth International Conference on Learning Representations,
  Cited by: [Appendix B](#A2.p1.1).
- T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. (2020)
  Language models are few-shot learners.
  Advances in neural information processing systems 33, pp. 1877–1901.
  Cited by: [Appendix B](#A2.p1.1),
  [§1](#S1.p1.1),
  [§2](#S2.p1.1).
- J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. (2024)
  Genie: Generative interactive environments.
  In Forty-first International Conference on Machine Learning,
  Cited by: [Appendix B](#A2.p1.1).
- J. R. Buchi and L. H. Landweber (1990)
  Solving sequential conditions by finite-state strategies.
  In The collected works of J. Richard Büchi,
  pp. 525–541.
  Cited by: [Appendix B](#A2.p2.1).
- R. Cao, F. Lei, H. Wu, J. Chen, Y. Fu, H. Gao, X. Xiong, H. Zhang, W. Hu, Y. Mao, et al. (2024)
  Spider2-v: how far are multimodal agents from automating data science and engineering workflows?.
  Advances in Neural Information Processing Systems 37, pp. 107703–107744.
  Cited by: [§4.2](#S4.SS2.p1.1).
- K. Chatzilygeroudis, A. Cully, V. Vassiliades, and J. Mouret (2021)
  Quality-diversity optimization: a novel branch of stochastic optimization.
  In Black Box Optimization, Machine Learning, and No-Free Lunch Theorems,
  pp. 109–135.
  Cited by: [Appendix B](#A2.p1.1).
- B. Chen, Z. Zhang, N. Langrené, and S. Zhu (2023)
  Unleashing the potential of prompt engineering in large language models: a comprehensive review.
  arXiv preprint arXiv:2310.14735.
  Cited by: [§2](#S2.p2.1).
- C. Cheng, A. Nie, and A. Swaminathan (2024)
  Trace is the next autodiff: generative optimization with rich feedback, execution traces, and llms.
  Advances in Neural Information Processing Systems 37, pp. 71596–71642.
  Cited by: [§2](#S2.p2.1).
- J. Clune (2019)
  AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence.
  arXiv preprint arXiv:1905.10985.
  Cited by: [Appendix I](#A9.p1.1),
  [§2](#S2.p1.1),
  [§4.3](#S4.SS3.p1.1),
  [§5](#S5.p1.1),
  [§5](#S5.p4.1),
  [§6](#S6.p4.1).
- C. Colas, P. Fournier, M. Chetouani, O. Sigaud, and P. Oudeyer (2019)
  Curious: intrinsically motivated modular multi-goal reinforcement learning.
  In International conference on machine learning,
  pp. 1331–1340.
  Cited by: [Appendix B](#A2.p1.1).
- C. Colas, T. Karch, C. Moulin-Frier, and P. Oudeyer (2022a)
  Language and culture internalization for human-like autotelic AI.
  Nature Machine Intelligence 4 (12), pp. 1068–1076.
  Cited by: [Appendix B](#A2.p1.1).
- C. Colas, T. Karch, O. Sigaud, and P. Oudeyer (2022b)
  Autotelic agents with intrinsically motivated goal-conditioned reinforcement learning: a short survey.
  Journal of Artificial Intelligence Research 74, pp. 1159–1199.
  Cited by: [Appendix B](#A2.p1.1),
  [§4.3](#S4.SS3.p1.1).
- C. Colas, L. Teodorescu, P. Oudeyer, X. Yuan, and M. Côté (2023)
  Augmenting autotelic agents with large language models.
  In Conference on Lifelong Learning Agents,
  pp. 205–226.
  Cited by: [Appendix B](#A2.p1.1).
- R. Coulom (2006)
  Efficient selectivity and backup operators in monte-carlo tree search.
  In International conference on computers and games,
  pp. 72–83.
  Cited by: [§C.2](#A3.SS2.p1.1).
- C. Darwin (2023)
  Origin of the species.
  In British Politics and the environment in the long nineteenth century,
  pp. 47–55.
  Cited by: [Appendix B](#A2.p3.1),
  [§1](#S1.p2.1),
  [§3](#S3.p1.1).
- R. Dawkins (2019)
  The evolution of evolvability.
  In Artificial life,
  pp. 201–220.
  Cited by: [§2](#S2.p1.1).
- M. Dennis, N. Jaques, E. Vinitsky, A. Bayen, S. Russell, A. Critch, and S. Levine (2020)
  Emergent complexity and zero-shot transfer via unsupervised environment design.
  Advances in neural information processing systems 33, pp. 13049–13061.
  Cited by: [Appendix B](#A2.p1.1).
- A. Dharna, C. Lu, and J. Clune (2024)
  Quality-Diversity Self-Play: Open-Ended Strategy Innovation via Foundation Models.
  In NeurIPS 2024 Workshop on Open-World Agents,
  Cited by: [Appendix B](#A2.p1.1).
- L. Ding, J. Zhang, J. Clune, L. Spector, and J. Lehman (2024)
  Quality diversity through human feedback: towards open-ended diversity-driven optimization.
  In Proceedings of the 41st International Conference on Machine Learning,
  pp. 11072–11090.
  Cited by: [Appendix B](#A2.p1.1).
- T. Dobzhansky (1970)
  Genetics of the evolutionary process.
  Vol. 139, Columbia University Press.
  Cited by: [Appendix B](#A2.p3.1).
- A. Ecoffet, J. Clune, and J. Lehman (2020)
  Open questions in creating safe open-ended AI: tensions between control and creativity.
  In Artificial Life Conference Proceedings 32,
  pp. 27–35.
  Cited by: [Appendix I](#A9.p1.1),
  [§5](#S5.p4.1).
- A. Ecoffet, J. Huizinga, J. Lehman, K. O. Stanley, and J. Clune (2019)
  Go-explore: a new approach for hard-exploration problems.
  arXiv preprint arXiv:1901.10995.
  Cited by: [Appendix B](#A2.p1.1),
  [§C.2](#A3.SS2.p1.1),
  [§2](#S2.p1.1).
- A. Ecoffet, J. Huizinga, J. Lehman, K. O. Stanley, and J. Clune (2021)
  First return, then explore.
  Nature 590 (7847), pp. 580–586.
  Cited by: [Appendix B](#A2.p1.1).
- A. W. F. Edwards (2000)
  The genetical theory of natural selection.
  Genetics 154 (4), pp. 1419–1426.
  Cited by: [Appendix B](#A2.p3.1).
- K. Ellis, C. Wong, M. Nye, M. Sablé-Meyer, L. Morales, L. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum (2021)
  Dreamcoder: bootstrapping inductive program synthesis with wake-sleep library learning.
  In Proceedings of the 42nd acm sigplan international conference on programming language design and implementation,
  pp. 835–850.
  Cited by: [Appendix B](#A2.p2.1).
- B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine (2018)
  Diversity is all you need: Learning skills without a reward function.
  arXiv preprint arXiv:1802.06070.
  Cited by: [Appendix B](#A2.p1.1).
- M. Faldor, J. Zhang, A. Cully, and J. Clune (2025)
  OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code.
  In The Thirteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=Y1XkzMJpPd)
  Cited by: [Appendix J](#A10.p2.1),
  [Appendix B](#A2.p1.1),
  [Appendix H](#A8.p8.1),
  [§1](#S1.p3.1),
  [§3](#S3.p1.1),
  [§6](#S6.p3.1).
- C. Fernando, D. S. Banarse, H. Michalewski, S. Osindero, and T. Rocktäschel (2024)
  Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution.
  In Forty-first International Conference on Machine Learning,
  Cited by: [Appendix B](#A2.p1.1),
  [§1](#S1.p3.1),
  [§2](#S2.p2.1),
  [§4.3](#S4.SS3.p1.1).
- D. Ganguli, L. Lovitt, J. Kernion, A. Askell, Y. Bai, S. Kadavath, B. Mann, E. Perez, N. Schiefer, K. Ndousse, et al. (2022)
  Red teaming language models to reduce harms: methods, scaling behaviors, and lessons learned.
  arXiv preprint arXiv:2209.07858.
  Cited by: [§5](#S5.p1.1).
- H. Gao, Y. Liu, Y. He, L. Dou, C. Du, Z. Deng, B. Hooi, M. Lin, and T. Pang (2025)
  FlowReasoner: reinforcing query-level meta-agents.
  arXiv preprint arXiv:2504.15257.
  Cited by: [§2](#S2.p2.1).
- P. Gauthier (2024)
  Aider: ai pair programming in your terminal.
  GitHub.
  Note: [https://github.com/Aider-AI/aider](https://github.com/Aider-AI/aider)Accessed: 2025-05-14
  Cited by: [§E.5](#A5.SS5.p1.1),
  [§4.2](#S4.SS2.p1.1),
  [§4.2](#S4.SS2.p3.1),
  [§4.3](#S4.SS3.p2.1).
- L. Gaven, T. Carta, C. Romac, C. Colas, S. Lamprier, O. Sigaud, and P. Oudeyer (2025)
  MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces.
  arXiv preprint arXiv:2502.07709.
  Cited by: [Appendix B](#A2.p1.1).
- J. Gerhart and M. Kirschner (2007)
  The theory of facilitated variation.
  Proceedings of the National Academy of Sciences 104 (suppl_1), pp. 8582–8589.
  Cited by: [§2](#S2.p1.1).
- I. J. Good (1966)
  Speculations concerning the first ultraintelligent machine.
  In Advances in computers,
  Vol. 6, pp. 31–88.
  Cited by: [§2](#S2.p3.1).
- Google DeepMind (2025)
  Gemini model “thinking” updates — march 2025.
  Note: [https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking)Accessed: 2025-05-11
  Cited by: [§4.2](#S4.SS2.p1.1).
- R. Greenblatt, C. Denison, B. Wright, F. Roger, M. MacDiarmid, S. Marks, J. Treutlein, T. Belonax, J. Chen, D. Duvenaud, et al. (2024)
  Alignment faking in large language models.
  arXiv preprint arXiv:2412.14093.
  Cited by: [§5](#S5.p1.1).
- S. Gulwani (2011)
  Automating string processing in spreadsheets using input-output examples.
  ACM Sigplan Notices 46 (1), pp. 317–330.
  Cited by: [Appendix B](#A2.p2.1).
- D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025)
  Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning.
  arXiv preprint arXiv:2501.12948.
  Cited by: [§2](#S2.p2.1).
- J. S. Hall (2007)
  Self-improving AI: An analysis.
  Minds and Machines 17 (3), pp. 249–259.
  Cited by: [§2](#S2.p3.1).
- A. Havrilla, A. Dai, L. O’Mahony, K. Oostermeijer, V. Zisler, A. Albalak, F. Milo, S. C. Raparthy, K. Gandhi, B. Abbasi, et al. (2024a)
  Surveying the effects of quality, diversity, and complexity in synthetic data from large language models.
  arXiv preprint arXiv:2412.02980.
  Cited by: [Appendix B](#A2.p1.1).
- A. Havrilla, S. Raparthy, C. Nalmpantis, J. Dwivedi-Yu, M. Zhuravinskyi, E. Hambro, and R. Raileanu (2024b)
  Glore: When, where, and how to improve llm reasoning via global and local refinements.
  arXiv preprint arXiv:2402.10963.
  Cited by: [§2](#S2.p3.1).
- J. L. Hendrikse, T. E. Parsons, and B. Hallgrímsson (2007)
  Evolvability as the proper focus of evolutionary developmental biology.
  Evolution & development 9 (4), pp. 393–401.
  Cited by: [§2](#S2.p1.1).
- N. Herr, T. Rocktäschel, and R. Raileanu (2025)
  LLM-first search: self-guided exploration of the solution space.
  arXiv preprint arXiv:2506.05213.
  Cited by: [Appendix J](#A10.p2.1),
  [§C.2](#A3.SS2.p1.1).
- M. Hobbhahn (2025)
  SWE-bench verified mini.
  Note: [https://github.com/mariushobbhahn/SWEBench-verified-mini](https://github.com/mariushobbhahn/SWEBench-verified-mini)Accessed: 2025-04-16
  Cited by: [§2](#S2.p3.1),
  [§4.2](#S4.SS2.p4.1).
- J. J. Hopfield (1982)
  Neural networks and physical systems with emergent collective computational abilities..
  Proceedings of the national academy of sciences 79 (8), pp. 2554–2558.
  Cited by: [§1](#S1.p1.1).
- S. Hu and J. Clune (2024)
  Thought Cloning: learning to think while acting by imitating human thinking.
  Advances in Neural Information Processing Systems 36.
  Cited by: [§2](#S2.p2.1).
- S. Hu, C. Lu, and J. Clune (2025)
  Automated Design of Agentic Systems.
  In The Thirteenth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=t9U3LW7JVX)
  Cited by: [Appendix B](#A2.p1.1),
  [§2](#S2.p1.1),
  [§2](#S2.p2.1),
  [§3](#S3.p2.1),
  [§4.3](#S4.SS3.p1.1).
- Y. Hu, Y. Cai, Y. Du, X. Zhu, X. Liu, Z. Yu, Y. Hou, S. Tang, and S. Chen (2024)
  Self-evolving multi-agent collaboration networks for software development.
  arXiv preprint arXiv:2410.16946.
  Cited by: [§2](#S2.p3.1).
- J. Huang, S. S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han (2022)
  Large language models can self-improve.
  arXiv preprint arXiv:2210.11610.
  Cited by: [§2](#S2.p3.1).
- E. Hughes, M. Dennis, J. Parker-Holder, F. Behbahani, A. Mavalankar, Y. Shi, T. Schaul, and T. Rocktaschel (2024)
  Open-endedness is essential for artificial superhuman intelligence.
  arXiv preprint arXiv:2406.04268.
  Cited by: [§2](#S2.p1.1).
- K. Irie, R. Csordás, and J. Schmidhuber (2025)
  Metalearning continual learning algorithms.
  Transactions on Machine Learning Research.
  Cited by: [§2](#S2.p3.1).
- K. Irie, I. Schlag, R. Csordás, and J. Schmidhuber (2022)
  A modern self-referential weight matrix that learns to modify itself.
  In International Conference on Machine Learning,
  pp. 9660–9677.
  Cited by: [§2](#S2.p3.1).
- M. Jaderberg, V. Dalibard, S. Osindero, W. M. Czarnecki, J. Donahue, A. Razavi, O. Vinyals, T. Green, I. Dunning, K. Simonyan, et al. (2017)
  Population based training of neural networks.
  arXiv preprint arXiv:1711.09846.
  Cited by: [§2](#S2.p3.1).
- M. Jiang, E. Grefenstette, and T. Rocktäschel (2021)
  Prioritized level replay.
  In International Conference on Machine Learning,
  pp. 4940–4950.
  Cited by: [Appendix B](#A2.p1.1).
- M. Jiang, T. Rocktäschel, and E. Grefenstette (2023)
  General intelligence requires rethinking exploration.
  Royal Society Open Science 10 (6), pp. 230539.
  Cited by: [§2](#S2.p1.1).
- C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan (2024)
  SWE-bench: Can Language Models Resolve Real-world Github Issues?.
  In The Twelfth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=VTF8yNQM66)
  Cited by: [§1](#S1.p4.1),
  [§4.2](#S4.SS2.p2.1),
  [§4](#S4.p1.1).
- I. Kanitscheider, J. Huizinga, D. Farhi, W. H. Guss, B. Houghton, R. Sampedro, P. Zhokhov, B. Baker, A. Ecoffet, J. Tang, O. Klimov, and J. Clune (2021)
  Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft.
  arXiv preprint arXiv:2106.14876.
  Cited by: [Appendix B](#A2.p1.1).
- A. Khan, J. Hughes, D. Valentine, L. Ruis, K. Sachan, A. Radhakrishnan, E. Grefenstette, S. R. Bowman, T. Rocktäschel, and E. Perez (2024)
  Debating with more persuasive llms leads to more truthful answers.
  arXiv preprint arXiv:2402.06782.
  Cited by: [Appendix B](#A2.p1.1),
  [§2](#S2.p2.1).
- O. Khattab, A. Singhvi, P. Maheshwari, Z. Zhang, K. Santhanam, S. Vardhamanan, S. Haq, A. Sharma, T. T. Joshi, H. Moazam, et al. (2023)
  Dspy: Compiling declarative language model calls into self-improving pipelines.
  arXiv preprint arXiv:2310.03714.
  Cited by: [§2](#S2.p2.1).
- Y. Kim, C. Denton, L. Hoang, and A. M. Rush (2017)
  Structured Attention Networks.
  In International Conference on Learning Representations,
  Cited by: [§1](#S1.p1.1).
- M. Kimura (1979)
  The neutral theory of molecular evolution.
  Scientific American 241 (5), pp. 98–129.
  Cited by: [Appendix B](#A2.p3.1).
- L. Kirsch and J. Schmidhuber (2022)
  Self-referential meta learning.
  In First Conference on Automated Machine Learning (Late-Breaking Workshop),
  Cited by: [§2](#S2.p3.1).
- M. Klissarov, P. D’Oro, S. Sodhani, R. Raileanu, P. Bacon, P. Vincent, A. Zhang, and M. Henaff (2023)
  Motif: Intrinsic motivation from artificial intelligence feedback.
  arXiv preprint arXiv:2310.00166.
  Cited by: [Appendix B](#A2.p1.1).
- M. Klissarov, M. Henaff, R. Raileanu, S. Sodhani, P. Vincent, A. Zhang, P. Bacon, D. Precup, M. C. Machado, and P. D’Oro (2024)
  MaestroMotif: Skill Design from Artificial Intelligence Feedback.
  arXiv preprint arXiv:2412.08542.
  Cited by: [Appendix B](#A2.p1.1).
- V. R. Kompella, M. Stollenga, M. Luciw, and J. Schmidhuber (2017)
  Continual curiosity-driven skill acquisition from high-dimensional video inputs for humanoid robots.
  Artificial Intelligence 247, pp. 313–335.
  Cited by: [Appendix B](#A2.p1.1).
- R. Lange, T. Schaul, Y. Chen, T. Zahavy, V. Dalibard, C. Lu, S. Singh, and S. Flennerhag (2023)
  Discovering evolution strategies via meta-black-box optimization.
  In Proceedings of the Companion Conference on Genetic and Evolutionary Computation,
  pp. 29–30.
  Cited by: [§2](#S2.p3.1).
- R. Lange, Y. Tian, and Y. Tang (2024)
  Large language models as evolution strategies.
  In Proceedings of the Genetic and Evolutionary Computation Conference Companion,
  pp. 579–582.
  Cited by: [Appendix B](#A2.p1.1).
- J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter (2020)
  Learning quadrupedal locomotion over challenging terrain.
  Science robotics 5 (47), pp. eabc5986.
  Cited by: [§4.3](#S4.SS3.p1.1).
- J. Lehman, J. Gordon, S. Jain, K. Ndousse, C. Yeh, and K. O. Stanley (2023)
  Evolution through large models.
  In Handbook of Evolutionary Machine Learning,
  pp. 331–366.
  Cited by: [Appendix B](#A2.p1.1).
- J. Lehman and K. O. Stanley (2011)
  Novelty search and the problem with objectives.
  Genetic programming theory and practice IX, pp. 37–56.
  Cited by: [Appendix B](#A2.p1.1),
  [§2](#S2.p1.1).
- J. Lehman (2023)
  Machine love.
  arXiv preprint arXiv:2302.09248.
  Cited by: [§5](#S5.p1.1).
- P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, et al. (2020)
  Retrieval-augmented generation for knowledge-intensive nlp tasks.
  Advances in neural information processing systems 33, pp. 9459–9474.
  Cited by: [§2](#S2.p2.1).
- J. Li, S. J., and J. Clune (2014)
  Encouraging creative thinking in robots improves their ability to solve challenging problems.
  In Proceedings of the Genetic and Evolutionary Computation Conference,
  pp. 193–200.
  Cited by: [Appendix B](#A2.p1.1).
- Y. Li, J. Parsert, and E. Polgreen (2024)
  Guiding enumerative program synthesis with large language models.
  In International Conference on Computer Aided Verification,
  pp. 280–301.
  Cited by: [Appendix B](#A2.p2.1).
- T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, S. Shi, and Z. Tu (2023)
  Encouraging divergent thinking in large language models through multi-agent debate.
  arXiv preprint arXiv:2305.19118.
  Cited by: [§2](#S2.p2.1).
- H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe (2023)
  Let’s verify step by step.
  In The Twelfth International Conference on Learning Representations,
  Cited by: [§2](#S2.p2.1).
- B. Lim, M. Flageat, and A. Cully (2024)
  Large language models as in-context ai generators for quality-diversity.
  In ALIFE 2024: Proceedings of the 2024 Artificial Life Conference,
  Cited by: [Appendix B](#A2.p1.1).
- S. Linnainmaa (1970)
  The representation of the cumulative rounding error of an algorithm as a taylor expansion of the local rounding errors.
  Ph.D. Thesis, Master’s Thesis (in Finnish), Univ. Helsinki.
  Cited by: [§1](#S1.p1.1).
- F. Liu, X. Tong, M. Yuan, X. Lin, F. Luo, Z. Wang, Z. Lu, and Q. Zhang (2024)
  Evolution of heuristics: towards efficient automatic algorithm design using large language model.
  arXiv preprint arXiv:2401.02051.
  Cited by: [Appendix B](#A2.p1.1).
- L. Liu, X. Yang, Y. Shen, B. Hu, Z. Zhang, J. Gu, and G. Zhang (2023)
  Think-in-memory: Recalling and post-thinking enable llms with long-term memory.
  arXiv preprint arXiv:2311.08719.
  Cited by: [§2](#S2.p2.1).
- C. Lu, C. Lu, R. T. Lange, J. Foerster, J. Clune, and D. Ha (2024a)
  The ai scientist: Towards fully automated open-ended scientific discovery.
  arXiv preprint arXiv:2408.06292.
  Cited by: [Appendix B](#A2.p1.1).
- C. Lu, S. Towers, and J. Foerster (2023)
  Arbitrary order meta-learning with simple population-based evolution.
  In Artificial Life Conference Proceedings 35,
  Vol. 2023, pp. 67.
  Cited by: [§2](#S2.p3.1).
- C. Lu, S. Hu, and J. Clune (2024b)
  Intelligent go-explore: standing on the shoulders of giant foundation models.
  arXiv preprint arXiv:2405.15143.
  Cited by: [Appendix B](#A2.p1.1).
- C. Lu, S. Hu, and J. Clune (2025)
  Automated capability discovery via model self-exploration.
  arXiv preprint arXiv:2502.0757.
  Cited by: [Appendix B](#A2.p1.1).
- Y. J. Ma, W. Liang, G. Wang, D. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar (2023)
  Eureka: Human-level reward design via coding large language models.
  arXiv preprint arXiv:2310.12931.
  Cited by: [Appendix B](#A2.p1.1).
- A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. (2023)
  Self-refine: Iterative refinement with self-feedback, 2023.
  URL https://arxiv. org/abs/2303.17651.
  Cited by: [§2](#S2.p2.1).
- J. Markoff (2016)
  Machines of loving grace: the quest for common ground between humans and robots.
  HarperCollins Publishers.
  Cited by: [§5](#S5.p1.1).
- E. Mayr (1982)
  The growth of biological thought: diversity, evolution, and inheritance.
  Harvard University Press.
  Cited by: [Appendix B](#A2.p3.1).
- L. Metz, C. D. Freeman, N. Maheswaranathan, and J. Sohl-Dickstein (2021)
  Training learned optimizers with randomly initialized learned optimizers.
  arXiv preprint arXiv:2101.07367.
  Cited by: [§2](#S2.p3.1).
- A. Modarressi, A. Imani, M. Fayyaz, and H. Schütze (2023)
  Ret-llm: Towards a general read-write memory for large language models.
  arXiv preprint arXiv:2305.14322.
  Cited by: [§2](#S2.p2.1).
- J. Mouret and J. Clune (2015)
  Illuminating search spaces by mapping elites.
  arXiv preprint arXiv:1504.04909.
  Cited by: [Appendix B](#A2.p1.1).
- N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès, and T. Hashimoto (2025)
  S1: simple test-time scaling.
  arXiv preprint arXiv:2501.19393.
  Cited by: [§2](#S2.p2.1).
- M. U. Nasir, S. James, and J. Togelius (2024)
  Word2world: Generating stories and worlds through large language models.
  arXiv preprint arXiv:2405.06686.
  Cited by: [Appendix B](#A2.p1.1).
- M. U. Nasir and J. Togelius (2023)
  Practical PCG through large language models.
  In 2023 IEEE Conference on Games (CoG),
  pp. 1–4.
  Cited by: [Appendix B](#A2.p1.1).
- A. M. Nguyen, J. Yosinski, and J. Clune (2015)
  Innovation engines: automated creativity and improved stochastic optimization via deep learning.
  In Proceedings of the 2015 annual conference on genetic and evolutionary computation,
  pp. 959–966.
  Cited by: [Appendix B](#A2.p1.1).
- F. Nie, L. Feng, H. Ye, W. Liang, P. Lu, H. Yao, A. Alahi, and J. Zou (2025)
  Weak-for-strong: training weak meta-agent to harness strong executors.
  arXiv preprint arXiv:2504.04785.
  Cited by: [§2](#S2.p2.1).
- B. Niu, Y. Song, K. Lian, Y. Shen, Y. Yao, K. Zhang, and T. Liu (2025)
  Flow: modularized agentic workflow automation.
  In The Thirteenth International Conference on Learning Representations,
  Cited by: [§2](#S2.p2.1).
- A. Novikov, N. Vũ, M. Eisenberger, E. Dupont, P. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog (2025)
  AlphaEvolve: a coding agent for scientific and algorithmic discovery.
  Technical report
  Google DeepMind.
  Cited by: [Appendix B](#A2.p1.1).
- OpenAI (2024)
  Introducing swe-bench verified.
  Note: [https://openai.com/index/introducing-swe-bench-verified/](https://openai.com/index/introducing-swe-bench-verified/)Accessed: 2025-04-16
  Cited by: [§4.2](#S4.SS2.p2.1).
- OpenAI (2025)
  OpenAI o3-mini.
  Note: [https://openai.com/index/openai-o3-mini/](https://openai.com/index/openai-o3-mini/)Accessed: 2025-05-01
  Cited by: [§4.1](#S4.SS1.p1.1).
- P. Oudeyer, F. Kaplan, and V. V. Hafner (2007)
  Intrinsic motivation systems for autonomous mental development.
  IEEE transactions on evolutionary computation 11 (2), pp. 265–286.
  Cited by: [Appendix B](#A2.p1.1),
  [§2](#S2.p1.1).
- L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. (2022)
  Training language models to follow instructions with human feedback.
  Advances in neural information processing systems 35, pp. 27730–27744.
  Cited by: [Appendix J](#A10.p3.1).
- A. Parikh, O. Täckström, D. Das, and J. Uszkoreit (2016)
  A Decomposable Attention Model for Natural Language Inference.
  In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing,
  pp. 2249–2255.
  Cited by: [§1](#S1.p1.1).
- J. Parker-Holder, P. Ball, J. Bruce, V. Dasagi, K. Holsheimer, C. Kaplanis, A. Moufarek, G. Scully, J. Shar, J. Shi, S. Spencer, J. Yung, M. Dennis, S. Kenjeyev, S. Long, V. Mnih, H. Chan, M. Gazeau, B. Li, F. Pardo, L. Wang, L. Zhang, F. Besse, T. Harley, A. Mitenkova, J. Wang, J. Clune, D. Hassabis, R. Hadsell, A. Bolton, S. Singh, and T. Rocktäschel (2024)
  Genie 2: a large-scale foundation world model.
  External Links: [Link](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
  Cited by: [Appendix B](#A2.p1.1).
- D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell (2017)
  Curiosity-driven exploration by self-supervised prediction.
  In International conference on machine learning,
  pp. 2778–2787.
  Cited by: [Appendix B](#A2.p1.1).
- Paul Gauthier (2024)
  O1 tops aider’s new polyglot leaderboard.
  Note: [https://aider.chat/2024/12/21/polyglot.html](https://aider.chat/2024/12/21/polyglot.html)Accessed: 2025-04-16
  Cited by: [§1](#S1.p4.1),
  [§4.2](#S4.SS2.p3.1),
  [§4](#S4.p1.1).
- O. Polozov and S. Gulwani (2015)
  Flashmeta: a framework for inductive program synthesis.
  In Proceedings of the 2015 ACM SIGPLAN International Conference on Object-Oriented Programming, Systems, Languages, and Applications,
  pp. 107–126.
  Cited by: [Appendix B](#A2.p2.1).
- J. K. Pugh, L. B. Soros, and K. O. Stanley (2016)
  Quality diversity: A new frontier for evolutionary computation.
  Frontiers in Robotics and AI 3, pp. 40.
  Cited by: [Appendix B](#A2.p1.1),
  [§2](#S2.p1.1).
- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. (2019)
  Language models are unsupervised multitask learners.
  OpenAI blog 1 (8), pp. 9.
  Cited by: [Appendix B](#A2.p1.1).
- M. Robeyns, M. Szummer, and L. Aitchison (2025)
  A Self-Improving Coding Agent.
  arXiv preprint arXiv:2504.15228.
  Cited by: [§A.3](#A1.SS3.p1.1),
  [§2](#S2.p3.1),
  [§3](#S3.p3.1).
- B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, et al. (2024)
  Mathematical discoveries from program search with large language models.
  Nature 625 (7995), pp. 468–475.
  Cited by: [Appendix B](#A2.p1.1).
- J. Rosser and J. N. Foerster (2025)
  AgentBreeder: mitigating the AI safety impact of multi-agent scaffolds via self-improvement.
  In Scaling Self-Improving Foundation Models without Human Supervision,
  External Links: [Link](https://openreview.net/forum?id=j0n3BJJTcT)
  Cited by: [§2](#S2.p2.1),
  [§5](#S5.p3.1).
- D. E. Rumelhart, G. E. Hinton, R. J. Williams, et al. (1985)
  Learning internal representations by error propagation.
  Institute for Cognitive Science, University of California, San Diego.
  Cited by: [§1](#S1.p1.1).
- M. Samvelyan, S. C. Raparthy, A. Lupu, E. Hambro, A. Markosyan, M. Bhatt, Y. Mao, M. Jiang, J. Parker-Holder, J. Foerster, et al. (2024)
  Rainbow teaming: Open-ended generation of diverse adversarial prompts.
  Advances in Neural Information Processing Systems 37, pp. 69747–69786.
  Cited by: [Appendix J](#A10.p2.1),
  [Appendix B](#A2.p1.1),
  [§4.3](#S4.SS3.p1.1).
- C. Sancaktar, C. Gumbsch, A. Zadaianchuk, P. Kolev, and G. Martius (2025)
  SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models.
  arXiv preprint arXiv:2503.01584.
  Cited by: [Appendix B](#A2.p1.1).
- T. Schaul, D. Horgan, K. Gregor, and D. Silver (2015)
  Universal value function approximators.
  In International conference on machine learning,
  pp. 1312–1320.
  Cited by: [Appendix B](#A2.p1.1).
- T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom (2023)
  Toolformer: Language models can teach themselves to use tools.
  Advances in Neural Information Processing Systems 36, pp. 68539–68551.
  Cited by: [§4.1](#S4.SS1.p1.1).
- J. Schmidhuber and R. Huber (1990)
  Learning to generate focus trajectories for attentive vision.
  Institut für Informatik.
  Cited by: [§1](#S1.p1.1).
- J. Schmidhuber (1987)
  Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-… hook.
  Ph.D. Thesis, Technische Universität München.
  Cited by: [§2](#S2.p3.1),
  [§5](#S5.p1.1).
- J. Schmidhuber (1993)
  A ‘self-referential’weight matrix.
  In International conference on artificial neural networks,
  pp. 446–450.
  Cited by: [§2](#S2.p3.1).
- J. Schmidhuber (2007)
  Gödel machines: fully self-referential optimal universal self-improvers.
  In Artificial general intelligence,
  pp. 199–226.
  Cited by: [§1](#S1.p2.1),
  [§2](#S2.p3.1),
  [§3](#S3.p1.1).
- J. Schmidhuber (2008)
  Driven by compression progress: a simple principle explains essential aspects of subjective beauty, novelty, surprise, interestingness, attention, curiosity, creativity, art, science, music, jokes.
  In Workshop on anticipatory behavior in adaptive learning systems,
  pp. 48–76.
  Cited by: [Appendix B](#A2.p1.1).
- J. Schmidhuber (2013)
  Powerplay: training an increasingly general problem solver by continually searching for the simplest still unsolvable problem.
  Frontiers in psychology 4, pp. 313.
  Cited by: [Appendix B](#A2.p1.1).
- S. Schulhoff, M. Ilie, N. Balepur, K. Kahadze, A. Liu, C. Si, Y. Li, A. Gupta, H. Han, S. Schulhoff, et al. (2024)
  The prompt report: a systematic survey of prompting techniques.
  arXiv preprint arXiv:2406.06608.
  Cited by: [§2](#S2.p2.1).
- I. Sheth, J. Wehner, S. Abdelnabi, R. Binkyte, and M. Fritz (2025)
  Safety is Essential for Responsible Open-Ended Systems.
  arXiv preprint arXiv:2502.04512.
  Cited by: [§5](#S5.p1.1).
- K. Shi, H. Dai, W. Li, K. Ellis, and C. Sutton (2023)
  Lambdabeam: neural program search with higher-order functions and lambdas.
  Advances in Neural Information Processing Systems 36, pp. 51327–51346.
  Cited by: [Appendix B](#A2.p2.1).
- N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao (2023)
  Reflexion: Language agents with verbal reinforcement learning.
  Advances in Neural Information Processing Systems 36, pp. 8634–8652.
  Cited by: [§2](#S2.p2.1).
- D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. (2016)
  Mastering the game of go with deep neural networks and tree search.
  nature 529 (7587), pp. 484–489.
  Cited by: [§C.2](#A3.SS2.p1.1).
- D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, et al. (2017)
  Mastering chess and shogi by self-play with a general reinforcement learning algorithm.
  arXiv preprint arXiv:1712.01815.
  Cited by: [§2](#S2.p3.1).
- A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, X. Garcia, P. J. Liu, J. Harrison, J. Lee, K. Xu, et al. (2023)
  Beyond human data: scaling self-training for problem-solving with language models.
  arXiv preprint arXiv:2312.06585.
  Cited by: [§2](#S2.p3.1).
- J. Skalse, N. Howe, D. Krasheninnikov, and D. Krueger (2022)
  Defining and characterizing reward gaming.
  Advances in Neural Information Processing Systems 35, pp. 9460–9471.
  Cited by: [Appendix H](#A8.p8.1).
- K. O. Stanley, J. Lehman, and L. Soros (2017)
  Open-endedness: The last grand challenge you’ve never heard of.
  While open-endedness could be a force for discovering intelligence, it could also be a component of AI itself.
  Cited by: [§2](#S2.p1.1).
- K. O. Stanley and J. Lehman (2015)
  Why greatness cannot be planned: The myth of the objective.
  Springer.
  Cited by: [§3](#S3.p1.1),
  [§4.3](#S4.SS3.p1.1).
- M. Strathern (1997)
  ‘Improving ratings’: audit in the British University system.
  European review 5 (3), pp. 305–321.
  Cited by: [Appendix H](#A8.p8.1).
- J. Su, Y. Xia, R. Shi, J. Wang, J. Huang, Y. Wang, T. Shi, Y. Jingsong, and L. He (2025)
  DebFlow: automating agent creation via agent debate.
  arXiv preprint arXiv:2503.23781.
  Cited by: [§2](#S2.p2.1).
- S. Sudhakaran, M. González-Duque, M. Freiberger, C. Glanois, E. Najarro, and S. Risi (2023)
  Mariogpt: Open-ended text2level generation through large language models.
  Advances in Neural Information Processing Systems 36, pp. 54213–54227.
  Cited by: [Appendix B](#A2.p1.1).
- O. Team, A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. (2024)
  Openai o1 system card.
  arXiv preprint arXiv:2412.16720.
  Cited by: [§C.3](#A3.SS3.p1.1).
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017)
  Attention is all you need.
  Advances in neural information processing systems 30.
  Cited by: [§1](#S1.p1.1).
- G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar (2023a)
  Voyager: An open-ended embodied agent with large language models.
  arXiv preprint arXiv:2305.16291.
  Cited by: [Appendix B](#A2.p1.1).
- R. Wang, K. Xue, Y. Wang, P. Yang, H. Fu, Q. Fu, and C. Qian (2023b)
  Diversity from human feedback.
  arXiv preprint arXiv:2310.06648.
  Cited by: [Appendix B](#A2.p1.1).
- R. Wang, J. Lehman, J. Clune, and K. O. Stanley (2019)
  Paired open-ended trailblazer (poet): Endlessly generating increasingly complex and diverse learning environments and their solutions.
  arXiv preprint arXiv:1901.01753.
  Cited by: [§1](#S1.p3.1).
- X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, et al. (2024)
  Openhands: An open platform for ai software developers as generalist agents.
  In The Thirteenth International Conference on Learning Representations,
  Cited by: [§E.4](#A5.SS4.p1.1).
- Y. Wang, Z. Xian, F. Chen, T. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan (2023c)
  Robogen: towards unleashing infinite data for automated robot learning via generative simulation.
  arXiv preprint arXiv:2311.01455.
  Cited by: [§6](#S6.p3.1).
- J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022)
  Chain-of-thought prompting elicits reasoning in large language models.
  Advances in neural information processing systems 35, pp. 24824–24837.
  Cited by: [§2](#S2.p2.1).
- M. Wiering and J. Schmidhuber (1997)
  HQ-learning.
  Adaptive behavior 6 (2), pp. 219–246.
  Cited by: [Appendix B](#A2.p1.1).
- S. Wright (1932)
  The roles of mutation, inbreeding, crossbreeding and selection in evolution, proceedings of the sixth international congress of genetics. proc sixth int congr genet [internet].
  New York356366.
  Cited by: [Appendix B](#A2.p3.1).
- C. S. Xia, Y. Deng, S. Dunn, and L. Zhang (2024)
  Agentless: demystifying llm-based software engineering agents.
  arXiv preprint arXiv:2407.01489.
  Cited by: [§4.2](#S4.SS2.p1.1).
- J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. R. Narasimhan, and O. Press (2024)
  SWE-agent: agent-computer interfaces enable automated software engineering.
  In The Thirty-eighth Annual Conference on Neural Information Processing Systems,
  External Links: [Link](https://arxiv.org/abs/2405.15793)
  Cited by: [Appendix J](#A10.p4.1).
- S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao (2023)
  React: Synergizing reasoning and acting in language models.
  In International Conference on Learning Representations (ICLR),
  Cited by: [§2](#S2.p2.1).
- R. Ye, S. Tang, R. Ge, Y. Du, Z. Yin, S. Chen, and J. Shao (2025)
  Mas-gpt: training llms to build llm-based multi-agent systems.
  arXiv preprint arXiv:2503.03686.
  Cited by: [§2](#S2.p2.1).
- X. Yin, X. Wang, L. Pan, X. Wan, and W. Y. Wang (2024)
  G$\backslash$" odel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement.
  arXiv preprint arXiv:2410.04444.
  Cited by: [§2](#S2.p2.1),
  [§2](#S2.p3.1).
- S. Yuan, K. Song, J. Chen, X. Tan, D. Li, and D. Yang (2024)
  EvoAgent: towards automatic multi-agent generation via evolutionary algorithms.
  arXiv preprint arXiv:2406.14228.
  Cited by: [§2](#S2.p2.1).
- E. Yudkowsky et al. (2008)
  Artificial Intelligence as a positive and negative factor in global risk.
  Global catastrophic risks 1 (303), pp. 184.
  Cited by: [§5](#S5.p4.1).
- M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, Z. Huang, C. Guestrin, and J. Zou (2024)
  Textgrad: automatic" differentiation" via text.
  arXiv preprint arXiv:2406.07496.
  Cited by: [§2](#S2.p2.1).
- E. Zelikman, G. Harik, Y. Shao, V. Jayasiri, N. Haber, and N. D. Goodman (2024a)
  Quiet-star: language models can teach themselves to think before speaking.
  arXiv preprint arXiv:2403.09629.
  Cited by: [§2](#S2.p2.1).
- E. Zelikman, E. Lorch, L. Mackey, and A. T. Kalai (2024b)
  Self-taught optimizer (stop): Recursively self-improving code generation.
  In First Conference on Language Modeling,
  Cited by: [§2](#S2.p3.1).
- D. Zhang, S. Zhoubian, Z. Hu, Y. Yue, Y. Dong, and J. Tang (2024a)
  Rest-mcts\*: llm self-training via process reward guided tree search.
  Advances in Neural Information Processing Systems 37, pp. 64735–64772.
  Cited by: [§4.2](#S4.SS2.p1.1).
- G. Zhang, L. Niu, J. Fang, K. Wang, L. Bai, and X. Wang (2025a)
  Multi-agent architecture search via agentic supernet.
  arXiv preprint arXiv:2502.04180.
  Cited by: [§2](#S2.p2.1).
- J. Zhang, J. Lehman, K. Stanley, and J. Clune (2024b)
  OMNI: Open-endedness via Models of human Notions of Interestingness.
  In The Twelfth International Conference on Learning Representations,
  External Links: [Link](https://openreview.net/forum?id=AgM3MzT99c)
  Cited by: [Appendix B](#A2.p1.1),
  [Appendix H](#A8.p8.1),
  [§2](#S2.p1.1),
  [§3](#S3.p1.1),
  [§4.3](#S4.SS3.p1.1).
- J. Zhang, J. Xiang, Z. Yu, F. Teng, X. Chen, J. Chen, M. Zhuge, X. Cheng, S. Hong, J. Wang, et al. (2024c)
  Aflow: Automating agentic workflow generation.
  arXiv preprint arXiv:2410.10762.
  Cited by: [§2](#S2.p2.1),
  [§3](#S3.p2.1).
- Y. Zhang, Y. Hou, B. Tang, S. Chen, M. Zhang, X. Dong, and S. Chen (2025b)
  GNNs as predictors of agentic workflow performances.
  arXiv preprint arXiv:2503.11301.
  Cited by: [§2](#S2.p2.1).
- Y. Zhang, H. Ruan, Z. Fan, and A. Roychoudhury (2024d)
  Autocoderover: autonomous program improvement.
  In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis,
  pp. 1592–1604.
  Cited by: [§4.2](#S4.SS2.p1.1).
- W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang (2024)
  Memorybank: Enhancing large language models with long-term memory.
  In Proceedings of the AAAI Conference on Artificial Intelligence,
  Vol. 38, pp. 19724–19731.
  Cited by: [§2](#S2.p2.1).
- A. Zhou, K. Wu, F. Pinto, Z. Chen, Y. Zeng, Y. Yang, S. Yang, S. Koyejo, J. Zou, and B. Li (2025)
  AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration.
  arXiv preprint arXiv:2503.15754.
  Cited by: [Appendix B](#A2.p1.1).
- W. Zhou, Y. Ou, S. Ding, L. Li, J. Wu, T. Wang, J. Chen, S. Wang, X. Xu, N. Zhang, et al. (2024)
  Symbolic learning enables self-evolving agents.
  arXiv preprint arXiv:2406.18532.
  Cited by: [§2](#S2.p2.1).
- Y. Zhu, J. Li, G. Li, Y. Zhao, Z. Jin, and H. Mei (2024)
  Hot or cold? adaptive temperature sampling for code generation with large language models.
  In Proceedings of the AAAI Conference on Artificial Intelligence,
  Vol. 38, pp. 437–445.
  Cited by: [§2](#S2.p2.1).
- M. Zhuge, H. Liu, F. Faccio, D. R. Ashley, R. Csordás, A. Gopalakrishnan, A. Hamdi, H. A. A. K. Hammoud, V. Herrmann, K. Irie, L. Kirsch, B. Li, G. Li, S. Liu, J. Mai, P. Piékos, A. Ramesh, I. Schlag, W. Shi, A. Stanic, W. Wang, Y. Wang, M. Xu, D. Fan, B. Ghanem, and J. Schmidhuber (2023)
  Mindstorms in natural language-based societies of mind.
  arXiv preprint arXiv:2305.17066.
  Cited by: [§2](#S2.p2.1).
- M. Zhuge, W. Wang, L. Kirsch, F. Faccio, D. Khizbullin, and J. Schmidhuber (2024)
  GPTSwarm: language agents as optimizable graphs.
  In Forty-first International Conference on Machine Learning,
  Cited by: [§2](#S2.p2.1).

## Appendix

## Table of Contents

<a id="appendix-a"></a>

## Appendix A Additional Results

### A.1 Baselines on SWE-bench

<a id="figure-5"></a>

![dgm_progress](images/dgm_progress.png)

> Figure 5: DGM without self-improving agents. Keeping the meta-agent that is modifying and producing the next coding agents the same, DGM w/o self-improve is unable to continuously improve over time. (Left) Archive of coding agents generated during the DGM w/o self-improve run on SWE-bench. Each node represents a coding agent, with node 0 corresponding to the base agent. Node color indicates performance on SWE-bench (percentage of solved tasks), while border color reflects the number of tasks for which the agent was evaluated. Edges show which agents self-modified to produce the offsprings. (Right) Progress plot of the DGM w/o self-improve on SWE-bench. The light green line shows the average score of all agents possessing basic codebase-editing functionality. The green line tracks the best score achieved by any agent in the archive at each iteration. The dark line shows the lineage of the final best-discovered agent and its precursor nodes.

<a id="figure-6"></a>

![transfer_model_task](images/transfer_model_task.png)

> Figure 6: DGM without open-ended exploration. Removing the archive, DGM w/o open-ended exploration always uses the most recent agent to self-modify and makes very little progress on SWE-bench. (Left) Archive of coding agents generated during the DGM w/o open-ended exploration run on SWE-bench. Each node represents a coding agent, with node 0 corresponding to the base agent. Node color indicates performance on SWE-bench (percentage of solved tasks), while border color reflects the number of tasks for which the agent was evaluated. Edges show which agents self-modified to produce the offsprings. (Right) Progress plot of the DGM w/o open-ended on SWE-bench. The orange line shows the average score of all agents possessing basic codebase-editing functionality. The light orange line tracks the best score achieved by any agent in the archive at each iteration. The dark line shows the lineage of the final best-discovered agent and its precursor nodes.

### A.2 Generality across models on Polyglot

<a id="figure-7"></a>

![dgm_wo_selfimprove](images/dgm_wo_selfimprove.png)

> Figure 7: Transfer between Models on Polyglot

In addition to testing the transfer models on SWE-bench (see [Section˜4.4](#S4.SS4), [Figure˜2](#S4.F2)), we also present the transfer results on Polyglot in this section. On Polyglot ([Figure˜7](#A1.F7)), where the DGM was run with o3-mini, we replaced the FM with Claude 3.5 Sonnet (New) or Claude 3.7 Sonnet, and evaluated on the full benchmark ([Figure˜4](#S4.F4), Middle). With Claude 3.5 Sonnet (New), the initial agent achieved 32.0% and the DGM-discovered agent 33.3%. With Claude 3.7 Sonnet, the initial agent achieved 35.6% and the DGM-discovered agent 36.8%. These results suggest that the DGM yields improvements that generalize across FMs, rather than being tightly coupled to the specific FM used during its run ([Figure˜4](#S4.F4)).

### A.3 Ablation of Parent Selection

To further study the impact of the parent selection mechanism in DGM, we introduce DGM Greedy. DGM Greedy always selects the best-performing node as the parent to branch from, rather than giving every node a non-zero probability of being branched off (roughly proportional to their performance score and number of children) as in this implementation of the DGM ([Section˜C.2](#A3.SS2)). This ablation replicates the approach of Robeyns et al. (2025) in this setting. As shown in [Table˜1](#A1.T1), DGM Greedy achieves 39.7% and 30.0% on SWE-bench and Polyglot, respectively, compared to 50.0% and 38.0% by this implementation of DGM. These results demonstrate that allowing all solutions in the archive to serve as potential stepping stones can lead to greater improvements over time, underscoring the importance of open-ended exploration.

<a id="table-1"></a>

> Table 1: Comparison of DGM, its ablations, and baselines on SWE-bench and Polyglot benchmarks.

| Method                         | SWE-bench | Polyglot |
| ------------------------------ | --------- | -------- |
| DGM                            | 50.0%     | 38.0%    |
| DGM w/o Open-ended exploration | 23.0%     | 14.0%    |
| DGM w/o Self-improve           | 39.0%     | 28.0%    |
| DGM Greedy                     | 39.7%     | 30.0%    |

### A.4 Additional Statistics of DGM runs

Percentage of Generated Agents with Basic Code-Editing Functionality.
To gain deeper insights into the DGM process, we analyze the percentage of generated agents that possess basic code-editing functionality on the SWE-bench benchmark. As shown in [Table˜2](#A1.T2), DGM exhibits the highest percentage of producing agents with basic codebase-editing functionality. These results highlight the effectiveness of both open-ended exploration and self-improvement components in the DGM, where open-ended exploration enables the search to escape local optima, while self-improvement enhances the ability to generate better agents.

<a id="table-2"></a>

> Table 2: Percentage of generated agents with basic code-editing functionality on SWE-bench.

| Method                         | Percentage with Basic Code-Editing Functionality |
| ------------------------------ | ------------------------------------------------ |
| DGM                            | 51.3%                                            |
| DGM w/o Open-ended exploration | 32.5%                                            |
| DGM w/o Self-improve           | 32.5%                                            |

Stability of DGM Runs.
To evaluate the stability of DGM, we run the DGM algorithm three times on the Polyglot benchmark and analyze the variance in performance. The DGM achieved a mean accuracy of 40.7% with a standard deviation of 2.3%, indicating that the DGM can achieve consistent and reproducible results across runs.

<a id="appendix-b"></a>

## Appendix B Additional Related Work

Open-Endedness (part 2).
Early approaches to open-endedness explored different mechanisms to balance learnability and interestingness. Quality-diversity algorithms sought to illuminate vast solution spaces with diverse, high-performing behaviors (Pugh et al., 2016; Chatzilygeroudis et al., 2021; Mouret and Clune, 2015; Nguyen et al., 2015). Other methods emphasized goal-directed exploration (Ecoffet et al., 2019; 2021; Schaul et al., 2015; Andrychowicz et al., 2017; Eysenbach et al., 2018), intrinsic motivation (Lehman and Stanley, 2011; Oudeyer et al., 2007; Li et al., 2014; Pathak et al., 2017), or learning progress frameworks (Kanitscheider et al., 2021; Gaven et al., 2025; Baranes and Oudeyer, 2013; Colas et al., 2019; 2022b; Jiang et al., 2021; Dennis et al., 2020; Schmidhuber, 2008; 2013; Kompella et al., 2017). More recently, large-scale foundation models (FMs) (Brown et al., 2020; Radford et al., 2019) have emerged as powerful proxies for human notions of interestingness (Zhang et al., 2024b; Faldor et al., 2025; Sancaktar et al., 2025) and effective mutation operators to propose novel solutions in code (Romera-Paredes et al., 2024; Novikov et al., 2025; Lehman et al., 2023; Faldor et al., 2025; Hu et al., 2025). FMs can guide autotelic agents (Colas et al., 2022b; 2023; a), model human preferences for quality and diversity (Bradley et al., 2024; Ding et al., 2024; Wang et al., 2023b; Klissarov et al., 2023; 2024; Samvelyan et al., 2024; Lim et al., 2024; Havrilla et al., 2024a), design reward functions (Wiering and Schmidhuber, 1997; Wang et al., 2023a; Ma et al., 2023; Faldor et al., 2025), create simulated environments (Sudhakaran et al., 2023; Nasir and Togelius, 2023; Aki et al., 2024; Nasir et al., 2024; Bruce et al., 2024; Parker-Holder et al., 2024; Schmidhuber, 2013), drive ever-evolving multi-agent dynamics (Dharna et al., 2024; Zhou et al., 2025), search diverse ambulating robot morphologies (Lehman et al., 2023), and search expansive solution spaces for benchmark or objective optimization (Lange et al., 2024; Zhang et al., 2024b; Faldor et al., 2025; Hu et al., 2025; Lu et al., 2024b; Romera-Paredes et al., 2024; Fernando et al., 2024; Lu et al., 2024a; Khan et al., 2024; Lu et al., 2025; Liu et al., 2024; Novikov et al., 2025).

Program Synthesis.
Program synthesis (Alur et al., 2018; Polozov and Gulwani, 2015; Buchi and Landweber, 1990; Gulwani, 2011; Ellis et al., 2021) seeks to generate code meeting external specifications such as input-output examples or logical formulas. Hybrid approaches combine symbolic methods with neural or FM guidance: for instance, Li et al. (2024) uses LLM suggestions to steer symbolic search in SyGuS settings, improving over pure enumeration. Barke et al. (2024) blends LLM completions with a learned surrogate model to guide synthesis in DSLs. Shi et al. (2023) uses neural policies to build higher-order and lambda abstractions during search, outperforming both LLM-only and symbolic baselines on list manipulation tasks. The DGM differs in focusing not just on producing programs for external tasks, but also on agent self-modification, rewriting its own implementation to improve its capacity for future self-improvement.

Inspiration from Darwinian Evolution.
This work is heavily inspired by the mechanisms of Darwinian evolution (Darwin, 2023), notably variation (mutation), selection, and the preservation of lineages (stepping stones), and brings them into the realm of self-modifying coding agents. In DGM, an archive of past agent versions is maintained, from which parent agents are sampled; then mutations (i.e., code edits) generate new offspring agents, which are empirically evaluated on coding benchmarks; successful ones are added to the archive, enabling parallel exploration of multiple evolutionary trajectories ([Section˜3](#S3)). This mirrors how biological evolution (Edwards, 2000; Wright, 1932) preserves genetic diversity (Mayr, 1982), leverages variation (Kimura, 1979), and uses natural selection to retain beneficial mutations (Dobzhansky, 1970).

<a id="appendix-c"></a>

## Appendix C Algorithmic Details

### C.1 Initial Coding Agent

In this section, we present the details of the tools available to the initial coding agent ([Section˜4.1](#S4.SS1)) and its task prompt.

Information of the given Bash tool:

Information of the given Edit tool:

Task prompt:

### C.2 Parent Selection

At each DGM iteration, we select a subset of agents from the archive as parents to self-modify and produce new child agents ([Section˜3](#S3)). The details of the parent selection process, inspired by Ecoffet et al. (2019), are outlined below. Future work could explore alternative search and exploration methods (Coulom, 2006; Silver et al., 2016; Herr et al., 2025).

At each iteration $t$ of the DGM run, let

$$
\mathcal{A}^{t}=\{a_{1}^{t},a_{2}^{t},\dots,a_{N}^{t}\}
$$

be the archive of candidate agents. We first define the eligible set

$$
\mathcal{E}^{t}=\bigl\{\,a_{i}^{t}\in\mathcal{A}^{t}:\alpha_{i}<1\bigr\},
$$

i.e. only those agents whose performance score is not yet perfect. We then sample $k$ parents (with replacement) from $\mathcal{E}^{t}$ in proportion to a combined measure of performance and number of children with codebase-editing functionality. Concretely, for each agent $a_{i}^{t}\in\mathcal{E}^{t}$:

$$
(performance) \displaystyle\alpha_{i}\;=\;\mathrm{performance}(a_{i}^{t}), (1) (children count) \displaystyle n_{i}\;=\;\mathrm{functioning_children_count}(a_{i}^{t}), (2) (sigmoid‐scaled performance) \displaystyle s_{i}\;=\;\frac{1}{1+\exp\!\bigl(-\lambda\,(\alpha_{i}-\alpha_{0})\bigr)}, (3) (novelty bonus) \displaystyle h_{i}\;=\;\frac{1}{1+n_{i}}, (4) (unnormalized weight) \displaystyle w_{i}\;=\;s_{i}\,h_{i}, (5) (selection probability) \displaystyle p_{i}\;=\;\frac{w_{i}}{\sum_{j=1}^{N}w_{j}}, (6) (sampling) \displaystyle\{\,\text{parents}\,\}\;\sim\;\mathrm{Categorical}\bigl(\{p_{i}\}_{a_{i}^{t}\in\mathcal{E}^{t}}\bigr)\quad\text{(draw k samples).}(7)
$$

Here, $\lambda>0$ controls the sharpness of the sigmoid, and $\alpha_{0}$ is the midpoint accuracy. We set $\lambda=10$ and $\alpha_{0}=0.5$. $k$ determines how many self-improvements are done in parallel during each iteration of the DGM run. We set $k=2$ in SWE-bench experiments, and $k=4$ in Polyglot experiments.

### C.3 Self-Improve Prompts

In every iteration, an agent is chosen to self-modify. For this step, the current implementation of DGM leverages a separate FM to diagnose improvement opportunities, which are then passed to the coding agent for implementation. To identify the next potential improvement, the evaluation logs from the chosen agent’s performance on the coding benchmark are fed into the FM. While the current implementation relies on an additional diagnostic step, there are no fundamental limitations preventing the DGM from autonomously analyzing its own performance, generating improvement hypotheses, and implementing subsequent modifications. We chose to use OpenAI’s o1 model (Team et al., 2024) for this purpose, as we found it had better reasoning capabilities than other FMs available at the time of our experiments. The prompts used for this diagnosis are provided below. From the diagnostic output, the implementation_suggestion and problem_description are extracted and passed to the coding agent as its self-modification task.

Self-improve diagnosis prompt for SWE-bench experiments:

While the private test patch is included in the diagnosis prompt, one could imagine that an easy way for the agent to score well on the task would be to hard code the solutions. However, from manual inspection of the logs and agent modifications, we have not observed any problematic logic or behavior indicative of memorization or overfitting to specific private test cases. Furthermore, improvements discovered on one task or LLM model often generalize well to others ([Figure˜4](#S4.F4)), showing that these improvements are not task-specific but instead reflect general changes that enhance overall performance.

Additional prompt for Polyglot experiments:

### C.4 Pseudocode

This is the pseudocode of the DGM algorithm, described in [Section˜3](#S3):

This is the pseudocode of the baseline DGM without self-improving agents, described in [Section˜4.3](#S4.SS3):

This is the pseudocode of the baseline DGM without open-ended exploration, described in [Section˜4.3](#S4.SS3):

<a id="appendix-d"></a>

## Appendix D Experiment Details

### D.1 Hyperparameters for Foundation Models

[Table˜3](#A4.T3) shows the foundation model used in each experiment setting, as described in [Section˜4.1](#S4.SS1). Since SWE-bench is a more challenging coding benchmark, we use a stronger coding model, Claude 3.5 Sonnet (New) (based on our preliminary testing). To enable faster iterations and avoid the same rate limits as Claude, we use o3-mini for Polyglot experiments. The temperature for all FMs in every setting is set to 1.0.

<a id="table-3"></a>

> Table 3: Foundation models used in each experiment setting.

| Benchmark         | SWE-bench               | Polyglot                |
| ----------------- | ----------------------- | ----------------------- |
| Self-modification | Claude 3.5 Sonnet (New) | Claude 3.5 Sonnet (New) |
| Evaluation        | Claude 3.5 Sonnet (New) | o3-mini                 |

<a id="appendix-e"></a>

## Appendix E Benchmark Details

### E.1 Cost Estimate

The estimated cost of completing a single run of the DGM on SWE-bench, as presented in [Section˜4](#S4), is about USD 22,000. In comparison, the estimated cost of completing a single run of either baseline (DGM w/o self-improve or DGM w/o open-ended exploration) on SWE-bench is about USD 10,000. Although the DGM is considerably more costly than the baselines, a method that can continuously improve, even at a higher cost, is preferable to one that fails to improve or stagnates at a level of performance that may never match that of the DGM. A more granular break down is:

| LLM                     | Benchmark | Number of Tasks | Cost Estimate (USD) |
| ----------------------- | --------- | --------------- | ------------------- |
| Claude 3.5 Sonnet (New) | SWE-bench | 60              | $350                |
| o3-mini                 | Polyglot  | 60              | $5                  |

We acknowledge that the current experiments on SWE-bench require considerable compute. Hence, we also include experiments on another benchmark, Polyglot, with significantly lower costs. This suggests that expenses vary greatly by task complexity, with SWE-bench being among the more complex and resource-intensive coding benchmarks. Moreover, several impactful methods (e.g., LLM training at its inception) were characterized substantial computational demands initially. Similar to these pioneering works, we hope to open the door to future research on improving the efficiency and scalability of our approach. In addition, many leading coding agents on the SWE-bench leaderboard are backed by industrial companies employing expert full-time researchers and engineers, which incurs substantial human labor costs. In contrast, our approach achieves SoTA-level performance through fully autonomous self-improvement without human intervention, potentially offering greater efficiency when considering the comparative costs of specialized AI development talent versus API usage. Finally, as FMs continue to improve and compute costs continue to decline, methods like the DGM will become increasingly efficient and accessible.

Also, higher-performing agents discovered by the DGM do indeed incur greater inference costs than the initial agent, but cost and performance are not strictly correlated, where some expensive agents underperform cheaper ones.

### E.2 SWE-bench Tasks

Initial 10 tasks for verifying basic functionality of a coding agent:

- •
  django\_\_django-10973
- •
  django\_\_django-11066
- •
  django\_\_django-12754
- •
  django\_\_django-15930
- •
  django\_\_django-13279
- •
  django\_\_django-16661
- •
  django\_\_django-13346
- •
  django\_\_django-10880
- •
  django\_\_django-10999
- •
  django\_\_django-11087

Additional 50 tasks for estimating general effectiveness of a coding agent:

- •
  django\_\_django-9296
- •
  django\_\_django-11790
- •
  django\_\_django-11815
- •
  django\_\_django-11848
- •
  django\_\_django-11880
- •
  django\_\_django-11885
- •
  django\_\_django-11951
- •
  django\_\_django-11964
- •
  django\_\_django-11999
- •
  django\_\_django-12039
- •
  django\_\_django-12050
- •
  django\_\_django-12143
- •
  django\_\_django-12155
- •
  django\_\_django-12193
- •
  django\_\_django-12209
- •
  django\_\_django-12262
- •
  django\_\_django-12273
- •
  django\_\_django-12276
- •
  django\_\_django-12304
- •
  django\_\_django-12308
- •
  django\_\_django-12325
- •
  django\_\_django-12406
- •
  django\_\_django-12708
- •
  django\_\_django-12713
- •
  django\_\_django-12774
- •
  sphinx-doc\_\_sphinx-7454
- •
  sphinx-doc\_\_sphinx-7590
- •
  sphinx-doc\_\_sphinx-7748
- •
  sphinx-doc\_\_sphinx-7757
- •
  sphinx-doc\_\_sphinx-7985
- •
  sphinx-doc\_\_sphinx-8035
- •
  sphinx-doc\_\_sphinx-8056
- •
  sphinx-doc\_\_sphinx-8265
- •
  sphinx-doc\_\_sphinx-8269
- •
  sphinx-doc\_\_sphinx-8475
- •
  sphinx-doc\_\_sphinx-8548
- •
  sphinx-doc\_\_sphinx-8551
- •
  sphinx-doc\_\_sphinx-8638
- •
  sphinx-doc\_\_sphinx-8721
- •
  sphinx-doc\_\_sphinx-9229
- •
  sphinx-doc\_\_sphinx-9230
- •
  sphinx-doc\_\_sphinx-9281
- •
  sphinx-doc\_\_sphinx-9320
- •
  sphinx-doc\_\_sphinx-9367
- •
  sphinx-doc\_\_sphinx-9461
- •
  sphinx-doc\_\_sphinx-9698
- •
  sphinx-doc\_\_sphinx-10449
- •
  sphinx-doc\_\_sphinx-10466
- •
  sphinx-doc\_\_sphinx-10673
- •
  sphinx-doc\_\_sphinx-11510

Additional 140 tasks for more accurate assessment of a coding agent’s performance:

- •
  astropy\_\_astropy-12907
- •
  astropy\_\_astropy-13033
- •
  astropy\_\_astropy-13236
- •
  astropy\_\_astropy-13398
- •
  astropy\_\_astropy-13453
- •
  astropy\_\_astropy-13579
- •
  astropy\_\_astropy-13977
- •
  astropy\_\_astropy-14096
- •
  astropy\_\_astropy-14182
- •
  astropy\_\_astropy-14309
- •
  astropy\_\_astropy-14365
- •
  astropy\_\_astropy-14369
- •
  astropy\_\_astropy-14508
- •
  astropy\_\_astropy-14539
- •
  astropy\_\_astropy-14598
- •
  astropy\_\_astropy-14995
- •
  astropy\_\_astropy-7166
- •
  astropy\_\_astropy-7336
- •
  astropy\_\_astropy-7606
- •
  astropy\_\_astropy-7671
- •
  astropy\_\_astropy-8707
- •
  astropy\_\_astropy-8872
- •
  django\_\_django-10097
- •
  django\_\_django-10554
- •
  django\_\_django-10914
- •
  django\_\_django-11095
- •
  django\_\_django-11099
- •
  django\_\_django-11119
- •
  django\_\_django-11133
- •
  django\_\_django-11138
- •
  django\_\_django-11141
- •
  django\_\_django-11149
- •
  django\_\_django-11163
- •
  django\_\_django-11179
- •
  django\_\_django-11206
- •
  django\_\_django-11211
- •
  django\_\_django-11239
- •
  django\_\_django-11265
- •
  django\_\_django-11276
- •
  django\_\_django-11292
- •
  django\_\_django-11299
- •
  django\_\_django-11333
- •
  django\_\_django-11400
- •
  django\_\_django-11433
- •
  django\_\_django-11451
- •
  django\_\_django-11477
- •
  django\_\_django-11490
- •
  django\_\_django-11532
- •
  django\_\_django-11551
- •
  django\_\_django-11555
- •
  django\_\_django-11603
- •
  django\_\_django-11728
- •
  django\_\_django-11734
- •
  django\_\_django-11740
- •
  django\_\_django-11749
- •
  django\_\_django-11820
- •
  django\_\_django-12125
- •
  django\_\_django-12419
- •
  django\_\_django-12663
- •
  django\_\_django-12741
- •
  django\_\_django-12858
- •
  django\_\_django-12965
- •
  django\_\_django-13012
- •
  django\_\_django-13023
- •
  django\_\_django-13028
- •
  django\_\_django-13033
- •
  django\_\_django-13089
- •
  django\_\_django-13109
- •
  django\_\_django-13112
- •
  django\_\_django-13121
- •
  django\_\_django-13128
- •
  django\_\_django-13158
- •
  django\_\_django-13195
- •
  django\_\_django-13212
- •
  django\_\_django-13297
- •
  django\_\_django-13315
- •
  django\_\_django-13343
- •
  django\_\_django-13344
- •
  django\_\_django-13363
- •
  django\_\_django-13401
- •
  django\_\_django-13406
- •
  django\_\_django-13410
- •
  django\_\_django-13417
- •
  django\_\_django-13449
- •
  django\_\_django-13512
- •
  django\_\_django-13513
- •
  django\_\_django-13516
- •
  django\_\_django-13551
- •
  django\_\_django-13568
- •
  django\_\_django-13569
- •
  django\_\_django-13590
- •
  django\_\_django-13658
- •
  django\_\_django-13670
- •
  django\_\_django-13741
- •
  django\_\_django-13786
- •
  django\_\_django-13794
- •
  django\_\_django-13807
- •
  django\_\_django-13809
- •
  django\_\_django-13810
- •
  django\_\_django-13820
- •
  django\_\_django-13821
- •
  django\_\_django-13837
- •
  django\_\_django-13925
- •
  django\_\_django-13933
- •
  django\_\_django-13964
- •
  django\_\_django-14007
- •
  django\_\_django-14011
- •
  django\_\_django-14017
- •
  django\_\_django-14034
- •
  django\_\_django-14053
- •
  django\_\_django-14089
- •
  django\_\_django-14122
- •
  django\_\_django-14140
- •
  django\_\_django-14155
- •
  django\_\_django-14170
- •
  django\_\_django-14238
- •
  django\_\_django-14311
- •
  django\_\_django-14315
- •
  django\_\_django-14349
- •
  django\_\_django-14351
- •
  django\_\_django-14373
- •
  django\_\_django-14376
- •
  django\_\_django-14404
- •
  django\_\_django-14434
- •
  django\_\_django-14493
- •
  django\_\_django-14500
- •
  django\_\_django-14534
- •
  django\_\_django-14539
- •
  django\_\_django-14559
- •
  django\_\_django-14580
- •
  django\_\_django-14608
- •
  django\_\_django-14631
- •
  django\_\_django-14672
- •
  django\_\_django-14725
- •
  django\_\_django-14752
- •
  django\_\_django-14765
- •
  django\_\_django-14771
- •
  django\_\_django-14787
- •
  django\_\_django-14792
- •
  django\_\_django-14855

### E.3 Polyglot Tasks

Initial 10 tasks for verifying basic functionality of a coding agent:

- •
  go\_\_dominoes
- •
  cpp\_\_all-your-base
- •
  python\_\_dominoes
- •
  java\_\_sgf-parsing
- •
  javascript\_\_robot-name
- •
  rust\_\_variable-length-quantity
- •
  python\_\_beer-song
- •
  go\_\_book-store
- •
  javascript\_\_bottle-song
- •
  rust\_\_bowling

Additional 50 tasks for estimating general effectiveness of a coding agent:

- •
  javascript\_\_queen-attack
- •
  rust\_\_wordy
- •
  python\_\_dot-dsl
- •
  java\_\_satellite
- •
  cpp\_\_diamond
- •
  rust\_\_accumulate
- •
  go\_\_error-handling
- •
  cpp\_\_queen-attack
- •
  rust\_\_poker
- •
  python\_\_sgf-parsing
- •
  rust\_\_react
- •
  java\_\_ledger
- •
  go\_\_connect
- •
  rust\_\_macros
- •
  javascript\_\_triangle
- •
  java\_\_zipper
- •
  java\_\_bowling
- •
  python\_\_tree-building
- •
  javascript\_\_say
- •
  java\_\_wordy
- •
  python\_\_food-chain
- •
  javascript\_\_wordy
- •
  python\_\_poker
- •
  javascript\_\_grade-school
- •
  cpp\_\_gigasecond
- •
  java\_\_forth
- •
  python\_\_dominoes
- •
  go\_\_word-search
- •
  javascript\_\_simple-linked-list
- •
  go\_\_counter
- •
  java\_\_react
- •
  javascript\_\_ocr-numbers
- •
  python\_\_scale-generator
- •
  java\_\_go-counting
- •
  rust\_\_doubly-linked-list
- •
  python\_\_grade-school
- •
  javascript\_\_forth
- •
  python\_\_wordy
- •
  java\_\_mazy-mice
- •
  cpp\_\_bank-account
- •
  python\_\_zipper
- •
  java\_\_custom-set
- •
  java\_\_rest-api
- •
  go\_\_transpose
- •
  rust\_\_gigasecond
- •
  rust\_\_say
- •
  go\_\_food-chain
- •
  rust\_\_pig-latin
- •
  go\_\_markdown
- •
  go\_\_crypto-square

### E.4 SWE-bench State-of-The-Art

At the time of writing this paper (16 April 2025), the highest performing, checked (i.e., the SWE-bench team received access to the system and were able to reproduce the patch generations), open-source entry on SWE-bench Verified is OpenHands + CodeAct v2.1 (claude-3-5-sonnet-20241022) (Wang et al., 2024), achieving 53.0%. Only considering the same subset of 200 tasks used by the DGM ([Section˜E.2](#A5.SS2)), OpenHands + CodeAct v2.1 (claude-3-5-sonnet-20241022) achieves 51.0%.

### E.5 Polyglot Representative Agent

Aider (Gauthier, 2024), a popular coding agent in the community, was published in Spring 2024. It has garnered over 33,000 stars on GitHub and has been continuously developed and tested against the Polyglot benchmark for over a year by human developers, primarily to evaluate its performance. Aider has also become a standard baseline for assessing the performance of different models, with the current top performers on the Polyglot benchmark being a mix of o3 (high) and GPT-4.1. We adopt a setup similar to that of the Polyglot leaderboard, with one key difference: the leaderboard reports pass@2 performance, where the agent can view feedback from ground-truth tests once. In contrast, we use a pass@1 setting, where the agent never sees the results of ground-truth tests, as we believe this more closely reflects realistic coding applications.

<a id="appendix-f"></a>

## Appendix F Best-Discovered Agents

### F.1 DGM on SWE-bench

Diff patches contributing to the best agent discovered by the DGM on SWE-bench:

### F.2 DGM on Polyglot

Diff patches contributing to the best agent discovered by the DGM on Polyglot:

<a id="appendix-g"></a>

## Appendix G Similar Target Functionality, Different Implementations

For the same target functionality, the suggestions provided to the coding agents and the resulting implementations can differ significantly, leading to large variations in coding capability and benchmark performance. For example, consider nodes 6 and 24 in the DGM run on SWE-bench ([Figure˜3](#S4.F3)). Both aimed to enhance the existing editor tool to support finer‑grained file editing. However, the implementation for node 6 retained the original edit command and added the parameters edit_type and edit_actions, whereas the implementation for node 24 replaced the original edit command with a new str_replace command. Despite targeting similar functionality, the feature suggestions and thus the implementation details differed greatly, as did their performance. Node 6 achieved only 23.3%, while node 24 achieved 40.5%.

Feature suggestion to obtain node 6:

Feature suggestion to obtain node 24:

Implementation of suggested feature to obtain node 6:

Implementation of suggested feature to obtain node 24:

<a id="appendix-h"></a>

## Appendix H Case Study: Solving Hallucination

The DGM can be used to optimize objectives beyond just coding, as discussed as a potential direction for future work in [Section˜5](#S5). In this section, we show that the DGM can address hallucinations of tool use by FMs. Through feedback from evaluation logs, the DGM improves hallucination detection mechanisms and ultimately discovers ways to resolve these hallucinations. We provide a more detailed discussion of when these hallucinations occur, the DGM setup, and the resulting solutions. Finally, we present an example of objective hacking, where the agent optimizes for the quantifiable metric rather than truly fulfilling the spirit of the task and solving the underlying problem.

<a id="figure-8"></a>

![dgm_wo_openended](images/dgm_wo_openended.png)

> Figure 8: The DGM solving hallucination of tool use in FMs. Archive of coding agents generated during the DGM run on SWE-bench to solve hallucination from FMs. Each node represents an agent, with node 0 corresponding to the base agent. Node color indicates solved hallucination score, while border color reflects whether the agent has basic codebase-editing functionality. Edges show which agents self-modified to produce the offsprings.

When hallucination occurs.
When the input context to Claude 3.5 Sonnet (New) (Claude) is very long, including all previous tool use inputs and outputs, we found that Claude tends to hallucinate tool usage. Instead of actually making a tool call, it outputs the tool input and output as plain text. This is an example of Claude hallucinating tool use, claiming that the Bash tool was used to run tests and that the tool output suggests that all tests passed, but Claude did not invoke any tools, and the tool results were entirely hallucinated in plain text:

DGM setup.
We run this experiment on SWE-bench. Other than the number of SWE-bench tasks we evaluate each coding agent on, and the final scoring function of the DGM, all other settings remain the same as in the main experiments ([Section˜3](#S3)). The initial coding agent and all agents generated during the DGM run are evaluated on only 3 SWE-bench tasks. We use just 3 tasks because if hallucination occurs in one, it is likely to occur in others as well. Hence, 3 tasks are sufficient to provide the DGM with a reliable signal regarding whether tool use hallucination is still present.

Solve hallucination score.
We score each generated agent based on how effectively it prevents hallucinations, with higher scores indicating fewer hallucinations and better performance. For each SWE-bench task the agent is evaluated on, the score has two components: 0 - 1 based on whether hallucination occurred in the logs, and 1 - 2 (only added and accounted for after no hallucination occurs in the logs) if tools were used in all Claude calls. The final solve hallucination score is computed as the average across all evaluated tasks. To detect hallucination in the logs, we insert special tokens when actual tool use occurs and then check if these tokens appear in the model’s plain-text output, indicating that the model mimicked tool use without actually invoking any tools. Importantly, the hallucination checking functions are hidden from the coding agent during self-modification. We found that objective hacking (i.e., optimizing for the measurable objective instead of truly solving the underlying problem) occurs more frequently when these functions are not hidden.

Results.
We run the DGM for 150 iterations. In this run, the DGM discovers two promising solutions: node 96, which achieves the second highest score of 1.67, and node 114, which achieves a perfect score of 2.0 ([Figure˜8](#A8.F8)).
Node 96 addresses hallucination to some extent, achieving a score of 1.67 without objective hacking. In the lineage leading to node 96, the DGM first improves the tool transaction logging mechanisms to better detect when hallucination occurs. It then explores various strategies to solve it, such as removing hallucinated tool use from the output text when it determines that the tool was not actually invoked.
Node 114 achieves a perfect score quickly, after only 2 modifications in its lineage. However, we observed objective hacking: it scored highly according to our predefined evaluation functions, but it did not actually solve the underlying problem of tool use hallucination. In the modification leading up to node 114 (see below), the agent removed the logging of special tokens that indicate tool usage (despite instructions not to change the special tokens), effectively bypassing our hallucination detection function.

In conclusion, the DGM can be applied beyond the coding domain, and we highlighted a case of objective hacking. Similar to reward hacking in reinforcement learning (Skalse et al., 2022), objective hacking occurs when a system optimizes for a predefined, quantifiable objective rather than fulfilling the spirit of the task or solving the intended problem. This observation supports arguments made in prior works (Zhang et al., 2024b; Faldor et al., 2025), which suggest that optimizing quantitative measures often leads to undesirable or pathological outcomes, and aligns with Goodhart’s law (Strathern, 1997) – "When a measure becomes a target, it ceases to be a good measure."

<a id="appendix-i"></a>

## Appendix I Additional Safety Discussion

Any advancement that increases the autonomous capabilities of AI systems introduces its own set of safety considerations (Bengio et al., 2024), especially for systems that improve in an open-ended way (Ecoffet et al., 2020; Clune, 2019). [Section˜5](#S5) discusses these concerns and outlines concrete, actionable steps for mitigating them. We call for much more research into and discussion regarding AI safety, including deep thought and discussion amongst all stakeholders in society on the complicated question of what exactly counts as safe AI. We are confident the work we have done was never unsafe ([Section˜5](#S5)), but scaled up versions of it could be. As with all transformative technologies, the ultimate impact of such AI systems remains deeply uncertain, and good arguments can be made both for the case that it will bring about tremendous good and tremendous harm. These uncertainties highlight the need for sustained, inclusive, and multidisciplinary discussion (not only from current experts but also from a wider and more diverse community) on how to navigate these developments.

<a id="appendix-j"></a>

## Appendix J Additional Future Work Directions

While this paper has shown the potential of the Darwin Gödel Machine in iteratively improving coding agents via open‐ended exploration and empirical validation, several extensions could address current limitations and push AI beyond its already growing role in inspiring culture and advancing science. The following directions outline promising avenues for further research.

Autonomously Improving the Open-ended Exploration Process.
In this version of the DGM, the open-ended exploration process described in [Section˜3](#S3) is kept fixed, which might hence impede the system’s self-acceleration potential. This design choice was made due to limited computational budget. If we were to evolve this part of the algorithm, it could require exponentially more compute to identify processes that yield the same improvements shown in [Section˜4.4](#S4.SS4). Nevertheless, since the open-ended exploration loop itself is implemented in code, it can in principle be edited and improved by a coding agent. There are many possible implementations of open-ended exploration, for example, using alternative search mechanisms that balance exploration and exploitation (Herr et al., 2025), keeping only the most interesting agents in the archive (Faldor et al., 2025), or leveraging the generated agent population as an ensemble (Samvelyan et al., 2024). A promising future work direction is to allow the agent to modify the open-ended exploration process, thereby autonomously improving not only its own capabilities but also the meta-process that allocates limited compute to drive self-improvement and self-acceleration.

Role of Humans in Autonomous AI Systems.
In the current formulation of the DGM, proposed self-modifications are autonomously evaluated without any human intervention. However, as autonomous systems grow in complexity and influence, the question of how humans should remain involved becomes increasingly pressing. Should human oversight be framed as an optimization objective, incorporated through techniques such as reinforcement learning from human feedback (Ouyang et al., 2022), or distilled into FMs that act as preference judges (Bai et al., 2022)? Each of these approaches raises challenges in terms of scalability, reliability, and alignment with evolving human values. The role of humans in guiding, constraining, or co-evolving with autonomous AI remains an open question. Exploring this dynamic is a promising avenue for future research, as it touches not only on technical feasibility but also on broader philosophical and societal considerations.

DGM with Advanced Foundation Models.
Recent FMs have advanced dramatically, enabling scaffolds to become simpler on current coding benchmarks (Yang et al., 2024). It is possible that in some settings, like current coding benchmarks, certain engineering efforts in scaffolding might be downplayed by the improvement of the FMs. However, many scaffolding components (advanced tools, parallel workflows, external memory, proxy verification, etc.) still fundamentally can not be internalized by FMs and will be essential for more complex real-world tasks beyond today’s benchmarks. Future work to explore how different components in agents will emerge with different FMs could be a promising direction.

Evolving Generalist Agent.
We believe some degree of task-specific adaptation is indeed expected and even desirable, since fundamentally different types of tasks (e.g., in our case, multi-file Python repository edits vs. primarily single-file, multi-language implementations) naturally require distinct scaffolding components. Crucially, this very property highlights a unique advantage of self-improving systems like the DGM: it replaces laborious manual efforts to design specialized agents for diverse tasks with a fully automated evolutionary process. This motivates an exciting future direction of running the DGM on a large, diverse set of tasks to evolve a true generalist agent. Also, currently we only evaluated DGM on two coding benchmarks. While we believe these two benchmarks differ substantially in task structure (multi-file Python repository edits vs. primarily single-file, multi-language implementations), providing strong evidence of generality, additional benchmarks would further strengthen evaluations.
