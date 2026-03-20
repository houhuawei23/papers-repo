# Title: Efficient Reasoning on the Edge

- ArXiv: 2603.16867
- Authors: Yelysei Bondarenko , Thomas Hehn , Rob Hesselink, Romain Lepert, Fabio Valerio Massoli, Evgeny Mironov, Leyla Mirvakhabova, Tribhuvanesh Orekondy, Spyridon Stasis, Andrey Kuzmin, Anna Kuzina, Markus Nagel, Ankita Nayak, Corrado Rainone, Ork de Rooij, Paul N Whatmough, Arash Behboodi% , Babak Ehteshami Bejnordi
- Sections: 1
- Estimated tokens: 27.0k

# Efficient Reasoning on the Edge

# Introduction

<a id="fig:overview"></a>

![](./images/fig1_overview.png)

> Figure 6: **Overview of the proposed efficient reasoning framework for edge devices.** (a) The model architecture utilizes parameter-efficient LoRA adapters and a lightweight switcher to dynamically route queries. This design allows the base model and the reasoning-activated mode to seamlessly share a reusable KV cache during prefill. (b) Parallel test-time scaling strategy, generating multiple reasoning streams concurrently to improve accuracy without severe latency penalties. (c) The end-to-end deployment pipeline, illustrating the progression from multi-stage training (SFT and budget-forced RL) to quantization, model export, and final on-device execution.

The success and impact of large language models (LLMs) continue to expand, with reasoning emerging as a fundamental component of their achievements. Commercial-grade coding agents reason about code structure, refactoring, debugging, and dependency resolution [@anthropic2024claude; @jaech2024openai]. In scientific domains, LLMs are increasingly used to assist professional mathematicians with research level problems [@Abouzaid2026-first-proof; @Cao2025-ai4math; @Feng2026-ai4math]. On mobile devices, reasoning-capable LLMs unlock a new class of intelligent personal assistants able to plan multi-step tasks, respond contextually to user queries, and operate autonomously across apps and interfaces. Recent advances from Gemini and OpenAI demonstrate increasingly capable standalone reasoning [@OpenAI2026-first-proof; @Woodruff2026-gemini], and a growing wave of agentic use cases is now emerging where models interact directly with device UIs and real-world services [@Yao2022-react; @Zhou2025-maiui; @Wang2025-uitars2; @Veuns-Team2026-uivenus1.5]. These achievements, however, come at the cost of generating massive numbers of tokens, with reasoning traces accounting for a large fraction of the overall computation [@thinking-efficiency2025].

Deploying reasoning models on edge devices is attractive for mobile scenarios because it keeps sensitive data on-device, reduces round-trip latency, and remains available even under limited connectivity. In practice, however, on-device reasoning faces several key limitations. The first is the memory bottleneck: mobile devices, constrained by current DRAM capacities, can typically support smaller models with moderate quantization, and larger models only with more aggressive quantization schemes [@Alizadeh2024-llm-flash; @Xiao2026-llm-mobile]. The second limitation is the cost of token generation in terms of power consumption, latency, and memory footprint. Long reasoning traces and large context lengths substantially increase KV cache size, quickly exhausting available memory. Finally, general purpose LLMs with broad capability scopes are difficult to realize within the model sizes supported by edge devices. While specialized small language models (SLMs) can match the performance of larger models on targeted tasks, model switching introduces additional memory movement overhead. Edge deployed models must therefore operate within strict memory budgets while achieving usable tokens per second (TPS) and acceptable time to response.

This work proposes an end-to-end pipeline for deploying reasoning-capable language models on edge devices under strict token, latency, and memory budgets. Our design starts from a base non-reasoning instruct model and enables a "reasoning mode" via LoRA adapters, so the same backbone can run either in standard chat mode (no adapters) or in reasoning mode (reasoning adapters enabled). A lightweight switcher routes each incoming query to the appropriate mode, enabling reasoning only when it is likely to help.

At training time, we use parameter-efficient fine-tuning to specialize the base model across domains while keeping deployment practical. We demonstrate that LoRA [@hulora] is effective for enabling domain targeted reasoning and task specialization. Selecting the LoRA rank as a function of the base model and desired performance is a central question, which we address through extensive analysis. Because LoRA adapters can be toggled at runtime, a single base model can be loaded once and then dynamically adapted to different tasks by enabling or disabling adapters. To enable KV-cache sharing between the base model and the LoRA augmented reasoning model, we propose masked LoRA training during the prefill phase, which we show has no significant impact on accuracy.

Our training recipe follows a two-stage structure commonly used for reasoning models: supervised fine-tuning (SFT) on high-quality reasoning traces, followed by a reinforcement learning (RL) phase for further alignment [@deepseekai2025deepseekr1]. Since, reasoning verbosity is not explicitly addressed during SFT and models often become verbose and repetitive after initial training, a key objective of our RL phase is to penalize excessively long reasoning traces via budget forcing [@muennighoff2025s1; @li2025budgetguidance]. We apply budget forcing during RL and explore different training strategies, reward designs, prompting methods, and hard enforcement mechanisms, distilling best practices from our experiments.

At inference time, we target memory-bound decoding as a primary on-device bottleneck. We leverage parallel test-time scaling with neural verification to improve accuracy without incurring significant latency overhead, particularly in typical on-device settings. Concretely, because on-device inference splits into a compute-bound prefill phase and a memory-bound decoding phase, we can better utilize compute units (e.g., NPUs) by running parallel decoding paths with minimal incremental overhead. We further show that neural verification in an outcome reward model style can be implemented using a lightweight verifier head trained on the latent representations of the base model.

Together, these components allow us to start from a non-reasoning model and progressively harvest reasoning performance while maintaining deployability on edge devices. Figure [1](#fig:overview){reference-type="ref" reference="fig:overview"} provides a comprehensive overview of our proposed end-to-end framework, illustrating the complete progression from adapter training to on-device deployment. We instantiate this pipeline on Qwen2.5 series of models, and we enable reasoning for these models using our proposed pipeline. On device deployment requires additional considerations. We discuss a range of quantization schemes supporting 4- to 8-bit weight quantization while allowing higher precision for activations to accommodate their dynamic range, minimizing quantization loss throughout the pipeline. The resulting quantized models are then exported and compiled for on-device execution. The entire workflow is implemented using Qualcomm open source tooling, including Qualcomm FastForward [@fastforward] and Qualcomm GENIE SDK [@GENIE]. In this paper, we provide deeper insights into the design choices and empirical analyses underpinning this work. We position the resulting system as a practical blueprint for deploying reasoning capable language models on resource constrained edge devices.

# Reasoning on Edge: System Design

<a id="fig:system_design"></a>

> Figure 1: <strong>Architecture of the Hybrid Reasoning Model.</strong> The pipeline begins with a compact base LLM, which is specialized for reasoning via LoRA-based supervised fine-tuning (SFT). To enforce concise generation and prevent excessive verbosity, these adapters undergo reinforcement learning (RL) with Budget Forcing. Finally, a lightweight Switcher module is introduced to act as a reasoning-needed classifier, creating a hybrid model that dynamically routes incoming queries to either the fast base model or the specialized reasoning adapters based on task complexity.

Developing compact yet capable reasoning models requires a training pipeline that can reliably elicit high-quality chain-of-thought (CoT) behavior from a relatively small base LLM while avoiding unnecessary verbosity and excessive computation. Although pretrained LLMs can sometimes function as zero-shot reasoners [@kojima2022large], explicit reasoning via approaches such as scratchpads [@nye2022show] or supervised CoT [@wei2022chain] remains essential for strong performance on mathematics and coding tasks. Recent systems such as OpenAI O1 [@jaech2024openai] and specialized small reasoning models, including Tina [@wang2025tina], Phi-4-mini [@xu2025phi], and hybrid reasoning architectures [@jiang2025think], illustrate that such capabilities can be distilled into relatively small models when combined with targeted fine-tuning and alignment techniques.

Our objective is to fine-tune a modest base LLM so that it performs competitively on complex reasoning tasks while producing concise outputs and minimizing token generation. To keep adaptation lightweight and deployable, we perform training primarily through Low-Rank Adapters (LoRA) [@hulora], which preserve a reusable frozen backbone and enable modular reasoning specialization. The overall pipeline, presented in Figure [2](#fig:system_design){reference-type="ref" reference="fig:system_design"}, integrates supervised fine-tuning, reinforcement learning, and lightweight routing in a manner that preserves efficiency and supports deployment under strict memory and latency constraints.

**Supervised fine-tuning** constitutes the first stage of our pipeline and is designed to unlock the reasoning capabilities of the pretrained base LLM. Rather than performing dense fine-tuning, we adopt parameter-efficient fine-tuning using LoRA. LoRA has been shown to match or even surpass dense fine-tuning in reasoning settings [@schulman2025lora], while enabling the base model to remain frozen and reusable for multiple domains. This stage equips the model with the fundamental ability to reason through multi-step problems, but, as widely observed, also increases verbosity and can lead to unnecessarily long or repetitive traces [@deepseekai2025deepseekr1].

To refine the reasoning behavior and control verbosity, we apply **Reinforcement Learning**  [@chen2025acereason; @dang2025reinforcement] (RL) using a custom reward function tailored to two objectives: accuracy and efficiency. First, we use budget forcing [@alomrani2025reasoning], a mechanism that penalizes excessively long responses. This constraint encourages the model to produce concise reasoning traces without sacrificing correctness. Second, we incorporate an answer-based reward, which directly incentivizes the model to generate correct final answers. For optimization, we employ the group-based relative policy optimization (GRPO) algorithm [@shao2024deepseekmath], which updates the LoRA parameters.

Finally, not all user queries require multi-step reasoning. To avoid unnecessary computation, we introduce a lightweight **Switcher module** that predicts whether reasoning is needed based on hidden prompt representations. When reasoning is unnecessary, the model bypasses the LoRA adapters and relies on the base model directly, reducing latency and limiting KV cache growth, an important consideration for edge deployment.

In the following sections, we decompose our end-to-end pipeline into its main components, LoRA-based adaptation (Section [3](#sec:lora){reference-type="ref" reference="sec:lora"}), dynamic inference-time routing using a Switcher module to activate or bypass these adapters (Section [4](#sec:switcher){reference-type="ref" reference="sec:switcher"}), budget-forced RL for verbosity control (Section [5](#sec:budget_forcing){reference-type="ref" reference="sec:budget_forcing"}), and parallel test-time scaling (Section [6](#sec:parallel){reference-type="ref" reference="sec:parallel"}), and the deployment path including quantization and on-device execution (Section [7](#sec:quantization){reference-type="ref" reference="sec:quantization"}). In each section, we describe the component and then quantify its impact with targeted experiments, highlighting key insights and focused ablations to isolate which design choices drive accuracy and which improve on-device efficiency.

# LoRA for Modular Reasoning {#sec:lora}

In our reasoning framework, we adopt parameter-efficient fine-tuning (PEFT) for two reasons: first, it enables scalable experimentation at low training costs, and second, it produces modular adapters that can be enabled or disabled at runtime, allowing a single base model to switch between general-purpose chat and enhanced reasoning modes. To elicit reasoning behavior, we perform SFT on datasets composed of reasoning traces generated by stronger teacher models such as DeepSeek-R1 [@deepseekai2025deepseekr1] and QwQ-32B [@QwQ-32B].

Our experiments show that 3B and 7B models can acquire strong reasoning ability using straightforward SFT on curated trace datasets in a cost-efficient setup, reaching performance comparable to substantially larger distilled baselines (e.g., DeepSeek-R1-Distill-Qwen-7B). These results suggest that strong reasoning does not require heavy distillation pipelines or large training budgets; high-quality trace data (e.g., OpenThoughts3 [@guha2025openthoughts]) combined with lightweight fine-tuning is sufficient to close most of the gap. Overall, this provides a practical and scalable path to building capable reasoning models without expensive infrastructure.

## Experimental Setup

The primary goal of our LoRA adaptation stage is to determine if small, general-purpose instruct models can acquire expert-level reasoning capabilities without full-parameter distillation. Accordingly, we adopt the Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct models [@qwen2025qwen25technicalreport] as our core experimental backbones. In this section, we outline the datasets and optimization strategies used to elicit reasoning behavior via LoRA, as well as the diverse suite of math, science, and coding benchmarks used to rigorously evaluate the resulting performance trade-offs. We additionally provide an extensive LoRA hyperparameter study designed to identify the most stable and compute-efficient adaptation strategies for both the 3B and 7B backbones.

## Training Details

#### Data.

In our experiments, we utilize two SFT datasets. The first is **Mixture of Thoughts** (MoT) [@openr1], which contains 350k reasoning traces distilled from the DeepSeek-R1 model. This dataset covers three core domains: Math (93.7k traces), Code (83.1k traces), and Science (173k traces). The second dataset is **OpenThoughts3-1.2M** (OT3) [@guha2025openthoughts], comprising 850k Math questions, 250k Code questions, and 100k Science questions. The annotation traces for OT3 were generated using the QwQ-32B model.

#### Training configuration.

All models were trained for 5 epochs using the bfloat16 data type with the DeepSpeed zero2 configuration and CPU offloading enabled. Across all configurations, we applied a cosine learning rate schedule with a warmup ratio of 0.1, and weight decay was set to 0.

For the baseline dense training on the MoT dataset, the learning rate was set to $1\mbox{e$-$}5$ for Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct models, with a global batch size of 128. Model weights were optimized using the AdamW optimizer with $(\beta_1,\beta_2) = (0.9, 0.95)$. For dense training on the OT3 dataset, we followed the recipe described in [@guha2025openthoughts]. Dense models were trained with a learning rate of $8\mbox{e$-$}5$ and a batch size of 512, using AdamW with $(\beta_1,\beta_2) = (0.9, 0.999)$.

In our LoRA training setup, we follow the common practice of employing relatively larger learning rates. We also find that reducing the batch size generally leads to more stable optimization and improved results. Throughout all PEFT experiments, we use a LoRA rank of $128$, set the LoRA alpha to twice this value, adopt a learning rate of $2\mbox{e$-$}{4}$, and train with a batch size of $64$.

## Evaluation Details

#### Benchmarks.

We assess the reasoning capabilities of our trained models using a diverse set of benchmarks that span tasks across mathematics, science, and coding domains. To evaluate multi-step mathematical problem-solving, we use challenging competition datasets including AIME 24/25 [@aime], AMC23 [@amc23], and MATH500 [@hendrycks2021measuring]. Scientific reasoning is measured using the PhD-level GPQA Diamond dataset [@rein2024gpqa]. Finally, to evaluate code generation, we utilize LiveCodeBench (v2, code generation scenario only) [@livecodebench] for recent competitive programming problems, alongside standard Python programming tasks from HumanEval [@humaneval] and MBPP [@mbpp], including their rigorously verified EvalPlus variants (HumanEval+ and MBPP+) [@evalplus]. Comprehensive descriptions of each benchmark, including problem counts and specific testing scenarios, are provided in Appendix [10](#App:benchmark){reference-type="ref" reference="App:benchmark"}.

#### Evaluation pipeline.

Our evaluation pipeline is structured as follows. For reasoning benchmarks including AIME24, AIME25, MATH500, GPQA, and AMC, we allow a generation length of up to 32,768 tokens, with the temperature set to 0.6 and $\mathrm{top}_{\_}\mathrm{p}$ to 0.95. For models trained on the OT3 dataset, we adopt the generation parameters recommended by the authors in [@guha2025openthoughts], specifically setting the temperature to 0.7 and $\mathrm{top}_{\_}\mathrm{p}$ to 1.0 and keeping generation length up to 32,768 tokens. Also for LCB, MBPP and HumanEval (and their + counterparts) we set the generation parameters as recommended by the models' creators; we set the maximal generation length to 32768 tokens for LCB, and 1024 tokens for HumanEval and MBPP.

Given that some benchmarks contain a limited number of questions and are therefore more susceptible to accuracy variance, we perform multiple evaluation runs to ensure robustness similarly to the protocol outlined in [@guha2025openthoughts]. Specifically, we evaluate the AIME24, AIME25 and AMC datasets 10 times and report the averaged results. For GPQA, we conduct 4 evaluation runs and for MATH500 we run the evaluation once. For coding benchkmarks, the pass@1 score is estimated from a pool of 16 candidate solutions for LCB, and of 200 candidate solutions for HumanEval and MBPP, using the unbiased pass@$k$ estimator first proposed in [@humaneval]. All evaluations are conducted using the lighteval framework [@lighteval] with vLLM support [@kwon2023efficient], except for those on HumanEval, MBPP and their enhanced variants, for which we used the Evalplus package[^4].

## Results

As shown in [
$$tab:lora_main_results$$
](#tab:lora_main_results){reference-type="ref+Label" reference="tab:lora_main_results"}, we evaluate Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct models under dense and LoRA-based fine-tuning, and compare against key baselines. Finetuning on the OpenThoughts3 (OT3) dataset yields the largest and most consistent gains in reasoning performance across both backbones, improving accuracy substantially on math and science benchmarks and also boosting performance on the more reasoning-sensitive coding benchmark (LiveCodeBench). In contrast, Mixture of Thoughts (MoT) provides clear improvements over the base models, but its gains are consistently smaller than OT3. In addition, the densely trained Qwen2.5-3B model on OT3 performs on par with, or slightly above, the densely trained Qwen2.5-7B model on MoT, suggesting that higher data quality can partially compensate for smaller backbone size.

For Qwen2.5-7B, LoRA fine-tuning on OT3 recovers most of the dense OT3 improvements, and increasing the adapter rank from 64 to 128 generally narrows the gap on core reasoning benchmarks (e.g., AIME24 and LCB). Notably, OT3 with LoRA rank 128, which requires updating only 4.24% of the parameters compared to dense fine-tuning, reaches performance close to the R1-Distill-Qwen-7B baseline on several reasoning benchmarks, indicating that lightweight adapter training can recover much of a distilled model's capability at significantly lower adaptation cost. For Qwen2.5-3B, however, OT3 LoRA (rank 128) underperforms the dense OT3 model by a large margin across reasoning benchmarks, highlighting that adapter capacity and/or optimization details matter more at smaller scales and motivating our ablations in the subsequent sections.

The coding results reveal a trade-off between reasoning specialization and "direct-answer" code generation. While SFT consistently improves performance on LCB, it always results in some degradation for MBPP, HumanEval and their respective .+ variants. This pattern is consistent with a shift toward explicit multi-step reasoning: it helps on harder, reasoning-sensitive coding tasks like LCB, but can be counterproductive on benchmarks that reward short, direct code outputs. Concretely, our SFT stage is designed to elicit reasoning behavior into an otherwise non-reasoning model, and LCB is a setting where such explicit reasoning can be beneficial. In contrast, MBPP and HumanEval typically do not prompt the model to reason, and a response must be given directly[^5]. Interestingly, for Qwen2.5-7B, OT3 LoRA (rank 64/128) often retains stronger HumanEval/MBPP performance than dense OT3, suggesting that PEFT can partially mitigate the specialization/forgetting trade-off relative to dense fine-tuning. Prior work has similarly observed [@huan2025does] that SFT on reasoning traces can result in partial forgetting of general capabilities, and that his phenomenon could potentially be mitigated by employing RL fine-tuning [@lai2025reinforcement].

We also explored a two-stage training strategy for the 7B model to see if we could combine the distinct benefits of both datasets. Our motivation was that while the OT3 dataset (generated by the smaller QwQ model) provided the strongest baseline performance, the MoT dataset might contain complementary, highly complex reasoning patterns that the model could absorb in a subsequent training phase. However, interestingly, additional training on MoT following training on OT3 results in minimal changes across benchmarks. Accuracies remain largely stable, except for a 0.12-point decrease on AIME24 and a 0.05-point improvement on GPQA.

### LoRA Hyperparameter Study: Rank, Learning Rate, and Batch Size {#sec:ablations}

To study the effect of hyperparameters on the performance of PEFT models, we consider a range of values for learning rate, batch size, and LoRA rank. We trained Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct on a subset of OT3 consisting of 50,000 entries for 1 epoch. For each model, we vary the values within the following ranges: learning rate in $\{1\mbox{e$-$}4, 2\mbox{e$-$}4, 5\mbox{e$-$}4\}$, batch size in $\{32, 64, 128\}$, and LoRA rank in $\{32, 64, 128, 256\}$. LoRA adapters were applied to all linear layers, with $\alpha$ set to 2$\times \text{rank}$ and dropout fixed at $0.1$. For our ablation study, we evaluate the trained models on MATH and Science benchmarks including AIME24, AIME25, MATH500, GPQA Diamond and AMC23. Similarly to the setup used in [@guha2025openthoughts], we average the accuracy values over 10 runs for smaller datasets like AIME24, AIME25 and AMC23, over 4 runs for GPQA and once for MATH500. We set the temperature to 0.7 and $top\_p$ value to 1.0. Full evaluation results for this ablation study can be found in Appendix [11](#sec:lora_ablation){reference-type="ref" reference="sec:lora_ablation"}.

#### Qwen2.5-3B-Instruct results.

The full performance of the models trained on a set of considered hyperparameters is summarized in [10](#tab:ablations_qwen3b){reference-type="ref+Label" reference="tab:ablations_qwen3b"} in the Appendix. To isolate the effect of each hyperparameter, we summarize results by averaging accuracy over the other two dimensions (learning rate, batch size, and LoRA rank), highlighting the main trends. As shown in [1](#tab:lr_summary){reference-type="ref+Label" reference="tab:lr_summary"}, learning rate has a noticeable impact on performance, with $2\mbox{e$-$}4$ providing the highest overall average accuracy across tasks. We also observe opposite sensitivities across benchmarks: AIME24 and AIME25 improve as the learning rate increases, whereas MATH500 degrades, suggesting that larger learning rates can lead to over-adaptation on more complex mathematical reasoning.

[]{#tab:lr_summary label="tab:lr_summary"}

**LR** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

$1\mbox{e$-$}4$ 0.040 0.025 0.573 0.277 0.281 0.239
$2\mbox{e$-$}4$ **0.051** 0.033 **0.574** 0.274 **0.299** **0.246**
$5\mbox{e$-$}4$ **0.051** **0.038** 0.556 **0.280** 0.275 0.240
: Average performance grouped by learning rate for Qwen2.5-3B-Instruct.

As presented in [2](#tab:rank_summary){reference-type="ref+Label" reference="tab:rank_summary"}, LoRA rank shows the most consistent positive influence on aggregate performance. Increasing rank generally improves accuracy, with rank $256$ achieving the highest overall average. However, rank $128$ already performs strongly on most benchmarks, making it a practical operating point when adapter memory is constrained. Across tasks, AIME25 is particularly sensitive to rank, while MATH500 remains comparatively stable. Overall, higher ranks are preferable when resources permit, but rank $128$ offers a favorable accuracy--efficiency trade-off for edge deployment.

[]{#tab:rank_summary label="tab:rank_summary"}

**Rank** **%TP** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

32 1.94% 0.047 0.020 0.568 0.270 0.284 0.238
64 3.88% 0.040 0.023 0.570 **0.283** 0.284 0.240
128 7.76% **0.054** 0.038 0.560 0.272 **0.287** 0.242
256 15.52% 0.048 **0.048** **0.573** 0.282 0.284 **0.247**
: Average performance grouped by LoRA rank for Qwen2.5-3B-Instruct. %TP denotes the percentage of trainable parameters.

Finally, batch size had a relatively minor effect compared to learning rate and rank as shown in [3](#tab:bs_summary){reference-type="ref+Label" reference="tab:bs_summary"}. The best overall performance was observed at batch size set to 64, though the differences were small. MATH500 benefited slightly from larger batches like 128, while AIME25 peaked at 64. These results indicate that batch size can be chosen primarily based on computational constraints without significant impact on overall performance.

[]{#tab:bs_summary label="tab:bs_summary"}

**Batch size** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

32 **0.054** 0.030 0.562 0.277 **0.288** 0.242
64 0.048 **0.038** 0.566 **0.278** 0.287 **0.243**
128 0.040 0.028 **0.575** 0.277 0.281 0.240
: Average performance grouped by batch size for Qwen2.5-3B-Instruct.

#### Qwen2.5-7B-Instruct results.

[11](#tab:qwen7_ablations){reference-type="ref+Label" reference="tab:qwen7_ablations"} in the Appendix summarizes the Qwen2.5-7B-Instruct LoRA ablations over learning rate, batch size, and adapter rank. In contrast to the 3B setting, where learning rate affects performance but the averages vary only modestly across the tested values, the 7B backbone exhibits a narrower stable learning-rate range (Table [4](#tab:lr_summary_second_model_full){reference-type="ref" reference="tab:lr_summary_second_model_full"}): LR=$1\mbox{e$-$}4$ and $2\mbox{e$-$}4$ trains reliably, whereas LR=$5\mbox{e$-$}4$ is often unstable and can lead to collapsed runs. Overall, a practical guideline is to use a lower learning rate for stable training on larger backbones, and tune the remaining hyperparameters within that stable regime. For the individual tables presented below, we account for this distinction explicitly. The table reporting averages across different learning rate settings includes all runs, highlighting the instability introduced by higher learning rates. In contrast, the tables reporting averaged accuracies for batch size and LoRA rank exclude runs with a learning rate of $5\mbox{e$-$}4$, as those configurations contain diverged results.

[]{#tab:lr_summary_second_model_full label="tab:lr_summary_second_model_full"}

**LR** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

$1\mbox{e$-$}4$ 0.158 0.132 0.775 **0.366** 0.533 0.393
$2\mbox{e$-$}4$ **0.170** **0.148** **0.779** 0.354 **0.543** **0.399**
$5\mbox{e$-$}4$ 0.148 0.112 0.635 0.350 0.439 0.337
: Average performance grouped by learning rate for Qwen2.5-7B-Instruct.

As shown in Table [5](#tab:rank_summary_second_model_full){reference-type="ref" reference="tab:rank_summary_second_model_full"}, LoRA rank has a measurable but relatively small impact on Qwen2.5-7B performance: the average score improves from 0.388 (rank 32) to 0.402 (rank 128), while rank 256 is comparable at 0.397. Overall, ranks 64--128 form a tight trade-off region, with rank 128 best on average but only marginally better than lower ranks. Compared to the 3B model, the 7B results are more tightly clustered across ranks, indicating that rank is a weaker lever here than it is for smaller backbones.

[]{#tab:rank_summary_second_model_full label="tab:rank_summary_second_model_full"}

**Rank** **%TP** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

32 1.06% 0.152 0.119 0.776 0.357 0.534 0.388
64 2.12% 0.159 0.150 0.779 0.355 0.535 0.396
128 4.24% **0.178** 0.141 **0.784** **0.367** **0.541** **0.402**
256 8.48% 0.165 **0.152** 0.771 0.359 0.536 0.397
: Average performance grouped by LoRA rank for Qwen2.5-7B-Instruct. %TP denotes the percentage of trainable parameters.

Similarly, Table [6](#tab:batch_summary_second_model_full){reference-type="ref" reference="tab:batch_summary_second_model_full"} shows that batch size has a negligible effect on Qwen2.5-7B performance in our sweep. The average accuracy varies only from 0.394 to 0.398. This mirrors the 3B trend, suggesting batch size can largely be chosen based on computational constraints once learning rate is in a stable regime.

[]{#tab:batch_summary_second_model_full label="tab:batch_summary_second_model_full"}

**Batch size** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

32 0.154 **0.148** **0.781** 0.355 0.530 0.394
64 0.164 0.146 0.774 **0.366** 0.538 **0.398**
128 **0.172** 0.127 0.776 0.358 **0.541** 0.395
: Average performance grouped by batch size for Qwen2.5-7B-Instruct.

As a result of the hyperparameter study, it follows that the setup with learning rate value of $2\mbox{e$-$}4$, batch size of 64, and LoRA rank set to 128 results in consistently good results while providing a good compromise between efficiency and training stability.

# Dynamic LoRA Routing via the Switcher Module {#sec:switcher}

While reasoning models excel at complex problem-solving, not every user query requires an exhaustive, multi-step CoT. For standard conversational prompts or straightforward factual questions, generating long reasoning traces incurs unnecessary latency and computational overhead which can be a critical bottleneck for edge devices. To address this, we introduce a lightweight _Switcher_ module that enables dynamic adapter routing. By analyzing the user prompt, the switcher decides whether to bypass or activate the reasoning-specific LoRA adapters. When disabled, the system operates as the highly efficient original instruct model for regular conversation; when activated, it seamlessly transitions into a specialized reasoning engine.

Architecturally, the switcher serves as an auxiliary classification head on top of the base LLM and operates during the prefilling stage. It processes the hidden states from the final transformer layer and computes an averaged sequence representation. Based on this representation, the switcher performs binary classification to determine whether the input sequence corresponds to a reasoning-oriented task. If classified as such, the reasoning LoRA adapters are activated for subsequent decoding.

The switcher head is implemented as a lightweight multilayer perceptron (MLP) with a single hidden dimension of $8$, a ReLU activation function, and a dropout rate of $p = 0.2$. This compact architecture ensures negligible overhead for efficient on-device inference while maintaining sufficient expressive capacity for sequence-level classification.

On edge devices, the prefill phase is heavily compute-bound. Processing a long input prompt in a single pass can incur prohibitive computational overhead. To mitigate this, practical on-device implementations typically divide the prefill sequence into smaller, discrete chunks. The switcher module is explicitly designed to support this chunked prefill strategy. Rather than buffering the hidden states of the entire prompt to compute a global average, the switcher updates its sequence representation on the fly. Specifically, we compute a running exponential moving average of the hidden states across these chunks. In our setup, we use a chunk size of $128$ tokens with a smoothing coefficient of $\alpha = 0.5$. To enhance robustness to quantization artifacts, we inject independent Gaussian noise with zero mean and standard deviation $\sigma = 0.5$ into the averaged representation during training.

#### Masked LoRA training for KV-cache reuse.

A major challenge with dynamically activating LoRA adapters at inference time is KV-cache compatibility. Under standard LoRA training, the model expects the KV cache for the prompt tokens (the prefill phase) to be generated with the LoRA adapters fully active. If a query is routed to the reasoning mode after the base model has already encoded the prompt, the system would typically need to re-encode the entire prompt with the LoRA adapters activated to generate a compatible KV cache. On edge devices, this re-encoding incurs a severe latency and compute penalty. To eliminate this inefficiency, we introduce a _masked LoRA_ training strategy. During the fine-tuning of the reasoning adapters, we mask (disable) the LoRA weights during the forward pass of the prompt tokens, activating them only for the generation of the response tokens. This forces the LoRA adapters to adapt to the prompt KV cache generated strictly by the base model. Empirically, we observe that this strategy incurs no drop in reasoning accuracy, while allowing the base model and reasoning mode to seamlessly share a single prefill KV cache, entirely obviating the need to re-encode prompt tokens when switching.

## Training details

To train the switcher, we constructed a small dataset that combines both straightforward conversational or knowledge based queries and complex queries, enabling the model to learn how to distinguish when reasoning is required. The dataset is structured as follows: each entry consists of a question prompt paired with a label 0 or 1, indicating low or high complexity, respectively. Complexity is determined based on the source of the dataset from which the prompt was sampled. We included prompts from datasets covering math and non-math domains to reduce the risk that the switcher overfits to domain-specific cues (e.g., assuming all math questions are inherently more difficult).

The final dataset contains approximately 2k samples. Easy queries were randomly drawn from the SQuAD2.0 dataset (600 questions) [@rajpurkar2018knowdontknowunanswerable], which consists primarily of general-knowledge comprehension questions, and from the MMLU math subset (419 questions) [@hendrycks2021measuring], which includes straightforward mathematical problems. Hard prompts were sourced from a subset of the S1K dataset (500 questions) [@muennighoff2025s1], encompassing challenging questions spanning math, science, and crosswords, as well as from StrategyQA (500 questions) [@geva2021didaristotleuselaptop] covering reasoning questions from non-scientific domains.

## Results

The primary motivation for the switcher module is to optimize standard, day-to-day user interactions. In real-world edge deployments, the vast majority of user queries are simple conversations, factual lookups, or basic instructions that do not require multi-step reasoning. By aggressively thresholding the switcher to route these simple queries to the base instruct model, we can achieve massive aggregate savings in token generation, latency, and power consumption, reserving the LoRA reasoning adapters strictly for complex tasks.

<a id="fig:switcher_pareto"></a>

![](./images/switcher_pareto.png)

> Figure 7: **Impact of the Switcher module on MATH500.** **Left:** Combined model accuracy as the fraction of queries routed to the reasoning adapters. **Right:** Average completion length versus overall accuracy across different switcher thresholds.

To rigorously assess the impact of this dynamic routing, we evaluated the switcher's performance on the challenging MATH500 benchmark as it contains questions with varying complexity. We swept across different switcher confidence thresholds to vary the fraction of prompts routed to the reasoning adapters versus the base model. Figure [3](#fig:switcher_pareto){reference-type="ref" reference="fig:switcher_pareto"} illustrates how overall model performance changes as a larger fraction of queries is routed to the reasoning model. Here, we study the base Qwen2.5-7B-Instruct model and its counterpart trained on OT3 with LoRA adapters of rank 128 without any budget forcing.

As more answers are generated with reasoning, accuracy rises smoothly from the base model baseline toward the reasoning-only upper bound. This demonstrates that the switcher effectively prioritizes the reasoning model on more complicated queries where it is most beneficial, allowing accuracy levels that cannot be achieved by the base model alone.

The right panel shows the corresponding computational cost, measured as average completion length. Choosing a higher-accuracy operating model requires a proportional increase in computational costs, while lower-cost regimes are possible when accuracy demands are modest. The switcher thus provides a flexible mechanism for navigating this tradeoff.

# Budget Forcing and Inference-Time Compute Optimization {#sec:budget_forcing}

CoT prompting [@wei2022chain] scales Inference-Time Compute (ITC) by decoding intermediate reasoning steps, substantially improving LLM performance on complex tasks, but often at the cost of high latency and large token footprints. Theoretical analysis [@zhang2025laws] argue that the optimal test-time compute should scale linearly with problem difficulty. Yet unconstrained models routinely violate this optimality, exhibiting degenerate verbosity and overthinking even on trivial tasks [@muennighoff2025s1].

To reconcile reasoning capability with computational efficiency, _Budget Forcing_ [@muennighoff2025s1] methods aim to align generation with explicit token/compute constraints. RL-based methods, among others [@xu2025cod; @renze2024ccot; @wang2025ton; @huang2025hapo], achieved impressive trade-off between performance and CoT length reduction. These methods typically require to augment the reward with a length-based penalty term [@aggarwal2025l1] or to enforces hard truncation constraints upon reaching a target budget [@liu2025dler]. Formally, a standard budget-forced reward objective can be expressed as an additive penalty: $$\label{eq:penalty}
    R(y,x) = R_{\text{accuracy}}(y,x) - \lambda \cdot R_{\text{budget}}(L)$$ where $x$ is the prompt, $y$ is the generated response, $R_{\text{accuracy}}(y,x)$ is the accuracy reward, $L$ represents the total token length, and $R_{\text{budget}}(L)$ is a penalty function scaled by the hyperparameter $\lambda$.

## Soft-Barrier Reward Formulation {#subsec:reward_design}

Building upon the foundations of budget-penalized RL, our reward is based on three core rationales:

1.  **Avoidance of strict token matching:** We do not force the model to exactly match a predefined budget. Doing so assumes perfect a priori knowledge of the optimal compute required for a specific task, which contradicts the premise of generative exploration.

2.  **Trajectory exploration:** The model must retain sufficient degrees of freedom to explore diverse reasoning paths without premature truncation.

3.  **Prompt-adherent budget compliance:** The model must reliably satisfy the user-defined budget constraints provided in the prompt.

To realize these principles, we prompt the model with discrete generation length budgets, specifically bucketing constraints into $1000$, $3000$, $4000$, and $6000$ tokens. Instead of an additive penalty, we introduce a multiplicative, piecewise-linear _soft barrier_. This barrier decays the budget reward from $1.0$ to $0.0$ as the generation length exceeds the prompted bucket. The linear decay serves as a buffer, discouraging the model from exceeding the limit without inflicting catastrophic penalties for minor budget infractions.

This decay operates within a symmetric window centered around the target budget $B$, where the half-size of the window, $m \in [0,1]$, is treated as a tunable hyperparameter. Formally, we define the budget reward modifier as: $$\label{eq:budget}
    R_{\text{budget}}(L) = 
    \begin{cases}
        1 & L \le L_{\text{low}} \\
        p & L > L_{\text{high}} \\
        1 - (1-p)\frac{L - L_{\text{low}}}{L_{\text{high}} - L_{\text{low}}} & L_{\text{low}} < L \le L_{\text{high}}
    \end{cases}$$ where $L$ is the total length of the generated response, $p$ is the maximum budget penalty floor, $L_{\text{low}} = (1-m)B$, and $L_{\text{high}} = (1+m)B$. Through empirical observation, we noted that setting a negative penalty floor provided no optimization benefits; thus, we set $p = 0$.

The final holistic reward $R$ is then defined as the product of the task accuracy and the budget compliance modifier: $$\label{eqn:bf_reward}
    R(y,x) = R_{\text{accuracy}}(y,x) \times R_{\text{budget}}(L)$$ where $R_{\text{accuracy}}(y,x) \in \{0, 1\}$ represents the binary accuracy reward.

#### Challenges and reward hacking.

Because budget forcing imposes a strict constraint on the optimization manifold, it inherently induces a trade-off between reasoning performance and compute cost. We observed that naively applying a length penalty (for instance, penalizing _only_ the tokens within the CoT reasoning trace) is highly susceptible to reward hacking. During early iterations, the policy rapidly collapses into a degenerate, "lazy" strategy: the model learns to circumvent the penalty by prematurely closing the reasoning block with a `</think>` token, only to continue its verbose CoT in the final response output. By penalizing the total generation length $L$ rather than just the reasoning trace, our multiplicative formulation effectively neutralizes this exploit. Furthermore, while our final reward formulation strips away explicit format-following rewards, empirical evaluations confirm that the model consistently maintains the desired structural formatting throughout training.

## Experimental Setup {#subsec:experimental_setup}

### Training Details

To demonstrate the efficacy of our soft-barrier reward formulation in compressing CoT reasoning, we conduct extensive experiments on state-of-the-art reasoning models. We utilize the DeepScaleR dataset [@luo2025deepscaler] as our primary training corpus. To maximize training stability and prevent degenerate optimization steps, we apply a rigorous filtering criterion to the dataset: any prompt exhibiting a group reward standard deviation of zero is removed, ensuring that the model always receives a meaningful comparative signal during policy updates.

We optimize our models using GRPO [@shao2024deepseekmath]. GRPO is particularly well-suited for reasoning tasks as it bypasses the need for a separate value model by leveraging group-scaled rewards. For a given prompt $x$, the objective minimizes the following loss: $$\mathcal{L}_{\mathrm{GRPO}}(\theta \mid x) = -\frac{1}{G}\sum_{i=1}^{G} \min\!\Big( \rho_i\,A_i,\; \operatorname{clip}(\rho_i,\,1-\epsilon,\,1+\epsilon)\,A_i \Big) + \beta\,D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\big\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\big),$$ where $G$ denotes the group size, and the probability ratio $\rho_i$ and the advantage $A_i$ are defined as: $$\rho_i = \frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}, \qquad A_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}$$ Here, $\mu_r$ and $\sigma_r$ represent the mean and standard deviation of the rewards within the group, respectively: $$\mu_r = \frac{1}{G}\sum_{j=1}^{G} r_j, \qquad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^{G} (r_j - \mu_r)^2}$$

Our implementation is built on the `trl` library (version 0.26.2) [@vonwerra2020trl]. We execute training on a single compute node equipped with 8 NVIDIA H100 (80GB) GPUs. We sample 8 generations per prompt ($G=8$) during the GRPO rollouts. A comprehensive summary of the training hyperparameters is detailed in Table [12](#tab:hyperparams){reference-type="ref" reference="tab:hyperparams"} in [12](#app:budget_forcing).

### Evaluation Details

[To assess the impact of our budget forcing technique on mathematical reasoning, we use the large-scale Math500 [@lightman2023lets] benchmark as our main testbed. ]{style="color: black"} For robust and reproducible evaluation, we employ the `lighteval` framework (version 0.8.1). All inference processes are accelerated using vLLM (version 0.10.2). To standardize the assessment of pass@1 accuracy, we apply a consistent sampling strategy across all benchmarks: generations are sampled with a temperature of 0.6, a $\text{top}\_\text{p}$ of 0.95, and a maximum completion length extended to 32K tokens to accommodate any lingering verbose trajectories from the baseline models.

## Results: Efficiency-Accuracy Trade-off {#subsec:main_results}

As established, our primary objective is to compress the generated reasoning trajectories with minimal degradation in task performance. Because our soft-barrier reward formulation deliberately omits a strict regularizer weight for the budget penalty (see eq. [
$$eq:penalty$$
](#eq:penalty){reference-type="ref" reference="eq:penalty"}), we discovered that the Kullback-Leibler (KL) divergence penalty coefficient in GRPO ($\beta_{\text{KL}}$) serves as an effective control mechanism to enforce budget-friendly behavior. Empirically, setting $\beta_{\text{KL}} = 10^{-3}$ yields the optimal balance, significantly reducing generation length with negligible performance drops. Conversely, a relaxed penalty of $\beta_{\text{KL}} = 10^{-4}$ improves formatting adherence at very short completion lengths, albeit at the cost of a slightly higher performance regression when evaluated on larger, unbounded contexts.

Figure [4](#fig:cdf_avg_len){reference-type="ref" reference="fig:cdf*avg_len"} illustrates the average completion length distributions for the unconstrained baseline (purple) and two intermediate checkpoints trained with $\beta*{\text{KL}} = 10^{-3}$. The left and right panels depict evaluations where the maximum completion length is strictly capped at 4K and 6K tokens, respectively. To enforce this hard budget during inference, we abruptly truncate the generation upon hitting the token limit and subsequently append a prompt forcing the model to immediately output its final answer.

![ **Average Completion Length Distributions.**   **Left:**  Evaluation with a forced maximum completion length of 4K tokens.  **Right:**  Evaluation with a maximum of 6K tokens. Note that distribution tails extending below zero or above the maximum budget are standard artifacts of Kernel Density Estimation (KDE) curve smoothing. The progression from the baseline ([purple]{style="color: purple"}) through the intermediate ([blue]{style="color: blue"}) to the final RL fine-tuned checkpoint ([green]{style="color: green"}) demonstrates stable, progressive learning of concise generation ($\beta_{\text{KL}}=10^{-3}$).](media/cdf_avg_len.pdf){#fig:cdf_avg_len width="\\linewidth"}

As demonstrated in Figure [4](#fig:cdf_avg_len){reference-type="ref" reference="fig:cdf_avg_len"}, our RL fine-tuning effectively shifts the distribution density toward significantly shorter lengths. Crucially, the transition from the baseline (purple) through the intermediate checkpoint (blue) to the final policy (green) highlights a stable, progressive optimization trajectory. Rather than experiencing sudden, erratic policy collapse, the model smoothly and monotonically learns to generate more concise reasoning traces over the course of training to solve the tasks.

To quantify this compression, Figure [5](#fig:compl_dist_comparison){reference-type="ref" reference="fig:compl_dist_comparison"} provides a granular breakdown of the actual CoT length reductions achieved via our RL fine-tuning. Specifically, the right panel of Figure [5](#fig:compl_dist_comparison){reference-type="ref" reference="fig:compl_dist_comparison"} reveals that our approach yields an average completion length reduction factor of $\sim 2.4\times$, with maximum compression rates reaching up to $\sim8\times$ on certain queries. As noted previously, this aggressive reduction in verbosity is achieved while maintaining comparable performance to the base model, with only minimal-and in many instances, negligible-accuracy drops. Practically, this reduction directly translates to lower overall inference latency and a faster time-to-final-answer, making advanced reasoning models significantly more viable for deployment in resource-constrained environments.

![ **Average Completion Length Comparison.**   **Left:**  C.D.F. of the average completion length for base model, [orange]{style="color: orange"} curve, and RL fine-tuned one, [green]{style="color: green"} curve, with $\beta_{KL}=1.e^{-3}$. We considered a maximum completion length of 6K.  **Right:**  Reduction in completion length from the RL fine-tuned model. We use the same models in the left plot. The RL fine tuned achieved average reduction length of $2.38 \pm 0.07$.](media/compl_dist_comparison.pdf){#fig:compl_dist_comparison width="\\linewidth"}

\@l\*4YZc@ **Model** &\

& **Budget = 1K** & **Budget = 2K** & **Budget = 4K** & **Budget = 6K** & **Average (Budget = 32K)** \
SFT Baseline (r=128) & 34 & 57 & 73 & 83 & **95\
BF RL${}^{\beta_{KL}=1.e-3}$ & [62]{.underline} & [78]{.underline} & **85 & ** 90 & [92]{.underline}\
BF RL${}^{\beta_{KL}=1.e-4}$ & **72 & \*\* 80 & [84]{.underline} & [85]{.underline} & 90\

---

<a id="fig:cot_example_2"></a>

> Figure 2: <strong>Qualitative comparison on algebraic simplification.</strong> <strong>Middle:</strong> The Baseline trace correctly identifies the difference of squares strategy immediately but engages in excessive self-verification, re-calculating the result via expansion, direct computation, and alternative factorizations. <strong>Bottom:</strong> The Budget Forced trace recognizes the nested difference of squares structure and executes the solution linearly without redundant checking.

## Qualitative Analysis of Budget-Forced CoT {#subsec:qualitative_examples}

To examine the mechanics of our budget forcing objective at the trajectory level, we qualitatively compare the reasoning traces of the unconstrained baseline against our budget-forced model across four distinct mathematical domains: number theory, algebraic simplification, pattern recognition, and modular arithmetic (Figures [6](#fig:cot_example_2){reference-type="ref" reference="fig:cot_example_2"}--[7](#fig:cot_example_4){reference-type="ref" reference="fig:cot_example_4"} and Figures [9](#fig:cot_example_1){reference-type="ref" reference="fig:cot_example_1"}--[10](#fig:cot_example_3){reference-type="ref" reference="fig:cot_example_3"} in [12](#app:budget_forcing)). A consistent pattern emerges from this analysis: the unconstrained baseline frequently suffers from severe epistemic hesitation. While it typically identifies the correct logical strategy early in the generation process, it expends thousands of tokens on redundant self-verification, testing alternative (and often less efficient) methods, and hypothesizing trivial errors. For instance, in Figure [6](#fig:cot_example_2){reference-type="ref" reference="fig:cot_example_2"} and Figure [7](#fig:cot_example_4){reference-type="ref" reference="fig:cot_example_4"}, the baseline arrives at the correct answer almost immediately but falls into extensive validation loops, re-calculating the result using three to four different approaches.

In contrast, our budget-forced policy learns to confidently trust its initial, correct logical derivations. It successfully prunes these redundant validation loops and verbose syntactic parsing, drastically reducing the trace length while strictly preserving the essential reasoning backbone and human readability. Furthermore, although we omit special delimiter tokens (e.g., `<think>`) in the visualizations for brevity, we observe that the budget-forced model robustly generates them, maintaining the required CoT and final-answer formatting constraints.

<a id="fig:cot_example_4"></a>

> Figure 3: <strong>Qualitative comparison on modular arithmetic.</strong> <strong>Middle:</strong> The Baseline trace correctly computes the sum and remainder immediately but engages in extensive, redundant verification using four different methods (step-by-step addition, digit sum rule, pairing, and re-calculation). <strong>Bottom:</strong> The Budget Forced trace performs the direct calculation and returns the result without hesitation.

# Parallel Test-Time Scaling and Reasoning {#sec:parallel}

During autoregressive generation with LLMs, one of the dominant runtime bottlenecks is repeatedly loading model weights of the individual layers to produce the next token. As a result, there is often potential to increase the total amount of computation performed per user query without incurring significant overhead in runtime. A straightforward way to do this is to generate multiple independent samples in parallel: instead of producing one CoT trajectory and one final answer, this approach allocates additional compute to produce several trajectories concurrently.

Parallel generation is not only attractive from a systems perspective, but it also improves accuracy. Across benchmarks and models, generating multiple candidate solutions and then aggregating them has repeatedly shown consistent performance gains. A common and simple aggregation strategy is majority voting over individual answers, where the most frequently produced answer is selected as the final answer. In practice, majority voting is widely used as a reliable tool for achieving an additional boost in final model performance from the same underlying base model.

Initial works [@cobbe2021training; @wang2022self] indicated that the same LLM can be made to generate multiple diverse and independent responses towards finding a better solution. Scaling compute by sampling multiple independent responses prior to aggregation has shown [@brown2024large; @wu2025inference; @snell2025scaling] to be an effective paradigm enabling smaller models to outperform larger models for the same inference compute budget. In these cases, the key is to score the solutions (e.g., by frequency, or using an external verifier) and predict the highest-scoring solution. Parallel reasoning design relies on selecting a sampling scheme, design and employ reward models, and finally aggregation scheme to generate the final answer. Diversity among responses [@brown2024large] is crucial to increasing the probability of sampling the correct response. Increasing diversity is typically done by auto-regressive generation at a higher sampling temperature [@renze2024effect]. While responses are typically sampled independently, recent works explore inter-dependent sampling [@hsu2025group; @pan2025learning; @rodionov2025hogwild; @zheng2025parallel] and guided-search [@yao2023tree; @ning2023skeleton; @li2025enhancing] towards generating a better response candidate pool. In this work, we focus on independent sampling schemes. Reward models (also referred to as verifiers) estimate one or more scalar-valued scores for a given response. There is a long line of work on improving reward models, with more accurate reward models [@liu2024skywork; @wang2024helpsteer2], more granular score assignment (e.g., per-step scores with process reward models) [@lightman2023let; @wang2023math; @zhang2025lessons], scaling verification with more compute [@liu2025inference], and extending reward models to other reasoning domains [@chae2025web; @chen2025scaling] beyond mathematical reasoning. In this work, we use the reward models for scoring the outcome. Finally, aggregation is essential to drafting a specific response from a pool of multiple response-score pairs. Existing works have primarily looked at this from a rank-and-select lens: to rank the responses either by frequency of occurrence ('self-consistency' or 'majority voting') [@wang2022self], or using external reward models to score responses [@cobbe2021training; @wang2023math]. Note that these strategies lead to a zero-sum situation: the top-ranking solution is selected and the rest discarded. Consequently, a more recent line of work [@khairi2025making; @li2025drafts; @qi2025learning; @zhao2025majority] investigates synthesizing (rather than selecting) a final response based on the candidate responses.

Scaling compute at test-time generally implies additional latency and compute overhead and hence making it especially challenging to realize on resource-constrained edge devices. Some benefits of parallel TTS on edge devices were discussed in [@hao2025scaling] with a separate verifier. This separation prevents efficient reuse of intermediate computations, e.g., through the KV-cache. Consequently, scaling parallel compute _efficiently_ has gained some attention in the research community. Towards efficiently reasoning, several works have investigated generating solutions by performing fewer FLOPs [@wu2025inference] and reducing the memory footprint [@sun2024fast; @hooper2025ets] required during generation. Closest to our verifier design presented in this paper is GenRM [@zhang2025generative], where the authors investigate finetuning the base generative model to additionally perform prompt-based verification. Such a joint generation-verification paradigm is appealing for edge devices, since generation and verification can be performed with minimal movement of parameters between DRAM and flash memory.

## Efficient Verifier Design for Edge Compute

Parallel reasoning produces a set of $N$ independent CoT traces and corresponding final answers. The central question then becomes: given these $N$ candidates, how do we reliably choose the correct one? Majority voting provides a strong baseline, but it is not always sufficient, especially when the answer space is large or ambiguous. This motivates introducing an explicit selection mechanism that can score candidates and prefer those most likely to be correct, while still benefiting from the diversity created by independent sampling.

A natural approach is to add a verifier model that evaluates each candidate solution. However, deploying a separate verifier on edge devices at the same scale as the generator can be prohibitive in storage and memory footprint, and often becomes especially costly in latency. To avoid this, we aim to reuse the generator as effectively as possible. Concretely, we keep the same base model and add a lightweight verifier head: a separate linear layer applied to the final token embedding, followed by a sigmoid activation to yield a scalar "correctness" score. Figure [1](#fig:overview){reference-type="ref" reference="fig:overview"}b illustrates this approach. This design keeps additional parameters minimal and, crucially, allows KV-cache reuse for all generated responses, since the verifier head can operate on representations already computed during generation.

In addition to the linear head, we append a short verification prompt after each generated response, asking the model whether the proposed solution is correct. Empirically, this extra query has been beneficial compared to relying on a linear head alone without an explicit verification prompt. Operationally, this strategy only requires a small additional prefill step for the verification prompt for each individual generation, while preserving KV-cache reuse from the original generation. The verifier therefore adds only modest overhead per candidate while improving the reliability of the selection process.

To obtain the best results, we combine majority voting with verifier scoring into a weighted majority vote. Instead of counting each candidate answer equally, we weight each candidate's vote by its verifier score. Intuitively, candidates that the verifier deems more likely to be correct contribute more to the final decision, while still retaining the robustness benefits of aggregation across multiple samples. This hybrid approach preserves the simplicity and stability of voting while incorporating a learned notion of solution quality, which is particularly helpful when the candidate set contains a mix of superficially plausible but incorrect solutions alongside correct ones.

## Training and Evaluation Details {#sec:training_and_quantization}

Verification is treated as a binary classification problem where a candidate response of the generator is labeled as correct or incorrect with respect to the ground-truth answer. To generate training data for the verifier, we use 97.5% of the 7.5k questions from the MATH training set [@hendrycks2021measuring]. The remaining 2.5% of questions are reserved for validation of the verifier. To construct a diverse training set and a validation set for fast evaluation, we generate 16 candidate responses for each training question, while we limit ourselves to 4 generated responses in the validation set. The verifier head uses a sigmoid activation to produce a probability-like score, and we train it with binary cross-entropy loss. This setup aligns the verifier objective directly with the downstream selection problem: distinguishing correct from incorrect candidate solutions produced by the generator under the same sampling procedure used at inference time.

[]{#tab:wmv_results label="tab:wmv_results"}

**Parallel Responses** **1** **2** **4** **6** **8**

---

Greedy (baseline) 71.0 \- \- \- \-
Majority Vote 69.9 $\pm$ 1.3 70.0 $\pm$ 1.3 75.1 $\pm$ 1.0 76.6 $\pm$ 1.0 77.5 $\pm$ 0.8
Weighted MV (ours) 69.9 $\pm$ 1.3 72.7 $\pm$ 1.0 76.1 $\pm$ 0.9 77.5 $\pm$ 0.8 78.2 $\pm$ 0.7
: Accuracy of our lightweight verifier weighted majority voting compared to majority voting without verifier and greedy decoding on MATH500. The mean and standard deviation are computed from 20 random draws from 16 independent 4bit-weight-quantized Qwen-2.5-7B-Instruct responses.

## Results

We evaluate the proposed lightweight verifier on MATH500 using a 4-bit-weight-quantized Qwen-2.5-7B-Instruct model, comparing greedy decoding, standard majority voting, and our weighted majority voting. Table [7](#tab:wmv_results){reference-type="ref" reference="tab:wmv_results"} summarizes accuracy as a function of the number of parallel responses, sampled with temperature set to 0.7.

Even with very limited parallelism, parallel test-time scaling yields immediate benefits. With just two parallel responses, weighted majority voting improves accuracy to 72.7%, outperforming both the greedy baseline (71.0%) and standard majority voting (70.0%). This highlights a key advantage of incorporating verification: when two sampled responses disagree, majority voting cannot break ties, whereas the verifier-weighted scheme can consistently select the more reliable candidate.

As the degree of parallelism increases, both majority voting and weighted majority voting exhibit steady gains, confirming prior observations that parallel sampling improves the probability of generating a correct solution. However, weighted majority voting consistently outperforms unweighted majority voting across all parallelism levels, and at eight parallel responses, weighted majority voting improves upon the baselines by 10%. Importantly, the variance across random draws is also slightly reduced compared to majority voting, suggesting that verifier weighting provides a more stable aggregation mechanism.

Despite its simplicity, the verifier delivers strong benefits. Architecturally, it amounts to generating only a minimal amount of overhead, effectively one extra token per stream. Because the verifier reuses the generator's KV-cache, it does not require reprocessing the original prompt or response, avoiding the dominant memory and latency costs typically associated with separate verifier models. This makes the performance gains particularly compelling in edge settings, where memory bandwidth and storage are tightly constrained.

# Quantization {#sec:quantization}

In the following section, we discuss the details of our quantization methodology. We begin by providing a brief overview of neural network quantization and a summary of recent methods for quantizing LLMs in Section [7.1](#sec:quantization_background){reference-type="ref" reference="sec:quantization_background"}.

We outline our strategy for quantizing base LLM in Section [7.2](#sec:quantization_base){reference-type="ref" reference="sec:quantization_base"} demonstrated on Qwen2.5-7B-Instruct, and later equip it with reasoning capabilities in Section [7.3](#sec:qarm){reference-type="ref" reference="sec:qarm"}. Lastly, we provide the specifics of quantized model export and deployment on device in Section [7.4](#sec:quantization_on_device){reference-type="ref" reference="sec:quantization_on_device"}.

## Background and Related Work {#sec:quantization_background}

#### Quantization.

Neural network quantization is one of the most powerful ways to reduce model footprint, data transfer and compute requirements [@krishnamoorthi2018quantizing; @nagel2021white]. By quantizing a model, high bit-width floating point weights and activations can be represented using low-bit numbers. Next to reducing model size, the use of low-bit fixed-point representations, such as [INT8]{.sans-serif}, can also significantly reduce the latency and energy consumption [@horowitz].

We use the following definition of the quantization-dequantization function: $$\widehat{{\bm{x}}} := q\left({\bm{x}};\,{\bm{s}},{\bm{z}},b\right) = {\bm{s}}\cdot 
    \vphantom{\Bigg(} \Big(\,\smash{\underbrace{\mathop{\mathrm{clip}}\!\left(\ensuremath{\left\lfloor{\frac{{\bm{x}}}{{\bm{s}}}}\right\rceil}+{\bm{z}};-2^{b-1},2^{b-1}-1\right)}_{\text{\normalsize $=: {\bm{x}}_\mathbb{Z}$}}} - {\bm{z}}\Big),
    \label{eq:dequant}$$\
where ${\bm{x}}$ denotes the quantizer input (i.e., network weight or activation tensor), ${\bm{s}}$ the high precision ([FP32]{.sans-serif} / [FP16]{.sans-serif} / [BF16]{.sans-serif}) quantization scale, ${\bm{z}}$ the integer zero offset, and $b$ the bitwidth. $\ensuremath{\left\lfloor{\cdot}\right\rceil}$ denotes the round-to-nearest-integer operator. ${\bm{x}}_\mathbb{Z}$ is a $b$-bit integer *quantized representation* of the input ${\bm{x}}$. Quantization parameters ${\bm{s}}$, ${\bm{z}}$ can be shared across the components of ${\bm{x}}$ (typically per-channel or block-wise). This quantization scheme is called *uniform affine* or *asymmetric* quantization [@hubara2017quantized; @krishnamoorthi2018quantizing; @zhou2016dorefa] and is one of the most commonly used quantization schemes because it allows for efficient implementation of fixed-point arithmetic. In the case of *symmetric* quantization, we restrict the quantization grid to be symmetric around ${\bm{z}}={\bm{0}}$.

Quantization methods can generally be categorized into *post-training quantization* (PTQ) and *quantization-aware training* (QAT) families. PTQ algorithms convert pretrained high-precision networks directly into fixed-point models without the need for the original training pipeline [@banner2018post; @cai2020zeroq; @choukroun2019low; @hubara2020improving; @meller2019same; @zhao2019improving; @Nagel_2019_ICCV; @nagel_up_2020; @li2021brecq]. These approaches are fast, easy to use, and typically rely only on a small calibration dataset. In contrast, QAT methods [@gupta2015deep; @jacob2018quantization; @lsq; @nagel_oscillations_2022] simulate quantization during training to find more optimal solutions, but generally require longer training, more memory, labeled data, and careful hyperparameter tuning.

#### LLM Quantization.

The excessive training cost and memory usage of traditional QAT methods renders them less practical for quantizing modern LLMs, although some works such as LLM-QAT [@liu2023llm] and BitDistiller [@du2024bitdistiller] explore QAT with knowledge distillation. Notably, [@liu_paretoq_2025; @chen2025scaling_qat] are the only studies we are aware of that successfully scale QAT to billions of tokens. Several papers explored the combination of QAT and parameter-efficient fine-tuning (PEFT), including [@dettmers2024qlora; @xu2023qa; @li2023loftq; @guo2023lq; @kim2024memory; @bondarenko2024low]. Most of these approaches offer a substantial memory reduction compared to traditional QAT, but generally are not focused on inference efficiency. For instance, QLoRA [@dettmers2024qlora] quantizes the pretrained weights to 4 bit using (a non-uniform) [NF4]{.sans-serif} format but dequantizes them in the forward pass back to [BF16]{.sans-serif}.

Post-training quantization of LLMs is a challenging task due to presence of strong numerical outliers in weights and activations [@bondarenko_understanding_2021; @kovaleva_bert_2021; @dettmers_gpt3_int8_2022; @bondarenko_quantizable_2023; @sun_massive_2024]. The core challenge is that quantizing outliers onto a fixed-point grid forces a range-precision trade-off: increasing the dynamic range captures outliers but sacrifices precision near zero, while retaining precision requires clipping them -- both of which strongly degrade model performance.

Existing LLM PTQ methods can be broadly categorized into *weights-only* quantization and *weight-activation* quantization. Weights-only quantization focuses on converting weights to low-bit values. GPTQ [@frantar_gptq_2022] employs second-order information to iteratively round grouped weights and correct the quantization error in the remaining groups. SpQR [@dettmers2023spqr], AWQ [@lin2023awq] and OWQ [@lee2024owq] emphasize the importance of so-called "salient" weights that correspond to high-magnitude activations. Other recent W-only methods include [@jeon2023frustratingly; @lee2023flexround; @luo2023long; @chee2024quip]. Weight-activation quantization compresses both weights and activations. SmoothQuant [@xiao_smoothquant_2024], `LLM.int8()` / `GPT3.int8()` [@dettmers_gpt3_int8_2022] and Outlier Suppression [@wei2022outlier] achieve W8A8 quantization by managing activation outliers. `LLM.int8()` uses mixed-precision decomposition, while the other two employ channel-wise scaling. Some of the other recent W&A PTQ methods are [@lee2023enhancing; @liu2023qllm; @wei_outlier_2023; @yuan2023rptq; @tang2024easyquant; @yao2022zeroquant; @lin2024qserve].

#### LLM Quantization using FPTs.

A promising direction in LLM quantization is the use of *rotations* and other *function-preserving transformations* (FPTs). @Nagel_2019_ICCV first explored FPTs for CNN quantization, showing that ReLU and per-channel scaling commute, enabling cross-layer rescaling of weights. In the LLM setting, @xiao_smoothquant_2024 propose migrating problematic outliers from the activations into the weights through online per-channel scaling applied before linear layers. Follow-up work extends this idea by incorporating shifts into the scaling [@wei_outlier_2023], scaling vectors for queries and keys [@shao_omniquant_2024], channel-mixing transforms [@chee2024quip], randomized Hadamard transforms to reduce outliers [@ashkboos_quarot_2024; @liu_spinquant_2024], other online rotations [@lin_duquant_2024], combinations of scaling and rotations [@hu_ostquant_2025], and Kronecker-structured matrix transforms [@sun_flatquant_2024].

Recently, FPTQuant [@van2025fptquant] introduced three novel, lightweight, and expressive FPTs to facilitate quantization of transformers. By leveraging the equivariances and independencies inherent to modern transformers, these FPTs are designed to maintain the model's function while shaping the intermediate activation distributions to be more quantization friendly. As a result, FPTQuant enables static [INT4]{.sans-serif} quantization with virtually no overhead and no custom kernels, is very fast and performs on par or exceeds most prior work.

## Quantizing Base Language Model {#sec:quantization_base}

#### Quantization setup.

To run the model efficiently on our hardware, we quantize the weights of all linear layers including the final LM head using [INT4]{.sans-serif} per-channel uniform affine quantization (eq. [
$$eq:dequant$$
](#eq:dequant){reference-type="ref" reference="eq:dequant"}). To further improve efficiency and reduce latency, we use [INT8]{.sans-serif} KV-cache, [INT8]{.sans-serif} input embeddings and [INT16]{.sans-serif} for all remaining activations (all per-tensor). For brevity, we will refer to this configuration as 'W4A16KV8'. We use symmetric quantization for weights, KV-cache and embeddings, and asymmetric quantization for activations, which is a common setting.

#### Transformations.

To maximize the accuracy of the quantized model, we apply the subset of fully-mergeable transformations from FPTQuant [@van2025fptquant] (Figure [8](#fig:quantization_transforms){reference-type="ref" reference="fig:quantization_transforms"}):

- a pair of pre-RoPE transforms $(\mathscompmodern{T}_k$, $\bar{\mathscompmodern{T}}_k)$, where $\mathscompmodern{T}_k$ is applied to keys and $\bar{\mathscompmodern{T}}_k$ can be interpreted as an inverse of $\mathscompmodern{T}_k$, applied to the queries;

- $(\mathscompmodern{T}_u,\mathscompmodern{T}^{-1}_u)$ a per-channel scaler merged into up and down projection weights;

- multi-head value transforms $(\mathscompmodern{T}_v,\bar{\mathscompmodern{T}}_v)$, which consist of invertible matrices per head merged into value and output weights;

- and a rotation matrix $(\mathscompmodern{T}_r,\mathscompmodern{T}^{-1}_r)$ for rotating the residuals (applied at the beginning and the very end of each transformer block and is shared).

This set of transformations $\mathscompmodern{T}:= \{\mathscompmodern{T}_k, \mathscompmodern{T}_u, \mathscompmodern{T}_v, \mathscompmodern{T}_r\}$ will help shape the intermediate activation distributions to be more quantization friendly, while keeping the unquantized model outputs intact. All of the above FPTs are fully mergeable, so that we can run the model without any extra inference overhead on our hardware.

<a id="fig:quantization_transforms"></a>

![](./images/quantization_transforms.png)

> Figure 8: **Function-Preserving Transformations** . We use 4 transform types from FPTQuant: scale-and-rotate transform $\mathscompmodern{T}_k$ merged into query and key, a per-channel scaler $\mathscompmodern{T}_u$ merged into up and down projection, and $\mathscompmodern{T}_v$ that consists of invertible matrices per head merged into value and output weights, and a rotation matrix $\mathscompmodern{T}_r$ for rotating residuals (shared across layers). After training of the transforms is complete, the transformation parameters from each 'merge group' are merged into the original model weights ${\bm{\mathsfit{W}}}$.

#### Training and evaluation details.

Following literature [@van2025fptquant; @lee2025unifying], we train the model on DCLM-Edu [@allal2025smollm2], a cleaner filtered version of DCLM [@li2024datacomp] obtained by applying an educational quality classifier [@penedo2024fineweb]. We initialize quantization parameters by minimizing the $L^p$ ($p=2$) norm between quantized and unquantized tensors, and then train transformation $\mathscompmodern{T}$ and quantization parameters $\{{\bm{s}}, {\bm{z}}\}$ end-to-end, closely following the pipeline of FPTQuant. We simulate quantization using FastForward [@fastforward]. For brevity, we denote to the aforementioned quantization pipeline as $\textbf{FPTQuant}^{\boldsymbol{\circ}}$.

To assess the predictive performance of the quantized base language model, we follow the previous work [@frantar_gptq_2022; @xiao_smoothquant_2024; @shao_omniquant_2024; @van2025fptquant; @sun_flatquant_2024], and report WikiText-2 test perplexity (assuming a sequence length of $4096$). We also report an average zero-shot accuracy of a set of common sense reasoning (CSR) tasks, that includes PIQA [@bisk_piqa_2020], WinoGrande [@sakaguchi_winogrande_2021], HellaSwag [@zellers_hellaswag_2019], ARC-e and ARC-c [@clark_think_2018], and LAMBADA [@paperno_lambada_2016]. Finally, we report 5-shot accuracy on MMLU [@hendrycks2020measuring]. For CSR and MMLU evaluation, we use the LM Harness framework [@sutawika2025eleutherai].

#### Results.

We summarize our results for quantized base model in Table [8](#tbl:quantization_results_base_model){reference-type="ref" reference="tbl:quantization_results_base_model"}. As we can see, the simplest PTQ pipeline with min-max range estimation experiences an unacceptable accuracy/perplexity drop. Both strong numerical outliers in the activations (even with 16-bits!) but mainly the catastrophic loss of precision in the quantized 4-bit weights lead to such poor performance.

Employing the set of function-preserving mergeable transformations $\mathscompmodern{T}$ already significantly improves the distribution of weights and activations, leading to much better accuracy, even with min-max range setting. Further, using a better range initialization together with end-to-end learning both progressively recover a greater portion of the full-precision model performance. In the end, we match full-precision accuracy on CSR, and have about 0.4 perplexity drop on WikiText-2 and just under 1.5% accuracy drop on MMLU, where the latter is known to be quite a challenging benchmark. Overall, $\text{FPTQuant}^{\circ}$-quantized base model demonstrates strong performance, given that the entire process took less than 24 hours on a single Nvidia H100 80GB GPU.

**Method** **Bitwidth** $L^p$ $\mathscompmodern{T}$ **train** **WikiText-2** ($\downarrow$) **CSR** ($\uparrow$) **MMLU** ($\uparrow$)

---

Full-precision [BF16]{.sans-serif} - - - 6.85 72.90 74.28
Min-max quantization W4A16KV8 **-** **-** **-** 102.4 51.71 62.35
W4A16KV8 **-** **-** 9.18 65.83 67.59
W4A16KV8 **-** **-** 8.48 67.85 69.06
W4A16KV8 **-** 7.53 70.68 72.26
$\textbf{FPTQuant}^{\boldsymbol{\circ}}$ (ours) W4A16KV8 **7.26** **72.94** **72.81**
: **Quantized W4A16KV8 base model (Qwen2.5-7B-Instruct) results** . We report Wikitext perplexity, average 0-shot CSR, and 5-shot MMLU accuracies. '$L^p$' denotes the use of $L^p$ range initialization, $\mathscompmodern{T}$ = using the set of mergeable transforms, 'train' = end-to-end training of transformation and quantization parameters.

## Quantization-Aware Modular Reasoning {#sec:qarm}

Quantizing the base model will inevitably affect the underlying activation distributions. To achieve the best performance, it is crucial to account for these changes when applying subsequent fine-tuning, including our Reasoning LoRA protocol described in Section [3](#sec:lora){reference-type="ref" reference="sec:lora"}.

Our approach follows the general paradigm of QLoRA [@dettmers2024qlora] and related techniques [@xu2023qa; @li2023loftq], in which LoRA adapters are trained on top of a frozen, quantized base model. To further improve memory and runtime efficiency, we quantize the trained LoRA adapter weights to [INT8]{.sans-serif} and use [INT16]{.sans-serif} activations during inference. As with the base model, we use symmetric per-channel quantization for weights and asymmetric per-tensor quantization for activations. We denote the aforementioned technique as *Quantization-Aware Modular Reasoning* (QAMR).

#### Training and evaluation details.

We follow the training and evaluation protocols described in Section [3](#sec:lora){reference-type="ref" reference="sec:lora"}. For training, we use the OT3 [@guha2025openthoughts] dataset. To assess the reasoning capabilities of our quantized reasoning model, we use a comprehensive set of benchmarks, including AIME 24/25 [@aime], MATH500 [@hendrycks2021measuring], GPQA Diamond [@rein2024gpqa], and AMC23 [@amc23]. We conduct an ablation study to assess the contributions of both the improved base model quantization pipeline and the proposed QAMR approach using a random subset of 50k training examples, while reserving the full training dataset for the final set of results.

#### Results.

We observe in Table [9](#tbl:quantization_results_reasoning_model){reference-type="ref" reference="tbl:quantization_results_reasoning_model"} that a naïvely quantized base model combined with a full-precision reasoning module (i.e., without QARM) is essentially non-functional. Qualitatively such model outputs seemingly random tokens, without any structure or relevance to the tasks at hand.

In contrast, applying QARM -- even with relatively short training -- recovers a substantial portion of the performance on tasks such as MATH500 and GPQA. Notably, quantization-aware modular reasoning is *crucial for learning anything at all*.

Further, using $\text{FPTQuant}^{\circ}$ offers a stronger starting point for training the reasoning module and consistently improves performance across all benchmarks except AMC23, for which longer training is required. By shaping the underlying activation distributions to be more quantization-friendly, using $\text{FPTQuant}^{\circ}$-quantized base model leads to significantly fewer training instabilities and enables faster learning compared to a base model quantized with a standard min-max range setting.

Finally, when combined with extended training, our approach achieves performance within roughly 2% of an equivalently trained full-precision reasoning model on average, while being significantly more compact, and inference-efficient.

       **Bitwidth**        $\textbf{FPTQuant}^{\boldsymbol{\circ}}$    **QAMR**     **N**     **AIME24**     **AIME25**     **MATH500**     **GPQA**     **AMC23**      **Avg**

---

[BF16]{.sans-serif} - - 50k 21.8 20.3 82.6 38.6 65.2 45.70
W4A16KV8 **-** **-** 50k 0.0 0.0 0.0 0.0 0.0 0.00
W4A16KV8 **-** 50k 17.3 6.3 75.6 33.0 **64.0** 39.25
W4A16KV8 50k **23.3** **15.0** **79.6** **33.7** 57.0 **41.72**
[BF16]{.sans-serif} - - 1.2M 53.3 33.0 94.0 39.9 82.5 60.54
W4A16KV8 1.2M **46.6** **36.6** **89.6** **37.8** **80.0** **58.12**
: **Quantized W4A16KV8 reasoning model results (Qwen2.5-7B-Instruct base)** . We report AIME24/25, MATH500, GPQA, and AMC23 accuracies (higher is better). N = number of OT data examples used for training reasoning module.

## Verifier Quantization and On-Device Deployment {#sec:quantization_on_device}

As the final stage of our quantization pipeline, we address the quantization of the verifier, which is essential for deploying the proposed verifier on resource-constrained hardware. To minimize degradation from reduced numerical precision, we train the verifier directly on embeddings produced by the 4-bit weight-quantized Qwen-2.5-7B-Instruct model obtained in section [7.2](#sec:quantization_base){reference-type="ref" reference="sec:quantization_base"}. This choice reduces distribution shift between training and inference, ensuring that the verifier learns to operate on similar representations it will encounter at deployment time. After training the verifier head under this setting, we further quantize both activations and verifier head weights to 8 bit representations using FastForward [@fastforward].

Once all components including the base model, reasoning adapters, and verifier are quantized, we prepare the models for on-device deployment. The first step after quantization is the model transformation to assure compatibility at pytorch representation level with the format supported by GENIE SDK [@GENIE]. These pertains to aspects of autoregressive parallel and sequential generation for prefill and decoding, as well as handling the attention operations and masking, and position embeddings.

Next, we establish compatibility with GENIE at the ONNX representation level. We use Qualcomm FastForward [@fastforward] for this stage and implement transformations at linear layers and multi-head attention, as well as model partitioning. We use Pytorch with FastForward to get the ONNX graph and the associated quantization encodings. We export to Deep Learning Container format, and Quantize any remaining non-quantized nodes missed by FastForward (e.g. biases). We compile for the deployment target (e.g. aarch64-android) and upload to the device using _adb_ (Android Device Bridge).

# Discussions and Challenges

Deploying capable reasoning models on resource-constrained edge devices requires navigating a complex trade-off between task performance, latency, memory footprint, and power consumption. In this work, we proposed a practical end-to-end framework to overcome these limitations. By decoupling reasoning from the base weights using modular LoRA adapters, we demonstrate that parameter-efficient fine-tuning can achieve competitive reasoning accuracy relative to computationally expensive full-parameter distillation methods like DeepSeek-R1-Distill. To optimize day-to-day user interactions, our lightweight dynamic Switcher routes standard conversations to the highly efficient base model, reserving the reasoning adapters strictly for complex queries where multi-step logic is required. To further combat latency, our budget-forced RL alignment explicitly penalizes generation verbosity, yielding a 2.4$\times$ reduction in average reasoning tokens without sacrificing task accuracy. At inference time, we exploit the memory-bound nature of autoregressive decoding by introducing parallel test-time scaling, coupled with a lightweight verifier that provides up to a 10% accuracy boost on complex reasoning benchmarks. Finally, we show that 4-bit weight quantization via FPTQuant and Quantization-Aware Modular Reasoning preserves this robust performance, achieving within 2% of the full-precision reasoning model's accuracy while delivering massive memory savings. Below, we summarize the key insights derived from each component of our pipeline and outline the remaining challenges that pave the way for future research.

#### LoRA for modular reasoning.

Our experiments revealed that parameter-efficient fine-tuning via LoRA is highly effective at eliciting reasoning capabilities in small (3B and 7B) base models, often rivaling the performance of computationally expensive full-parameter distillation. A key insight is that the success of LoRA is heavily dependent on adapter capacity and base model scale. While a LoRA rank of 128 allowed the 7B model to nearly match dense fine-tuning baselines on challenging benchmarks, the 3B model exhibited a wider performance gap, indicating that smaller backbones are more sensitive to adapter capacity limits. Furthermore, while reasoning specialization improves performance on complex tasks (e.g., LiveCodeBench), it introduces a trade-off, occasionally degrading zero-shot performance on simpler coding tasks that require direct answers. Managing this specialization-forgetting trade-off remains an ongoing challenge.

#### Dynamic LoRA routing via the switcher.

We demonstrated that the computational overhead of reasoning can be drastically reduced by recognizing that not all queries require complex multi-step logic. The lightweight Switcher module successfully acts as an on-demand router, preserving the base model's speed for standard queries while activating reasoning LoRA adapters only when necessary. A major deployment insight was the necessity of _masked LoRA training_ during the prefill stage. This strategy ensures that the KV-cache generated by the base model can be seamlessly reused by the reasoning adapters, completely eliminating the severe latency penalty of re-encoding prompt tokens when switching modes.

While the current switcher relies on a supervised classifier head to evaluate query complexity, a promising direction for future work is to learn this routing policy via reinforcement learning. This would act synergistically with our budget-forcing objectives: while budget-forced RL explicitly shortens the reasoning traces when LoRA is active, an RL-driven router optimized for both accuracy and length would learn to bypass the adapters entirely whenever possible. Because the frozen base model natively produces direct answers without verbose chain-of-thought, successfully routing a query to the non-LoRA mode automatically yields a drastically shorter response, organically reserving the reasoning adapters strictly for complex queries where the base model would fail.

Future work could extend the switcher beyond binary routing and turn it into a general mechanism for dynamic LoRA selection. Instead of choosing only between the base model and a single reasoning adapter, the system could route each query to a bank of task-specific adapters [@huang2024mixture; @feng2024mixture], for example specialized for mathematics, coding, or other domains, allowing the same backbone to support richer capabilities while preserving modular deployment. An especially promising direction is to include adapters tailored for latent reasoning, since recent work suggests that latent-reasoning LoRA adapters can retain strong reasoning performance while substantially reducing the token overhead of explicit CoT generation [@hao2024training; @shen2025codi; @wu2025parallel; @kuzina2026kava; @wang2025system]. In this setting, the switcher would not only decide whether reasoning is needed, but also which form of reasoning is most efficient for a given query, making dynamic adapter routing a natural path toward more capable and more compute-efficient on-device systems.

#### Budget forcing.

We presented our RL recipe to finetune LLMs to generate shorter completions. By leveraging a multiplicative penalty (eq. [
$$eqn:bf_reward$$
](#eqn:bf_reward){reference-type="ref" reference="eqn:bf_reward"}), we successfully aligned LLMs to reduce the number of generated tokens to answers given questions. Our empirical evaluations demonstrate an average completion length reduction of $2.4\times$-and up to $8\times$ maximum compression-with minimal degradation in task accuracy. Crucially, our formulation structurally mitigates the reward hacking we observed in early experiments, ensuring that efficiency gains stem from genuine rationale compression rather than formatting exploits. While our "soft-barrier" approach offers a robust mechanism for compute-aware inference, it also opens several exciting avenues for future research. We outline these forward-looking directions, which naturally address the current boundaries of our methodology.

Our experiments provide a snapshot of budget forcing on state-of-the-art reasoning models. However, the relationship between base model scale (e.g., parameter count) and the capacity for rationale compression remains an open question. Investigating whether larger, more capable models exhibit greater "epistemic hesitation", and thus offer a larger margin for ITC reduction, will be critical for formulating generalized scaling laws for budget forcing.

A fundamental limitation of current budget forcing approaches, including ours, is the assumption of uniform token cost where every generated token contributes equally to the budget consumption regardless of its utility. However, a token representing a crucial logical leap carries significantly higher semantic value than a token used for syntactic glue or hedging (e.g., "Let me think about this\..."). A promising future direction lies in developing _semantic-aware_ budget priors that weight penalties dynamically based on information density or local entropy as explored in [@Massoli2026-eu]. By shifting the optimization objective from pure length minimization to _reasoning density maximization_, we can encourage models to prioritize high-utility tokens while disproportionately penalizing low-entropy filler, effectively decoupling computational cost from reasoning depth.

#### Parallel test-time scaling and reasoning.

We have demonstrated that even a light verifier design can substantially improve the effectiveness of parallel test-time scaling for reasoning on edge devices. By combining the robustness of aggregation with a learned notion of correctness at negligible additional cost, weighted majority voting offers a practical and efficient approach for deploying parallel reasoning on resource-constrained devices. The current design can be extended to score the steps with a process reward model. Another interesting extension is the parallel reasoning schemes with interdependent generation as in [@hsu2025group; @rodionov2025hogwild; @cesa2026lanerope].

#### Quantization {#quantization}

Our experiments on Qwen2.5-7B-Instruct highlight the importance of starting from a strong quantized base model. Leveraging function-preserving transformations, improved range initialization, and joint fine-tuning of transformation and quantization parameters offers a straightforward and lightweight path to quantizing modern LLMs while preserving robust predictive performance. When extending such models with reasoning capabilities, we further show that it is essential to account for distribution shifts induced by quantization. Inspired by prior PEFT literature, we address this challenge by proposing the Quantization-Aware Modular Reasoning (QAMR) approach, which mitigates the quantization noise and distribution shifts by training reasoning adapters directly on the quantized base model. Finally, our results demonstrate that a relatively compact 4-bit weight-quantized quantized 7B model can achieve reasoning performance comparable to that of substantially larger models.

Reasoning remains a token-generation-intensive task, making it fundamentally memory-bound rather than compute-bound. As a result, further improvements in efficiency and performance will likely depend on reducing the memory footprint of the model weights. A promising direction for future work is to push quantization below 4 bits by leveraging state-of-the-art compression techniques such as Quip# [@tseng2024quip], or exploring 2-3-bit QAT methods, such as ParetoQ [@liu_paretoq_2025].

# Conclusion

In this work, we presented an end-to-end framework that makes state-of-the-art LLM reasoning practical on resource-constrained edge devices. We demonstrated that parameter-efficient LoRA adaptation, governed by a dynamic routing switcher, unlocks powerful reasoning capabilities without compromising the speed of everyday interactions. By introducing budget-forced reinforcement learning, we successfully curbed model verbosity to fit strict on-device token limits. To further maximize hardware utilization, we leveraged parallel test-time scaling and a lightweight latent verifier to boost accuracy during the memory-bound generation phase. Throughout this pipeline, hardware-awareness remains the unifying principle: from maximizing KV-cache reuse to intertwining quantization directly into the training of the adapters, switcher, and verifier. Ultimately, this co-designed approach bridges the gap between cloud-based reasoning and the strict memory, latency, and power budgets of mobile hardware, providing a practical blueprint for on-device AI.

# Benchmark Description {#App:benchmark}

To comprehensively assess the reasoning capabilities of our fine-tuned models, we leverage a diverse suite of benchmarks spanning the mathematics, science, and coding domains.

- **AIME 24/25** [@aime] consists of 30 highly challenging mathematics competition problems from the 2024 and 2025 American Invitational Mathematics Examination. The questions cover a range of topics including algebra, geometry, and number theory, and are aimed for high-school students. The problems require multi-step reasoning, and the answers are integers between 0 and 999.

- **MATH500** [@hendrycks2021measuring] is a benchmark consisting of 500 mathematical questions spanning different topics including algebra, geometry, number theory, precalculus, and probability. The problems require multi-step solutions, and answers may include LaTeX-formatted expressions.

- **GPQA Diamond** [@rein2024gpqa] consists of 198 science PhD-level questions from physics, chemistry, and biology. All questions are presented in a multiple-choice format.

- **AMC23** [@amc23] contains 40 problems from the 2023 American Mathematics Competition with integer answers.

- **LiveCodeBench** [@livecodebench] is a continuously updated (hence \"live\") coding benchmark. In every release, a certain number of coding problems is sourced from competitive programming platforms (e.g. CodeForces, LeetCode), and each problem is used to build four coding \"scenarios\": Code Generation, Code Repair, Test Output Prediction, and Code Execution. In this work we use the v2 release, comprising 511 problems, and confine ourselves to the Code Generation scenario.

- **HumanEval and HumanEval+** [@humaneval; @evalplus] are the original and improved versions of the HumanEval benchmark, comprising 164 problems in which a model, based on a Python function's signature and docstring, must generate its body. The resulting function is then verified with several unit tests. **HumanEval+**  [@evalplus] improves the original benchmark by increasing the number unit tests for verification by 80 times.

- **MBPP and MBPP+** [@mbpp; @evalplus] are the original and improved versions of the Most Basic Python Programs benchmark, consisting of 1000 basic Python programming tasks sourced from human coders. Each task involves writing a simple Python function based on natural language requirements and three unit tests it must pass. The **MBPP+** enhancement selects a subset of 378 task and increases the number of unit tests by 35 times.

# LoRA ablation study {#sec:lora_ablation}

This appendix provides the complete, detailed results of the parameter-efficient fine-tuning (PEFT) ablation study introduced in Section [3.4.1](#sec:ablations){reference-type="ref+Label" reference="sec:ablations"}, in Tables [10](#tab:ablations_qwen3b){reference-type="ref" reference="tab:ablations_qwen3b"} and [11](#tab:qwen7_ablations){reference-type="ref" reference="tab:qwen7_ablations"}. The study explores the impact of varying learning rates, batch sizes, and LoRA adapter ranks on the reasoning capabilities of both the 3B and 7B model backbones. The models were trained on a 50,000-entry subset of the OpenThoughts3 (OT3) dataset for one epoch and evaluated across core mathematical and scientific benchmarks (AIME24, AIME25, MATH500, GPQA, and AMC23).

[]{#tab:ablations_qwen3b label="tab:ablations_qwen3b"}

    **LR**     **BS**     **Rank**     **AIME24**     **AIME25**     **MATH500**     **GPQA**     **AMC23**      **Avg**

---

0.0001 32 32 0.05 0.01 0.55 0.25 0.24 0.220
0.0001 32 64 0.03 0.02 0.58 0.29 0.28 0.240
0.0001 32 128 **0.08** 0.03 0.56 0.28 0.30 0.250
0.0001 32 256 0.03 0.04 0.57 0.28 0.28 0.240
0.0001 64 32 0.05 0.01 0.54 0.27 0.28 0.230
0.0001 64 64 0.03 0.03 0.57 0.29 0.29 0.242
0.0001 64 128 0.02 0.03 0.58 0.27 0.30 0.240
0.0001 64 256 0.04 **0.06** 0.59 **0.30** **0.34** **0.266**
0.0001 128 32 0.04 0.01 **0.61** 0.26 0.29 0.242
0.0001 128 64 0.05 0.00 0.59 0.29 0.27 0.240
0.0001 128 128 0.03 0.01 0.56 0.27 0.25 0.224
0.0001 128 256 0.03 0.05 0.58 0.27 0.25 0.236
0.0002 32 32 **0.08** 0.01 0.58 0.26 0.33 0.252
0.0002 32 64 0.05 0.03 0.59 **0.30** 0.32 0.258
0.0002 32 128 0.06 0.04 0.57 0.25 0.29 0.242
0.0002 32 256 0.07 0.04 **0.60** 0.28 **0.34** 0.266
0.0002 64 32 0.03 0.02 0.56 0.27 0.30 0.236
0.0002 64 64 0.07 0.03 0.58 0.26 0.28 0.244
0.0002 64 128 0.07 **0.06** 0.55 0.29 0.25 0.244
0.0002 64 256 0.06 0.04 0.58 0.27 0.30 0.250
0.0002 128 32 0.00 0.03 0.59 0.27 0.28 0.234
0.0002 128 64 0.00 0.02 0.55 0.28 0.28 0.226
0.0002 128 128 0.05 0.04 0.55 0.27 0.30 0.242
0.0002 128 256 0.07 0.04 0.59 0.29 0.32 **0.262**
0.0005 32 32 0.05 0.02 0.55 **0.31** 0.29 0.244
0.0005 32 64 0.05 0.03 0.53 0.28 0.27 0.232
0.0005 32 128 **0.06** 0.04 0.54 0.26 0.27 0.234
0.0005 32 256 0.04 0.05 0.53 0.28 0.24 0.228
0.0005 64 32 **0.06** 0.04 0.56 0.26 0.29 0.242
0.0005 64 64 0.05 0.05 0.57 0.28 0.30 0.250
0.0005 64 128 **0.06** 0.04 0.57 0.29 0.27 0.246
0.0005 64 256 0.03 0.05 0.54 0.28 0.24 0.228
0.0005 128 32 **0.06** 0.03 0.57 0.28 0.26 0.240
0.0005 128 64 0.03 0.00 0.57 0.28 0.27 0.230
0.0005 128 128 **0.06** 0.05 0.56 0.27 **0.35** **0.258**
0.0005 128 256 **0.06** **0.06** **0.58** 0.29 0.25 0.248
: Ablation results for Qwen2.5-3B-Instruct. LR denotes learning rate and BS stands for batch size. In each learning rate subgroup, the best performance is marked in bold.

[]{#tab:qwen7_ablations label="tab:qwen7_ablations"}

**LR** **BS** **Rank** **AIME24** **AIME25** **MATH500** **GPQA** **AMC23** **Avg**

---

0.0001 32 32 0.18 0.12 **0.79** 0.36 0.50 0.390
0.0001 32 64 0.13 0.16 0.78 0.36 0.51 0.388
0.0001 32 128 0.17 0.11 0.78 0.37 0.55 0.396
0.0001 32 256 0.14 0.14 **0.79** 0.35 0.54 0.392
0.0001 64 32 0.14 0.12 0.77 0.35 0.54 0.384
0.0001 64 64 0.16 **0.17** 0.76 0.38 **0.56** **0.406**
0.0001 64 128 0.15 0.15 0.78 **0.39** 0.55 0.404
0.0001 64 256 0.15 0.15 0.76 0.38 0.53 0.394
0.0001 128 32 0.12 0.12 0.77 0.37 0.55 0.386
0.0001 128 64 0.18 0.13 0.78 0.35 0.51 0.390
0.0001 128 128 **0.19** 0.11 0.78 0.36 0.52 0.392
0.0001 128 256 0.18 0.11 0.76 0.37 0.53 0.390
0.0002 32 32 0.12 0.14 0.77 0.36 0.54 0.386
0.0002 32 64 0.14 0.16 0.79 0.34 0.51 0.388
0.0002 32 128 0.18 **0.18** 0.78 0.35 0.54 0.406
0.0002 32 256 0.18 0.17 0.77 0.35 0.56 0.406
0.0002 64 32 **0.21** 0.11 0.79 0.36 0.52 0.398
0.0002 64 64 0.16 0.16 0.77 0.37 **0.58** 0.408
0.0002 64 128 0.20 0.14 **0.80** **0.38** 0.52 0.408
0.0002 64 256 0.14 0.17 0.76 0.33 0.52 0.384
0.0002 128 32 0.15 0.11 0.77 0.35 0.57 0.390
0.0002 128 64 0.18 0.12 0.79 0.33 0.55 0.394
0.0002 128 128 0.18 0.15 0.77 0.35 0.57 0.404
0.0002 128 256 0.20 0.17 0.79 **0.38** 0.54 **0.416**
0.0005 32 32 0.17 0.15 0.76 0.38 0.53 0.398
0.0005 32 64 0.11 0.09 0.67 **0.40** 0.50 0.354
0.0005 32 128 0.16 0.11 0.76 0.37 0.47 0.374
0.0005 32 256 0.00 0.00 0.00 0.24 0.00 0.048
0.0005 64 32 0.17 0.14 0.76 0.38 0.54 0.398
0.0005 64 64 0.15 0.14 0.77 0.34 0.50 0.380
0.0005 64 128 0.22 0.12 **0.79** 0.37 **0.59** **0.418**
0.0005 64 256 0.00 0.00 0.01 0.27 0.02 0.060
0.0005 128 32 0.19 0.13 **0.79** 0.34 0.55 0.400
0.0005 128 64 0.17 0.12 0.77 0.38 0.55 0.398
0.0005 128 128 **0.23** **0.17** 0.78 0.37 0.53 0.416
0.0005 128 256 0.21 **0.17** 0.76 0.36 0.49 0.398
: Ablation results for Qwen2.5-7B-Instruct. LR denotes learning rate and BS stands for batch size. In each learning rate subgroup, the best performance is marked in bold.

# Budget Forcing Details {#app:budget_forcing}

[12](#tab:hyperparams) reports the list of hyperparameters we used in the budget forcing RL training.

**Hyperparameter** **Value**

---

Optimizer AdamW
Learning Rate $2 \times 10^{-5}$
LR Scheduler Cosine
Warmup Ratio 0.05
Batch Size (Global) 256
Generations per Prompt ($G$) 8
Temperature 0.8
Max Completion Length 6144
Max Gradient Norm 1.0
KL Penalty Coefficient ($\beta_{\text{KL}}$) $\{10^{-3}, 10^{-4}\}$
Training Steps 200
: **GRPO Training Hyperparameters.** All experiments share these settings unless otherwise noted.

We report in Figures [9](#fig:cot_example_1){reference-type="ref" reference="fig:cot_example_1"} and [10](#fig:cot_example_3){reference-type="ref" reference="fig:cot_example_3"} additional qualitative comparisons between the reasoning traces of the unconstrained baseline against our budget-forced model.

<a id="fig:cot_example_1"></a>

> Figure 4: <strong>Qualitative comparison on number theory reasoning.</strong> Text highlighted in red denotes redundant verification and verbal parsing, while bold text identifies essential reasoning steps. We use “<strong>[<span class="math inline">…</span>]</strong>” as a placeholder for brevity. <strong>Top:</strong> Prompt. <strong>Middle:</strong> The Baseline trace correctly identifies the property (<span class="math inline"><em>p</em><sup>2</sup></span>) early on but falls into extensive, redundant verification loops checking composite numbers and re-listing primes (highlighted in red). <strong>Bottom:</strong> The Budget Forced trace directly applies the prime-square property and computes the result without unnecessary hesitation or syntactic noise.

<a id="fig:cot_example_3"></a>

> Figure 5: <strong>Qualitative comparison on pattern recognition.</strong> <strong>Middle:</strong> The Baseline trace correctly identifies the formula <span class="math inline">2<em>n</em> − 1</span> initially but spends nearly 1000 tokens validating it against alternative arithmetic formulas (<span class="math inline"><em>a</em> + (<em>n</em> − 1)<em>d</em></span>) and hypothetical user errors (confusing term number with value, testing <span class="math inline"><em>n</em><sup>2</sup></span>, etc.). <strong>Bottom:</strong> The Budget Forced trace directly retrieves the formula and computes the specific term requested.

[^1]: Core contributors

[^2]: Contributors

[^3]: Project leads

[^4]: <https://github.com/evalplus/evalplus>

[^5]: We have nonetheless confirmed that the model's answers are never cut short by the Evalplus harness.
