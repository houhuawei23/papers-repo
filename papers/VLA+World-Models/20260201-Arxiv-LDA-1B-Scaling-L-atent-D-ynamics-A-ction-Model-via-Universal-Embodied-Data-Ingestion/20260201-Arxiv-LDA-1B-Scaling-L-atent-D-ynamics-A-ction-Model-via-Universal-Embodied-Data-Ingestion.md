# Title: LDA-1B: Scaling L atent D ynamics A ction Model via Universal Embodied Data Ingestion
- ArXiv: 2602.12215
- Authors:
  - Jiangran Lyu — , Kai Liu
  - Jiayi Chen — , Jiazhao Zhang
  - Xuesong Shi — , Haoran Li
- Sections: 38
- Estimated tokens: 28.2k

## Abstract

Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data.
While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets.
We introduce  **LDA-1B** , a robot foundation model that scales through universal embodied data ingestion by jointly learning dynamics, policy, and visual forecasting, assigning distinct roles to data of varying quality.
To support this regime at scale, we assemble and standardize  **EI-30k** , an embodied interaction dataset comprising over  **30k hours**  of human and robot trajectories in a unified format.
Scalable dynamics learning over such heterogeneous data is enabled by prediction in a structured DINO latent space, which avoids redundant pixel-space appearance modeling.
Complementing this representation, LDA-1B employs a multi-modal diffusion transformer to handle asynchronous vision and action streams, enabling stable training at the  **1B-parameter**  scale.
Experiments in simulation and the real world show LDA-1B outperforms prior methods (e.g., $\pi_{0.5}$) by up to  **21%** ,  **48%** , and  **23%**  on contact-rich, dexterous, and long-horizon tasks, respectively.
Notably, LDA-1B enables data-efficient fine-tuning, gaining 10% by leveraging 30% low-quality trajectories typically harmful and discarded.

<a id="S1"></a>

## I Introduction

Inspired by the success of Large Language Models (LLMs) and Vision-Language Models (VLMs), the robotics community has increasingly pursued general-purpose robot foundation models through large-scale pretraining [[5](#ref-6), [41](#ref-14)]. Most existing approaches center on scaling behavior cloning (BC), which imitates expert actions but fundamentally restricts learning to high-quality demonstrations.
Consequently, a large portion of heterogeneous embodied data [[42](#ref-44)] is discarded or only weakly utilized, despite containing rich physical interaction dynamics [[26](#ref-45)].

Unified World Model (UWM) formulation [[30](#ref-25), [60](#ref-24)] provides an alternative by jointly optimizes dynamics, policy, and video generation within a single model, which can leverages not only expert data. Despite the potential value, existing UWM instantiations remain far from scaling to foundation-level.
A major limitation lies in coarse data usage: heterogeneous embodied data are often treated uniformly, without differentiating their roles by quality or supervision, which underutilizes transferable dynamics knowledge. In addition, the community lacks ready-to-use large-scale datasets that unify varying-quality data with consistent formats and aligned action representations.
Furthermore, UWM represent future state in pixel space, entangling dynamics learning with redundant appearance modeling. Subtle variations in illumination, texture, background clutter, or camera viewpoint can dominate the training objective, making large-scale training inefficient and hindering the learning of interaction-relevant dynamics.

To overcome these limitations, we introduce  **LDA-1B** , a robot foundation model that scales via *universal embodied data ingestion*. In this framework, heterogeneous data play distinct yet complementary roles: actionless human videos supervise visual forecasting [[38](#ref-48), [37](#ref-49), [25](#ref-50)], lower-quality trajectories primarily inform dynamics learning, and high-quality trajectories support both policy and dynamics. To realize this approach at scale, we assemble  **EI-30k** , a large-scale embodied interaction dataset with over 30k hours of human and robot trajectories across real and simulated environments, standardized in format and aligned in action representation. Scalable learning on such diverse data is facilitated by a structured  **DINO latent space** [[46](#ref-13), [59](#ref-2), [22](#ref-20)], which reduces redundant appearance modeling [[38](#ref-48), [25](#ref-50)], and a multi-modal diffusion transformer that aligns asynchronous visual and action prediction. By combining this ingestion strategy, dataset, latent representation, and model architecture, LDA-1B achieves stable training at the 1B-parameter scale while maximizing data utilization.

We evaluate LDA-1B on challenging RoboCasa-GR1 benchmark and a diverse set of real-world tasks involving both grippers and high-DoF dexterous hands [[56](#ref-47)]. LDA-1B consistently outperforms $\pi_{0.5}$, achieving  **21%**  gains on contact-rich manipulation, benefiting from improved dynamics understanding and  **48%**  gains on dexterous manipulation, benefiting from effective utilization of human data. Moreover, under a mixed-quality fine-tuning setting, LDA-1B improves data efficiency by  **10%**  through leveraging low-quality trajectories that are detrimental to baseline methods. These results highlight universal embodied data ingestion and unified latent dynamics learning as a scalable alternative to behavior-cloning-centric robot pretraining.
In summary, our contributions are threefold:

- We propose LDA-1B, a scalable robot foundation model that learns generalizable interaction dynamics through unified latent dynamics pretraining.
- We construct EI-30k, a large-scale embodied interaction dataset covering diverse embodiments, environments, data qualities, with aligned end effector coordinate system.
- We demonstrate that LDA-1B achieves superior generalization and robustness across a wide range of settings, including simulation and real-world environments, contact-rich manipulation, dexterous manipulation, and long-horizon manipulation.

<a id="figure-2"></a>

![Refer to caption](images/LDA_pipeline.png)

> Figure 2:  **Architecture of LDA.** 
LDA jointly denoises action chunks and future visual latent under multiple co-training objectives, including policy learning, forward dynamics, inverse dynamics, and visual forecasting.
Conditioned on VLM tokens, diffusion timesteps, and task embeddings, the model adopts a multimodal diffusion transformer architecture, where action and visual experts are decoupled and interact through a shared self-attention layer.
<a id="S2"></a>

## II Related Work

<a id="table-1"></a>

| Model | Data Src. | #Data | Action Quality | Train. | Param. |
| --- | --- | --- | --- | --- | --- |
| $\pi_{0.5}$[[23](#ref-15)] | Tele. | 10k+ | High | BC | 3B |
| RDT [[32](#ref-7)] | Tele. | <10k | High | BC | 1B |
| GraspVLA [[14](#ref-17)] | Sim. | 20k+ | High | BC | 2B |
| InternVLA-M1 [[11](#ref-16)] | Sim. | <10k | High | BC | 3B |
| Being-H0 [[35](#ref-28)] | Hum. | <10k | Mixed | Aln. + BC | 14B |
| InternVLA-A1 [[9](#ref-71)] | Het. | 10k+ | High | VF + BC | 3B |
| GR00T-N1.6 [[40](#ref-34)] | Het. | <10k | Mixed | LA + BC | 1B |
| UniVLA [[7](#ref-8)] | Het. | <10k | Mixed | LA + BC | 7B |
| LDA-1B | Het. |  **30k+**  | Mixed | UWM[[60](#ref-24)] | 1B |

 **Robot Foundation Models.**  Recent robot foundation models predominantly adopt the Behavior Cloning paradigm. As summarized in Table I, representative approaches—including $\pi_{0}$[[4](#ref-12)], RDT [[32](#ref-7)], and InternVLA [[11](#ref-16)]—rely heavily on high-quality teleoperation or simulation data, which fundamentally constrains their scalability. Hybrid methods such as Being-H0 [[35](#ref-28)] and UniVLA [[7](#ref-8)] attempt to incorporate heterogeneous data with mixed quality; however, they largely depend on action alignment or auxiliary pretrained latent action models, limiting the effective data scale to around 6k hour embodied data. In contrast, LDA-1B breaks this ceiling by adopting a unified world model formulation, enabling efficient ingestion of up to 30k hours of mixed-quality embodied data.

 **Unified Video Action Models.** 
Recent works have explored joint modeling dynamics and policy for embodied decision making. Methods such as DyWA [[36](#ref-21)], FLARE [[58](#ref-22)], and the WorldVLA series [[10](#ref-23), [24](#ref-5)] demonstrate that co-training next-state prediction and policy learning can improve generalization in interactive environments.
To enrich dynamics modeling, UWM [[60](#ref-24)] and UVA [[30](#ref-25)] further propose optimizing multiple objectives jointly, including video generation, forward and inverse dynamics, and action prediction. Concurrent with our work, Motus [[3](#ref-19)] adopt UWM paradigm and integrate priors from pretrained VLM and video generation models.
Despite their promising results, these approaches typically operate directly in pixel space and do not explicitly consider the roles of data quality, scale, or heterogeneity during training, which limits their ability to fully exploit large-scale, mixed-quality interaction data for robust dynamics learning.

 **Large-Scale Embodied Interaction Datasets.** 
The progress in embodied ai relys on large-scale embodied datasets.
Many widely used datasets are collected via teleoperation on real robots [[4](#ref-12), [23](#ref-15), [27](#ref-11), [31](#ref-10)] or generated in simulation [[14](#ref-17), [11](#ref-16)], providing high-quality action-labeled trajectories.
Beyond robot-collected data, recent works explore human-centric embodied datasets, such as egocentric recordings with hand actions [[53](#ref-27), [35](#ref-28)].
While these datasets significantly expand data diversity, many are either not publicly released or provide limited action supervision, making them difficult to directly integrate with robot learning pipelines.
More broadly, existing embodied datasets are highly fragmented: some are closed-source, others are open but vary substantially in data formats, sensor configurations, action representations, and annotation quality.
This lack of standardization poses a major obstacle to large-scale data aggregation and unified training.
In contrast, our work introduces EI-30k, a large-scale embodied interaction dataset that unifies diverse data sources—including robot and human trajectories from both real-world and simulated environments—under consistent data formats and aligned action representations.
<a id="S3"></a>

## III Latent Dynamics Action Model

<a id="S3.SS1"></a>

### III-A Preliminary: Unified World Models

Given the current observation $o_{t}$ (typically an RGB image), UWM [[60](#ref-24)] jointly models multiple conditional distributions over future observations $\boldsymbol{o}_{t+1:t+k}$ and action chunk $\boldsymbol{a}_{t+1:t+k}$, enabling unified learning of:

1. Policy: $p(\boldsymbol{a}_{t+1:t+k}\mid\boldsymbol{o}_{t})$
2. Forward Dynamics: $p(\boldsymbol{o}_{t+1:t+k}\mid\boldsymbol{o}_{t},\boldsymbol{a}_{t+1:t+k})$
3. Inverse Dynamics: $p(\boldsymbol{a}_{t+1:t+k}\mid\boldsymbol{o}_{t:t+k})$
4. Visual Planning: $p(\boldsymbol{o}_{t+1:t+k}\mid\boldsymbol{o}_{t})$

Concretely, UWM [[60](#ref-24)] instantiates this framework using a joint diffusion model that predicts noise for both actions and future observations:

<a id="eq-1"></a>

$$
(\epsilon_{a}^{\theta},\epsilon_{o}^{\theta})=s_{\theta}\!\left(o,\,a_{t_{a}},\,o^{\prime}_{t_{o}},\,t_{a},\,t_{o^{\prime}}\right), \tag{(1)}
$$

where $t_{a}$ and $t_{o}$ are independently sampled diffusion timesteps for actions and observations, and $\tilde{a}_{t_{a}}$, $\tilde{o}_{t_{o}}$ denote their corresponding noisy inputs. The model is trained with a standard DDPM [[20](#ref-29)] objective, jointly denoising future actions and observations conditioned on $o_{t}$.
We further extend this formulation by introducing language $\ell$ conditioning through a VLM, enabling instruction-guided action and observation prediction.

<a id="S3.SS2"></a>

### III-B Universal Data Ingestion via Multi-task Co-training

We adopt a *universal data ingestion* regime to jointly train the unified objectives described above, allowing heterogeneous embodied data to contribute according to their supervision quality.
Specifically, high-quality robot and human demonstrations are co-trained with all objectives, supporting both action policy learning and dynamics modeling.
Lower-quality trajectories, which may contain suboptimal or noisy actions, are used exclusively for dynamics and visual forecasting, where accurate action optimality is not required.
In addition, we leverage large-scale human manipulation videos without action annotations to train the visual forecasting objective, providing supervision for instruction-conditioned future state prediction.
This role-aware data usage prevents overfitting to expert-only behaviors and enables scalable learning of transferable dynamics and action representations.

To implement differentiated objectives within a single diffusion model, we introduce four learnable *task embeddings* and two learnable *register tokens*.
Each task embedding corresponds to a specific training objective (policy, forward dynamics, inverse dynamics, or visual forecasting) and is added to the diffusion timestep embedding $f_{t}$ to condition the denoising process.
The learnable register tokens—one for action and one for visual state—serve as placeholders for modalities that are absent in a given task.
For example, during policy training, the model receives noisy action tokens along with a visual register token representing the unobserved future state; in contrast, visual forecasting uses noisy future visual tokens with an action register token.
This design enables a unified architecture to flexibly support different input–output structures without modifying the network topology.
Overall, the model predicts a denoising vector field $v_{a}^{\theta}$ under different task conditions and is trained using a flow-matching objective:

<a id="eq-2"></a>

$$
\displaystyle l_{\mathrm{action}}^{\theta} \displaystyle=\mathbb{E}_{\begin{subarray}{c}(\boldsymbol{o}_{t:t+k},\boldsymbol{a}_{t+1:t+k},\ell)\sim\mathcal{D}\\
\tau_{a}\sim\mathcal{U}(0,T_{\tau})\\
\epsilon_{a}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{subarray}}\left\|v_{a}^{\theta}-(\epsilon_{a}-\boldsymbol{a}_{t+1:t+k})\right\|_{2}^{2}, \displaystyle l_{\mathrm{obs}}^{\theta} \displaystyle=\mathbb{E}_{\begin{subarray}{c}(\boldsymbol{o}_{t:t+k},\boldsymbol{a}_{t+1:t+k},\ell)\sim\mathcal{D}\\
\tau_{o}\sim\mathcal{U}(0,T_{\tau})\\
\epsilon_{o}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{subarray}}\left\|v_{o}^{\theta}-(\epsilon_{o}-\boldsymbol{o}_{t+1:t+k})\right\|_{2}^{2}, \displaystyle l^{\theta} \displaystyle=l_{\mathrm{action}}^{\theta}+l_{\mathrm{obs}}^{\theta}. \tag{(2)}
$$

During training, action and visual losses are selectively activated according to the task specification, allowing heterogeneous data to contribute under appropriate supervision.
At inference time, the same model can be flexibly invoked for different objectives by specifying the task embedding and corresponding inputs.

<a id="S3.SS3"></a>

### III-C Representation of Predictive Targets

We represent predictive targets—future visual states and actions—in a unified format to maximize knowledge sharing across heterogeneous datasets.
For visual prediction, we adopt latent features extracted from a pretrained DINO [[46](#ref-13)] encoder, rather than VAE-based pixel-space representations. DINO latents encode high-level semantic and spatial structure while suppressing background noise and low-level visual variations, which facilitates learning scene dynamics that generalize across diverse environments and object configurations.

For actions, we define a unified hand-centric action space based on end-effector motion, consisting of delta wrist poses and finger configurations. For parallel-jaw grippers, the finger state is represented by a single degree-of-freedom gripper width, while for multi-finger dexterous hands, finger configurations are described using keypoints expressed in the wrist coordinate frame. This design enables consistent action modeling across different embodiments and manipulation platforms.

To model temporal dynamics, visual states and actions are organized as two synchronized temporal streams with different sampling rates. Visual observations are sampled at 3hz, a lower frequency than actions, 10 hz. This reduces redundant computation from highly correlated consecutive frames while preserving fine-grained action dynamics, allowing the model to maintain coherent temporal alignment between fast-varying control signals and slower-evolving visual states.

<a id="S3.SS4"></a>

### III-D Architecture: MM-DiT

We adopt a Multi-Modal Diffusion Transformer (MM-DiT) to jointly denoise action chunks and predict future visual features within a unified diffusion framework (Fig. 2).
The model operates on heterogeneous tokens while sharing a common Transformer backbone.
Conditioning inputs include the current observation, language instruction, diffusion timestep, and task specification.
Observations and language are encoded by a pretrained VLM into conditioning tokens.
The diffusion timestep is encoded using a sinusoidal embedding, and task information is represented by a learned task embedding.
All conditioning signals are injected into each Transformer block via adaptive layer normalization (AdaLN [[43](#ref-30)]).

Actions are organized as fixed-length chunks and corrupted with Gaussian noise.
Future visual features (DINO [[46](#ref-13)] futures) are noised in parallel.
Both modalities are projected into token embeddings through modality-specific linear layers and processed jointly by MM-DiT.
Each MM-DiT block applies multi-modal self-attention over concatenated action and visual tokens, enabling cross-modal interaction.
Modality-specific QKV projections and FFNs are retained to preserve inductive biases, while attention is shared across modalities.
Language tokens are incorporated via cross-attention to provide high-level semantic guidance.
Finally, modality-specific output heads predict denoised action sequences and future visual features.

<a id="S3.SS5"></a>

### III-E Pre-training and Post-training

 **Pre-training Configurations.** 
Our model is trained on a server cluster equipped with 48 NVIDIA H800 GPUs. The training process contains 400k iterations, resulting in a total computational cost of 4,608 GPU hours. To preserve the generalization capability and visual representation quality of the pre-trained foundation models, we keep the parameters of the VLM [[52](#ref-35)] and the DINO [[46](#ref-13)] encoder frozen throughout the pre-training process, updating the MM-DiT and action encoder/decoder. This design ensures that the model can learn from new data without degrading the core abilities of the base models in cross-modal understanding and fine-grained visual feature extraction.

 **Data-Efficient Finetuning.**  To adapt the model to target embodiments and tasks for real-world deployment, we introduce a lightweight post-training stage. This stage follows the same data regime as pretraining and effectively leverages naturally collected teleoperation data of mixed quality, without requiring expert-level demonstrations. Compared to prior finetuning pipelines that rely on carefully curated expert datasets, our method directly utilizes unfiltered teleoperation data, substantially improving data efficiency and reducing the cost of data collection and annotation, thereby facilitating practical deployment.

<a id="figure-3"></a>

![Refer to caption](images/unified_eef.jpg)

> Figure 3:  **Aligned End Effector Coordinate Systems.**  We manually align coordinate frames across diverse robot and human embodiments to ensure consistency. This shared representation enables joint learning from heterogeneous interaction data.

<a id="figure-4"></a>

![Refer to caption](images/dataset.jpg)

> Figure 4:  **Statistics of EI-30K.**  The dataset contains more than 30k hours of diverse human and robot interaction data (right). It spans varying episode lengths (left) and a rich set of manipulation tasks (center).
<a id="S4"></a>

## IV Embodied Interaction Dataset (EI-30K)

We introduce the Embodied Interaction Dataset (EI-30K), a large-scale collection of embodied interaction trajectories totaling over 30k hours.
It consists of 8.03k hours of real-world robot data, 8.6k hours of simulated robot data, 7.2k hours of human demonstrations with actions, and 10k hours of actionless human videos.
All subdatasets are annotated with explicit quality labels, enabling systematic analysis across different fidelity levels and supporting quality-aware learning.

 **Data Unification.** 
EI-30K consolidates datasets from heterogeneous platforms and tasks, which vary in storage formats, sensor modalities, and annotations.
All data are converted into the LeRobot format, providing a unified representation of observations, actions, and language.
This standardization facilitates plug-and-play training, flexible data composition, and seamless integration of additional annotations, while greatly reducing engineering overhead for handling diverse sources.

 **Aligned Action Representation.** 
To support consistent modeling of physical interactions across embodiments, all available action annotations are expressed as hand-centric motion in a shared coordinate frame (Fig. [3](https://arxiv.org/html/2602.12215v1#S3.F3)).
For robots, this includes the 6-DoF end-effector pose plus gripper width or dexterous hand joints. For humans, the 6-DoF wrist pose and full MANO [[45](#ref-32)] hand parameters are recorded.
Camera extrinsics are retained to decouple hand motion from egocentric head motion. All coordinate frames are manually aligned to ensure geometric consistency across datasets, enabling joint learning from both human and robot trajectories.

 **Quality Annotation and Cleaning.** 
EI-30K applies systematic cleaning and quality-aware annotation.
Language annotations are normalized using a vision-language model to ensure semantic consistency. Motion segments without meaningful hand-object interaction are removed, e.g., head-only or idle segments in egocentric videos.
Each trajectory is assigned a quality label based on action accuracy, and annotation completeness.
Unlike aggressive filtering, low-quality trajectories are preserved, allowing downstream models to exploit the full spectrum of data through quality-aware training.
<a id="S5"></a>

## V Experiments

<a id="S5.SS1"></a>

### V-A Simulation Experiments

 **Benchmark and Baselines.** 
We evaluate our method on RoboCasa-GR1 [[39](#ref-33)], a simulated kitchen benchmark featuring 24 tabletop rearrangement and articulated-object manipulation tasks with the GR-1 humanoid robot and Fourier dexterous hands.
The benchmark provides challenging and realistic settings that require high-DoF dexterous manipulation from egocentric RGB observations captured by a head-mounted camera.
Following the GR00T [[40](#ref-34)] evaluation protocol, we finetune all models using 1,000 trajectories per task and evaluate each task with 51 trials, reporting average success rates.
We compare LDA against GR00T and its strong variants, as well as UWM [[60](#ref-24)], under matched training paradigms and data.
To ensure a fair comparison in terms of model capacity and pretraining, we reproduce a strong GR00T baseline (denoted as GR00T-EI10k) with 1B parameters, pretrained on our curated EI-30k high-quality subset and using Qwen3-VL as the VLM encoder.

<a id="table-2"></a>

| Model | Vis. Rep. | MMDiT | VLM | Success Rate $\uparrow$ |
| --- | --- | --- | --- | --- |
| GR00T-N1.6[[40](#ref-34)] | - | - | Cosmos | 47.6 |
| StarVLA[[47](#ref-72)] | - | - | Qwen3vl[[52](#ref-35)] | 47.8 |
| GR00T-EI30k | - | - | Qwen3vl | 51.3 |
| UWM-0.1B[[60](#ref-24)] | VAE | ✗ | - | 14.2 |
| UWM-1B | VAE | ✗ | Qwen3vl | 19.3 |
| UWM(MM-DiT) | VAE | ✓ | Qwen3vl | 20.0 |
| LDA(DiT) | DINO | ✗ | Qwen3vl | 48.9 |
| LDA-0.5B | DINO | ✓ | Qwen3vl | 50.7 |
| LDA-1B | DINO | ✓ | Qwen3vl |  **55.4**  |

<a id="figure-5"></a>

![Refer to caption](images/fig_gallery.png)

> Figure 5: Real-World Manipulation Demonstrations Across Multiple Robotic Platforms and End-Effectors.
Galbot G1 equipped with a Sharpa dexterous hand (top-left), Unitree G1 with a BrainCo dexterous hand (middle and bottom-left), and Galbot G1 with a two-finger gripper (right).

<a id="figure-6"></a>

![Refer to caption](images/fig_galbot.png)

> Figure 6:  **Success Rate Comparison on Real-World Gripper Manipulation Tasks.** 
All models are few-shot fine-tuned on Galbot and evaluated on eight tasks spanning Pick & Place, Contact-rich, Fine, and Long-horizon manipulation. LDA consistently outperforms GR00T-N1.6 [[40](#ref-34)] and $\pi_{0.5}$[[23](#ref-15)].

 **Comparison with Baselines.** 
As shown in Table [II](https://arxiv.org/html/2602.12215v1#S5.T2), the original GR00T-N1.6 [[40](#ref-34)] with 3B parameters achieves a success rate of 47.6%.
When pretrained on our curated EI-30k dataset, the reproduced GR00T-EI10k with 1B parameters shows a clear improvement, reaching 51.3%, highlighting the impact of high-quality embodied data.
Under the same parameter budget, LDA further improves the success rate to 55.4%.
These results indicate that, beyond data quality and parameter scaling, jointly learning actions and dynamics within a unified model provides additional gains when pretrained on mixed-quality data.

 **Ablation Study.** 
We further analyze key design choices under identical training data and optimization settings.
UWM [[60](#ref-24)], despite jointly predicting actions and dynamics, achieves only 14.2% success due to limited model capacity and the use of entangled VAE latent representations.
Scaling UWM to 1B parameters or replacing its DiT backbone with our MM-DiT yields only marginal improvements (19.3% and 20.0%, respectively), suggesting that architectural constraints fundamentally limit its performance.
In contrast, replacing pixel-space VAE latents with DINO [[46](#ref-13)] representations leads to a substantial performance gain (20.0% $\rightarrow$ 55.4%), highlighting the importance of semantically structured latent spaces for effective scaling.
Finally, removing the proposed MM-DiT architecture or reducing the model size to 0.5B parameters results in performance drops of 6.5% and 4.7%, respectively, confirming the effectiveness of the multi-expert design and its favorable scaling behavior.

<a id="S5.SS2"></a>

### V-B Real-world Experiments

To validate the scalability and robustness of LDA-1B, we conduct extensive real-world experiments focusing on few-shot adaptation to new embodiments, dexterous manipulation, and data efficiency under mixed-quality supervision.

 **Real-World Robot and Task Setup.** 
We evaluate our method on two humanoid platforms: Galbot G1 and Unitree G1.
Galbot G1 is equipped with either a two-finger gripper or 22-DoF Sharpa dexterous hands, while Unitree G1 uses 10-DoF BrainCo hands.
Across all configurations, the policy receives only egocentric RGB observations from a head-mounted camera.
We evaluate four categories of manipulation tasks under the gripper setting, *Pick and Place*, *Contact-rich Manipulation*, *Fine Manipulation*, and *Long-horizon Manipulation*—covering diverse contact dynamics and temporal horizons.
Representative tasks include *Beat Block, Flip Box, Handover, Pick-and-Place (Pepper), Sweep Table, Clean Rubbish, Water Flower,* and *Wipe Board*.
Dexterous manipulation further includes tool-use tasks such as pulling a nail with a hammer and flipping bread with a spatula, which require precise force control and coordinated finger motion.
Qualitative demonstrations are shown in Fig. [5](https://arxiv.org/html/2602.12215v1#S5.F5).
For each task, we collect 100 teleoperated trajectories without enforcing expert-level execution.
As a result, the dataset naturally exhibits mixed quality: approximately 50–80% of trajectories correspond to expert behavior, while the remainder contain suboptimal actions such as pauses, retries, or inefficient motion patterns.

 **Baselines and Finetuning Protocol.** 
We compare LDA-1B against two strong baselines, $\pi_{0.5}$[[23](#ref-15)] and GR00T [[40](#ref-34)].
To ensure stable and competitive performance, baseline models are finetuned exclusively on the filtered expert subset.
In contrast, LDA-1B leverages all collected trajectories and learns directly from the full mixed-quality distribution via our Universal Embodied Data Ingestion mechanism.

<a id="figure-7"></a>

![Refer to caption](images/fig_dexterous.png)

> Figure 7:  **Success Rate Comparison on Real-World Dexterous Manipulation Tasks**  We evaluate the real-world performance of our model against baselines (GR00T-N1.6 and $\pi_{0.5}$) on 3 low DoFs hand (BrainCo) tasks and 2 high DoFs hand (Sharpa) tasks. Ours (dark blue) consistently outperforms baselines especially on fine dexterous task (pull nails) and high DoFs tasks.

<a id="figure-8"></a>

![Refer to caption](images/fig_generalization.png)

> Figure 8: Generalization evaluation setup on Pick and Place task

<a id="table-3"></a>

| Method | Pick & Place |  |  |
| --- | --- | --- | --- |
| $\pi_{0.5}$ | 26.7 | 20.0 | 6.7 |
| GR00T | 40.0 | 40.0 | 20.0 |
| Ours |  **60.0**  |  **60.0**  |  **40.0**  |

 **Results on Gripper Manipulation.** 
We first evaluate few-shot adaptation by deploying LDA-1B on the Galbot G1, which is excluded from our EI-30k pretraining dataset.
As shown in Fig. [6](https://arxiv.org/html/2602.12215v1#S5.F6), LDA-1B consistently outperforms all baselines across task categories.
On simple pick-and-place tasks, LDA-1B achieves success rates of 80%–90%, indicating effective few-shot adaptation to a new robot embodiment.
The performance gap widens substantially in contact-rich and long-horizon scenarios.
For instance, the *Clean the Rubbish* task requires coordinated dual-arm manipulation, tool usage (dustpan), and sequential object transfer into a trash bin, where errors can easily accumulate over time.
In this setting, LDA-1B achieves a 35% success rate, while both GR00T and $\pi_{0.5}$ fail entirely (0%).
This result suggests that latent dynamics modeling enables LDA to better anticipate action-induced state transitions, maintain temporal consistency, and recover from intermediate failures in extended manipulation sequences.

 **Results on Dexterous Manipulation.** 
We further evaluate LDA-1B on both low-DoF and high-DoF dexterous manipulation tasks, as reported in Fig. [7](https://arxiv.org/html/2602.12215v1#S5.F7).
On low-DoF tasks such as *Pull Nail*, which requires precise motion direction and stable contact maintenance between the hammer and the nail, LDA-1B achieves 80% success, reliably localizing targets and adjusting sensitive actions, whereas $\pi_{0.5}$ largely fails.
On high-DoF tasks such as *Flip Bread*, which involve high-dimensional control, continuous contact, and coordinated wrist motion, LDA-1B attains 90% success, while $\pi_{0.5}$ reaches only 10%.
These results demonstrate that pretraining on large-scale human data provides strong latent priors for dexterous control, enabling precise finger coordination and object reorientation with limited robot data.
In contrast, baseline policies struggle to generalize as action dimensionality and contact complexity increase.

<a id="figure-9"></a>

![Refer to caption](images/dino.jpg)

> Figure 9:  **Visualization of latent forward dynamics.**  Our model generates accurate future visual representations (top) aligned with ground truth (bottom) across time steps, capturing semantic object structure and motion dynamics

<a id="figure-10"></a>

![Refer to caption](images/scaling.jpg)

> Figure 10: Scaling Analysis of LDA, evaluated by action prediction error on unseen test set. Top: Action prediction error decreases to 6.6 with 30k hours of training data, demonstrating effective utilization of diverse data sources. Bottom: LDA consistently outperforms UWM across model sizes (0.1B$\rightarrow$1B) with increasing training data, while the baseline saturates rapidly.

 **Generalization Ability.** 
To evaluate the generalization of our policy, we test pick and place task under three conditions: novel objects, unseen backgrounds, and out-of-distribution (OOD) starting position, shown as Fig [8](https://arxiv.org/html/2602.12215v1#S5.F8).
As summarized in Table [III](https://arxiv.org/html/2602.12215v1#S5.T3), our model maintains high success rates despite visual and spatial perturbations. The large-scale latent dynamics pretraining allows the model to ignore visual distractors (background changes) while focusing on relevant object affordances, demonstrating strong generalization relative to baselines.

 **Data-Efficient Finetuning.** 

<a id="table-4"></a>

| Method | Place the pen into the box | Bimanually remove the lid |  |  |
| --- | --- | --- | --- | --- |
| 63 High | 63 High + 37 Low | 66 High | 66 High + 34 Low |  |
| $\pi_{0.5}$ | 60 | 40 (20$\downarrow$) | 50 | 40 (10$\downarrow$) |
| Ours |  **70**  |  **80 (10$\uparrow$)**  |  **50**  |  **60 (10 $\uparrow$)**  |

We analyze the value of mixed-quality data ingestion during finetuning stage, by post-training on two splits: (1) High-Quality Only (expert data), and (2) High + Low Quality (all 100 trajectories).
As shown in Table [IV](https://arxiv.org/html/2602.12215v1#S5.T4), while baseline models degrade when low-quality data is added, LDA-1B effectively leverages these noisy trajectories, boosting performance with 10%, substantially improving data efficiency and reducing the cost of data collection and annotation for practical deployment.

<a id="S5.SS3"></a>

### V-C Analysis of Scaling Effects

To analyze the scaling behavior of LDA, we systematically vary model capacity, data composition, and training objectives.
All models are evaluated on an unseen test set sampled from a held-out subset of *Agibot World*[[6](#ref-37)].
We report the action prediction L1 error as the primary metric, which serves as a stable and reproducible proxy for real-world performance.
Fig. [10](https://arxiv.org/html/2602.12215v1#S5.F10) summarizes the results under four training configurations:
(i) *Policy Only*,
(ii) *Policy + Visual Forecasting*,
(iii) *Policy with Forward and Inverse Dynamics*, and
(iv) the full co-training framework (*Ours*).
These experiments jointly reveal how LDA scales under heterogeneous supervision and increasing model capacity.

 **Effectiveness of Universal Data Ingestion.** 
Effectively leveraging heterogeneous embodied data requires jointly scaling both data sources and training objectives.
As shown in Fig. [10](https://arxiv.org/html/2602.12215v1#S5.F10), LDA achieves its best performance only when all supervision signals—policy learning, dynamics modeling, and visual forecasting—are optimized together.
When either the data scale or the training objectives are reduced, performance degrades noticeably.
Using only action-labeled trajectories with a *Policy Only* objective (grey line), increasing the dataset size yields unstable behavior: while moderate scaling initially reduces error, incorporating lower-quality data leads to performance degradation.
Similarly, partial co-training variants that exclude either dynamics or visual forecasting objectives (green and brown lines) improve robustness but fail to fully exploit the available data.
In contrast, the full co-training framework (blue line) exhibits consistent improvement as additional heterogeneous data is introduced.
Notably, even after all action-labeled trajectories are exhausted, adding 10k actionless videos continues to reduce prediction error.
These indicates that LDA can extract useful supervisory signals from low-qaulity data and non-action data through latent dynamics and visual forecasting, rather than treating such data as noise.
Overall, these results demonstrate that Universal Data Ingestion is most effective when heterogeneous data and co-training objectives are scaled together, enabling LDA to fully utilize mixed-quality supervision.

 **Effectiveness of Latent Representation.** 
Although both LDA and UWM incorporate dynamics-related supervision, their scaling behaviors diverge substantially due to differences in the structure of their latent spaces.
As shown in Fig. [10](https://arxiv.org/html/2602.12215v1#S5.F10), UWM quickly saturates as data scale and model capacity increase, with additional supervision yielding diminishing or even negative returns.
This indicates that simply increasing data or parameters is insufficient when the latent space cannot support compositional and causal reasoning.
This limitation stems from UWM’s VAE-derived latent representation, which entangles appearance, geometry, and dynamics at a low-level feature granularity.
Such entanglement restricts the model’s ability to factorize action-induced state transitions and prevents effective reuse of heterogeneous supervision during scaling.
In contrast, LDA operates in a semantically structured latent space obtained from large-scale visual pretraining.
This representation preserves object-level semantics and spatial coherence, enabling dynamics learning to scale smoothly with increased model capacity, richer training objectives, and more diverse datasets.

 **Effectiveness of Model Scaling.** 
Beyond data scale, LDA exhibits consistent and predictable improvements as model capacity increases.
As shown in Fig. [10](https://arxiv.org/html/2602.12215v1#S5.F10), scaling the model from 0.1B to 0.5B and further to 1B parameters leads to monotonic reductions in action prediction error under the full co-training framework.
This indicates that LDA can effectively absorb additional capacity to model increasingly complex action–dynamics relationships when sufficient heterogeneous supervision is available. The results highlight a promising scaling paradigm in which model capacity, training objectives, and heterogeneous embodied data are jointly aligned, enabling reliable performance gains.

<a id="S5.SS4"></a>

### V-D Analysis of Dynamics Learning

 **Qualitative Analysis of Latent Forward Dynamics.** 
Beyond quantitative prediction errors, we qualitatively examine the forward dynamics learned by LDA, visualized via PCA projections of DINO feature embeddings.
As shown in Fig. [16](https://arxiv.org/html/2602.12215v1#A4.F16), the model produces coherent future-state predictions that respect physical constraints such as object permanence, contact continuity, and motion consistency under the applied action.
Notably, the predicted dynamics focus on task-relevant objects while remaining invariant to visual distractors that do not influence the control loop.
This suggests that LDA learns a dynamics-aware latent world model, capturing how actions causally propagate through the scene rather than merely extrapolating visual appearance.

<a id="figure-11"></a>

![Refer to caption](images/attn_diff.jpg)

> Figure 11: Attention Heat Map: ”Push Right”(top) highlights the mug’s leading edge and trajectory; ”Push Close”(bottom) concentrates on the contact surface. The model attends exclusively to movable regions while ignoring irrelevant background clutter.

 **Action-Conditioned Attention.** 
To interpret how LDA reasons about action-induced state transitions, we visualize attention maps conditioned on different action primitives.
As shown in Fig. [11](https://arxiv.org/html/2602.12215v1#S5.F11), we compare the attention patterns induced by an active motion command ($a_{1}$) with those under a static *No-Op* command ($a_{2}$), and compute their difference to reveal action-specific visual grounding.
Across tasks, LDA consistently attends to regions that are causally relevant to the commanded interaction.
In the *Push Right* scenario, the attention difference highlights the leading edge of the mug and the anticipated motion direction, reflecting awareness of object displacement.
In the *Push Close* task, attention concentrates on the drawer surface where contact and force application are expected.
Importantly, background clutter and visually salient but non-interactive regions are largely suppressed.
These results indicate that LDA conditions visual attention on the physical consequences of actions, selectively focusing on regions that drive state transitions rather than static appearance.
<a id="S6"></a>

## VI Conclusion, Limitation and Future Direction

We present  **LDA-1B** , a robot foundation model that scales latent dynamics learning via universal embodied data ingestion. By assigning heterogeneous data distinct roles and leveraging over  **30k hours**  of human and robot trajectories in the EI-30k dataset, LDA-1B learns dynamics in a structured DINO latent space and employs a mixed-frequency multimodal diffusion transformer, enabling stable training at the  **1B-parameter**  scale. Experiments show strong performance across diverse manipulation and long-horizon tasks, as well as data-efficient fine-tuning on imperfect trajectories.
Limitations include the reliance on fixed DINO visual features and predominantly egocentric camera viewpoints, which may constrain generalization to new visual perspectives and multi-modal signals. Future work includes jointly learning visual representations and latent dynamics, extending to richer sensory modalities, automatically optimizing data roles, and fostering broader community adoption of scalable, heterogeneous data-driven robot foundation models.
<a id="Sx1"></a>

## Acknowledgments

We thank Caowei Meng for collecting teleoperation data; Haoran Liu and Jiayi Su for their assistance in early-stage exploration; Yu-Wei Chao and Shengliang Deng for fruitful discussions; and Junkai Zhao for providing experimental equipment.