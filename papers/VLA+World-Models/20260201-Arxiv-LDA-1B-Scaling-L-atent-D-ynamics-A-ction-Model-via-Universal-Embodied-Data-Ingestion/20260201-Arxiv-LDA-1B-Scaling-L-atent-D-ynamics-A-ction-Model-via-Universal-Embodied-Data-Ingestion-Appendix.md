<a id="A1"></a>

## Appendix A Details of Model

We employ Qwen3-VL-4B-Instruct [[52](#ref-35)] as the joint language and vision encoder to extract high-level semantic representations. Visual observations are encoded using DINOv3-ViT-s [[46](#ref-13)]. During pretraining, we freeze both the VLM and the DINOv3 image encoder to leverage the strong priors from the pretrained language and vision models while allowing the MM-DiT to be trained thoroughly on the downstream structure. In the subsequent finetuning stage, we unfreeze the VLM to enable end-to-end adaptation and further improve overall performance.

Additionally, the MM-DiT is conditioned on a short history of two timesteps, comprising both past DINO-encoded observations and actions, to effectively capture temporal dynamics. Table [V](https://arxiv.org/html/2602.12215v1#A1.T5) presents the detailed configurations of the model and the hyperparameters used during training.

<a id="table-5"></a>

|  **Parameter**  |  **Value**  |
| --- | --- |
|  **Model**  |  |
| VLM | Qwen3-VL [[52](#ref-35)] |
| Observation Encoder | DINOv3-ViT-s [[46](#ref-13)] |
| Hidden Size | 1536 |
| Layers | 16 |
| Attention Heads | 32 |
| Image Shape | (224, 224, 3) |
| Latent Image Shape | (14, 14, 384) |
| Action Chunk | 16 |
|  **Training**  |  |
| Batch Size | 32 * 48 (pretraining) |
|  | 12 * 8 (finetuning) |
| Learning Rate | $1e^{-4}$ |
| Optimizer | AdamW |
| Weight Decay | $1e^{-5}$ |
| Betas | [0.9, 0.95] |
| Epsilon | $1e^{-8}$ |
| LR Schedule | cosine w/ min lr |
| Min LR | $5e^{-7}$ |

<a id="figure-12"></a>

![Refer to caption](images/fig_robocasa.png)

> Figure 12: Qualitative comparison between our model and GR00T [[40](#ref-34)] on RoboCasa-GR1 [[39](#ref-33)] manipulation tasks.
Three representative tasks demonstrate our model’s superior robustness in object grasping and placement accuracy. Critical failure modes of GR00T, including grasp slippage, misaligned object placement, and collision during manipulation, are highlighted with circles, while our model consistently achieves successful task completion.

<a id="table-6"></a>

| model | UWM | UWM-XL | UWM+MMDiT | GR00T | StarVLA | GR00T-EI10k | LDA(DiT) | LDA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PnP Bottle To Cabinet Close | 27 | 41 | 49 | 51.5 | 46 |  **69**  |  **65**  |  **76**  |
| PnP Can To Drawer Close | 22 | 53 | 55 | 13 |  **80**  |  **61**  | 59 |  **71**  |
| PnP Cup To Drawer Close | 18 | 12 | 43 | 8.5 |  **54**  |  **47**  | 40 | 41 |
| PnP Milk To Microwave Close | 22 | 25 | 33 | 14 |  **48**  |  **75**  | 47 |  **52**  |
| PnP Potato To Microwave Close | 16 | 29 | 18 |  **41.5**  | 28 |  **41**  | 39 |  **41**  |
| PnP Wine To Cabinet Close | 31 | 24 | 25 | 16.5 | 46 |  **51**  | 49 |  **57**  |
| PnP Novel From Cuttingboard To Basket | 8 | 18 | 10 |  **58**  | 48 | 43 | 55 |  **65**  |
| PnP Novel From Cuttingboard To Cardboardbox | 8 | 14 | 16 | 46.5 | 40 | 39 |  **57**  |  **69**  |
| PnP Novel From Cuttingboard To Pan | 24 | 20 | 27 |  **68.5**  |  **68**  | 67 | 65 |  **75**  |
| PnP Novel From Cuttingboard To Pot | 16 | 25 | 20 |  **65**  | 52 | 53 |  **57**  |  **61**  |
| PnP Novel From Cuttingboard To Tieredbasket | 10 | 10 | 6 | 46.5 |  **56**  | 29 | 39 |  **51**  |
| PnP Novel From Placemat To Basket | 8 | 16 | 14 |  **58.5**  | 42 | 45 | 37 |  **53**  |
| PnP Novel From Placemat To Bowl | 12 | 10 | 14 |  **57.5**  | 44 |  **55**  | 53 |  **55**  |
| PnP Novel From Placemat To Plate | 10 | 12 | 10 |  **63**  | 48 |  **57**  | 51 |  **59**  |
| PnP Novel From Placemat To Tieredshelf | 2 | 2 | 2 |  **28.5**  | 18 |  **20**  | 22 |  **24**  |
| PnP Novel From Plate To Bowl | 12 | 8 | 14 |  **57**  |  **60**  | 49 |  **57**  | 53 |
| PnP Novel From Plate To Cardboardbox | 2 | 10 | 8 | 43.5 |  **50**  |  **61**  | 43 | 43 |
| PnP Novel From Plate To Pan | 10 | 20 | 16 | 51 |  **54**  | 51 | 49 |  **55**  |
| PnP Novel From Plate To Plate | 22 | 27 | 25 |  **78.7**  |  **70**  | 67 | 59 | 61 |
| PnP Novel From Tray To Cardboardbox | 20 | 25 | 20 |  **51.5**  | 38 | 49 |  **59**  |  **65**  |
| PnP Novel From Tray To Plate | 12 | 18 | 16 |  **71**  | 56 | 57 |  **57**  |  **63**  |
| PnP Novel From Tray To Pot | 18 | 25 | 20 |  **64.5**  | 50 |  **63**  | 53 | 55 |
| PnP Novel From Tray To Tieredbasket | 6 | 16 | 16 |  **57**  | 36 |  **55**  | 39 | 51 |
| PnP Novel From Tray To Tieredshelf | 4 | 2 | 4 |  **31.5**  | 16 |  **31**  | 22 |  **33**  |
| Average | 14.3 | 19.3 | 20.0 | 47.6 | 47.8 |  **51.3**  | 48.9 |  **55.4**  |
<a id="A2"></a>

## Appendix B Detailed Results on the Simulation Benchmark

<a id="A2.SS1"></a>

### B-A Evaluation Setup and Model Description.

All methods are evaluated on the full set of 24 RoboCasa-GR1 [[39](#ref-33)] tasks, with 51 evaluation trials per task.
Unless otherwise specified, models are finetuned using 1,000 demonstrations per task and optimized under the same training paradigm to isolate architectural differences.

We summarize the evaluated models below:

-  **UWM** : A 140M-parameter Unified World Model [[60](#ref-24)], serving as a lightweight baseline.
-  **UWM-XL** : A 1B-parameter UWM variant equipped with Qwen3-VL [[52](#ref-35)] for joint language-vision encoding.
-  **UWM+MM-DiT** : UWM-L with its DiT backbone replaced by our MM-DiT architecture.
-  **GR00T-N1.6** [[40](#ref-34)]: The original GR00T policy model without explicit dynamics modeling.
-  **StarVLA** : A GR00T variant following StarVLA [[47](#ref-72)], replacing the original VLM with Qwen3-VL and trained from scratch on RoboCasa.
-  **GR00T-EI10k** : A strong reproduced baseline pretrained on our EI-10k high-qaulity subset with Qwen3-VL, where VLM parameters are unfrozen during finetuning.
-  **LDA (DiT)** : An ablated version of LDA replacing MM-DiT with a standard DiT backbone.
-  **LDA-1B** : The full Latent Dynamics Action model with MM-DiT, designed to model action-induced state transitions in a structured latent space.

<a id="A2.SS2"></a>

### B-B Task-Level Results and Analysis.

Table [VI](https://arxiv.org/html/2602.12215v1#A1.T6) reports detailed per-task success rates.
LDA consistently outperforms GR00T across contact-rich and cluttered rearrangement tasks, with particularly large gains in scenarios requiring precise placement and closing actions, such as
PnP Bottle To Cabinet Close (76 % vs. 51.5%),
PnP Can To Drawer Close (71% vs. 13%),
and PnP Milk To Microwave Close (52% vs. 14%).

As illustrated in Fig. [12](https://arxiv.org/html/2602.12215v1#A1.F12), GR00T frequently fails due to a lack of anticipation of post-action consequences.
For example, after placing an object inside a container, GR00T often retracts its arm along a trajectory that collides with the object, causing it to tip over.
In contrast, LDA anticipates such interactions and generates trajectories that preserve object stability throughout the entire manipulation sequence.

The largest improvements are observed in novel-object rearrangement tasks involving transfers across surfaces and containers
(e.g., *Cuttingboard*$\rightarrow$*Basket*/*Cardboardbox*,
*Placemat*$\rightarrow$*Plate*/*Tieredshelf*,
and *Tray*$\rightarrow$*Cardboardbox*/*Plate*/*Pot*).
These tasks require adaptive contact handling and trajectory correction under clutter, where LDA shows clear advantages.
While GR00T remains competitive on a small subset of simple pick-and-place tasks with minimal environmental interaction, these cases are limited.
Overall, LDA’s higher average success rate (55.4 %vs. 47.6 %) reflects a systematic advantage in complex and contact-rich manipulation scenarios rather than isolated gains.

<a id="figure-13"></a>

![Refer to caption](images/real_setup.jpg)

> Figure 13:  Real-world robot platforms used in our physical experiments. From left to right:
(1) Galbot G1 equipped with a standard two-finger parallel gripper for basic grasping tasks;
(2) Galbot G1 fitted with the SharpaWave dexterous hand (22 DoF) for fine manipulation;
(3) Unitree G1 mounted with the BrainCo dexterous hand (10 DoF) and a Zed Mini camera.
This multi-platform setup demonstrates the generalization capability of our LDA model across diverse robot morphologies and end-effectors.

<a id="figure-14"></a>

![Refer to caption](images/task_overview.jpg)

> Figure 14: Task descriptions for the Galbot G1 robot equipped with a standard two-finger parallel-jaw gripper, spanning four manipulation categories.
<a id="A3"></a>

## Appendix C Details regarding real-world experiment

<a id="A3.SS1"></a>

### C-A Real-world Setup.

We conduct real-world experiments on two humanoid platforms: the Galbot G1 and the Unitree G1, as shown in Fig.[13](https://arxiv.org/html/2602.12215v1#A2.F13). The Galbot G1, with two 7-DoF arms, is equipped with two interchangeable end-effectors: two-finger parallel-jaw grippers and 22-DoF SharpaWave dexterous hands. The Unitree G1 uses 10-DoF BrainCo hands. In all real-robot configurations, the policy receives visual input only from an egocentric head-mounted camera, providing a first-person view of the workspace.

<a id="A3.SS2"></a>

### C-B Task description and evaluation protocol.

To validate the effectiveness of our method on physical systems, we evaluate eight representative manipulation tasks involving single-arm, dual-arm coordination, tool use, and contact-rich interactions.
For object generalization, movable objects are randomized within predefined spatial regions while several supporting objects (e.g., baskets, dustpans, and trash bins) remain fixed to isolate task-specific manipulation challenges rather than compounding errors from initial grasp failures. All experiments are conducted in-domain, and each trial is terminated after 200 seconds if unsuccessful.
Task success is defined using task-specific criteria such as successful object placement, execution of full procedural steps, or normalized scoring metrics for partial completion in long-horizon tasks.
We evaluate each task over independent trials. The corresponding training data volume and success criteria for each task are summarized in Table [VII](https://arxiv.org/html/2602.12215v1#A3.T7).

<a id="figure-15"></a>

![Refer to caption](images/dex_demo.jpg)

> Figure 15:  Dexterous manipulation task description across two robotic platforms. Top three rows: Unitree robot equipped with BrainCo hands performing bottle placement, MacBook opening, and nail extraction. Bottom two rows: Galbot robot utilizing SharpaWave hands executing bread placement and flipping tasks.

<a id="table-7"></a>

|  **Task Abbreviation**  |  **Description**  |  **Test Protocol**  |
| --- | --- | --- |
| Pick Vegetable | Pick a plastic pepper and place it into a basket using the left gripper. Pepper is randomized within a 15$\times$30 cm region. | 10 trials; success if placed in basket |
| Handover | Left gripper grasps a bottle and passes it to the right gripper, which places it into a basket. Bottle randomized within 15$\times$30 cm. | 10 trials; success if placed in basket |
| Wipe Board | Use an eraser to remove marker writing from a whiteboard. Writing area randomized within 25$\times$40 cm. | 10 trials; scored from 0–5 based on cleaning completeness |
| Flip Box | Flip an upside-down storage box to upright using bimanual manipulation. Box randomized within 2$\times$4 cm. | 10 trials; success if fully flipped |
| Water Flower (pouring) | grasp a watering bottle and pour water into a flower pot. Pot randomized within 15$\times$15 cm. | 10 trials; success if pouring posture is achieved with spout above pot |
| Knock the block with hammer (pnp2) | grasp a hammer with a very thin handle and then knock the specific block. Hammer randomized within 15$\times$15 cm. | 60 trials; only if both the grasp and knock succeed. |
| Sweep Table | Sweep ten nails into a dustpan using a broom and dustpan. Nail positions randomized within 10$\times$25 cm. | 10 trials; success rate is computed as the proportion of nails collected in the dustpan. |
| Throw Rubbish | Pick paper balls, place them into a dustpan, and dump them into a trash can. Paper balls are randomized within a 20$\times$25 cm area. | 10 trials; success rate is computed as the proportion of paper balls successfully dumped into the trash can. |

<a id="A3.SS3"></a>

### C-C More Analysis.

To validate the efficacy of our proposed approach, we conducted a comprehensive comparison against two baseline policies: GR00T-N1.6 [[40](#ref-34)] and $\pi_{0.5}$[[23](#ref-15)]. Our method (LDA) demonstrates superior performance across all four evaluated categories: Pick & Place, Contact-rich Manipulation, Fine Manipulation, and Long-horizon Manipulation.

 **Performance on Basic Grasping Tasks** . In standard Pick & Place scenarios, LDA achieves a dominant success rate, reaching 90.0% on the ”handover” task, significantly outperforming $\pi_{0.5}$[[23](#ref-15)] (70.0%) and nearly doubling the success rate of GR00T-N1.6 [[40](#ref-34)] (50.0%). This indicates that our policy has learned a more robust grasping primitive and achieve better few-shot adaptation on unseen Galbot robot, benefiting from larger-scale cross-embodiment learning.

 **Robustness in Contact-Rich and Fine Manipulation** . The advantages of LDA become increasingly pronounced in tasks requiring precise dynamic interaction.
In Contact-rich Manipulation, such as ”flip the box,” LDA achieves a 60.0% success rate compared to just 20.0% for GR00T-N1.6 [[40](#ref-34)]. This suggests that LDA effectively models the complex contact dynamics required to manipulate objects without slippage or instability, whereas the baselines likely struggle with the discontinuous nature of the contact forces.
Similarly, in Fine Manipulation tasks like ”pouring,” which demand continuous closed-loop feedback, our method sustains an 80.0% success rate, surpassing the best baseline ($\pi_{0.5}$[[23](#ref-15)]) by 20 percentage points.

 **Capabilities in Long-Horizon Planning** .
The most striking distinction emerges in the Long-horizon Manipulation category. While baseline methods achieve moderate success on the relatively simple “sweep the table” task, they completely fail on the more complex “throw rubbish” task, registering a 0.0% success rate. In stark contrast, LDA achieves a 35.0% success rate, demonstrating robustness in multi-stage, temporally extended scenarios.
This performance gap reveals a fundamental limitation of existing approaches: their inability to manage compounding errors over long action sequences due to a lack of explicit dynamics modeling. LDA’s success stems from its capacity to reason about the physical consequences of actions across time, maintain temporal consistency in latent states, and recover from intermediate deviations—capabilities that are essential for real-world, multi-step manipulation. Crucially, this advantage is rooted in LDA’s dynamics-aware architecture, which aligns predicted visual features with underlying physical transitions and mitigates covariate shift through structured temporal modeling.
Collectively, these results validate that explicitly modeling latent dynamics is not merely beneficial but *necessary* for reliable, generalizable robotic manipulation in complex, real-world settings.

<a id="table-8"></a>

|  **Task Abbreviation**  |  **Description**  |  **Test Protocol**  |
| --- | --- | --- |
| Pick Bottle | Pick up a plastic bottle and place it onto a fixed target region using the right hand. Bottle position is randomized. | 20 trials; success if bottle is upright and its base overlaps at least half of the target region |
| Open MacBook | Left hand stabilizes the base while the right hand opens the hinge by pushing the upper edge. Initial opening angle is randomized. | 20 trials; success if opening angle exceeds 75% of maximum |
| Pull Nail | Use a claw hammer held by the right hand to extract a nail from the surface. Hammer pose is randomized. | 10 trials; scored with partial credit: 0.25 for locating, 0.5 for single-claw removal, 1.0 for full claw removal |
| Pick Bread | Pick a bread item and place it into a plate using the right hand. Three bread types are used with equal distribution. | 10 trials; success if bread is placed into the plate |
| Flip Bread | Flip a long bread using a spatula held by the right hand. Bread pose is randomized over a large region. | 10 trials; 1.0 if flipped on first attempt, 0.5 if second, 0 otherwise |

 **Capabilities in Dexterous Manipulation.** 
LDA consistently outperforms baselines on both low-DoF and high-DoF hands, with the performance gap becoming more pronounced as task difficulty and dexterity requirements increase.
For low-DoF hands, LDA already demonstrates strong robustness on tasks involving tool use and force-sensitive interactions. On *Pick Bottle*, LDA achieves a 90% success rate, substantially higher than $\pi_{0.5}$ (20%) and GR00T-N1.6 (75%). On *Pull Nail*, which requires precise force direction and stable contact maintenance, LDA reaches 80% success, while $\pi_{0.5}$ completely fails and GR00T-N1.6 achieves only 40%. Notably, all methods perform well on *Open Macbook*, suggesting that tasks with strong geometric affordances and limited contact ambiguity are less challenging even for baseline policies.
The advantage of LDA becomes even more evident with high-DoF hands, where action spaces are larger and control errors accumulate more easily. On *Pick Bread*, LDA attains a 70% success rate, outperforming GR00T-N1.6 (20%) and $\pi_{0.5}$ (10%). The gap further widens on *Flip Bread*, a highly dexterous task requiring coordinated finger motion and continuous contact reasoning, where LDA achieves 90% success while both baselines remain at only 10%.
These results highlight LDA’s superior ability for high-dimensional control and contact-rich dexterous manipulation. Unlike baseline methods that rely primarily on reactive policies, LDA benefits from dynamics-aware latent representations that capture fine-grained physical interactions over time. This enables more stable control, improved contact reasoning, and effective recovery from transient failures—capabilities that are critical for dexterous manipulation with complex, multi-DoF robotic hands.
<a id="A4"></a>

## Appendix D Details of EI-30k.

<a id="A4.SS1"></a>

### D-A Data Processing Pipeline for Robot and Human Datasets

To ensure consistency and usability across heterogeneous robot and human datasets, we design a standardized data processing pipeline that converts raw recordings into a unified representation suitable for effective learning of both policy and dynamics. The pipeline consists of three main stages: dataset standardization, coordinate alignment and cleaning, and post-processing for training.

<a id="A4.SS1.SSS0.Px1"></a>

#### Dataset Standardization

All raw datasets are first converted into the common LeRobot [[8](#ref-31)] 2.1 format. This format includes:

-  **End-effector poses** : 6D position and orientation for both hands (human) or manipulators (robot);
-  **Hand articulation** : 21-point MANO keypoints for human hands (when available) and binary or continuous gripper states for robots;
-  **Camera parameters** : intrinsic and extrinsic matrices enabling reprojection across coordinate frames;
-  **Task and temporal metadata** : task identifiers, episode boundaries and timestamps.

During this stage, all sequences are uniformly resampled to 10 Hz, and structured metadata files are generated to preserve the alignment between frames and their semantic annotations, ensuring temporal coherence and task-aware data organization for downstream training.

After standardization of LeRobot format, we implement an easy-to-use Dataset class for the following data process pipeline to harmonize heterogeneous action data across diverse datasets.

<a download="" href="data:text/plain;base64,ICAgIGRlZiBfX2luaXRfXyhzZWxmLCBkYXRhc2V0OiBzdHIsIGVlZl9pbl93b3JsZDoKICAgIGludCwgaGFzX21hbm86IGJvb2wpOgogICAgICAgIHNlbGYuZGF0YXNldCA9IGRhdGFzZXQKICAgICAgICBzZWxmLmVlZl9pbl93b3JsZCA9IGVlZl9pbl93b3JsZAogICAgICAgICMgMSBpZiB3cmlzdCBpbiB3b3JsZCBjb29yZGluYXRlcwogICAgICAgIHNlbGYuaGFzX21hbm8gPSBoYXNfbWFubwogICAgICAgIHNlbGYuZWVmX29mZnNldCA9IHtoYW5kOiBucC5leWUoNCkgZm9yIGhhbmQKICAgICAgICBpbiBIQU5EX0tFWVN9CiAgICAgICAgc2VsZi5lZWZfa2V5cyA9IEhBTkRfS0VZUwoKICAgIGRlZiBnZXRfd3Jpc3Qoc2VsZiwgZGY6IHBkLkRhdGFGcmFtZSkgLT4KICAgIGRpY3Rbc3RyLCBucC5uZGFycmF5XToKICAgICAgICBwYXNzCgogICAgZGVmIGdldF9tYW5vX29yX2dyaXBwZXIoc2VsZiwgZGY6IHBkLkRhdGFGcmFtZSk6CiAgICAgICAgcGFzcwo=">⬇</a>

def

__init__

(

self

,

dataset

:

str

,

eef_in_world

:

int

,

has_mano

:

bool

)

:

self

.

dataset

=

dataset

self

.

eef_in_world

=

eef_in_world

#

1

if

wrist

in

world

coordinates

self

.

has_mano

=

has_mano

self

.

eef_offset

=

{

hand

:

np

.

eye

(

4

)

for

hand

in

HAND_KEYS

}

self

.

eef_keys

=

HAND_KEYS

def

get_wrist

(

self

,

df

:

pd

.

DataFrame

)

-

>

dict

[

str

,

np

.

ndarray

]

:

pass

def

get_mano_or_gripper

(

self

,

df

:

pd

.

DataFrame

)

:

pass

<a id="A4.SS1.SSS0.Px2"></a>

#### Coordinate Alignment and Data Cleaning

Human and robot datasets often employ inconsistent coordinate frame definitions. To unify them, particularly the end-effector (EEF) representations, we apply the following alignment and cleaning steps:

-  **End-effector coordinate alignment** : For each dataset, we define a canonical EEF frame (e.g., at the wrist or gripper center). All recorded hand or manipulator poses are transformed into this common frame using a dataset-specific rigid offset, estimated through geometric inspection or visual validation.
-  **Camera motion decoupling** : For sequences captured in a moving camera frame, hand trajectories are reprojected into a fixed world coordinate system to eliminate artifacts caused by camera motion.
-  **Keypoint standardization** : Human hand poses without native MANO keypoints are converted into the standard 21-point MANO representation, expressed relative to the aligned wrist frame.
-  **Data validation** : Hand visibility is verified using an off-the-shelf detector; frames with occluded, truncated, or kinematically invalid hand data are discarded to ensure annotation reliability.

For robot datasets, we further normalize actuation signals: gripper widths are scaled to a consistent range (e.g., $[0,1]$), and joint encodings are harmonized to match a unified kinematic convention.

<a id="A4.SS1.SSS0.Px3"></a>

#### Data Cleaning

Textual annotations are unified into a structured format that explicitly describes the environmental context, per-hand actions (left/right), and high-level task objectives. When original annotations are inconsistent or missing, we leverage vision-language models to generate coherent, semantically aligned instructions.
Finally, all processed datasets are organized by agent type (human or robot) and accompanied by comprehensive metadata files detailing task definitions, episode boundaries, and dataset statistics. This standardized pipeline ensures a consistent, interoperable data representation across domains—enabling robust, scalable training of dexterous manipulation policies that generalize across embodiment and task complexity.

<a id="table-9"></a>

|  **Data Type**  |  **Source / Sub-dataset**  |  **Duration (h)**  |
| --- | --- | --- |
|  **Real-world Robot**  | Open X-Embodiment [[12](#ref-36)] | 3000 |
| Agibot World [[6](#ref-37)] | 3276 |  |
| RoboMIND [[50](#ref-38)] | 305 |  |
| Humanoid Everyday [[57](#ref-39)] |  30 |  |
| RoboCOIN [[51](#ref-40)] | 500 |  |
| Galaxea [[48](#ref-41)] | 500 |  |
| LET [[28](#ref-51)] | 1000 |  |
|  **Simulated Robot**  | InternData-A1 [[11](#ref-16)] | 7433 |
| Behavior-1k [[29](#ref-53)] | 1200 |  |
|  **Ego Human**  **(w/ Action)**  | Ego4D [[18](#ref-54)] | 3670 |
| Epic-Kitchens [[13](#ref-55)] | 100 |  |
| Ego-Exo4d [[19](#ref-56)] | 1286 |  |
| SSV2 [[17](#ref-69)] | 240 |  |
| EgoDex [[21](#ref-57)] | 830 |  |
| HOT3D [[2](#ref-70)] | 16 |  |
| HoloAssist [[49](#ref-58)] | 166 |  |
| OAKINK2 [[54](#ref-59)] | 6.5 |  |
| TACO [[33](#ref-61)] | 3.2 |  |
| HOI4D [[34](#ref-62)] | 7.6 |  |
| ARCTIC [[15](#ref-63)] | 2.3 |  |
|  **Ego Human**  **(Actionless)**  | Egocentric-10k [[1](#ref-64)] | 10000 |
| RH20T-human [[16](#ref-60)] | 100 |  |
| Egome [[44](#ref-66)] | 80 |  |
|  | Taste-Rob [[55](#ref-65)] | 130 |
|  |  **Total**  |  **30k+**  |

<a id="figure-16"></a>

![Refer to caption](images/dino.jpg)

> Figure 16: DINO Feature Prediction Visualization. Left column: Original RGB input images. Middle column: Ground-truth DINO features extracted by DINOv3 [[46](#ref-13)]. Right column: DINO features predicted by our model.

<a id="A4.SS2"></a>

### D-B Data Composition.

Our training data spans four complementary categories, totaling more than 30,000 hours of egocentric experience:

<a id="A4.SS2.SSS0.Px1"></a>

#### Real-world Robot Data

This category includes large-scale physical robot execution logs. We primarily leverage *Open X-Embodiment*[[12](#ref-36)] and *Agibot World*[[6](#ref-37)] for general-purpose manipulation. To enhance hardware-specific capabilities, we incorporate *Humanoid Everyday*[[57](#ref-39)] for bipedal locomotion dynamics and *Galaxea*[[48](#ref-41)] for high-fidelity dexterous tasks. Additionally, we include *RoboCOIN*[[51](#ref-40)] despite its noisier action labels, as it provides valuable diverse environment explorations.

<a id="A4.SS2.SSS0.Px2"></a>

#### Simulated Robot Data

To provide dense, noise-free supervision, we use high-quality simulated trajectories. The majority comes from *InternData-A1*[[11](#ref-16)], which offers large-scale automated generation of locomotion and basic manipulation sequences. *Behavior-1k*[[29](#ref-53)] further contributes long-horizon task demonstrations in simulated household environments, enabling the model to learn complex task hierarchies.

<a id="A4.SS2.SSS0.Px3"></a>

#### Egocentric Human Data with Actions

This subset bridges human intent and robot-executable actions. We draw from large-scale datasets such as *Ego4D*[[18](#ref-54)] , *Epic-Kitchens*[[13](#ref-55)] , *Ego-Exo4d*[[19](#ref-56)] and *SSV2*[[17](#ref-69)] focusing on object-interaction segments. High-precision sources like *EgoDex*[[21](#ref-57)] and *HOT3D*[[2](#ref-70)] provide fine-grained 3D hand poses and contact information, critical for learning extrinsic dexterity.

<a id="A4.SS2.SSS0.Px4"></a>

#### Egocentric Human Data without Actions

Representing the largest source of visual diversity, this category consists of first-person observations. *Egocentric-10k*[[1](#ref-64)] serves as the primary source, covering a broad spectrum of daily activities. Additional datasets like *RH20T-human*[[16](#ref-60)] and *Taste-Rob*[[55](#ref-65)] contribute domain-specific visual priors. Although these trajectories lack explicit action labels, they provide a powerful self-supervised signal for learning world dynamics, visual affordances, and temporal structure.
<a id="A5"></a>

## Appendix E Details of Other Experiments

<a id="A5.SS1"></a>

### E-A Action-Conditioned Attention Visualization.

We provide additional details on how action-conditioned attention maps are computed and interpreted.
Our visualization is based on the Diffusion Transformer (DiT) [[43](#ref-30)] backbone, where visual tokens and action embeddings interact through shared self-attention layers.

For a given observation, we extract attention maps from the middle transformer blocks, where high-level semantic and geometric information is most prominent.
Conditioned on an active action primitive (e.g., “Push Right”), we compute the attention weights $A_{1}$, which quantify the influence of each spatial token on the predicted latent transition.
To establish a reference, we generate a baseline attention map $A_{2}$ by replacing the action embedding with a *No-Op* (static) command.

We then compute the absolute difference:

<a id="eq-3"></a>

$$
\Delta A=|A_{1}-A_{2}|, \tag{(3)}
$$

which isolates attention changes induced purely by the action condition.
This subtraction effectively removes generic visual saliency (e.g., high-contrast edges or background objects) and highlights regions whose relevance emerges only when a specific action is applied.

As illustrated in Fig. [11](https://arxiv.org/html/2602.12215v1#S5.F11), the resulting difference maps consistently emphasize contact regions, force application points, and anticipated motion trajectories.
For example, in the “Push Right” task, attention shifts toward the gripper–object contact interface and the direction of expected displacement.
This behavior demonstrates that the DiT dynamically re-weights visual tokens based on the physics implied by the action, rather than passively encoding static appearance.

<a id="A5.SS2"></a>

### E-B Visualization of Latent Forward Dynamics

We provide additional qualitative visualizations to illustrate the forward dynamics learned by LDA.
These qualitative results complement the quantitative analysis and provide further evidence that LDA learns structured, dynamics-aware latent representations suitable for long-horizon reasoning and control.