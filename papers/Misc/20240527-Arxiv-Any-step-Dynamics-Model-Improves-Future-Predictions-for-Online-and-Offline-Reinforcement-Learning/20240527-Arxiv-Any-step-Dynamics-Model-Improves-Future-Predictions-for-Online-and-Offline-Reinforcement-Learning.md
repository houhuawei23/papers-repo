# Title: Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning

- ArXiv: 2405.17031
- Authors: Chengxing Jia, ,, Junyin Ye, Jiaji Zhang, Yang Yu, Corresponding Author
- Sections: 44
- Estimated tokens: 20.3k

## Contents

- 1 Introduction
- 2 Preliminaries
  - 2.1 Markov Decision Process and Reinforcement Learning
  - 2.2 Model-based Reinforcement Learning
- 3 Method
  - 3.1 Any-step Dynamics Model
    - Definition 3.1 (Any-step Dynamics Model) .
  - 3.2 ADMPO-ON: ADM for Policy Optimization in Online Setting
  - 3.3 ADMPO-OFF: ADM for Policy Optimization in Offline Setting
    - Definition 3.2 (ADM-Uncertainty Quantifier) .
    - Assumption 3.3 (Admissible Error Estimator) .
    - Theorem 3.4 .
    - Proof.
- 4 Experiments
  - 4.1 Dynamics Model Evaluation
  - 4.2 Evaluation in Online Setting
  - 4.3 Evaluation in Offline Setting
    - 4.3.1 D4RL Benchmark Results
    - 4.3.2 NeoRL Benchmark Results
    - 4.3.3 Uncertainty Quantification
- 5 Related Work
  - 5.1 Online Model-based Reinforcement Learning
  - 5.2 Offline Model-based Reinforcement Learning
- 6 Conclusion

## Abstract

###### Abstract

Model-based methods in reinforcement learning offer a promising approach to enhance data efficiency by facilitating policy exploration within a dynamics model. However, accurately predicting sequential steps in the dynamics model remains a challenge due to the bootstrapping prediction, which attributes the next state to the prediction of the current state. This leads to accumulated errors during model roll-out. In this paper, we propose the Any-step Dynamics Model (ADM) to mitigate the compounding error by reducing bootstrapping prediction to direct prediction. ADM allows for the use of variable-length plans as inputs for predicting future states without frequent bootstrapping. We design two algorithms, ADMPO-ON and ADMPO-OFF, which apply ADM in online and offline model-based frameworks, respectively. In the online setting, ADMPO-ON demonstrates improved sample efficiency compared to previous state-of-the-art methods. In the offline setting, ADMPO-OFF not only demonstrates superior performance compared to recent state-of-the-art offline approaches but also offers better quantification of model uncertainty using only a single ADM.

<a id="section-1"></a>

## 1 Introduction

Model-based Reinforcement Learning (MBRL) [34] has demonstrated empirical success in both online [15, 6, 11, 35, 21, 31] and offline [51, 25, 50, 40, 44, 33] settings. The essence of MBRL lies in the dynamics model, where extensive explorations and evaluations of the agent can occur, thereby reducing the reliance on real-world samples. Embedded in the model-based framework, online policy optimization can leverage a large Update-To-Data (UTD) ratio [8] to improve sample efficiency, while offline policy optimization can be completed using the model augmented data beyond the dataset.

Although some efforts aim to propose high-fidelity dynamics models, such as adversarial models [9, 5], causal models [53], and ensemble dynamics models [11, 21, 51] adopted by the majority of MBRL algorithms, it is challenging to generate high-quality imaginary samples via long-horizon model roll-out. In a dynamics model with the common form, the state-action pair at time step $t$, $(s_{t},a_{t})$, is used as input to predict the next state $s_{t+1}$. Thus, the bootstrapping prediction, which attributes the next state to the prediction of the current state, is inevitably employed to roll out states in the dynamics model. The deviation error of generated states increases with the roll-out length since the error accumulates gradually as the state transitions in imagination. If updated on the unreliable samples with a large compounding error, the policy will be misled by biased policy gradients.

In the online setting, the impact of compounding error [49] on policy optimization restricts the utilization of the model, thereby hindering further improvement in sample efficiency. In the offline setting, compounding error affects the accuracy of current model uncertainty estimation based on ensemble. For instance, using the behavior policy corresponding to the dataset for a long-horizon roll-out in the model still leads to a great compounding error, and the accumulated deviations of different learners cannot be similar. In this case, the divergence of the ensemble will inevitably be large, which is inconsistent with the situation that the actual trajectory is within the region covered by the dataset. Therefore, it is essential to reduce compounding error in both online and offline settings.

One potential way to deal with the issue of compounding error is to reduce bootstrapping prediction to direct prediction, considering the direct state transition after executing a multi-step action sequence [2, 3, 7, 36]. Although state $s_{t+1}$ is only dependent on state-action $(s_{t},a_{t})$ under the assumption of Markov property [46], the prediction of $s_{t+1}$ can actually leverage earlier information. Tracing back a prior $k$-step plan, $s_{t+1-k}$ and the intermediate $k$-step actions $(a_{t+1-k},a_{t+2-k},\cdots,a_{t})$ are sufficient to constitute an attribution to predict $s_{t+1}$.

To handle the variable-length plans, we introduce a special Any-step Dynamics Model (ADM) that allows for the use of $s_{t+1-k}$ and $(a_{t+1-k},a_{t+2-k},\cdots,a_{t})$ corresponding to any integer $k$ within a specified range as inputs for predicting $s_{t+1}$. When the agent faces changes occurring in the trajectory distribution, the state predictions from different backtracking lengths will exhibit noticeable divergence. This feature naturally enables ADM to estimate model uncertainty without ensemble. Replacing the ensemble dynamics model with ADM, we devise a unique model roll-out method with random backtracking, which can be plugged into any existing MBRL algorithmic frameworks. In this paper, our main purpose is to demonstrate how the augmented data generated by ADM exhibits excellent effectiveness, both in improving future predictions and measuring the model uncertainty.

In general, our contributions are summarized as follows. (1) We present a generalized dynamics model called ADM to replace the dynamics model used in existing online and offline MBRL algorithms and demonstrate its superiority in reducing the compounding error. (2) We propose a new online MBRL algorithm called ADMPO-ON based on ADM and show that it can outperform recent state-of-the-art online model-based algorithms in terms of sample efficiency while retaining competitive performance on MuJoCo [47] benchmarks. (3) We propose a new offline MBRL algorithm called ADMPO-OFF based on ADM and show that it can effectively quantify the model uncertainty, achieving superior performance compared to recent state-of-the-art offline algorithms on D4RL [16] and NeoRL [39] benchmarks.

<a id="section-2"></a>

## 2 Preliminaries

<a id="section-2-1"></a>

### 2.1 Markov Decision Process and Reinforcement Learning

We consider a standard Markov Decision Process (MDP) specified by a tuple $\mathcal{M}=(\mathcal{S},\mathcal{A},T,\rho_{0},\gamma)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $T(s_{t+1},r_{t+1}|s_{t},a_{t})$ is the dynamics function that calculates the conditioned distribution of $s_{t+1}\in\mathcal{S}$ and $r_{t+1}\in\mathbb{R}$ given $(s_{t},a_{t})$, $\rho_{0}$ is the initial state distribution, and $\gamma$ is the discount factor. We use $\rho^{\pi}$ to denote the on-policy distribution over states induced by the dynamic function $T$ and the policy $\pi$. From a multi-step perspective, the attribution of state $s_{t+1}$ and reward $r_{t+1}$ can be traced back to the earlier $k$-step plan, $s_{t-k+1}$ along with the action sequence $a_{t-k+1:t}=(a_{t-k+1},a_{t-k+2},\cdots,a_{t})$ in between. This relationship can be represented by the $k$-step dynamics model

$$
T^{k}(s_{t+1},r_{t+1}|s_{t-k+1},a_{t-k+1:t})=\sum_{(s_{t-k+2:t},r_{t-k+2:t})} \prod_{i=0}^{k-1}T(s_{t-i+1},r_{t-i+1}|s_{t-i},a_{t-i}).(1)
$$

We use $\Gamma^{k}_{\pi}(s_{t-k+1:t},a_{t-k+1:t}|s_{t+1})$ to denote the distribution over $(s_{t-k+1:t},a_{t-k+1:t})$ conditioned on $s_{t+1}$ induced by the dynamic function $T$ and the policy $\pi$.

The optimization goal of Reinforcement Learning (RL) is to find a policy $\pi$ that maximized the expected discounted return $\mathbb{E}_{\rho^{\pi}}\left[\sum_{t=1}^{\infty}\gamma^{t-1}r_{t}\right]$. Such a policy can be derived from the estimation of the state-action value function $Q^{\pi}(s_{t},a_{t})=\mathbb{E}_{(s_{t+1},r_{t+1})\sim T(\cdot|s_{t},a_{t})} \left[r_{t+1}+\gamma V^{\pi}(s_{t+1})\right]$, where $V^{\pi}(s_{t+1})=\mathbb{E}_{a_{t+1}\sim\pi(\cdot|s_{t+1})}\left[Q^{\pi}(s_{t+ 1},a_{t+1})\right]$ is the state value function.

<a id="section-2-2"></a>

### 2.2 Model-based Reinforcement Learning

MBRL aims to find the optimal policy while transferring the agent’s explorations and evaluations from the environment to the learned dynamics model. Given a dataset $\mathcal{D}_{\mathrm{env}}$ collected via interaction in the real environment, the dynamics model $\hat{T}$ is typically trained to maximize the expected likelihood $\mathbb{E}_{(s_{t},a_{t},r_{t+1},s_{t+1})\sim\mathcal{D}_{\mathrm{env}}}[\log \hat{T}(s_{t+1},r_{t+1}|s_{t},a_{t})]$. The estimated dynamics model defines a surrogate MDP $\hat{\mathcal{M}}=(\mathcal{S},\mathcal{A},\hat{T},\rho_{0},\gamma)$. Then any RL algorithm can be used to recover the optimal policy with the augmented dataset $\mathcal{D}_{\mathrm{env}}\cup\mathcal{D}_{\mathrm{model}}$, where $\mathcal{D}_{\mathrm{model}}$ is the synthetic data rolled out in $\hat{\mathcal{M}}$.

The above-mentioned paradigm is adopted by model-based policy optimization (MBPO) [21] and much of its follow-up work [31, 30, 38, 12] in the online setting. These works don’t need to consider the issue of model coverage, as the agent can explore online to fill in the regions where the dynamics model is uncertain. However, in the offline setting, the limited dataset causes $\hat{T}$ to cover only a part of the state-action space. Once the agent encounters out-of-distribution samples during roll-out in $\hat{\mathcal{M}}$, the learning process can collapse. Therefore, MOPO [51] and some of its subsequent offline MBRL algorithms [25, 44] incorporate a penalty term in the reward function to measure the model uncertainty, allowing the agent to sample within safe regions of $\hat{T}$.

<a id="section-3"></a>

## 3 Method

In this section, we propose a special Any-step Dynamics Model (ADM) to replace the mainstream ensemble dynamics models. Applying ADM to existing MBRL frameworks for policy optimization, we introduce two algorithms, namely online ADMPO-ON and offline ADMPO-OFF. ADM improves future state predictions since it reduces bootstrapping prediction to direct prediction by backtracking variable-length plans. Consequently, ADMPO-ON can improve the sample efficiency in the online setting, while ADMPO-OFF can accurately estimate the model uncertainty in the offline setting.

<a id="section-3-1"></a>

### 3.1 Any-step Dynamics Model

Currently, the prevalent dynamics models typically operate on a single-step basis, with $s_{t}$ and $a_{t}$ as inputs to predict $s_{t+1}$ and $r_{t+1}$. In a broader context, dynamics models can also be multi-step [2, 3, 7], where inputs encompass $s_{t}$ along with a $k$-step sequence of actions $(a_{t},a_{t+1},\cdots,a_{t+k-1})$ to predict $s_{t+k}$ and $r_{t+k}$. To introduce flexibility in the backtracking length of the model, we further extend the definition of the multi-step dynamics model to allow $k$ to be any positive integer within a specified range, as delineated in Definition [3.1](https://arxiv.org/html/2405.17031v1#S3.Thmtheorem1).

###### Definition 3.1 (Any-step Dynamics Model) .

Given the maximum backtracking length $m$, an any-step dynamics model $\hat{T}(s_{t+k},r_{t+k}|s_{t},a_{t:t+k-1})$ is the distribution of $s_{t+k}\in\mathcal{S}$ and $r_{t+k}\in\mathbb{R}$ conditioned on the $k$-step plan $(s_{t},a_{t:t+k-1})=(s_{t},a_{t},a_{t+1},\cdots,a_{t+k-1})\in\mathcal{S}\times \mathcal{A}^{k}$, where $k$ can be any integer between $[1,m]$.

<a id="figure-1"></a>

![admpo](images/admpo.png)

> Figure 1: Illustration of any-step dynamics model (left) structured using RNN and its application for next-step prediction with random backtracking (right).

To handle inputs with variable step sizes, we utilize an RNN [13] with a GRU [10] cell to implement the any-step dynamics model, as depicted in the left part of Figure [1](#figure-1). Certainly, Transformer [48] is also a feasible choice, but we do not consider it because the model structure is beyond the scope of this study. Since the input state consists of only one step, while the action may be a sequence of multiple steps, we duplicate the state to match the length of the action sequence, and then sequentially feed it into the RNN. The input $(s_{t},a_{t:t+k-1})$, after being represented by the RNN, yields the hidden $h^{k}_{t}$, which is then fed into an MLP to obtain the mean and standard deviation of $s_{t+k}$ and $r_{t+k}$, i.e., $(\boldsymbol{\mu}^{s}_{t+k},\boldsymbol{\Sigma}^{s}_{t+k})$ and $(\boldsymbol{\mu}^{r}_{t+k},\boldsymbol{\Sigma}^{r}_{t+k})$. Similar to previous model-based methods [21, 38, 31], we model the distributions of $s_{t+k}$ and $r_{t+k}$ as Gaussian distributions and predict them through sampling. We call the Any-step Dynamics Model as ADM and denote it as $\hat{T}_{\theta}(s_{t+k},r_{t+k}|s_{t},a_{t:t+k-1})$, where $\theta$ represents the neural parameters. With the real samples from the environment, $\hat{T}_{\theta}$ is trained to maximize the expected likelihood:

$$
J_{T}(\theta)=\frac{1}{m}\sum_{k=1}^{m}\mathbb{E}_{(s_{t},a_{t:t+k-1},r_{t+k}, s_{t+k})\sim\mathcal{D}_{\textrm{env}}}\left[\log\hat{T}_{\theta}(s_{t+k},r_{t +k}|s_{t},a_{t:t+k-1})\right].(2)
$$

With $\hat{T}_{\theta}$, the frequent bootstrapping during model roll-out can be reduced. Specifically, given the maximum backtracking length $m$, a state-action sequence of length $m$, $(s_{1},a_{1},s_{2},a_{2},\cdots,s_{m},a_{m})$, is sampled from the data buffer to start the roll-out in $\hat{T}_{\theta}$. For predicting $s_{m+1}$, an integer between $[1,m]$ is chosen uniformly at random as the backtracking length. If only one step is selected for backtracking, $(s_{m},a_{m})$ will be fed into $\hat{T}_{\theta}$ to obtain the prediction result; if $m-1$ steps are chosen, $(s_{2},a_{2:m})$ will be fed into $\hat{T}_{\theta}$, and so forth. The right part of Figure [1](#figure-1) illustrates the aforementioned process based on random backtracking. Further prediction for $s_{m+2}$ should backtrack to at most $(s_{2},a_{2:m+1})$, as $\hat{T}_{\theta}$ is trained with a maximum sequence length of $m$ steps. Similarly, subsequent state predictions can also backtrack at most $m$ steps. The backtracked state, one part of the attribution for next state prediction, is located several steps ahead in expectation. Thus, ADM reduces the actual bootstrapping count of a rolled-out trajectory. The complete $H$-step roll-out process in ADM is described in Algorithm [1](#algorithm-1).

<a id="algorithm-1"></a>

**Algorithm 1 Roll-out in ADM: ADM-Roll($\hat{T}_{\theta}$, $\pi_{\phi}$, $H$, $m$, $(s_{1},a_{1},s_{2},a_{2},\cdots,s_{m-1},a_{m-1},s_{m})$)**

```text

1:  for $\tau=0$ to $H-1$ do

2:     Sample $a_{m+\tau}\sim\pi_{\phi}(\cdot|s_{m+\tau})$

3:     Randomly sample an integer $k$ from $[1,m]$ uniformly

4:     Roll out the next step in ADM via $(s_{m+\tau+1},r_{m+\tau+1})\sim\hat{T}_{\theta}(\cdot|s_{m+\tau+1-k},a_{m+\tau
+1-k:m+\tau})$

5:  end for

6:  return  $(s_{m},a_{m},r_{m+1},s_{m+1},\cdots,s_{m+H-1},a_{m+H-1},r_{m+H},s_{m+H})$
```

Similar to existing MBRL algorithms, policy roll-out in ADM can generate a large number of fake samples for policy updates. We refer to the new dyna-style ADM-based policy optimization framework as ADMPO (ADM-based Policy Optimization). Any policy optimization algorithm can be plugged into this framework. In the subsequent subsections, we will introduce two foundational algorithms, AMRPO-ON and ADMPO-OFF, for online and offline settings, respectively.

###### Definition 3.1 (Any-step Dynamics Model) .

<a id="section-3-2"></a>

### 3.2 ADMPO-ON: ADM for Policy Optimization in Online Setting

In the online setting, the agent interacts with the real environment while simultaneously optimizing the policy. Like MBPO [21], ADMPO-ON can be divided into two alternating stages, namely updating the dynamics model with continuously collected samples and utilizing samples generated through model roll-outs additionally for policy optimization. ADMPO-ON replaces the ensemble dynamics model in MBPO framework with ADM. It trains ADM with the optimization objective shown in Equation ([2](https://arxiv.org/html/2405.17031v1#S3.E2)) and generates a large number of fake samples using the roll-out method depicted in Algorithm [1](#algorithm-1). The detailed pseudo-code is provided by Algorithm [2](#algorithm-2) in Appendix [C.1](https://arxiv.org/html/2405.17031v1#A3.SS1).

During roll-outs, ADM randomly selects a backtracking length at each step and attributes the states to be predicted to variable-length plans. While backtracking $k$ steps, we view the sampling process $(\hat{s}_{t+1},\hat{r}_{t+1})\sim\hat{T}_{\theta}(\cdot|s_{t-k+1},a_{t-k+1:t})$ as $(\hat{s}_{t+1},\hat{r}_{t+1})=\mu_{\theta}(s_{t-k+1},a_{t-k+1:t})+\eta_{t+1}$ with $\eta_{t+1}\sim\mathcal{N}(0,\Sigma_{\theta}(s_{t-k+1},a_{t-k+1:t}))$, where $\mu_{\theta}$ is the deterministic dynamics function and $\Sigma_{\theta}$ is the standard deviation function used to construct the noise distribution with zero mean. In expectation, the target value of $Q(s_{t},a_{t})$ is estimated as

$$
\mathbb{E}_{(s_{t-m+1:t-1},a_{t-m+1:t-1})\sim\Gamma^{m-1}_{\pi}(\cdot|s_{t})} \left[\frac{1}{m}\sum_{k=1}^{m}\mathbb{E}_{(\hat{s}_{t+1},\hat{r}_{t+1})\sim \hat{T}_{\theta}(\cdot|s_{t-k+1},a_{t-k+1:t})}\left[y(\hat{s}_{t+1},\hat{r}_{t +1})\right]\right],(3)
$$

where $y(\hat{s}_{t+1},\hat{r}_{t+1})=\hat{r}_{t+1}+\gamma\mathbb{E}_{a\sim\pi(\cdot| \hat{s}_{t+1})}\left[Q(\hat{s}_{t+1},a)\right]$. Data generated via roll-outs in our ADM can be viewed as an implicit augmentation. The augmentation stems from two sources: (i) variation of the backtracking-length while applying the learned ADM to predict the next state, and (ii) the noise introduced by the distribution $\mathcal{N}(0,\Sigma_{\theta}(s_{t-k+1},a_{t-k+1:t}))$ at each backtracking-length $k$. According to [52], variations of state predictions can effectively implicitly regularize the local Lipschitz condition of the Q network around regions where the model prediction is uncertain, thereby regulating the value-aware model error [14].

<a id="section-3-3"></a>

### 3.3 ADMPO-OFF: ADM for Policy Optimization in Offline Setting

In the offline setting, due to limitations of the behavior policy corresponding to the dataset, the learned ADM can only cover some regions of the state-action space. Beyond these safe regions lie the risky regions where the model is uncertain and unable to be fixed since online exploration is inaccessible to the agent. To prevent policy optimization collapse, exploitation of the learned model needs to be focused within the safe regions. Simultaneously, efforts should be made to explore beyond the boundaries of the risky regions to discover samples conducive to a better policy than the behavior policy. Achieving such a balance between conservatism and generalization often requires measuring model uncertainty. Based on ADM, next we will introduce a new uncertainty quantification method.

In our ADM, states predicted using different backtracking lengths exhibit discrepancies. Intuitively, these discrepancies are closely related to the data distribution. When the agent is in safe regions, the discrepancies are small. As the agent gradually explores towards risky regions, the discrepancies tend to increase. The difference among probabilistic predictions $\hat{T}_{\theta}(\cdot|s_{t-k+1},a_{t-k+1:t})$ obtained with different backtracking $k$ serve as a natural measure of model uncertainty, which can be quantified using variance (or standard deviation), as defined by Definition [3.2](https://arxiv.org/html/2405.17031v1#S3.Thmtheorem2).

###### Definition 3.2 (ADM-Uncertainty Quantifier) .

For any maximum backtracking length $m$ and the corresponding learned ADM $\hat{T}_{\theta}$, the uncertainty of $\hat{T}_{\theta}$ at $(s_{t},a_{t})$ is quantified as

$$
\displaystyle\mathcal{U}^{\mathrm{ADM}}(s_{t},a_{t})= \displaystyle\mathbb{E}_{\Gamma^{m-1}_{\pi}(\cdot|s_{t})}\left[\left\|\mathrm{Var}_{k\sim\mathrm{Uniform}(m),\hat{s}_{t+1}\sim\hat{T}_{\theta}(\cdot|s_{t-k+ 1},a_{t-k+1:t})}\left[\hat{s}_{t+1}\right]\right\|_{1}\right] (4) \displaystyle= \displaystyle\mathbb{E}_{\Gamma^{m-1}_{\pi}(\cdot|s_{t})}\left[\left\|\frac{1} {m}\sum_{k=1}^{m}\left((\Sigma_{\theta}^{k})^{2}+(\mu_{\theta}^{k})^{2}\right) -(\bar{\mu})^{2}\right\|_{1}\right]
$$

for any $s_{t}\in\mathcal{S}$ and $a_{t}\in\mathcal{A}$, where $\Sigma_{\theta}^{k}=\Sigma_{\theta}(s_{t-k+1},a_{t-k+1:t})$, $\mu_{\theta}^{k}=\mu_{\theta}(s_{t-k+1},a_{t-k+1:t})$ for convenience, and $\bar{\mu}=\frac{1}{m}\sum_{k=1}^{m}\mu_{\theta}^{k}$.

This uncertainty term corresponds to a combination of epistemic and aleatoric model uncertainty with a similar form to the ensemble standard deviation [32, 29]. However, the source of diversity has shifted from ensemble to variable backtracking lengths. Since estimating the approximation error via epistemic or aleatoric uncertainty has been applied in many works [51, 4, 44, 32], we assume that our ADM uncertainty ([4](https://arxiv.org/html/2405.17031v1#S3.E4)) is an admissible error estimator [51], as described in Assumption [3.3](https://arxiv.org/html/2405.17031v1#S3.Thmtheorem3).

###### Assumption 3.3 (Admissible Error Estimator) .

Assume that there exists a positive $b\in\mathbb{R}^{+}$ such that the following inequality ([5](https://arxiv.org/html/2405.17031v1#S3.E5)) holds for any maximum backtracking length $m$ and any $s_{t}\in\mathcal{S}$, $a_{t}\in\mathcal{A}$.

$$
D_{\mathrm{TV}}(\bar{T}_{\theta,m}(\cdot|s_{t},a_{t}),T(\cdot|s_{t},a_{t})) \leq b\cdot\mathcal{U}^{\mathrm{ADM}}(s_{t},a_{t}),(5)
$$

where $\bar{T}_{\theta,m}$ is the overall conditioned distribution coming from

$$
\bar{T}_{\theta,m}(\cdot|s_{t},a_{t})=\frac{1}{m}\sum_{k=1}^{m}\left[\sum_{\begin{subarray}{c}s_{t-k+1}\\ a_{t-k+1:t-1}\end{subarray}}\Gamma^{k-1}_{\pi}(s_{t-k+1},a_{t-k+1:t-1}|s_{t}) \hat{T}_{\theta}(\cdot|s_{t-k+1},a_{t-k+1:t}))\right].(6)
$$

Under Assumption [3.3](https://arxiv.org/html/2405.17031v1#S3.Thmtheorem3) and the $\xi$-uncertainty quantifier definition (see Appendix [A](#appendix-a) for details) proposed by PEVI [23], we present the following theorem, demonstrating that $\mathcal{U}^{\mathrm{ADM}}$ can serve as a $\xi$-uncertainty quantifier to bound the Bellman error.

###### Theorem 3.4 .

$\beta\cdot\mathcal{U}^{\mathrm{ADM}}$ is a valid $\xi$-uncertainty quantifier, with $\beta=b\frac{\gamma r_{\mathrm{max}}}{1-\gamma}$. Specifically,

$$
\left|\hat{\mathcal{T}}^{\pi}Q(s_{t},a_{t})-\mathcal{T}^{\pi}Q(s_{t},a_{t}) \right|\leq\beta\cdot\mathcal{U}^{\mathrm{ADM}}(s_{t},a_{t}),(7)
$$

where $\hat{\mathcal{T}}^{\pi}$ is the proxy Bellman operator induced by ADM to estimate the true Bellman operator $\mathcal{T}^{\pi}$.

###### Proof.

See Appendix [B](#appendix-b). ∎

According to the suboptimality theorem (see Appendix [A](#appendix-a) for details) presented by PEVI [23], the policy $\hat{\pi}$ derived via pessimistic value iteration, which incorporates any $\xi$-uncertainty quantifier as a penalty term into the value iteration process [46], has a bounded optimality gap to the optimal policy $\pi^{*}$. The optimality gap is dominated by the Bellman error and the uncertainty quantification. Intuitively, the Bellman error is usually small in safe regions where the dynamics model has been trained with rich data and tends to yield high consistency under different backtracking lengths, while large errors often appear in risky regions where data is scarce and the predictions via backtracking different lengths become inconsistent. The penalization prevents the policy from taking actions leading it to risky regions, otherwise the model will induce inaccurate value estimations on these actions. Thus, we can penalize the Bellman operator to obtain a pessimistic value estimation by

$$
\hat{\mathcal{T}}^{\mathrm{ADM}}Q(s_{t},a_{t}):=\hat{\mathcal{T}}^{\pi}Q(s_{t} ,a_{t})-\beta\cdot\mathcal{U}^{\mathrm{ADM}}(s_{t},a_{t}).(8)
$$

We expect the penalty term $\beta\cdot\mathcal{U}^{\mathrm{ADM}}(s_{t},a_{t})$ to be as small as possible thereby constraining the optimality gap. While our Assumption [3.3](https://arxiv.org/html/2405.17031v1#S3.Thmtheorem3) lacks theoretical guarantees and the tightness of the bound in Theorem [3.4](https://arxiv.org/html/2405.17031v1#S3.Thmtheorem4) is unclear, we have provided sufficient evidence in Section [4.3.3](https://arxiv.org/html/2405.17031v1#S4.SS3.SSS3) that our uncertainty quantification effectively estimates the model error.

Overall, ADMPO-OFF is the offline version of ADMPO-ON, which introduces a penalized Bellman operator ([8](https://arxiv.org/html/2405.17031v1#S3.E8)) into the policy optimization process of ADMPO-ON, following the algorithmic framework of MOPO [51]. The detailed pseudo-code is provided by Algorithm [3](#algorithm-3) in Appendix [C.2](https://arxiv.org/html/2405.17031v1#A3.SS2).

###### Definition 3.2 (ADM-Uncertainty Quantifier) .

###### Assumption 3.3 (Admissible Error Estimator) .

###### Theorem 3.4 .

###### Proof.

<a id="section-4"></a>

## 4 Experiments

In this section, we conduct several experiments to answer: (1) Does ADM roll-out samples with less compounding error than the ensemble dynamics model? (2) How well does ADMPO-ON perform in the online setting? (3) How well does ADMPO-OFF perform in the offline setting? Does ADM quantify the model uncertainty better than the ensemble dynamics model?

<a id="section-4-1"></a>

### 4.1 Dynamics Model Evaluation

An essential metric for evaluating dynamics model quality is the compounding error, which increases with the roll-out length. We selected four D4RL [16] datasets, hopper-medium-v2, hopper-medium-replay-v2, walker2d-medium-v2, and walker2d-medium-replay-v2, to compare the compounding error between ADM and the commonly used ensemble dynamics model. To eliminate the influence of the RNN structure, we also compare the bootstrapping RNN dynamic model, which shares the same structure as ADM but predicts $s_{t+m+1}$ using the historical state-action sequence $(s_{t},a_{t},\cdots,s_{t+m},a_{t+m})$ as input. Figure [2](#figure-2) shows the growth curves of the compounding error as the roll-out length increases. We observe that the curves of ADM remain close to zero, while the other two models exhibit exponential growth as the roll-out length exceeds a certain threshold. This phenomenon suggests ADM can improve predictions for future states due to its any-step backtracking mechanism during model roll-outs.

<a id="figure-2"></a>

![model_exp](images/model_exp.png)

> Figure 2: Comparison among ADM, ensemble dynamics model, and bootstrapping RNN dynamics model, in terms of the growth curve of the compounding error as roll-out length increases, after offline learning. The overflow value is regarded as the maximum value of float32.

<a id="figure-3"></a>

![mujoco_performance](images/mujoco_performance.png)

> Figure 3: Online learning curves of ADMPO-ON (red) and other five baselines on four MuJoCo-v3 tasks. The blue dashed lines indicate the asymptotic performance of SAC for reference. The solid lines indicate the mean while the shaded areas indicate the standard error over five different seeds.

<a id="section-4-2"></a>

### 4.2 Evaluation in Online Setting

We evaluate ADMPO-ON on four difficult MuJoCo continuous control tasks [47], including Hopper, Walker2d, Ant, and Humanoid. All the tasks adopt version v3 and follow the default settings. Four model-based methods and one model-free method are selected as our baselines. These include SAC [20], whcih is the state-of-the-art model-free RL algorithm; STEVE [6], which incorporates an ensemble into the model-based value expansion; MBPO [21], which updates the policy with a mixture of real environmental samples and branched roll-out data; BMPO [28], which builds upon MBPO and replaces the dynamics model with a bidirectional one; and DDPPO [30], which adopts a two-model-based learning method to control the prediction error and the gradient error.

Figure [3](#figure-3) shows learning curves of ADMPO-ON and other five baselines, along with SAC’s asymptotic performance. ADMPO-ON achieves competitive performance after fewer environmental steps than the baselines. Taking the most difficult Humanoid as an example, ADMPO-ON has achieved 100% of SAC convergence performance (about 6000) after 150k steps, while DDPPO needs about 200k steps, and the other four methods can’t get close to the blue dashed line even at step 300k. ADMPO-ON performs about 1.33x faster than DDPPO and dominates other baselines in terms of learning efficiency on the Humanoid task. After training, ADMPO-ON can achieve a final performance close to the asymptotic performance of SAC on all these four MuJoCo tasks. These results demonstrate that ADMPO-ON has both high sample efficiency and competitive performance. Further study on why ADMPO-ON performs well in the online setting can be found in Appendix [E.1](https://arxiv.org/html/2405.17031v1#A5.SS1).

<a id="section-4-3"></a>

### 4.3 Evaluation in Offline Setting

#### 4.3.1 D4RL Benchmark Results

We compare ADMPO-OFF with four model-free methods: BC (behavioral cloning), which simply imitates the behavior policy of the dataset; CQL [27], which equally penalized the Q values on out-of-the-distribution state-action pairs; TD3+BC [17], which simply incorporates a BC term into the policy optimization objective of TD3 [19]; and EDAC [1], which quantifies the Q uncertainty via ensemble; as well as five model-based methods: MOPO [51], which adds the uncertainty of the model prediction as a penalization term to the reward function; COMBO [50], which introduces the penalty function of CQL into the model-based framework; RAMBO [40], which adversarially trains the dynamics model and the policy; CBOP [22], which adopts the variance of values under an ensemble of dynamics models to estimate the Q value conservatively under MVE [15] regime; and MOBILE [44], which proposes Model-Bellman inconsistency to estimate the Bellman error.

Table [1](#table-1) reports the results on twelve D4RL [16] MuJoCo datasets (v2 version). The normalized score for each dataset is obtained via online evaluation after offline learning. The source of the reported performance in provided in Appendix [D.4](https://arxiv.org/html/2405.17031v1#A4.SS4). We observe that ADMPO-OFF outperforms the other nine baselines in most tasks and achieves the highest average score.

<a id="table-1"></a>

> Table 1: Normalized scores after offline learning on D4RL MuJoCo tasks, averaged over five seeds.

| Task Name                 | BC   | CQL   | TD3+BC | EDAC  | MOPO  | COMBO | RAMBO | CBOP  | MOBILE | ADMPO-OFF (ours) |
| ------------------------- | ---- | ----- | ------ | ----- | ----- | ----- | ----- | ----- | ------ | ---------------- |
| hopper-random             | 3.7  | 5.3   | 8.5    | 25.3  | 31.7  | 17.9  | 25.4  | 31.4  | 31.9   | 32.7$\pm$0.2     |
| halfcheetah-random        | 2.2  | 31.3  | 11.0   | 28.4  | 38.5  | 38.8  | 39.5  | 32.8  | 39.3   | 45.4$\pm$2.8     |
| walker2d-random           | 1.3  | 5.4   | 1.6    | 16.6  | 7.4   | 7.0   | 0.0   | 17.8  | 17.9   | 22.2$\pm$0.2     |
| hopper-medium             | 54.1 | 61.9  | 59.3   | 101.6 | 62.8  | 97.2  | 87.0  | 102.6 | 106.6  | 107.4$\pm$0.6    |
| halfcheetah-medium        | 43.2 | 46.9  | 48.3   | 65.9  | 73.0  | 54.2  | 77.9  | 74.3  | 74.6   | 72.2$\pm$0.6     |
| walker2d-medium           | 70.9 | 79.5  | 83.7   | 92.5  | 84.1  | 81.9  | 84.9  | 95.5  | 87.7   | 93.2$\pm$1.1     |
| hopper-medium-replay      | 16.6 | 86.3  | 60.9   | 101.0 | 103.5 | 89.5  | 99.5  | 104.3 | 103.9  | 104.4$\pm$0.4    |
| halfcheetah-medium-replay | 37.6 | 45.3  | 44.6   | 61.3  | 72.1  | 55.1  | 68.7  | 66.4  | 71.7   | 67.6$\pm$3.4     |
| walker2d-medium-replay    | 20.3 | 76.8  | 81.8   | 87.1  | 85.6  | 56.0  | 89.2  | 92.7  | 89.9   | 95.6$\pm$2.1     |
| hopper-medium-expert      | 53.9 | 96.9  | 98.0   | 110.7 | 81.6  | 111.1 | 88.2  | 111.6 | 112.6  | 112.7$\pm$0.3    |
| halfcheetah-medium-expert | 44.0 | 95.0  | 90.7   | 106.3 | 90.8  | 90.0  | 95.4  | 105.4 | 108.2  | 103.7$\pm$0.2    |
| walker2d-medium-expert    | 90.1 | 109.1 | 110.1  | 114.7 | 112.9 | 103.3 | 56.7  | 117.2 | 115.2  | 114.9$\pm$0.3    |
| Average                   | 36.5 | 61.6  | 58.2   | 76.0  | 70.3  | 66.8  | 67.7  | 79.3  | 80.0   | 81.0             |

#### 4.3.2 NeoRL Benchmark Results

NeoRL [39], is an offline RL benchmark that collects the data in a manner more conservative and closer to real-world data-collection scenarios. We focus on nine datasets collected using policies of three different qualities (low, medium, and high) in three environments Hopper-v3, HalfCheetah-v3, and Walker2d-v3, respectively. In our evaluation, each dataset contains 1000 trajectories.

We compare our ADMPO-OFF with six baselines, including BC, CQL, TD3+BC, EDAC, MOPO, and MOBILE. Table [2](#table-2) presents the normalized scores of these methods. Due to the narrow and limited coverage of the NeoRL data, all the baselines experience a decline in performance. In contrast, our ADMPO-OFF maintains a relatively high-level average score, still achieving superior performance in most tasks. This remarkable out-performance indicates the potential of our algorithm in more challenging real-world tasks.

<a id="table-2"></a>

> Table 2: Normalized scores after offline learning on NeoRL tasks, averaged over five seeds.

| Task Name                | BC   | CQL  | TD3+BC | EDAC | MOPO | MOBILE | ADMPO-OFF (ours) |
| ------------------------ | ---- | ---- | ------ | ---- | ---- | ------ | ---------------- |
| neorl-hopper-low         | 15.1 | 16.0 | 15.8   | 18.3 | 6.2  | 17.4   | 22.3$\pm$0.1     |
| neorl-halfcheetah-low    | 29.1 | 38.2 | 30.0   | 31.3 | 40.1 | 54.7   | 52.8$\pm$1.2     |
| neorl-walker2d-low       | 28.5 | 44.7 | 43.0   | 40.2 | 11.6 | 37.6   | 55.9$\pm$3.8     |
| neorl-hopper-medium      | 51.3 | 64.5 | 70.3   | 44.9 | 1.0  | 51.1   | 51.5$\pm$5.0     |
| neorl-halfcheetah-medium | 49.0 | 54.6 | 52.3   | 54.9 | 62.3 | 77.8   | 69.3$\pm$1.7     |
| neorl-walker2d-medium    | 48.7 | 57.3 | 58.5   | 57.6 | 39.9 | 62.2   | 70.1$\pm$2.4     |
| neorl-hopper-high        | 43.1 | 76.6 | 75.3   | 52.5 | 11.5 | 87.8   | 87.6$\pm$4.9     |
| neorl-halfcheetah-high   | 71.3 | 77.4 | 75.3   | 81.4 | 65.9 | 83.0   | 84.0$\pm$0.8     |
| neorl-walker2d-high      | 72.6 | 75.3 | 69.6   | 75.5 | 18.0 | 74.9   | 82.2$\pm$1.9     |
| Average                  | 45.4 | 56.1 | 54.5   | 50.7 | 28.5 | 60.7   | 64.0             |

#### 4.3.3 Uncertainty Quantification

In our analysis, we sample a lot of state-action pairs in the learned ADM and the ensemble dynamics model respectively. These samples are obtained by model roll-out with three types of policy: random action selection, the learned policy after offline training, and the behavior policy of the dataset. Subsequently, we measure their model uncertainty and model error. The resulting scatter plots on two D4RL tasks, hopper-medium-replay-v2 and walker2d-medium-replay-v2, are illustrated in Figure [4](#figure-4). We observe that our ADM provides a better quantification for the model uncertainty. On the one hand, points sampled in ADM with greater model errors tend to exhibit greater quantified model uncertainty. The correlation coefficient of 0.98, observed across both tasks, surpasses that of the ensemble dynamics model. On the other hand, ADM can distinguish the samples from different policy better than the ensemble model. Samples generated from random actions deviate from the dataset distribution, whose uncertainty should be maximum in expectation. Conversely, when the learned policy is optimized within the safe regions covered by the dataset, model uncertainty is expected to be minimal. The experimental plots of ADM illustrate this phenomenon more clearly.

![uquant1](images/uquant1.png)

> (a) hopper-medium-replay-v2

<a id="section-5"></a>

## 5 Related Work

This work is related to online and offline dyna-style MBRL [45].

<a id="section-5-1"></a>

### 5.1 Online Model-based Reinforcement Learning

In the online setting, MBRL algorithms aim to accelerate value estimation or policy optimization with model roll-out data. MVE [15] enhances Q-value target estimation by allowing short-term imagination to a fixed depth using the dynamics model. STEVE [6] builds upon MVE by incorporating an ensemble into the value expansion to better estimate the Q value. SLBO [35] directly utilizes TRPO [41] to optimize the policy with synthetic data generated by rolling out to the end of trajectories in the dynamics model. MBPO [21] proposes a branched roll-out scheme to truncate unreliable samples, thereby reducing the influence of compounding error [49], and employs SAC [20] to update the policy with a mixture of real-world data and model-generated data.

Recent work improves MBRL performance mainly from two perspectives. One focuses on learning a better dynamics model, such as bidirectional models [28], adversarial models [9, 5], causal models [53], and multi-step models [2, 3, 7]. The other pursues a better utilization of the learned model, enhancing the reliability of model-generated samples [38] or applying model-based multi-step planning techniques [11, 12, 24, 37, 42, 31].

<a id="section-5-2"></a>

### 5.2 Offline Model-based Reinforcement Learning

Although some model-free RL algorithms [26, 18, 17, 27, 1, 4] have made significant contributions to offline RL research, MBRL algorithms appear to be more promising for the offline setting since they can utilize the dynamics model to extend the dataset and largely improve the data efficiency.

The core issue of offline MBRL lies in how to effectively leverage the model. MOPO [51] and MOReL [25] add the uncertainty of the model prediction as a penalization term to the original reward function to achieve a pessimistic value estimation. MOBILE [44] improves the uncertainty quantification by introducing Model-Bellman inconsistency into the offline model-based framework. COMBO [50] applies CQL [27] to force the Q value to be small on model-generated out-of-distribution samples. RAMBO [40] achieves conservatism by adversarial model learning for value minimization while keeping fitting the transition function. CBOP [22] introduces adaptive weighting of short-horizon roll-out into MVE [15] technique and adopts the variance of values under an ensemble of dynamics models to estimate the Q value conservatively. MOREC [33] designs a reward-consistent dynamics model using an adversarial discriminator to let the model-generated samples be more reliable.

<a id="section-6"></a>

## 6 Conclusion

In this work, we propose a new method for environment model learning and utilization, namely Any-step Dynamics Model (ADM). ADM is applicable in both online and offline MBRL frameworks and yields two algorithms, ADMPO-ON and ADMPO-OFF, respectively. Several analysis and experiments show that ADM outperforms the ensemble dynamics model applied in previous MBRL approaches widely. The only problem is that RNN may consume more resources during the training process. We believe ADM has powerful potential beyond the capabilities demonstrated in this paper. In the future, we will explore the scalability of ADM in non-Markovian visual RL scenarios, considering both online and offline settings.
