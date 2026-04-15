<a id="appendix-a"></a>

## Appendix A Additional Introduction to Pessimistic Value Iteration (PEVI)

Pessimistic Value Iteration (PEVI) [23] is a meta-algorithm for offline RL settings. It constructs an estimated Bellman operator $\hat{\mathcal{T}}^{\pi}$ based on the given dataset $\mathcal{D}_{\mathrm{env}}$ to approximate the true Bellman operator $\mathcal{T}^{\pi}$ that satisfies

$$
\mathcal{T}^{\pi}V_{h+1}(s_{h},a_{h})=\mathbb{E}_{(s_{h+1},r_{h+1})\sim T( \cdot|s_{h},a_{h})}\left[r_{h+1}+V_{h+1}(s_{h+1})\right],(9)
$$

where $h$ is the step index less than the horizon $\mathcal{H}$. Then the state-action value function is updated with

$$
Q_{h}(s_{h},a_{h})\leftarrow\hat{\mathcal{T}}^{\pi}V_{h+1}(s_{h},a_{h})- \Lambda_{h}(s_{h},a_{h})(10)
$$

for each $(s_{h},a_{h})$, where $\Lambda_{h}$ is the penalty function that guarantees the conservatism of the learned policy. Especially, $\Lambda_{h}$ should be a $\xi$-uncertainty quantifier as follows.

###### Definition A.1 ( $\xi$ -Uncertainty Quantifier (proposed by [ 23 ] )) .

The set of penalization $\{\Lambda_{h}\}_{h\in[\mathcal{H}]}$ forms a $\xi$-uncertainty quantifier if

$$
\left|\hat{\mathcal{T}}^{\pi}V_{h+1}(s_{h},a_{h})-\mathcal{T}^{\pi}V_{h+1}(s_{h},a_{h})\right|\leq\Lambda_{h}(s_{h},a_{h})(11)
$$

holds with probability at least $1-\xi$ for all $(s_{h},a_{h})\in\mathcal{S}\times\mathcal{A}$.

The following theorem characterizes the suboptimality of PEVI.

###### Theorem A.2 (Suboptimality of PEVI (proposed by [ 23 ] )) .

Suppose $\{\Lambda_{h}\}_{h=1}^{\mathcal{H}}$ in PEVI is a set of $\xi$-uncertainty quantifier. Then the derived policy $\hat{\pi}$ satisfies

$$
\left|V_{1}^{\pi^{*}}(s_{1})-V_{1}^{\hat{\pi}}(s_{1})\right|\leq 2\sum_{h=1}^{\mathcal{H}}\mathbb{E}_{\rho^{\pi^{*}}}\left[\Lambda_{h}(s_{h},a_{h})\right](12)
$$

with probability at least $1-\xi$ for all starting $s_{1}\in\mathcal{S}$. Here $\mathbb{E}_{\rho^{\pi^{*}}}$ is with respect to the trajectory induced by the optimal policy $\pi^{*}$ in the underlying MDP given the fixed function $\Lambda_{h}$.

###### Proof.

See PEVI [23] for detailed proof. ∎

###### Definition A.1 ( ξ 𝜉 \xi italic_ξ -Uncertainty Quantifier (proposed by [ 23 ] )) .

###### Theorem A.2 (Suboptimality of PEVI (proposed by [ 23 ] )) .

###### Proof.

<a id="appendix-b"></a>

## Appendix B Theoretical Results

###### Theorem B.1 .

$\beta\cdot\mathcal{U}^{\mathrm{ADM}}$ is a valid $\xi$-uncertainty quantifier, with $\beta=b\frac{\gamma r_{\mathrm{max}}}{1-\gamma}$. Specifically,

$$
\left|\hat{\mathcal{T}}^{\pi}Q(s_{t},a_{t})-\mathcal{T}^{\pi}Q(s_{t},a_{t}) \right|\leq\beta\cdot\mathcal{U}^{\mathrm{ADM}}(s_{t},a_{t}),(13)
$$

where $\hat{\mathcal{T}}^{\pi}$ is the proxy Bellman operator induced by ADM to estimate the true Bellman operator $\mathcal{T}^{\pi}$.

###### Proof.

First, we define $y(\hat{s}_{t+1},\hat{r}_{t+1})=\hat{r}_{t+1}+\gamma\mathbb{E}_{a\sim\pi(\cdot| \hat{s}_{t+1})}\left[Q(\hat{s}_{t+1},a)\right]$ and expand these two Bellman operator to

$$
\displaystyle\hat{\mathcal{T}}^{\pi}Q(s_{t},a_{t}) (14) \displaystyle= \displaystyle\mathbb{E}_{(s_{t-m+1:t-1},a_{t-m+1:t-1})\sim\Gamma^{m-1}_{\pi}( \cdot|s_{t})}\left[\frac{1}{m}\sum_{k=1}^{m}\mathbb{E}_{(\hat{s}_{t+1},\hat{r} _{t+1})\sim\hat{T}_{\theta}(\cdot|s_{t-k+1},a_{t-k+1:t})}\left[y(\hat{s}_{t+1} ,\hat{r}_{t+1})\right]\right] \displaystyle= \displaystyle\sum_{\begin{subarray}{c}s_{t-m+1}\\ a_{t-m+1:t-1}\end{subarray}}\Gamma^{m-1}_{\pi}(s_{t-m+1},a_{t-m+1:t-1}|s_{t}) \left[\frac{1}{m}\sum_{k=1}^{m}\sum_{\begin{subarray}{c}\hat{s}_{t+1}\\ \hat{r}_{t+1}\end{subarray}}\hat{T}_{\theta}(\cdot|s_{t-k+1},a_{t-k+1:t})y( \hat{s}_{t+1},\hat{r}_{t+1})\right] \displaystyle= \displaystyle\frac{1}{m}\sum_{k=1}^{m}\left[\sum_{\begin{subarray}{c}s_{t-k+1} \\ a_{t-k+1:t-1}\end{subarray}}\Gamma^{k-1}_{\pi}(s_{t-k+1},a_{t-k+1:t-1}|s_{t}) \sum_{\begin{subarray}{c}\hat{s}_{t+1}\\ \hat{r}_{t+1}\end{subarray}}\hat{T}_{\theta}(\cdot|s_{t-k+1},a_{t-k+1:t}))y( \hat{s}_{t+1},\hat{r}_{t+1})\right] \displaystyle= \displaystyle\sum_{\hat{s}_{t+1},\hat{r}_{t+1}}\bar{T}_{\theta,m}(\hat{s}_{t+1},\hat{r}_{t+1}|s_{t},a_{t})y(\hat{s}_{t+1},\hat{r}_{t+1}),
$$

and

$$
\displaystyle\mathcal{T}^{\pi}Q(s_{t},a_{t}) (15) \displaystyle= \displaystyle\mathbb{E}_{\hat{s}_{t+1},\hat{r}_{t+1}\sim T(\cdot|s_{t},a_{t})} \left[y(\hat{s}_{t+1},\hat{r}_{t+1})\right] \displaystyle= \displaystyle\sum_{\hat{s}_{t+1},\hat{r}_{t+1}}T(\hat{s}_{t+1},\hat{r}_{t+1}|s _{t},a_{t})y(\hat{s}_{t+1},\hat{r}_{t+1}).
$$

Then, we can obtain

$$
\displaystyle\left|\hat{\mathcal{T}}^{\pi}Q(s_{t},a_{t})-\mathcal{T}^{\pi}Q(s_ {t},a_{t})\right| (16) \displaystyle= \displaystyle\sum_{\hat{s}_{t+1},\hat{r}_{t+1}}\left|\bar{T}_{\theta,m}(\hat{s}_{t+1},\hat{r}_{t+1}|s_{t},a_{t})-T(\hat{s}_{t+1},\hat{r}_{t+1}|s_{t},a_{t}) \right|\cdot\left|y(\hat{s}_{t+1},\hat{r}_{t+1})\right| \displaystyle= \displaystyle\gamma\sum_{\hat{s}_{t+1},\hat{r}_{t+1}}\left|\bar{T}_{\theta,m}( \hat{s}_{t+1},\hat{r}_{t+1}|s_{t},a_{t})-T(\hat{s}_{t+1},\hat{r}_{t+1}|s_{t},a _{t})\right|\cdot\left|\mathbb{E}_{a\sim\pi(\cdot|\hat{s}_{t+1})}\left[Q(\hat{s}_{t+1},a)\right]\right| \displaystyle\leq \displaystyle\dfrac{\gamma r_{\mathrm{max}}}{1-\gamma}\sum_{\hat{s}_{t+1},\hat {r}_{t+1}}\left|\bar{T}_{\theta,m}(\hat{s}_{t+1},\hat{r}_{t+1}|s_{t},a_{t})-T( \hat{s}_{t+1},\hat{r}_{t+1}|s_{t},a_{t})\right| \displaystyle= \displaystyle\dfrac{\gamma r_{\mathrm{max}}}{1-\gamma}D_{\mathrm{TV}}(\bar{T}_ {\theta,m}(\cdot|s_{t},a_{t}),T(\cdot|s_{t},a_{t})) \displaystyle\leq \displaystyle b\dfrac{\gamma r_{\mathrm{max}}}{1-\gamma}\mathcal{U}^{\mathrm{ADM}}(s_{t},a_{t}).
$$

Thus, let $\beta=b\frac{\gamma r_{\mathrm{max}}}{1-\gamma}$, we can say that $\beta\cdot\mathcal{U}^{\mathrm{ADM}}$ is a valid $\xi$-uncertainty quantifier, as defined by Definition [A.1](https://arxiv.org/html/2405.17031v1#A1.Thmtheorem1). ∎

###### Theorem B.1 .

###### Proof.

<a id="appendix-c"></a>

## Appendix C Implementation Details

### C.1 ADMPO-ON

Our ADMPO-ON algorithm follows the framework of MBPO [21], as shown in Algorithm [2](#algorithm-2). The only difference between ADMPO-ON and MBPO lies in the way the dynamics model is trained and utilized, as indicated by the blue-highlighted parts in the pseudo-code.

<a id="algorithm-2"></a>

**Algorithm 2 ADMPO-ON**

```text

1:  Explore for $U$ environmental steps and add data to $\mathcal{D}_{\mathrm{env}}$

2:  for $N$ epochs do

3:     Train ADM $\hat{T}_{\theta}$ on $\mathcal{D}_{\mathrm{env}}$ by maximizing Equation ([2](https://arxiv.org/html/2405.17031v1#S3.E2))

4:     for $t=1$ to $E$ do

5:        Sample action $a_{t}$ according to $\pi_{\phi}(\cdot|s_{t})$

6:        Perform $a_{t}$ in the environment and add the real sample $(s_{t},a_{t},r_{t+1},s_{t+1})$ to $\mathcal{D}_{\mathrm{env}}$

7:        for $M$ model roll-outs do

8:           Sample initial $m$-step state-action sequence $(s_{i:i+m-1},a_{i:i+m-2})$ from $\mathcal{D}_{\mathrm{env}}$

9:           Roll out $H$ steps in $\hat{T}_{\theta}$ via ADM-Roll($\hat{T}_{\theta}$, $\pi_{\phi}$, $H$, $m$, $(s_{i:i+m-1},a_{i:i+m-2})$) and add the model roll-out data to $\mathcal{D}_{\mathrm{model}}$

10:        end for

11:        for $G$ policy updates do

12:           Update current policy $\pi_{\phi}$ using samples from $\mathcal{D}_{\mathrm{env}}\cup\mathcal{D}_{\mathrm{model}}$

13:        end for

14:     end for

15:  end for
```

### C.2 ADMPO-OFF

Our ADMPO-OFF algorithm follows the framework of MOPO [51], as shown in Algorithm [3](#algorithm-3). The only difference between ADMPO-OFF and MOPO lies lies in the way the dynamics model is trained and utilized, as indicated by the blue-highlighted parts in the pseudo-code.

<a id="algorithm-3"></a>

**Algorithm 3 ADMPO-OFF**

```text

1:  Train ADM $\hat{T}_{\theta}$ on $\mathcal{D}_{\mathrm{env}}$ by maximizing Equation ([2](https://arxiv.org/html/2405.17031v1#S3.E2))

2:  for $N$ iterations do

3:     for $M$ model roll-outs do

4:        Sample initial $m$-step state-action sequence $(s_{i:i+m-1},a_{i:i+m-2})$ from $\mathcal{D}_{\mathrm{env}}$

5:        Roll out $H$ steps in $\hat{T}_{\theta}$ via ADM-Roll($\hat{T}_{\theta}$, $\pi_{\phi}$, $H$, $m$, $(s_{i:i+m-1},a_{i:i+m-2})$)

6:        Penalize the reward via $\tilde{r}=r-\beta\mathcal{U}^{\mathrm{ADM}}(s,a)$ for each rolled-out step

7:        Add the penalized model roll-out data to $\mathcal{D}_{\mathrm{model}}$

8:     end for

9:     for $G$ policy updates do

10:        Update current policy $\pi_{\phi}$ using samples from $\mathcal{D}_{\mathrm{env}}\cup\mathcal{D}_{\mathrm{model}}$

11:     end for

12:  end for
```

### C.3 Policy Optimization

The policy optimization method used in our ADMPO-ON and ADMPO-OFF is SAC [20], following MBPO [21] and MOPO [51]. The hyper-parameters about SAC follow its standard implementation, as listed in Table [3](#table-3).

<a id="table-3"></a>

> Table 3: Hyper-parameters of Policy Optimization in ADMPO-ON and ADMPO-OFF.

| Hyper-parameter        | Value             | Description                                        |
| ---------------------- | ----------------- | -------------------------------------------------- |
| $N_{Q}$                | 2                 | the number of critics.                             |
| actor network          | FC(256,256)       | fully connected (FC) layers with ReLU activations. |
| critic network         | FC(256,256)       | fully connected (FC) layers with ReLU activations. |
| $\tau$                 | $5\times 10^{-3}$ | target network smoothing coefficient.              |
| $\gamma$               | 0.99              | discount factor.                                   |
| $lr_{\textrm{actor}}$  | $1\times 10^{-4}$ | learning rate of actor.                            |
| $lr_{\textrm{critic}}$ | $3\times 10^{-4}$ | learning rate of critic.                           |
| optimizer              | Adam              | optimizers of the actor and critics.               |
| batch size             | 256               | batch size for each update.                        |

<a id="appendix-d"></a>

## Appendix D Experimental Details

### D.1 Resource Requirements

All experiments can be completed with just one NVIDIA GeForce RTX 2080 Ti or any other type of GPU with larger graphic memory. There are no additional resource requirements. The time of execution for each task is about 24 hours.

### D.2 ADMPO-ON Settings

The experimental settings of our ADMPO-ON in Section [4.2](#section-4-2) are listed in Table [4](#table-4).

<a id="table-4"></a>

> Table 4: Hyper-parameter settings of ADMPO-ON results presented in Figure [3](#figure-3). $x\rightarrow y$ over $a\rightarrow b$ denotes a thresholded linear increasing schedule, i.e. the length of model rollouts at step $t$ is calculated by $f(t)=\min\left(\max\left(x+\frac{t-a}{b-a}\cdot(y-x),x\right),y\right)$.

| environment                                  | Hopper                | Walker2d              | Ant                  | Humanoid              |
| -------------------------------------------- | --------------------- | --------------------- | -------------------- | --------------------- |
| steps                                        | 50k                   | 200k                  | 300k                 |                       |
| Update-To-Date ratio                         | 20                    |                       |                      |                       |
| maximum backtracking length $\boldsymbol{m}$ | 5                     | 2                     |                      |                       |
| model rollout schedule                       | 1$\rightarrow$15 over | 1$\rightarrow$10 over | 1$\rightarrow$5 over | 1$\rightarrow$10 over |
| 0$\rightarrow$50k                            | 0$\rightarrow$100k    | 10k$\rightarrow$100k  | 10k$\rightarrow$100k |                       |
| target entropy                               | -1                    | -3                    | -4                   | -8                    |

### D.3 ADMPO-OFF Settings

The experimental settings of our ADMPO-OFF in Section [4.3](#section-4-3) are listed in Table [5](#table-5).

<a id="table-5"></a>

> Table 5: Hyper-parameter settings of ADMPO-OFF results presented in Section [4.3](#section-4-3).

| Domain Name               | Task Name        | $\boldsymbol{m}$ | $\boldsymbol{H}$ | $\boldsymbol{\beta}$ |
| ------------------------- | ---------------- | ---------------- | ---------------- | -------------------- |
| D4RL MuJoCo               | hopper-random    | 5                | 50               | 5                    |
| halfcheetah-random        | 2                | 10               | 2.5              |                      |
| walker2d-random           | 2                | 50               | 2.5              |                      |
| hopper-medium             | 5                | 10               | 1                |                      |
| halfcheetah-medium        | 2                | 5                | 2.5              |                      |
| walker2d-medium           | 5                | 10               | 5                |                      |
| hopper-medium-replay      | 5                | 5                | 0.1              |                      |
| halfcheetah-medium-replay | 2                | 5                | 2.5              |                      |
| walker2d-medium-replay    | 5                | 5                | 0.1              |                      |
| hopper-medium-expert      | 2                | 20               | 20               |                      |
| halfcheetah-medium-expert | 2                | 50               | 10               |                      |
| walker2d-medium-expert    | 3                | 2                | 6                |                      |
| NeoRL MuJoCo              | neorl-hopper-low | 5                | 20               | 5                    |
| neorl-halfcheetah-low     | 2                | 20               | 10               |                      |
| neorl-walker2d-low        | 5                | 10               | 2.5              |                      |
| neorl-hopper-medium       | 5                | 20               | 50               |                      |
| neorl-halfcheetah-medium  | 2                | 5                | 20               |                      |
| neorl-Walker2d-medium     | 5                | 10               | 5                |                      |
| neorl-hopper-high         | 5                | 20               | 50               |                      |
| neorl-halfcheetah-high    | 2                | 10               | 50               |                      |
| neorl-walker2d-high       | 5                | 10               | 2.5              |                      |

### D.4 Source of Baselines’ Results

For the evaluation on D4RL [16] benchmarks, the results of the compared baselines come from two sources:

- Retraining on D4RL datasets of v2 version with OfflineRL-Kit [43], for the algorithms whose original papers only report the performance on the v0 version, such as CQL [27], MOPO [51].
- Including the scores in their papers, for the algorithms whose original papers report the performance on the v2 version, such as TD3+BC [17], EDAC [1], RAMBO [40], CBOP [22], and MOBILE [44], or who does not provide source codes, such as COMBO [50].

For the evaluation on NeoRL [39] benchmarks, we report the scores of BC, CQL, and MOPO from the original paper of NeoRL and retrain TD3+BC and EDAC with OfflineRL-Kit [43].

<a id="appendix-e"></a>

## Appendix E Additional Experiments

### E.1 Study on Why ADMPO-ON Performs Well in Online Setting

Value-aware model error [14] is a dependable metric for measuring the learning quality of the dynamics model and the suboptimality of the MBRL algorithm. We conduct a study to verify how well ADMPO-ON regulates the value-aware model error. Without loss of rigor, we only choose MBPO for comparison since most other model-based methods follow the same way of learning and utilizing the ensemble dynamics model. Figure [5](https://arxiv.org/html/2405.17031v1#A5.F5) shows the results on the most difficult Humanoid task. The learned ADM in ADMPO-ON and the ensemble dynamics model in MBPO achieve similar mean squared errors, indicating their similar fitting abilities. However, ADMPO-ON provides greater model roll-out standard deviation over diverse state prediction, forcing the agent to explore more uncertain areas. Therefore, since the variation of state prediction helps smoothen the Q target, the Q network in ADMPO-ON has a significantly smaller Lipschitz constant, and afterwards the value-aware model error, which measures the suboptimality of MBRL becomes smaller. This phenomenon explains why ADMPO-ON performs significantly better then MBPO in Figure [3](#figure-3). For details of the metrics used in this experiment, refer to [52].

<a id="figure-5"></a>

![uquant2](images/uquant2.png)

> Figure 5: Comparison between ADMPO-ON and MBPO on Humanoid, in terms of (a) model mean squared error, (b) model roll-out standard deviation over diverse predictions, (c) estimated Lipschitz constant [52] of Q, and (d) value-aware model error [14]. Results are averaged over five seeds.

### E.2 Study on Maximum Backtracking Length

<a id="figure-6"></a>

![humanoid_exp](images/humanoid_exp.png)

> Figure 6: Illustration of ADMPO-OFF’s performance under different maximum backtracking length.

Maximum backtracking length $m$ is an important hyper-parameter in our algorithms. Figure [5](https://arxiv.org/html/2405.17031v1#A5.F5) shows the influence of $m$ on the performance of hopper-medium-v2 and walker2d-medium-v2 tasks, respectively. We observe that the performance of ADMPO-OFF is not very sensitive to the hyper-parameter $m$.
