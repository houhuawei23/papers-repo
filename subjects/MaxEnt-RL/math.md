下面我们来系统性地展开最大熵强化学习的数学框架，从目标函数的构造出发，逐步推导软贝尔曼方程，并解释最优策略玻尔兹曼形式的由来。

---

## 一、从标准 MDP 到熵正则化目标

### 1. 标准强化学习的累积回报

在标准马尔可夫决策过程中，智能体与环境的交互产生轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots)$。

传统强化学习的目标是最大化期望累积折扣奖励：

$$
J_{\text{std}}(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \, r(s_t, a_t) \right],
$$

其中 $\gamma \in [0,1)$ 是折扣因子，$r(s_t,a_t)$ 是即时奖励。

### 2. 引入熵正则项

最大熵强化学习修改了这一目标，将每个时间步的奖励扩展为 **奖励 + 熵奖励**：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \big( r(s_t, a_t) + \alpha \, \mathcal{H}(\pi(\cdot|s_t)) \big) \right].
$$

这里：

- $\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi(\cdot|s)} \log \pi(a|s)$ 是策略在状态 $s$ 下的熵，度量动作分布的随机性；
- $\alpha \ge 0$ 是**温度参数**，用于调节熵项相对于奖励的重要性。
  - $\alpha = 0$ 退化为标准强化学习；
  - $\alpha$ 越大，策略越倾向于均匀分布（高熵）；
  - $\alpha \to 0^+$ 时，最优策略趋近于确定性贪心策略。

该目标函数的直观解释：智能体不仅要追逐高奖励，还要**尽可能地保持行动选择的多样性**。这一修改使得探索变成优化目标的内在组成部分，而不是外加的启发式策略（如 $\varepsilon$-greedy）。

---

## 二、软贝尔曼方程（Soft Bellman Equation）

在标准 MDP 中，值函数 $V^\pi$ 和 $Q^\pi$ 通过贝尔曼方程相互关联。为了处理熵正则化项，我们需要定义**软值函数**和**软 Q 函数**。

### 1. 软状态值函数 $V^\pi$

给定策略 $\pi$，定义状态 $s$ 下的**软值函数**为从该状态开始，按策略 $\pi$ 采样动作，并在每一步累加“奖励 + 熵”的期望折扣总和：

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi, s_0 = s} \left[ \sum_{t=0}^{\infty} \gamma^t \big( r(s_t, a_t) + \alpha \, \mathcal{H}(\pi(\cdot|s_t)) \big) \right].
$$

根据动态规划原理，我们可以将第一项拆分出来：

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} \big[ r(s,a) + \alpha \, \mathcal{H}(\pi(\cdot|s)) + \gamma \, \mathbb{E}_{s' \sim p(\cdot|s,a)} V^\pi(s') \big].
$$

注意到熵项 $\mathcal{H}(\pi(\cdot|s))$ 只依赖于当前状态和策略，不依赖于下一状态 $s'$，因此可以将其与奖励合并。

### 2. 软动作值函数 $Q^\pi$

定义**软 Q 函数**为：在状态 $s$ 执行动作 $a$，之后遵循策略 $\pi$ 所能获得的期望折扣总和（**不包括**当前动作带来的熵奖励，因为当前动作已经确定，其熵对后续状态的贡献通过策略体现）：

$$
Q^\pi(s,a) = r(s,a) + \gamma \, \mathbb{E}_{s' \sim p(\cdot|s,a)} V^\pi(s').
$$

由此，软状态值函数可以表达为软 Q 函数与策略熵的组合：

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} \big[ Q^\pi(s,a) - \alpha \log \pi(a|s) \big].
$$

这个关系是最大熵框架的核心等式之一。它表明状态的价值等于在该状态下按策略选动作得到的 Q 值，再减去因动作不确定性带来的熵惩罚（注意负号，实际上是要加上熵奖励，所以是 $-\alpha \log \pi$）。

### 3. 软贝尔曼方程的形式

将 $V^\pi(s')$ 用 $Q^\pi(s',a')$ 表示，并代回软 Q 函数的定义中，即得**软 Q 函数的贝尔曼方程**：

$$
Q^\pi(s,a) = r(s,a) + \gamma \, \mathbb{E}_{s' \sim p, a' \sim \pi} \left[ Q^\pi(s',a') - \alpha \log \pi(a'|s') \right].
$$

与标准贝尔曼方程 $Q(s,a) = r(s,a) + \gamma \mathbb{E}[V(s')]$ 相比，差别仅在于 $V(s')$ 中多了一项 $-\alpha \log \pi(a'|s')$，这正是熵奖励的体现。

### 4. 最优软值函数与最优策略

定义最优软 Q 函数 $Q^*(s,a) = \max_{\pi} Q^\pi(s,a)$，以及最优软值函数 $V^*(s) = \max_{\pi} V^\pi(s)$。最优策略 $\pi^*$ 应满足：

$$
\pi^*(\cdot|s) = \arg\max_{\pi} \mathbb{E}_{a \sim \pi} \left[ Q^*(s,a) - \alpha \log \pi(a|s) \right].
$$

这是一个带熵正则的优化问题，可以通过拉格朗日乘子法或直接使用变分法求解。

---

## 三、最优策略的玻尔兹曼形式推导

考虑固定状态 $s$ 下的优化问题：

$$
\max_{\pi(\cdot|s)} \; \mathbb{E}_{a \sim \pi} \left[ Q^*(s,a) - \alpha \log \pi(a|s) \right],
$$

约束条件为 $\int \pi(a|s) \, da = 1$ 且 $\pi(a|s) \ge 0$。

构造拉格朗日函数：

$$
\mathcal{L} = \int \pi(a|s) \left[ Q^*(s,a) - \alpha \log \pi(a|s) \right] da + \lambda \left( 1 - \int \pi(a|s) da \right).
$$

对 $\pi(a|s)$ 求泛函导数并令其为零（忽略归一化常数，先考虑内部最大化）：

$$
\frac{\partial}{\partial \pi(a|s)} \left[ \pi (Q^* - \alpha \log \pi) \right] = Q^*(s,a) - \alpha \log \pi(a|s) - \alpha = 0.
$$

解得：

$$
\log \pi(a|s) = \frac{1}{\alpha} Q^*(s,a) - 1.
$$

即：

$$
\pi(a|s) \propto \exp\left( \frac{Q^*(s,a)}{\alpha} \right).
$$

加上归一化条件，得到最优策略的解析形式：

$$
\pi^*(a|s) = \frac{\exp\left( \frac{Q^*(s,a)}{\alpha} \right)}{Z(s)},
\quad \text{其中 } Z(s) = \int \exp\left( \frac{Q^*(s,a)}{\alpha} \right) da.
$$

这就是**玻尔兹曼分布（或吉布斯分布）**，其中 $Q^*(s,a)$ 充当负能量函数，$\alpha$ 充当温度参数。

### 重要推论

- 当 $\alpha \to 0$ 时，$\exp(Q^*/\alpha)$ 在最大 $Q^*$ 处趋于无穷，策略退化为确定性贪心策略（在最大值处 Dirac delta）。
- 当 $\alpha \to \infty$ 时，指数项趋近于 1，策略退化为均匀分布。
- 对于有限的 $\alpha$，每个动作都有非零概率被选中，但高 Q 值动作概率更高，这自然地实现了“软探索”：次优动作也有机会被尝试，从而避免陷入局部最优。

---

## 四、软最优贝尔曼方程与最优值函数的关系

将最优策略的玻尔兹曼形式代入 $V^*(s)$ 的定义中：

$$
\begin{aligned}
V^*(s) &= \mathbb{E}_{a \sim \pi^*} \left[ Q^*(s,a) - \alpha \log \pi^*(a|s) \right] \\
&= \int \pi^*(a|s) \left[ Q^*(s,a) - \alpha \log \pi^*(a|s) \right] da.
\end{aligned}
$$

利用 $\log \pi^*(a|s) = \frac{Q^*(s,a)}{\alpha} - \log Z(s)$，代入得：

$$
V^*(s) = \int \pi^*(a|s) \left[ Q^*(s,a) - \alpha \left( \frac{Q^*(s,a)}{\alpha} - \log Z(s) \right) \right] da = \alpha \log Z(s).
$$

因此，**最优软值函数等于配分函数对数的 $\alpha$ 倍**。这一关系与统计力学中的自由能概念紧密相连，也使得 $V^*$ 有了信息论的解释。

进一步，软 Q 函数的贝尔曼最优方程变为：

$$
Q^*(s,a) = r(s,a) + \gamma \, \mathbb{E}_{s' \sim p} \left[ \alpha \log \int \exp\left( \frac{Q^*(s',a')}{\alpha} \right) da' \right].
$$

该方程中的积分算子被称为 **LogSumExp**（LSE）或软最大化，因此最大熵强化学习有时也被称为 **Soft Q-Learning**。

---

## 五、总结：数学框架的逻辑链条

1. **目标函数**：在每步奖励中加入策略熵 $\alpha \mathcal{H}(\pi)$。
2. **软值函数定义**：将熵纳入动态规划，得到 $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a) - \alpha \log \pi(a|s)]$。
3. **软贝尔曼方程**：递归关系 $Q^\pi(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[V^\pi(s')]$。
4. **最优策略的变分形式**：通过最大化 $Q - \alpha \log \pi$ 的期望得到玻尔兹曼分布 $\pi^* \propto \exp(Q^*/\alpha)$。
5. **最优值函数的解析表达**：$V^*(s) = \alpha \log \int \exp(Q^*(s,a)/\alpha) da$。

这套数学框架不仅赋予了强化学习“探索即目标”的理论基础，还催生了诸如 **Soft Actor-Critic (SAC)** 等高效实用的算法，其中温度参数 $\alpha$ 可被自动调节以维持目标熵水平，从而在复杂连续控制任务中表现卓越。
