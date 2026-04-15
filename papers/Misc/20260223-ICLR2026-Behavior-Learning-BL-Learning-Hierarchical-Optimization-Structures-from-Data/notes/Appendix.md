
## 附录 A 架构细节

<a id="appendix-a"></a>

### A.1 学习方案细节

##### BL 函数的输入与输出。

我们将 BL（Bilinear，双线性）函数表述为一个从输入-输出对到组合效用表示的直接映射：

$$
\mathrm{BL}:\ \mathcal{X}\times\mathcal{Y}\to\mathbb{R}^{d_{\mathrm{out}}},\qquad(x,y)\mapsto\mathrm{BL}(x,y)\in\mathbb{R}^{d_{\mathrm{out}}},
$$

其中输出维度 $d_{\mathrm{out}}$ 根据建模选择而定。这种表述方式有意允许 BL 为每个 $(x,y)$ 返回标量或向量；以下几种情况最为常见：

- **每个候选的标量（逐点评估）** 。
  设 $d_{\mathrm{out}}=1$。此时 $\mathrm{BL}(x,y)\in\mathbb{R}$ 是为单个候选 $y$ 评估的标量组合效用。这种视角对于连续的 $y$（回归或密度估计）或当倾向于单独评估候选时是很自然的。
- **在有限候选集上的向量化** 。
  如果 $\mathcal{Y}=\{y_{1},\dots,y_{m}\}$ 是有限的，可以选择 $d_{\mathrm{out}}=m$，并通过堆叠对候选集的评估来定义向量值输出：
  $\text{BL}(x):=\begin{bmatrix}\mathrm{BL}(x,y\_{1})
  $$
  4.0pt]
  \vdots
  $$4.0pt]
  \mathrm{BL}(x,y_{m})\end{bmatrix}\in\mathbb{R}^{m}.$
  这种向量化形式对于分类很方便：它一次性评估所有类别候选，并为每个 $x$ 产生一个组合效用向量。
  $$
- **灵活性与等价性** 。
  标量和向量模式是兼容的：向量化形式本质上就是一批逐点评估。反之，一个标量逐点评估器可以通过在候选集上重复调用来组装成一个向量。因此，在逐点（标量）输出和向量化输出之间的选择是一个权衡计算效率与便利性的工程选择。

给定数据集 $\mathcal{D}=\{(x_{i},y_{i})\}_{i=1}^{n}$，训练和推理可以使用任一模式：在可行时（例如，小的有限 $\mathcal{Y}$）使用向量化计算，或在 $\mathcal{Y}$ 很大或连续时使用逐点评估。

##### 条件吉布斯模型。

令 $(x,y)\sim\mathcal{D}$，其中 $x\in\mathbb{R}^{d}$ 且 $y=(y^{\mathrm{disc}},y^{\mathrm{cont}})\in\mathcal{Y}_{\mathrm{disc}}\times\mathbb{R}^{m_{c}}$（离散、连续或混合）。
BL 诱导出一个具有温度 $\tau>0$ 的条件吉布斯分布：

$$
p_{\tau}(y\mid x)=\frac{\exp\{\mathrm{BL}(x,y)/\tau\}}{Z_{\tau}(x)},\qquad Z_{\tau}(x)=\!\int_{\mathcal{Y}}\!\exp\{\mathrm{BL}(x,y^{\prime})/\tau\}\,dy^{\prime}.
$$

对于离散的 $\mathcal{Y}=\{y_{1},\dots,y_{m}\}$，如果我们选择向量输出表述，则定义

$$
\mathrm{BL}(x):=\bigl[\mathrm{BL}(x,y_{1}),\dots,\mathrm{BL}(x,y_{m})\bigr]\in\mathbb{R}^{m},
$$

使得条件分布简化为对该组合效用向量的 softmax（归一化指数函数）：

$$
p_{\tau}(y=k\mid x)=\mathrm{softmax}_{k}\!\left(\tfrac{1}{\tau}\,\mathrm{BL}(x)\right).
$$

从行为上看，$\tau$ 编码了 **噪声理性（noisy rationality）** ；当 $\tau\!\to 0$ 时，$p_{\tau}(\cdot\mid x)$ 集中在 $\arg\max_{y}\mathrm{BL}(x,y)$ 上，对应于学习模型所隐含的确定性最优选择。

##### 监督、无监督及生成式用途。

BL 适应多种模式。
(i) **监督式** ：将 $x$ 作为输入，$y$ 作为标签。
对于离散的 $y$，可以 (a) 采用向量输出表述，其中 $\mathrm{BL}(x)\in\mathbb{R}^{m}$ 产生一个覆盖所有类别的组合效用向量，似然由 softmax 给出；或者 (b) 采用标量输出表述，其中 $\mathrm{BL}(x,y)$ 为每个候选单独评估，然后在类别间进行归一化。
对于连续的 $y$，BL 自然在标量输出模式下运行，将 $\mathrm{BL}(x,y)\in\mathbb{R}$ 视为一个组合效用场。
(ii) **无监督 / 生成式** ：对边际分布 $p(y)\propto\exp\{\mathrm{BL}(y)/\tau\}$（空 $x$）或联合分布 $p(x,y)\propto\exp\{\mathrm{BL}(x,y)/\tau\}$ 进行建模；对吉布斯分布进行采样即可得到一个生成器。

##### 学习目标。

由于响应 $\mathbf{y}$ 可能同时包含离散和连续分量，我们通过最小化类型特定的风险来估计参数 $\theta$：

$$
\mathcal{L}(\theta)=\gamma_{\mathrm{d}}\,\mathbb{E}\bigl[-\log p_{\tau}(y^{\mathrm{disc}}\mid x)\bigr]\;+\;\gamma_{\mathrm{c}}\,\mathbb{E}\Bigl\|\nabla_{\tilde{y}^{\mathrm{cont}}}\log p_{\tau}(\tilde{y}^{\mathrm{cont}}\mid x)+\sigma^{-2}(\tilde{y}^{\mathrm{cont}}-y^{\mathrm{cont}})\Bigr\|^{2},
$$

其中第一项是离散分量上的交叉熵（Cross-entropy），第二项是连续分量上的 **去噪分数匹配（Denoising Score Matching, DSM）** ，其中 $\tilde{y}^{\mathrm{cont}}=y^{\mathrm{cont}}+\varepsilon$，$\varepsilon\sim\mathcal{N}(0,\sigma^{2}I)$。
对于纯离散输出，设 $(\gamma_{\mathrm{d}},\gamma_{\mathrm{c}})=(1,0)$；对于纯连续输出，设 $(0,1)$；对于混合输出，设 $(>0,>0)$。

### A.2 模型结构细节

在正文中，我们采用了 BL 的紧凑记法；在此，我们提出一个等价的、更明确的矩阵/向量表述，它明确了维度、线性映射以及每个头（head）的参数化，这对于形式化证明和实现细节很有用。

##### **固定基与头部预激活（Fixed bases and head pre-activations）**

对于模块输入 $z$（具体定义见下文），令

$$
m_{u}(z)\in\mathbb{R}^{d_{u}},\qquad m_{c}(z)\in\mathbb{R}^{d_{c}},\qquad m_{t}(z)\in\mathbb{R}^{d_{t}}
$$

表示固定的基（例如，单项式）向量。
可学习的线性映射产生头部预激活：

$$
u(z):=M_{u}\,m_{u}(z)+b_{u}\in\mathbb{R}^{r_{u}},\quad c(z):=M_{c}\,m_{c}(z)+b_{c}\in\mathbb{R}^{r_{c}},\quad t(z):=M_{t}\,m_{t}(z)+b_{t}\in\mathbb{R}^{r_{t}},
$$

其中 $M_{u}\in\mathbb{R}^{r_{u}\times d_{u}}$， $M_{c}\in\mathbb{R}^{r_{c}\times d_{c}}$， $M_{t}\in\mathbb{R}^{r_{t}\times d_{t}}$，且 $b_{\bullet}$ 为可选偏置项。

##### **单个 BL 模块（Single BL block）**

单个模块化块定义为

$$
\mathcal{B}(z)\;=\;\lambda_{0}^{\top}\,\phi\!\big(u(z)\big)\;-\;\lambda_{1}^{\top}\,\rho\!\big(c(z)\big)\;-\;\lambda_{2}^{\top}\,\psi\!\big(t(z)\big),(12)
$$

其中 $\lambda_{0}\in\mathbb{R}^{r_{u}}$， $\lambda_{1}\in\mathbb{R}^{r_{c}}$， $\lambda_{2}\in\mathbb{R}^{r_{t}}$ 为可学习的权重，且 $\phi,\rho,\psi$ 按坐标作用，其角色在定理 [2.1](https://arxiv.org/html/2602.20152v1#S2.Thmtheorem1) 中指定（$\phi$ 递增用于效用，$\rho$ 作为违反不等式约束的惩罚，$\psi$ 对称用于等式约束）。
通过识别

$$
U_{\theta_{U}}(x,y)=u\bigl(z=(x,y)\bigr),\quad\mathcal{C}_{\theta_{C}}(x,y)=c\bigl(z=(x,y)\bigr),\quad\mathcal{T}_{\theta_{T}}(x,y)=t\bigl(z=(x,y)\bigr),
$$

并代入方程 [12](https://arxiv.org/html/2602.20152v1#A1.E12)，即可恢复正文方程 [4](https://arxiv.org/html/2602.20152v1#S2.E4) 中的参数化形式。

##### **并行模块层（Layer of parallel blocks）**

一个层 $\mathbb{B}_{\ell}$ 将方程 [12](https://arxiv.org/html/2602.20152v1#A1.E12) 的 $d_{\ell}$ 个并行副本（可能具有不同的参数 $\theta_{\ell,i}$）堆叠起来：

$$
\mathbb{B}_{\ell}(z_{\ell})\;:=\;\begin{bmatrix}\mathcal{B}_{\theta_{\ell,1}}(z_{\ell})
$$-1.0pt] \vdots
$$-1.0pt] \mathcal{B}_{\theta_{\ell,d_{\ell}}}(z_{\ell})\end{bmatrix}\in\mathbb{R}^{d_{\ell}}.
$$

我们采用标准的分层（前馈）形式：

$$
z_{1}:=(x,y),\qquad z_{\ell+1}:=\mathbb{B}_{\ell}(z_{\ell})\quad(\ell=1,\dots,L-1),
$$

因此每一层的输入就是前一层的输出。这是典型的前馈架构。

可选地，可以允许每一层显式访问原始输入：

$$
z_{1}:=(x,y),\qquad z_{\ell+1}:=\mathbb{B}_{\ell}\big((x,y),\,z_{\ell}\big).
$$

为了提高可训练性，也可以使用残差连接（Residual connections）：

$$
z_{\ell+1}:=z_{\ell}+\mathbb{B}_{\ell}(z_{\ell}).
$$

##### **浅层/深层组合与最终仿射读出（Shallow/Deep composition and final affine readout）**

对于深度 $L\geq 1$，BL 组合效用（BL compositional utility）通过对顶层进行最终的可学习仿射变换产生：

$$
\mathrm{BL}(x,y)\;=\;W_{L}\,\mathbb{B}_{L}\bigl(z_{L}\bigr)+b_{L},(13)
$$

其中，对于标量输出，$W_{L}\in\mathbb{R}^{1\times d_{L}}$；对于向量输出，$W_{L}\in\mathbb{R}^{m\times d_{L}}$；$b_{L}$ 为匹配维度的偏置项。
$L=1$（此时 $d_{1}=1$）、$L\leq 2$ 和 $L>2$ 的情况分别对应 **BL(Single)** 、 **BL(Shallow)** 和 **BL(Deep)** ，与正文中的描述完全一致。

### A.3 实现细节

#### A.3.1 函数实例化

##### 默认实例化。

在实践中，我们使用特定的选择 $(\phi,\rho,\psi)=(\tanh,\mathrm{ReLU},|\cdot|)$ 来实例化方程 [4](https://arxiv.org/html/2602.20152v1#S2.E4)：

$$
\mathcal{B}(x,y;\theta)=\lambda_{0}^{\!\top}\tanh\!\bigl(U_{\theta_{U}}(x,y)\bigr)-\lambda_{1}^{\!\top}\mathrm{ReLU}\!\bigl(\mathcal{C}_{\theta_{C}}(x,y)\bigr)-\lambda_{2}^{\!\top}\bigl|\mathcal{T}_{\theta_{T}}(x,y)\bigr|.(14)
$$

此处 $\lambda_{0},\lambda_{1},\lambda_{2}$ 为可学习的非负权重。
有界的 $\tanh$ 函数捕捉了效用头中的饱和效应和收益递减规律（Jevons， 2013），而 $\mathrm{ReLU}$ 和 $|\cdot|$ 则分别对不等式和等式违反施加了非对称（单侧）和对称（双侧）惩罚。

##### 变体与简化。

方程 [14](https://arxiv.org/html/2602.20152v1#A1.E14) 的几种变体通常很有用：

- **恒等效用头** 。
  设 $\phi=\mathrm{id}$，使效用头使用原始多项式：
  $\mathcal{B}=\lambda_{0}^{\!\top}U_{\theta_{U}}-\lambda_{1}^{\!\top}\mathrm{ReLU}(\mathcal{C}_{\theta_{C}})-\lambda_{2}^{\!\top}|\mathcal{T}_{\theta_{T}}|.$
- **平滑惩罚替代方案** 。
  将 $\mathrm{ReLU}$ 替换为 $\mathrm{softplus}$ 以产生平滑的不等式惩罚，或将 $|\cdot|$ 替换为 Huber 或平方惩罚，以调节等式项在零附近的敏感性。
- **舍弃头** 。
  该框架是模块化的，因此可以根据任务省略某些头：
- **无 $T$ 头** ：忽略对称偏差，产生一个仅包含不等式惩罚的约束最大化问题。
- **无 $C$ 头** ：如果保留了 $T$ 头，模型则简化为一个仅包含等式约束的最大化问题；如果 $T$ 也被移除，则变为完全无约束的最大化问题。
- **无 $U$ 头** ：产生一个专注于可行性的纯（软）约束模型。
  值得注意的是，同时移除 $U$ 和 $T$ 头后，仅剩下分段线性的 $\mathrm{ReLU}$ 惩罚；当其后接一个最终的仿射读出层时，所得架构变得与标准 **多层感知机（Multilayer Perceptron, MLP）** 高度相似——这表明 MLP 可被视为更广泛的 BL 框架中一个密切相关的特殊实例。

#### A.3.2 多项式特征映射与线性约简

我们采用一种务实的默认方案：对于单块模型，使用低次多项式映射以最大化可解释性；对于浅层/深层堆叠模型，在块内使用仿射（一次）映射以控制参数增长和计算量。下面我们陈述实例化方案，并给出实验中使用的最终块公式。

##### BL(Single) — 多项式实例化。

令 $m_{D}(x,y)$ 表示一个固定的、总次数不超过 $D$ 的单项式基（例如 $D\leq 2$）：

$$
m_{D}(x,y)=\bigl[x,\;y,\;\mathrm{vec}(xx^{\top}),\;\mathrm{vec}(xy^{\top}),\;\mathrm{vec}(yy^{\top}),\dots\bigr]^{\top}.
$$

将每个映射参数化为该基上的线性映射：

$$
U_{\theta_{U}}(x,y)=M_{U}\,m_{D}(x,y)+b_{U},\quad\mathcal{C}_{\theta_{C}}(x,y)=M_{C}\,m_{D}(x,y)+b_{C},\quad\mathcal{T}_{\theta_{T}}(x,y)=M_{T}\,m_{D}(x,y)+b_{T},
$$

其中 $M_{\bullet}$ 和 $b_{\bullet}$ 为可学习的矩阵和偏置。该块变为：

$$
\mathcal{B}(x,y;\theta)=\lambda_{0}^{\top}\phi(M_{U}m_{D}+b_{U})-\lambda_{1}^{\top}\rho(M_{C}m_{D}+b_{C})-\lambda_{2}^{\top}\psi(M_{T}m_{D}+b_{T}).
$$

##### BL (Shallow/Deep) — 逐层线性实例化。

对于堆叠架构（Shallow/Deep），我们在每个块内部使用仿射映射以保持每层复杂度较低：

$$
U_{\theta_{U}}(x,y)=A_{U}\,[x;y]+b_{U},\quad\mathcal{C}_{\theta_{C}}(x,y)=A_{C}\,[x;y]+b_{C},\quad\mathcal{T}_{\theta_{T}}(x,y)=A_{T}\,[x;y]+b_{T},
$$

其中 $A_{\bullet}$ 和 $b_{\bullet}$ 为可学习的参数。对应的块为：

$$
\mathcal{B}(x,y;\theta)=\lambda_{0}^{\top}\phi(A_{U}[x;y]+b_{U})-\lambda_{1}^{\top}\rho(A_{C}[x;y]+b_{C})-\lambda_{2}^{\top}\psi(A_{T}[x;y]+b_{T}).
$$

##### **按需高阶项（On-demand higher-order terms）**

如果诊断或领域知识表明存在欠拟合，我们可选择性地用选定的高阶项或交互项来增强仿射映射。具体而言，这是通过向输入向量 $[x;y]$ 追加一小组单项式（例如 $x_{i}y_{j}$, $x_{i}^{2}$, $y_{k}^{2}$）并重新估计相同的仿射映射 $A_{\bullet}$ 来实现的。
这种有针对性的增强保留了基础的仿射参数化，仅在需要的地方增加表达能力，并在保持可解释性的同时，将计算和统计成本控制在适度水平。

<a id="figure-7"></a>
![penalty_test_energycons](images/penalty_test_energycons.png)

> 图 7 | 多项式特征映射作为计算图的可视化，其中节点代表变量或输出，边代表它们的影响。左图展示了线性形式 $\mathcal{F}=ax+b$，其中单一边 $x\!\to\!\mathcal{F}$ 直接编码了 $x$ 对 $\mathcal{F}$ 的边际效应。中图展示了二次形式 $\mathcal{F}=ax^{2}+bx+c$，其中 $x$ 不仅有一条直接边 $x\!\to\!\mathcal{F}$，还作用于其自身的边（“$x\!\to\!\mathcal{F}$”），从而通过高阶贡献修改其自身效应的强度。右图描绘了交互形式 $\mathcal{F}=ax+by+cxy+d$，其中 $y$ 有一条边 $y\!\to\!\mathcal{F}$，此外，$x$ 作用于这条边（“$y\!\to\!\mathcal{F}$”），从而调节 $y$ 对 $\mathcal{F}$ 贡献的强度。对称地，$y$ 也可能作用于边（“$x\!\to\!\mathcal{F}$”），因此每个变量都可以通过交互项重塑对方的影响。

#### **A.3.3 跳跃连接（Skip Connections）**

跳跃连接在我们的实现中是可选的。当有益时，我们通常会考虑两种为 BL（Bilinear Layer）量身定制的模式：一种 DenseNet 风格（拼接式）变体和一种 ResNet 风格（加性）变体。

##### **密集跳跃连接（DenseNet 风格，拼接）**

该变体将之前所有表示的拼接作为每一层的输入，模仿了 DenseNet（Huang 等人，2017）[1]。
令

$$
z_{1}:=[\,x;\,y\,],\qquad s_{1}:=\mathbb{B}_{1}(z_{1})\in\mathbb{R}^{d_{1}}.
$$

对于 $\ell\geq 2$，

$$
z_{\ell}:=[\,x;\,y;\,s_{1};\dots;s_{\ell-1}\,],\qquad s_{\ell}:=\mathbb{B}_{\ell}(z_{\ell})\in\mathbb{R}^{d_{\ell}}.
$$

最终的组合效用（compositional utility）读出为

$$
\mathrm{BL}(x,y)\;=\;W_{L}\,s_{L}+b_{L}.
$$

_优点_：通过将所有早期块输出显式地暴露给后续块作为输入，密集跳跃连接保留了一条透明的特征轨迹：可以追踪哪些中间 $\mathcal{B}$-块输出进入了下游计算和最终的仿射读出。这通常能改善特征重用，并在块级别获得良好的可解释性。

##### **残差跳跃连接（ResNet 风格，加法）**

该变体为每一层添加一个恒等（或投影）捷径，如 ResNet（He 等人，2016）[2] 中所示。
定义

$$
z_{1}:=[\,x;\,y\,],\qquad s_{1}:=\mathbb{B}_{1}(z_{1})\in\mathbb{R}^{d_{1}},
$$

对于 $\ell\geq 2$，

$$
s_{\ell}:=\mathbb{B}_{\ell}(s_{\ell-1})\;+\;\Pi_{\ell}\,s_{\ell-1},\qquad\Pi_{\ell}\in\mathbb{R}^{d_{\ell}\times d_{\ell-1}},
$$

其中，如果 $d_{\ell}=d_{\ell-1}$，则 $\Pi_{\ell}$ 是恒等矩阵；否则是一个无偏置的可学习投影矩阵。读出同样为

$$
\mathrm{BL}(x,y)\;=\;W_{L}\,s_{L}+b_{L}.
$$

##### **跳跃连接与可解释性（Skip Connections and Interpretability）**

跳跃连接引入了显式的跨层依赖结构，这种形式在统计物理学和其他科学领域被广泛研究。这种结构通过使长程影响变得透明，增强了科学的可解释性。在行为和组织科学中，它们捕捉了低层级代理无需通过中间层路由而直接影响高层级决策者的情境。在物理学中，微观参数可以对跨多个尺度的宏观行为产生直接影响。在架构上，ResNet 风格的跳跃连接建模了线性的跨层依赖关系，而 DenseNet 风格的连接则实现了拼接式（信息复制）的依赖关系。这些机制为表示层次化交互提供了灵活且可解释的途径。

<a id="appendix-b"></a>

**参考文献**

1. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 4700-4708).
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 770-778).

## 附录 B 定理证明

### B.1 效用最大化问题（Utility Maximization Problem, UMP）

**定理 [2.1]** （UMP 的局部精确惩罚重构）。

令 $\mathcal{X}\subset\mathbb{R}^{d_{x}}$ 和 $\mathcal{Y}\subset\mathbb{R}^{d_{y}}$ 为非空紧集，并令 $U:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}$、$\mathcal{C}:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}^{m}$ 和 $\mathcal{T}:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}^{p}$ 为 $C^{1}$ 函数。
考虑 **效用最大化问题（Utility Maximization Problem, UMP）** ：

$$
\max_{\mathbf{y}\in\mathcal{Y}}\,U(\mathbf{x},\mathbf{y})\quad\text{s.t.}\quad\mathcal{C}(\mathbf{x},\mathbf{y})\leq 0,\;\;\mathcal{T}(\mathbf{x},\mathbf{y})=0.(15)
$$

假设存在一个可行点 $\mathbf{y}^{\star}\in\mathrm{int}(\mathcal{Y})$，它是方程 [15] 的一个严格局部极大值点，并且在 $\mathbf{y}^{\star}$ 处满足 **韩-曼加萨里安约束规格（Han–Mangasarian constraint qualification）** (2.1)（采用 Han & Mangasarian (1979) 的记法）。
令 $\phi:\mathbb{R}\to\mathbb{R}$ 为严格递增的 $C^{1}$ 函数，并定义 $\rho(z):=\max\{z,0\}$ 和 $\psi(z):=|z|$（在 $\mathbb{R}^{m}$ 和 $\mathbb{R}^{p}$ 上按分量定义）。
则存在 $\lambda_{0}>0$、$\lambda_{1}\in\mathbb{R}^{m}_{++}$ 和 $\lambda_{2}\in\mathbb{R}^{p}_{++}$，使得 $\mathbf{y}^{\star}$ 是以下问题的局部极大值点：

$$
\max_{\mathbf{y}\in\mathcal{Y}}\;\lambda_{0}\,\phi\!\bigl(U(\mathbf{x},\mathbf{y})\bigr)-\lambda_{1}^{\top}\rho\!\bigl(\mathcal{C}(\mathbf{x},\mathbf{y})\bigr)-\lambda_{2}^{\top}\psi\!\bigl(\mathcal{T}(\mathbf{x},\mathbf{y})\bigr).(16)
$$

###### 证明。

固定 $\mathbf{x}\in\mathcal{X}$ 并简记：

$$
g(\mathbf{y}):=\mathcal{C}(\mathbf{x},\mathbf{y})\in\mathbb{R}^{m},\qquad h(\mathbf{y}):=\mathcal{T}(\mathbf{x},\mathbf{y})\in\mathbb{R}^{p}.
$$

$\mathbf{y}^{\star}$ 的可行性意味着 $g(\mathbf{y}^{\star})\leq 0$（分量意义）且 $h(\mathbf{y}^{\star})=0$。

**步骤 1：转换为环境空间中的约束局部最小化问题。**
任取 $\lambda_{0}>0$ 并定义：

$$
f(\mathbf{y}):=-\lambda_{0}\,\phi\!\bigl(U(\mathbf{x},\mathbf{y})\bigr).(17)
$$

由于 $\phi$ 严格递增，对于任意 $\mathbf{y}_{1},\mathbf{y}_{2}$，我们有 $U(\mathbf{x},\mathbf{y}_{1})>U(\mathbf{x},\mathbf{y}_{2})$ 当且仅当 $f(\mathbf{y}_{1})<f(\mathbf{y}_{2})$。
因此，$\mathbf{y}^{\star}$ 是方程 [15] 的严格局部极大值点，当且仅当 $\mathbf{y}^{\star}$ 是以下问题的严格局部极小值点：

$$
\min_{\mathbf{y}\in\mathcal{Y}}\,f(\mathbf{y})\quad\text{s.t.}\quad g(\mathbf{y})\leq 0,\ \ h(\mathbf{y})=0.(18)
$$

现在利用内点假设 $\mathbf{y}^{\star}\in\mathrm{int}(\mathcal{Y})$：存在 $\varepsilon_{0}>0$ 使得 $B_{\varepsilon_{0}}(\mathbf{y}^{\star})\subset\mathcal{Y}$。
因此，在 $\mathbf{y}^{\star}$ 处关于 $\mathcal{Y}$ 的严格局部极小值点概念与环境空间中的概念一致：对于任意函数 $F$，存在 $\varepsilon\in(0,\varepsilon_{0}]$ 使得

$$
F(\mathbf{y}^{\star})<F(\mathbf{y})\quad\forall\,\mathbf{y}\in\bigl(\mathcal{Y}\cap B_{\varepsilon}(\mathbf{y}^{\star})\bigr)\setminus\{\mathbf{y}^{\star}\}
$$

当且仅当

$$
F(\mathbf{y}^{\star})<F(\mathbf{y})\quad\forall\,\mathbf{y}\in B_{\varepsilon}(\mathbf{y}^{\star})\setminus\{\mathbf{y}^{\star}\}.
$$

因此，$\mathbf{y}^{\star}$ 在环境空间意义下是约束问题方程 [18] 的严格局部极小值点。

此外，根据假设，三元组 $(f,g,h)$ 在 $\mathbf{y}^{\star}$ 的某个邻域内连续可微。

**步骤 2：将向量权重嵌入范数并构建韩-曼加萨里安惩罚。**
按分量定义正部 $g_{+}(\mathbf{y})\in\mathbb{R}^{m}$：$(g_{+}(\mathbf{y}))_{i}:=\max\{g_{i}(\mathbf{y}),0\}$。
暂时任取 $\lambda_{1}\in\mathbb{R}^{m}_{++}$ 和 $\lambda_{2}\in\mathbb{R}^{p}_{++}$，并通过加权 $\ell_{1}$-范数在 $\mathbb{R}^{m+p}$ 上定义一个范数：

$$
\|(u,v)\|_{\lambda}:=\lambda_{1}^{\top}|u|+\lambda_{2}^{\top}|v|,\qquad(u,v)\in\mathbb{R}^{m}\times\mathbb{R}^{p},(19)
$$

其中 $|\cdot|$ 是分量绝对值。由于 $\lambda_{1},\lambda_{2}$ 的各个分量均为严格正数，$\|\cdot\|_{\lambda}$ 确实是一个范数。

选择标量惩罚函数 $Q:[0,\infty)\to[0,\infty)$ 为 $Q(t)=t$。
则 $Q$ 满足 Han & Mangasarian (1979) 中的惩罚正则性条件 (1.3)，特别地，$Q^{\prime}(0+)=1>0$。
对于 $\alpha\geq 0$，定义惩罚函数

$$
P(\mathbf{y},\alpha):=f(\mathbf{y})+\alpha\,Q\!\left(\left\|\bigl(g_{+}(\mathbf{y}),h(\mathbf{y})\bigr)\right\|_{\lambda}\right).(20)
$$

利用方程 [19](https://arxiv.org/html/2602.20152v1#A2.E19) 和 $Q(t)=t$ 展开方程 [20](https://arxiv.org/html/2602.20152v1#A2.E20)，得到

$$
P(\mathbf{y},\alpha)=-\lambda_{0}\,\phi\!\bigl(U(\mathbf{x},\mathbf{y})\bigr)+\alpha\Bigl[\lambda_{1}^{\top}g_{+}(\mathbf{y})+\lambda_{2}^{\top}|h(\mathbf{y})|\Bigr].(21)
$$

由于 $\rho(g(\mathbf{y}))=g_{+}(\mathbf{y})$ 且 $\psi(h(\mathbf{y}))=|h(\mathbf{y})|$ 在分量意义上成立，方程 [21](https://arxiv.org/html/2602.20152v1#A2.E21) 可以重写为

$$
P(\mathbf{y},\alpha)=-\lambda_{0}\,\phi\!\bigl(U(\mathbf{x},\mathbf{y})\bigr)+\alpha\Bigl[\lambda_{1}^{\top}\rho\!\bigl(g(\mathbf{y})\bigr)+\lambda_{2}^{\top}\psi\!\bigl(h(\mathbf{y})\bigr)\Bigr].(22)
$$

**步骤 3：应用 Han–Mangasarian 定理 4.4。**
根据步骤 1–2，函数 $f,g,h$ 在 $\mathbf{y}^{\star}$ 的某个邻域上是 $C^{1}$ 的，$\mathbf{y}^{\star}$ 是约束问题方程 [18](https://arxiv.org/html/2602.20152v1#A2.E18) 的一个严格局部极小值点（在环境空间意义下），并且 Han–Mangasarian 约束规格（2.1）在 $\mathbf{y}^{\star}$ 处成立。
因此，根据 (Han & Mangasarian, 1979, Thm. 4.4)，存在 $\bar{\alpha}\geq 0$，使得对于每个 $\alpha\geq\bar{\alpha}$，$\mathbf{y}^{\star}$ 都是 $P(\cdot,\alpha)$ 的一个局部极小值点。

**步骤 4：回到局部最大化并确保向量权重严格为正。**
选择任意

$$
\alpha>\max\{\bar{\alpha},0\},(23)
$$

因此特别地有 $\alpha>0$。定义惩罚后的最大化目标

$$
\widetilde{F}(\mathbf{y}):=-P(\mathbf{y},\alpha)=\lambda_{0}\,\phi\!\bigl(U(\mathbf{x},\mathbf{y})\bigr)-\alpha\,\lambda_{1}^{\top}\rho\!\bigl(g(\mathbf{y})\bigr)-\alpha\,\lambda_{2}^{\top}\psi\!\bigl(h(\mathbf{y})\bigr).
$$

由于 $\mathbf{y}^{\star}$ 是 $P(\cdot,\alpha)$ 的一个局部极小值点，因此它是 $\widetilde{F}$ 的一个局部极大值点。
最后，令

$$
\lambda_{1}^{\prime}:=\alpha\,\lambda_{1}\in\mathbb{R}^{m}_{++},\qquad\lambda_{2}^{\prime}:=\alpha\,\lambda_{2}\in\mathbb{R}^{p}_{++}.
$$

那么 $\widetilde{F}(\mathbf{y})$ 等于

$$
\lambda_{0}\,\phi\!\bigl(U(\mathbf{x},\mathbf{y})\bigr)-(\lambda_{1}^{\prime})^{\top}\rho\!\bigl(\mathcal{C}(\mathbf{x},\mathbf{y})\bigr)-(\lambda_{2}^{\prime})^{\top}\psi\!\bigl(\mathcal{T}(\mathbf{x},\mathbf{y})\bigr),
$$

这正是方程 [16](https://arxiv.org/html/2602.20152v1#A2.E16) 中的目标函数。
因此，$\mathbf{y}^{\star}$ 是方程 [16](https://arxiv.org/html/2602.20152v1#A2.E16) 在 $\mathcal{Y}$ 上的一个局部极大值点。
由于 $\mathbf{y}^{\star}\in\mathrm{int}(\mathcal{Y})$，这等价于在环境空间意义下的局部极大性。
证明完成。
∎

**定理 [2.2](https://arxiv.org/html/2602.20152v1#S2.Thmtheorem2) （UMP 的普适性）。**
设 $\mathcal{X}$ 和 $\mathcal{Y}$ 是任意的非空集合。
设 $f:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ 为一个目标函数，并令

$$
\{g_{i}\}_{i\in I_{\leq}},\quad\{\tilde{g}_{k}\}_{k\in I_{\geq}},\quad\{h_{j}\}_{j\in J}
$$

是 $\mathcal{X}\times\mathcal{Y}$ 上的（可能为空、可数或不可数的）实值约束函数族。
对于每个固定的 $\mathbf{x}\in\mathcal{X}$，考虑优化问题

$$
\sup_{\mathbf{y}\in\mathcal{Y}}f(\mathbf{x},\mathbf{y})\quad\text{s.t.}\quad g_{i}(\mathbf{x},\mathbf{y})\leq 0~(i\in I_{\leq}),\ \ \tilde{g}_{k}(\mathbf{x},\mathbf{y})\geq 0~(k\in I_{\geq}),\ \ h_{j}(\mathbf{x},\mathbf{y})=0~(j\in J).(24)
$$

定义（约定 $\sup\varnothing:=-\infty$，且上确界在扩展实数中取值）

$$
U(\mathbf{x},\mathbf{y}):=f(\mathbf{x},\mathbf{y}),\qquad\mathcal{C}(\mathbf{x},\mathbf{y}):=\max\Bigl\{\,0,\ \sup_{i\in I_{\leq}}g_{i}(\mathbf{x},\mathbf{y}),\ \sup_{k\in I_{\geq}}\bigl(-\tilde{g}_{k}(\mathbf{x},\mathbf{y})\bigr)\Bigr\},
$$

$$
\mathcal{T}(\mathbf{x},\mathbf{y}):=\max\Bigl\{\,0,\ \sup_{j\in J}\,|h_{j}(\mathbf{x},\mathbf{y})|\Bigr\}.
$$

那么对于每个 $\mathbf{x}\in\mathcal{X}$，问题方程 [24](https://arxiv.org/html/2602.20152v1#A2.E24) **\*等价于** \* 效用最大化问题

$$
\sup_{\mathbf{y}\in\mathcal{Y}}U(\mathbf{x},\mathbf{y})\quad\text{s.t.}\quad\mathcal{C}(\mathbf{x},\mathbf{y})\leq 0,\qquad\mathcal{T}(\mathbf{x},\mathbf{y})=0,(25)
$$

这种等价性在于方程 [24](https://arxiv.org/html/2602.20152v1#A2.E24) 和方程 [25](https://arxiv.org/html/2602.20152v1#A2.E25) 的可行集重合；因此最优值重合，并且只要极大值点存在，其 **极大值点集（argmax sets）** 也重合。
对于最小化问题，将 $U$ 替换为 $-f$。

###### 证明（Proof）.

固定 $\mathbf{x}\in\mathcal{X}$。令

$$
F(\mathbf{x}):=\Bigl\{\mathbf{y}\in\mathcal{Y}:\ g_{i}(\mathbf{x},\mathbf{y})\leq 0~\forall i\in I_{\leq},\ \tilde{g}_{k}(\mathbf{x},\mathbf{y})\geq 0~\forall k\in I_{\geq},\ h_{j}(\mathbf{x},\mathbf{y})=0~\forall j\in J\Bigr\}
$$

表示方程 [24] 的可行集，并令

$$
\hat{F}(\mathbf{x}):=\Bigl\{\mathbf{y}\in\mathcal{Y}:\ \mathcal{C}(\mathbf{x},\mathbf{y})\leq 0,\ \mathcal{T}(\mathbf{x},\mathbf{y})=0\Bigr\}
$$

表示方程 [25] 的可行集。我们将证明 $F(\mathbf{x})=\hat{F}(\mathbf{x})$。

_(i) $F(\mathbf{x})\subseteq\hat{F}(\mathbf{x})$._
设 $\mathbf{y}\in F(\mathbf{x})$。则对所有 $i\in I_{\leq}$ 有 $g_{i}(\mathbf{x},\mathbf{y})\leq 0$，因此

$$
\sup_{i\in I_{\leq}}g_{i}(\mathbf{x},\mathbf{y})\leq 0.
$$

类似地，对所有 $k\in I_{\geq}$ 有 $\tilde{g}_{k}(\mathbf{x},\mathbf{y})\geq 0$ 意味着对所有 $k$ 有 $-\tilde{g}_{k}(\mathbf{x},\mathbf{y})\leq 0$，因此

$$
\sup_{k\in I_{\geq}}\bigl(-\tilde{g}_{k}(\mathbf{x},\mathbf{y})\bigr)\leq 0.
$$

此外，对所有 $j\in J$ 有 $h_{j}(\mathbf{x},\mathbf{y})=0$ 意味着对所有 $j\in J$ 有 $|h_{j}(\mathbf{x},\mathbf{y})|=0$，因此

$$
\sup_{j\in J}|h_{j}(\mathbf{x},\mathbf{y})|\leq 0\qquad\text{(约定 \sup\varnothing=-\infty$)}.$
$$

根据定义，

$$
\mathcal{C}(\mathbf{x},\mathbf{y})=\max\Bigl\{\,0,\ \sup_{i\in I_{\leq}}g_{i}(\mathbf{x},\mathbf{y}),\ \sup_{k\in I_{\geq}}\bigl(-\tilde{g}_{k}(\mathbf{x},\mathbf{y})\bigr)\Bigr\}=0,
$$

且

$$
\mathcal{T}(\mathbf{x},\mathbf{y})=\max\Bigl\{\,0,\ \sup_{j\in J}|h_{j}(\mathbf{x},\mathbf{y})|\Bigr\}=0.
$$

因此 $\mathbf{y}\in\hat{F}(\mathbf{x})$。

_(ii) $\hat{F}(\mathbf{x})\subseteq F(\mathbf{x})$._
设 $\mathbf{y}\in\hat{F}(\mathbf{x})$。令

$$
A:=\sup_{i\in I_{\leq}}g_{i}(\mathbf{x},\mathbf{y}),\qquad B:=\sup_{k\in I_{\geq}}\bigl(-\tilde{g}_{k}(\mathbf{x},\mathbf{y})\bigr),\qquad S:=\sup_{j\in J}|h_{j}(\mathbf{x},\mathbf{y})|.
$$

则

$$
\mathcal{C}(\mathbf{x},\mathbf{y})=\max\{0,A,B\}\leq 0.
$$

由于 $0\leq\max\{0,A,B\}$ 恒成立，我们有 $\max\{0,A,B\}=0$，特别地，$A\leq 0$ 且 $B\leq 0$。
利用上确界的基本性质，对每个 $i\in I_{\leq}$，我们有

$$
g_{i}(\mathbf{x},\mathbf{y})\leq\sup_{i\in I_{\leq}}g_{i}(\mathbf{x},\mathbf{y})=A\leq 0,
$$

且对每个 $k\in I_{\geq}$，我们有

$$
-\tilde{g}_{k}(\mathbf{x},\mathbf{y})\leq\sup_{k\in I_{\geq}}\bigl(-\tilde{g}_{k}(\mathbf{x},\mathbf{y})\bigr)=B\leq 0,
$$

即，$\tilde{g}_{k}(\mathbf{x},\mathbf{y})\geq 0$。

接下来，$\mathcal{T}(\mathbf{x},\mathbf{y})=0$ 意味着

$$
0=\mathcal{T}(\mathbf{x},\mathbf{y})=\max\{0,S\},
$$

因此 $S\leq 0$。由于对每个 $j\in J$ 都有 $|h_{j}(\mathbf{x},\mathbf{y})|\geq 0$，且 $|h_{j}(\mathbf{x},\mathbf{y})|\leq S\leq 0$，可得

$$
|h_{j}(\mathbf{x},\mathbf{y})|=0\quad\text{对所有 }j\in J,
$$

等价地，对所有 $j\in J$ 有 $h_{j}(\mathbf{x},\mathbf{y})=0$。因此 $\mathbf{y}\in F(\mathbf{x})$。

结合 (i) 和 (ii) 可得 $F(\mathbf{x})=\hat{F}(\mathbf{x})$。由于 $U(\mathbf{x},\mathbf{y})=f(\mathbf{x},\mathbf{y})$（并且对于最小化问题，可以等价地优化 $-f$），两个问题在相同的可行集上优化相同的目标函数。因此，它们的最优值一致，并且只要存在最大化器，它们的 argmax 集合也一致。
∎

###### 证明。

###### 证明。

### B.2 BL 架构

**定理 [2.3](https://arxiv.org/html/2602.20152v1#S2.Thmtheorem3) (BL 的通用逼近性)** 。
令 $\mathcal{X}\subset\mathbb{R}^{d}$ 和 $\mathcal{Y}\subset\mathbb{R}^{m}$ 为紧集，并令 $p^{\star}(\mathbf{y}\mid\mathbf{x})$ 为任意连续条件密度，使得对所有 $(\mathbf{x},\mathbf{y})\in\mathcal{X}\times\mathcal{Y}$ 有 $p^{\star}(\mathbf{y}\mid\mathbf{x})>0$。那么，对于任意 $\tau>0$ 和 $\varepsilon>0$，存在一个有限的 BL 架构（其深度和宽度取决于 $\varepsilon$）和一个参数 $\theta^{\star}$，使得吉布斯分布（Gibbs distribution）

$$
p_{\tau}(\mathbf{y}\mid\mathbf{x};\theta^{\star})=\frac{\exp\bigl(\text{BL}_{\theta^{\star}}(\mathbf{x},\mathbf{y})/\tau\bigr)}{\int_{\mathcal{Y}}\exp\bigl(\text{BL}_{\theta^{\star}}(\mathbf{x},\mathbf{y}^{\prime})/\tau\bigr)\mathrm{d}\mathbf{y}^{\prime}}(26)
$$

满足

$$
\sup_{\mathbf{x}\in\mathcal{X}}\mathrm{KL}\bigl(p^{\star}(\cdot\mid\mathbf{x})\,\|\,p_{\tau}(\cdot\mid\mathbf{x};\theta^{\star})\bigr)<\varepsilon.(27)
$$

###### 证明。

**步骤 0 (有界对数密度)** 。
定义 $f(\mathbf{x},\mathbf{y}):=\log p^{\star}(\mathbf{y}\mid\mathbf{x})$。
由于 $p^{\star}$ 在紧集 $\mathcal{X}\times\mathcal{Y}$ 上连续且严格为正，它达到一个正的最小值和有限的最大值。
因此 $f\in C(\mathcal{X}\times\mathcal{Y})$ 且有界。

**步骤 1 (BL 块包含一个单隐藏层 $\tanh$ 网络)** 。
回顾基本块（elementary block）

$$
\mathcal{B}(\mathbf{x},\mathbf{y};\theta):=\lambda_{0}^{\!\top}\tanh\bigl(\mathbf{p}_{u}(\mathbf{x},\mathbf{y})\bigr)-\lambda_{1}^{\!\top}\mathrm{ReLU}\bigl(\mathbf{p}_{c}(\mathbf{x},\mathbf{y})\bigr)-\lambda_{2}^{\!\top}\bigl|\mathbf{p}_{t}(\mathbf{x},\mathbf{y})\bigr|.(28)
$$

设 $\lambda_{1}=\mathbf{0}$ 和 $\lambda_{2}=\mathbf{0}$。
选择 $\mathbf{p}_{u}(\mathbf{x},\mathbf{y})$ 为 $[\mathbf{x};\mathbf{y}]$ 的仿射函数，即
$\mathbf{p}_{u}(\mathbf{x},\mathbf{y})=W[\mathbf{x};\mathbf{y}]+b\in\mathbb{R}^{k}$，其中 $k\in\mathbb{N}$。
那么

$$
\mathcal{B}(\mathbf{x},\mathbf{y};\theta)=\lambda_{0}^{\top}\tanh\!\bigl(W[\mathbf{x};\mathbf{y}]+b\bigr),(29)
$$

这是在紧域 $\mathcal{X}\times\mathcal{Y}$ 上的标准单隐藏层 $\tanh$ 网络。

如果 $\lambda_{0}\in\mathbb{R}^{k}$ 无约束，方程 [29](https://arxiv.org/html/2602.20152v1#A2.E29) 就是经典的通用逼近类。
如果改为施加分量约束 $\lambda_{0}\geq\mathbf{0}$，由于 $\tanh$ 是奇函数，表达能力得以保留：
对于任意标量 $a\in\mathbb{R}$，将其写为 $a=a^{+}-a^{-}$，其中 $a^{\pm}\geq 0$，并注意
$a\tanh(h)=a^{+}\tanh(h)+a^{-}\tanh(-h).$
由于当 $h$ 是仿射时 $-h$ 也是仿射的，负系数可以通过复制隐藏单元并保持相应的输出权重非负来实现。因此，在宽度上最多增加一个常数因子，该块类就包含了 $\tanh$ 单元的有符号线性组合。

**步骤 2 (目标能量的均匀逼近)** 。
根据具有非多项式激活函数（例如 $\tanh$）的单隐藏层网络的通用逼近定理（universal approximation theorem），
对于任意 $\delta>0$，存在一个宽度 $k$ 和参数 $\theta$，使得

$$
\sup_{(\mathbf{x},\mathbf{y})\in\mathcal{X}\times\mathcal{Y}}\Bigl|\mathcal{B}(\mathbf{x},\mathbf{y};\theta)-\tau f(\mathbf{x},\mathbf{y})\Bigr|<\delta.(30)
$$

定义 $g(\mathbf{x},\mathbf{y}):=\mathcal{B}(\mathbf{x},\mathbf{y};\theta)/\tau$ 和 $\eta:=\delta/\tau$。
那么方程 [30](https://arxiv.org/html/2602.20152v1#A2.E30) 等价于

$$
\sup_{(\mathbf{x},\mathbf{y})\in\mathcal{X}\times\mathcal{Y}}\bigl|g(\mathbf{x},\mathbf{y})-f(\mathbf{x},\mathbf{y})\bigr|<\eta.(31)
$$

**步骤 3（均匀 KL 控制（Uniform KL Control））** 。
对于每个 $\mathbf{x}\in\mathcal{X}$，定义

$$
q(\mathbf{y}\mid\mathbf{x}):=\frac{\exp\bigl(g(\mathbf{x},\mathbf{y})\bigr)}{\int_{\mathcal{Y}}\exp\bigl(g(\mathbf{x},\mathbf{y}^{\prime})\bigr)\,d\mathbf{y}^{\prime}}。(32)
$$

方程 [32] 中的归一化因子是有限的，因为 $g$ 是连续的且 $\mathcal{Y}$ 是紧致的。
令
$Z_{g}(\mathbf{x}):=\int_{\mathcal{Y}}\exp\bigl(g(\mathbf{x},\mathbf{y}^{\prime})\bigr)\,d\mathbf{y}^{\prime}$。
由于 $p^{\star}(\cdot\mid\mathbf{x})$ 是一个密度函数，不失一般性，我们可以对能量进行归一化，使得 $\int_{\mathcal{Y}}e^{f(\mathbf{x},\mathbf{y}^{\prime})}d\mathbf{y}^{\prime}=1$。
由方程 [31]，对所有 $(\mathbf{x},\mathbf{y})$ 有

$$
e^{-\eta}\leq\frac{e^{g(\mathbf{x},\mathbf{y})}}{e^{f(\mathbf{x},\mathbf{y})}}\leq e^{\eta}。
$$

对 $\mathbf{y}\in\mathcal{Y}$ 积分得到

$$
e^{-\eta}\leq Z_{g}(\mathbf{x})\leq e^{\eta},\qquad\text{因此}\qquad|\log Z_{g}(\mathbf{x})|\leq\eta,\quad\forall\,\mathbf{x}\in\mathcal{X}。(33)
$$

此外，

$$
\log\frac{p^{\star}(\mathbf{y}\mid\mathbf{x})}{q(\mathbf{y}\mid\mathbf{x})}=\log\frac{e^{f(\mathbf{x},\mathbf{y})}}{e^{g(\mathbf{x},\mathbf{y})}/Z_{g}(\mathbf{x})}=\bigl(f(\mathbf{x},\mathbf{y})-g(\mathbf{x},\mathbf{y})\bigr)+\log Z_{g}(\mathbf{x})。
$$

在 $p^{\star}(\cdot\mid\mathbf{x})$ 下取期望，并利用方程 [31] 和方程 [33] 可得

$$
\displaystyle\mathrm{KL}\!\left(p^{\star}(\cdot\mid\mathbf{x})\,\big\|\,q(\cdot\mid\mathbf{x})\right) $\displaystyle=\mathbb{E}_{p^{\star}(\cdot\mid\mathbf{x})}\!\left[f(\mathbf{x},\mathbf{Y})-g(\mathbf{x},\mathbf{Y})\right]+\log Z_{g}(\mathbf{x})$ $\displaystyle\leq\eta+\eta=2\eta,\qquad\forall\,\mathbf{x}\in\mathcal{X}。$ (34)
$$

**步骤 4（选择 $\delta$ 并嵌入到 BL 中）** 。
选择 $\delta:=\varepsilon\tau/4$，使得 $\eta=\delta/\tau=\varepsilon/4$。
那么方程 [34] 意味着

$$
\sup_{\mathbf{x}\in\mathcal{X}}\mathrm{KL}\!\left(p^{\star}(\cdot\mid\mathbf{x})\,\big\|\,q(\cdot\mid\mathbf{x})\right)\leq 2\eta=\varepsilon/2<\varepsilon。
$$

最后，密度 $q(\cdot\mid\mathbf{x})$ 等于 **吉布斯分布（Gibbs distribution）** 方程 [26]，其能量为
$\mathrm{BL}_{\theta^{\star}}(\mathbf{x},\mathbf{y}):=\mathcal{B}(\mathbf{x},\mathbf{y};\theta)$（一个包含单个块的有限 BL 架构），
温度为 $\tau$。
这证明了该论断。
∎

###### 证明。

### B.3 可识别行为学习（Identifiable Behavior Learning, IBL）

#### B.3.1 设置与假设

##### 输入-输出空间与数据。

令 $\mathcal{X}\subset\mathbb{R}^{d_{x}}$ 和 $\mathcal{Y}\subset\mathbb{R}^{d_{y}}$ 为紧致集。
假设数据分布 $P_{X,Y}$ 支撑在 $\mathcal{X}\times\mathcal{Y}$ 上，并且在其支撑的内部存在一点 $z_{0}=(x_{0},y_{0})$；也就是说，$z_{0}$ 的某个开邻域具有正的 $P_{X,Y}$ 测度。
除非另有说明，所有期望均关于 $P_{X,Y}$ 计算。

##### 参数空间与多项式特征映射。

参数空间可分解为

$$
\Theta\;:=\;\Theta_{U}\times\Theta_{C}\times\Theta_{T}\times\mathcal{W}_{\circ}。
$$

对于 $\theta_{U}\in\Theta_{U}$、$\theta_{C}\in\Theta_{C}$ 和 $\theta_{T}\in\Theta_{T}$，
我们定义多项式特征映射

$$
p_{u}:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}^{d_{u}},\quad p_{c}:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}^{d_{c}},\quad p_{t}:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}^{d_{t}},
$$

每个映射具有固定的次数，并且在其系数上是单射的（即，不同的系数产生不同的函数）。对于单个块，$\theta_{U},\theta_{C},\theta_{T}$ 对应于 $U$、$C$ 和 $T$ 项的参数及其各自的外部乘子（例如，惩罚权重 $\lambda$）。
对于由多个块组成的深度网络，$\theta=(\theta_{U},\theta_{C},\theta_{T})$ 表示跨层次的所有块级参数的集合，其中 $\theta_{U}$ 聚合所有 $U$ 项的参数，$\theta_{C}$ 聚合所有 $C$ 项的参数，$\theta_{T}$ 聚合所有 $T$ 项的参数（每个都包括其关联的乘子）。

输出分量 $\mathcal{W}_{\circ}$ 对应于最终层的仿射变换：
对于单输出预测，$\mathcal{W}_{\circ}=\mathbb{R}^{d^{\prime}}$，
对于 $m$ 路分类，$\mathcal{W}_{\circ}=\mathbb{R}^{d^{\prime}\times m}$，
其中 $d^{\prime}$ 是由前面的网络（无论是浅层还是深层）诱导的输出维度。

##### **可识别基础模块（Identifiable base block）**

令 $\lambda_{0}\in\mathbb{R}^{d_{u}}$、$\lambda_{1}\in\mathbb{R}^{d_{c}}$ 和 $\lambda_{2}\in\mathbb{R}^{d_{t}}$ 表示非负权重向量，它们被视为可学习参数。我们将可识别模块化块实例化为：

$$
\mathcal{B}^{\mathrm{id}}(x,y;\theta)\;=\;\lambda_{0}^{\!\top}\tanh\!\bigl(p_{u}(x,y)\bigr)\;-\;\lambda_{1}^{\!\top}\mathrm{softplus}\!\bigl(p_{c}(x,y)\bigr)\;-\;\lambda_{2}^{\!\top}\bigl(p_{t}(x,y)\bigr)^{\odot 2},(35)
$$

其中 $(\cdot)^{\odot 2}$ 表示逐元素平方。根据构造，$\tanh$ 和 $\mathrm{softplus}$ 头在其参数上是严格单调的，而二次头是偶函数。

我们假设每个多项式特征映射 $\mathbf{p}_{\bullet}(x,y)$ 不包含任何与 $y$ 无关的非零单项式；也就是说，没有特征是 $x$ 的纯函数或常数。这确保了除非所有权重都为零，否则 $\mathcal{B}^{\mathrm{id}}(x,y)$ 在 $y$ 上是非恒定的。

##### **架构（Architectures）**

我们以三种架构形式实现 IBL（Identifiable Basis Learning），每种都产生一个关于 $(x,y)$ 的组合效用函数。

- **IBL(Single)** ：
  使用单个块作为组合效用函数，
  $\mathrm{IBL}(x,y):=\mathcal{B}^{\mathrm{id}}(x,y).$
- **IBL(Shallow)** ：
  浅层 IBL 使用一层或两层堆叠的并行块。例如，第一层
  $\mathbb{B}^{\mathrm{id}}_{1}(x,y):=[\,\mathcal{B}^{\mathrm{id}}_{{1,1}}(x,y),\ldots,\mathcal{B}^{\mathrm{id}}_{{1,d_{1}}}(x,y)\,]^{\top}\in\mathbb{R}^{d_{1}}$
  输入到一个无偏置的仿射映射
  $\mathrm{IBL}_{\text{Shallow}}(x,y):=\mathbf{W}^{\circ}_{1}\,\mathbb{B}^{\mathrm{id}}_{1}(x,y),$
  其中对于分类任务 $\mathbf{W}^{\circ}_{1}\in\mathbb{R}^{m\times d_{1}}$，对于标量输出 $\mathbf{W}^{\circ}_{1}\in\mathbb{R}^{1\times d_{1}}$。
- **IBL(Deep)** ：
  深层 IBL 将构造扩展到深度 $L>2$，递归定义为
  $\mathrm{IBL}(x,y):=\mathbf{W}^{\circ}_{L}\cdot\mathbb{B}^{\mathrm{id}}_{L}\big(\cdots\,\mathbb{B}^{\mathrm{id}}_{2}(\mathbb{B}^{\mathrm{id}}_{1}(x,y))\cdots\big),$
  其中每个 $\mathbb{B}^{\mathrm{id}}_{\ell}$ 堆叠了并行块 $\mathcal{B}^{\mathrm{id}}_{{\ell,i}}(x,y)$，而 $\mathbf{W}^{\circ}_{L}$ 是一个无偏置的仿射变换。当 $L=1$ 和 $L=2$ 时，分别对应恢复 Single 和 Shallow 架构。

##### **诱导的条件模型（Induced conditional model）**

令 $\mathrm{IBL}(x,y)$ 表示由所选架构（Single、Shallow 或 Deep）产生的组合效用函数。它诱导出条件吉布斯分布：

$$
\displaystyle\text{(离散 $y\in[m]$)}\qquad p(y\mid x)$ $\displaystyle=\mathrm{softmax}_{y}\{\mathrm{IBL}(x,y)\},$ (36) $\displaystyle\text{(连续 $y$)}\qquad p(y\mid x)$ $\displaystyle=\frac{\exp\{\mathrm{IBL}(x,y)/\tau\}}{\int_{\mathcal{Y}}\exp\{\mathrm{IBL}(x,\tilde{y})/\tau\}\,d\tilde{y}},\quad\tau>0\text{ 固定}.$ (37)
$$

这里 $\tau$ 是一个固定的温度参数。因此，IBL 通过定义一个组合效用景观来进行预测，该景观的吉布斯分布决定了给定 $x$ 时的 $y$。

##### **商参数空间（Quotient parameter space）**

###### **定义 B.1（对称商空间，Symmetry Quotient Space）**

在 $\Theta$ 上定义等价关系 $\sim$ 为满足以下条件的最小关系：

$$
\theta_{t}\sim\theta_{t}^{\prime}\quad\Longleftrightarrow\quad p_{t}^{(i)}(x,y;\theta_{t}^{(i)})^{\odot 2}=p_{t}^{(i)}(x,y;\theta_{t}^{\prime(i)})^{\odot 2}\quad\text{对所有 }i\text{ 和 }(x,y)\text{ 成立}.
$$

对应的商空间为：

$$
\bar{\Theta}:=\Theta/\!\sim.
$$

**解释** ：
$T$ 分量旨在编码等式约束，这些约束在符号上是方程。翻转此类约束的整体符号不会改变方程，因此仅符号不同的不同参数化应被视为等价。

###### **定义 B.2（尺度不变商空间，Scale-Invariant Quotient Space）**

在 $\bar{\Theta}$ 上通过下式定义等价关系 $\approx$：

$$
\bar{\theta}\approx\bar{\theta}^{\prime}\quad\Longleftrightarrow\quad\exists\,c>0\;\text{ 使得 }\;\mathsf{s}(x,y;\bar{\theta})=c\,\mathsf{s}(x,y;{\bar{\theta}^{\prime}}).
$$

尺度不变商空间则由下式给出：

$$
\widetilde{\Theta}:=\bar{\Theta}/\!\approx.
$$

**解释** ：
在分类中，预测仅取决于候选标签之间相对的组合效用差异。从技术角度来看，商掉全局偏移或均匀缩放是必要的：如果没有这种识别，交叉熵损失会允许仅因此类变换而不同的冗余参数化。同时，这个商是自然且无害的，因为它不会消除类别之间信息丰富的比率，而只是丢弃在 softmax 决策规则中不起作用的绝对水平或尺度。

###### 定义 B.2 （尺度不变商空间， Scale-Invariant Quotient Space）

##### 损失函数（Loss Functions）

我们采用混合损失函数，以同时适应离散和连续输出。
具体而言， **交叉熵（Cross-Entropy, CE）** 应用于离散目标，而 **去噪分数匹配（Denoising Score Matching, DSM）** 应用于连续目标。
令 $\gamma_{c},\gamma_{d}\geq 0$ 且 $\gamma_{c}+\gamma_{d}>0$。
定义在商参数空间上的总体风险为：

$$
\mathcal{M}(\bar{\theta})\;=\;\gamma_{d}\,\mathbb{E}\!\left[-\log p_{\theta}(Y\mid X)\right]\;+\;\gamma_{c}\,\mathbb{E}\!\left[\mathcal{S}_{\mathrm{DSM}}(\theta;X)\right],\qquad\theta\in\pi^{-1}(\bar{\theta}),(38)
$$

其中 $\pi$ 表示从原始参数空间到其商空间的 **典范投影（canonical projection）** 。

对于连续输出 $Y\in\mathcal{Y}\subseteq\mathbb{R}^{d_{y}}$，
DSM 通过向目标添加高斯噪声 $\tilde{Y}=Y+\varepsilon$， $\varepsilon\sim\mathcal{N}(0,\sigma^{2}I)$ 来实现，
并惩罚模型分数与相应去噪分数之间的平方差异：

$$
\mathcal{S}_{\mathrm{DSM}}(\theta;X)=\frac{1}{2\sigma^{2}}\,\mathbb{E}_{\varepsilon}\!\left[\Bigl\|\nabla_{\tilde{y}}\log p_{\theta}(\tilde{y}\mid X)+\tfrac{1}{\sigma^{2}}(Y-\tilde{Y})\Bigr\|^{2}\,\Big|\,X,Y\right].(39)
$$

在纯分类设置中，我们令 $\gamma_{c}=0$（纯 CE），
而在纯回归设置中，我们令 $\gamma_{d}=0$（纯 DSM）。

对于单个观测 $Z=(X,Y)$，我们定义*单样本损失（per-sample loss）*为：

$$
\ell(\theta;Z):=\gamma_{d}\,\big[-\log p_{\theta}(Y\mid X)\big]+\gamma_{c}\,\mathcal{S}_{\mathrm{DSM}}(\theta;X).(40)
$$

经验准则则采用标准的 $M$ 估计形式：

$$
\hat{Q}_{n}(\theta)\;=\;\frac{1}{n}\sum_{i=1}^{n}\ell(\theta;Z_{i}),\qquad Z_{i}=(X_{i},Y_{i}).(41)
$$

##### 关键假设（Key Assumptions）

###### 假设 B.1 （全局原子独立性与单射性， Global Atomic Independence and Injectivity）

令 $\bar{\Psi}$ 为原子参数商空间。

1.  **商空间上的单射性（Injectivity on the quotient）** 。映射 $\bar{\Psi}\to\mathbb{R}^{\mathcal{X}\times\mathcal{Y}}$， $\bar{\psi}\mapsto g_{\bar{\psi}}$ 是单射。
2.  **线性独立性（Linear Independence）** 。原子线性独立性。任何由 $\bar{\psi}_{i}\in\bar{\Psi}$ 构成的有限个两两不同的原子集合 $\{g_{\bar{\psi}_{i}}\}_{i=1}^{r}$ 在 $\mathbb{R}^{\mathcal{X}\times\mathcal{Y}}$ 中是线性独立的。
3.  **极小性（Minimality）** 。在所有模型实例中，我们只考虑极小表示：没有重复原子，且其在混合中的对应线性系数非零。
4.  **典范排序（Canonical ordering）** 。对于每个模型实例，原子列表都施加一个固定的典范排序。

**解释（Explanation）** 。
假设 [B.1] 将每个可识别块 $\mathcal{B}^{\mathrm{id}}$ 视为一个*原子*构建单元，并对由这些原子构建的表示施加了四个结构性要求。这四个条件共同定义了一个无歧义、无冗余且典范的原子代数：在通过自然对称性进行商化之后，由 $\mathcal{B}$ 块构建的每个模型都承认一个唯一的极小表示（在规定的等价关系下）。这种结构规律性是构建可识别性陈述的基础：它保证了原则上，通过观察模型输出（或其优化的目标），可以在适当的商意义下恢复底层的原子成分及其系数。

**实践备注（Practical remark）** 。
在实践中，可以通过两种互补的方式来鼓励或近似强制执行这些条件。首先，原子类的设计（多项式基、交互项和激活头的选择）可以使得单射性和线性独立性在构造上更合理。其次，可以在训练后应用模型选择和后处理（例如，剪枝接近零系数的原子，强制执行确定性的平局决胜规则进行排序）来实现极小性和典范排序。这些实际措施使得理论假设在经验应用中具有可操作性意义。

###### 假设 B.1 （全局原子独立性与单射性， Global Atomic Independence and Injectivity）

#### B.3.2 定理证明（Proof of Theorems）

###### 引理 B.1 （线性组合的可识别性， Identifiability of Linear Combinations）

令 $Z$ 为一个集合。
对于每个 $j=1,\dots,m$，令 $\Phi_{j}$ 为一个参数空间，并定义原子函数：

$$
g_{\psi}:=f(\cdot;\phi_{j}),\qquad\psi=(j,\phi_{j})\in\Psi,
$$

其中 $\Psi:=\bigsqcup_{j=1}^{m}\Phi_{j}$ 是 **不交并（disjoint union）** 。
令 $\bar{\Psi}$ 为商原子参数空间，其元素记为 $\bar{\psi}\in\bar{\Psi}$。

定义模型的商参数空间为：

$$
\bar{\Xi}:=\prod_{j=1}^{m}\bigl((\mathbb{R}\setminus\{0\})\times\bar{\Psi}\bigr),\qquad\bar{\xi}=((a_{1},\bar{\psi}_{1}),\dots,(a_{m},\bar{\psi}_{m})).
$$

其对应的线性组合模型为

$$
S_{\bar{\xi}}:=\sum_{j=1}^{m}a_{j}g_{\bar{\psi}_{j}}.
$$

根据假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1)，该模型在商参数空间 $\bar{\Xi}$ 中是可识别的：如果在 $Z$ 上 $S_{\bar{\xi}}\equiv S_{\bar{\xi}^{\prime}}$，则 $\bar{\xi}=\bar{\xi}^{\prime}$。

###### 证明。

假设在 $Z$ 上 $S_{\bar{\xi}}\equiv S_{\bar{\xi}^{\prime}}$，即

$$
\sum_{j=1}^{m}a_{j}\,g_{(j,\phi_{j})}-\sum_{j=1}^{m}a^{\prime}_{j}\,g_{(j,\phi^{\prime}_{j})}\equiv 0.
$$

令 $\mathcal{U}$ 为商空间 $\bar{\Psi}$ 中出现在等式任一侧的 **不同** 原子（Atom）的集合，并对每个 $\bar{\psi}\in\mathcal{U}$，令

$$
\beta(\bar{\psi}):=\sum_{\begin{subarray}{c}j:\,[j,\phi_{j}]=\bar{\psi}\end{subarray}}a_{j}\;-\!\!\!\!\!\sum_{\begin{subarray}{c}j^{\prime}:\,[j^{\prime},\phi^{\prime}_{j^{\prime}}]=\bar{\psi}\end{subarray}}a^{\prime}_{j^{\prime}}
$$

为 $g_{\bar{\psi}}$ 的净系数。于是有

$$
\sum_{\bar{\psi}\in\mathcal{U}}\beta(\bar{\psi})\,g_{\bar{\psi}}\;\equiv\;0.
$$

根据 $\bar{\Psi}$ 中两两不同原子的 **线性无关性** 条件（假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1):2），我们必须有对所有 $\bar{\psi}\in\mathcal{U}$，$\beta(\bar{\psi})=0$。

此外，根据 **极小性** 要求（假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1):3），每个 $\bar{\psi}$ 在等式两侧恰好各出现一次，且系数非零。
因此，等式两侧必须包含完全相同的系数-原子对列表 $\{(a_{j},\bar{\psi}_{j})\}_{j=1}^{m}$，并且由于施加了规范排序（假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1):4），由此可得

$$
\bar{\xi}=\bar{\xi}^{\prime}.
$$

∎

###### 定理 B.1 (IBL(Single) 的可识别性) 。

IBL(Single) 架构使用的原子集为

$$
\bigl\{\,\tanh(p_{u,i}),\ \operatorname{softplus}(p_{c,i}),\ (p_{t,i})^{2}:\ i=1,\dots,d_{u};\ i=1,\dots,d_{c};\ i=1,\dots,d_{t}\,\bigr\}.
$$

在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，该模型在商空间 $\bar{\Theta}$ 中是可识别的：如果在 $\mathcal{X}\times\mathcal{Y}$ 上 $\mathcal{B}^{\mathrm{id}}_{\theta}\equiv\mathcal{B}^{\mathrm{id}}_{\theta^{\prime}}$，则在 $\bar{\Theta}$ 中 $\theta=\theta^{\prime}$。

###### 证明。

记

$$
\mathcal{B}^{\mathrm{id}}_{\theta}=\sum_{j=1}^{m}a_{j}\,f(\cdot;\phi_{j}),\qquad m:=d_{u}+d_{c}+d_{t},
$$

其中每个 $f(\cdot;\phi_{j})$ 是原子 $\tanh(p_{u,i})$、$\operatorname{softplus}(p_{c,i})$ 或 $(p_{t,i})^{2}$ 中的一个，$a_{j}$ 是 $(\lambda_{0},\lambda_{1},\lambda_{2})$ 中对应的条目，且所有索引采用固定排序。

如果在 $\mathcal{X}\times\mathcal{Y}$ 上 $\mathcal{B}^{\mathrm{id}}_{\theta}\equiv\mathcal{B}^{\mathrm{id}}_{\theta^{\prime}}$，那么引理 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmlemma1) 和假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 意味着所有原子和系数在商原子空间 $\bar{\Psi}$ 中必须一致。
由于排序是固定的，这意味着在 $\bar{\Theta}$ 中 $\theta=\theta^{\prime}$。
∎

###### 定理 B.2 (IBL(Shallow) 的可识别性) 。

IBL(Shallow) 架构使用的原子集为

$$
\bigl\{\,\mathcal{B}^{\mathrm{id}}_{\theta_{1,j}}(x,y)\,\bigr\}_{j=1}^{d_{1}},
$$

其中每个 $\mathcal{B}^{\mathrm{id}}_{\theta_{1,j}}:\mathcal{X}\times\mathcal{Y}\to\mathbb{R}$ 是一个由 $\theta_{1,j}\in\Theta_{1}$ 参数化的单块 **可识别块学习（Identifiable Block Learning, IBL）** 模块。完整参数记为

$$
\theta:=\bigl((\theta_{1,1},\dots,\theta_{1,d_{1}}),\ \mathbf{W}_{1}^{\circ}\bigr)\in\Theta:=(\Theta_{1})^{d_{1}}\times\mathbb{R}^{m\times d_{1}}.
$$

在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，映射 $\theta\mapsto\mathrm{IBL}_{\text{Shallow}}$ 在商空间 $\bar{\Theta}$ 中是 **可识别的（Identifiable）** ：
如果

$$
\mathrm{IBL}_{\text{Shallow}}(x,y;\theta)\equiv\mathrm{IBL}_{\text{Shallow}}(x,y;\theta^{\prime})\quad\text{在 }\mathcal{X}\times\mathcal{Y} \text{ 上成立},
$$

那么

$$
\theta=\theta^{\prime}\quad\text{在 }\bar{\Theta} \text{ 中}.
$$

###### 证明。

将第 $k$ 个输出分量写为原子（Atoms）的线性组合：

$$
s^{(k)}_{\theta}(x,y)=\sum_{j=1}^{d_{1}}w^{(k)}_{j}\,\mathcal{B}^{\mathrm{id}}_{\theta_{1,j}}(x,y),\qquad k=1,\ldots,m,
$$

其中 $w^{(k)}_{j}$ 表示 $\mathbf{W}_{1}^{\circ}$ 的第 $(k,j)$ 个元素。

假设两个参数元组 $(\mathbf{W}_{1}^{\circ},\{\theta_{1,j}\}_{j=1}^{d_{1}})$ 和 $(\mathbf{W}_{1}^{\circ\,\prime},\{\theta^{\prime}_{1,j}\}_{j=1}^{d_{1}})$ 在 $\mathcal{X}\times\mathcal{Y}$ 上产生相同的向量得分。
那么对于每个 $k$，我们在 $\mathcal{X}\times\mathcal{Y}$ 上有 $s^{(k)}_{\theta}\equiv s^{(k)}_{\theta^{\prime}}$。

固定任意 $k$。在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，引理 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmlemma1) 确保了系数-原子对 $\{(w^{(k)}_{j},\,\mathcal{B}^{\mathrm{id}}_{\theta_{1,j}})\}_{j=1}^{d_{1}}$ 是唯一确定的（在商空间 $\bar{\Theta}$ 中的等价意义下）。特别地，对于每个 $j=1,\dots,d_{1}$，我们必须有

$$
w^{(k)}_{j}=w^{\prime(k)}_{j},\qquad\mathcal{B}^{\mathrm{id}}_{\theta_{1,j}}\equiv\mathcal{B}^{\mathrm{id}}_{\theta^{\prime}_{1,j}}.
$$

由于这对所有 $k=1,\dots,m$ 都成立，因此可以推出 $\mathbf{W}_{1}^{\circ}=\mathbf{W}_{1}^{\circ\,\prime}$，并且在商参数空间中，对所有 $j$ 有 $\theta_{1,j}=\theta^{\prime}_{1,j}$。

因此，在 $\bar{\Theta}$ 中 $\theta=\theta^{\prime}$，从而在固定排序下建立了完全可识别性。
∎

###### **定理 B.3（IBL(Deep) 的可识别性）** 。

固定整数 $L>2$ 和宽度 $d_{1},\dots,d_{L-1}$。
**IBL(Deep)** 架构使用最后一层的原子集

$$
\bigl\{\,\mathcal{B}^{\mathrm{id}}_{\vartheta_{L,j}}(x,y)\,\bigr\}_{j=1}^{d_{L}}\subset\mathbb{R}^{\mathcal{X}\times\mathcal{Y}},
$$

其中每个 $\mathcal{B}^{\mathrm{id}}_{\vartheta_{L,j}}:\mathbb{R}^{d_{L-1}}\to\mathbb{R}$ 是一个应用于第 $L{-}1$ 层输出的标量值块。只有第一层的块（$\ell=1$）是如定理 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem1) 中所述的 **IBL(Single)** 模块。对于具有 **跳跃连接（Skip Connections）** 的架构，最后一层的原子可以扩展以包含跳跃的特征（例如，来自更早的层），这些特征被视为 $\bigl\{\,\mathcal{B}^{\mathrm{id}}_{\vartheta_{L,j}}(x,y)\,\bigr\}_{j=1}^{d_{L}}$ 的元素。

完整参数为

$$
\theta:=\bigl(\{\vartheta_{\ell,j}\}_{\ell=1,j=1}^{L,d_{\ell}},\ \mathbf{W}_{\text{out}}\bigr)\in\Theta:=\prod_{\ell=1}^{L}(\Theta_{1})^{d_{\ell}}\times\mathbb{R}^{m\times d_{L}}.
$$

在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，映射 $\theta\mapsto\mathrm{IBL}_{\text{Deep}}(x,y;\theta)$ 在商空间 $\bar{\Theta}$ 中是可识别的。

###### 证明。

在给定的架构下， **IBL(Deep)** 模型最终呈现为以下形式

$$
s^{(k)}(x,y)\;=\;\sum_{j=1}^{d_{L}}w^{(k)}_{j}\,\mathcal{B}^{\mathrm{id}}_{\vartheta_{L,j}}(x,y),\qquad k=1,\dots,m,
$$

其中，每个 $\mathcal{B}^{\mathrm{id}}_{\vartheta_{L,j}}$ 都是应用于前一层输出的标量值函数。
通过将集合 $\{\mathcal{B}^{\mathrm{id}}_{\vartheta_{L,j}}(x,y)\}_{j=1}^{d_{L}}$ 视为原子集，我们将模型简化为 **IBL(Shallow)** 形式：

$$
\mathbf{s}(x,y)=\mathbf{W}_{\text{out}}\,\mathbf{B}_{L}(x,y).
$$

在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，定理 [B.2](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem2) 适用，这意味着完整参数 $\theta=(\{\vartheta_{\ell,j}\}_{\ell,j},\,\mathbf{W}_{\text{out}})$ 在商空间 $\bar{\Theta}$ 中是可识别的。
∎

**定理 2.4（Identifiability of IBL）** 。
在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，架构 **IBL(Single)** 、 **IBL(Shallow)** 和 **IBL(Deep)** 在商空间 $\bar{\Theta}$ 中都是可识别的。

###### 证明。

直接由定理 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem1)、[B.2](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem2) 和 [B.3](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem3) 得出。
∎

**定理 2.5（Loss Identifiability of IBL）** 。
令 $\mathrm{IBL}_{\theta}(x,y)$ 表示一个 **IBL** 模型，并考虑条件吉布斯分布（Conditional Gibbs distribution）

$$
p_{\theta}(y\mid x)=\frac{\exp\!\big(\mathrm{IBL}_{\theta}(x,y)\big)}{\int_{\mathcal{Y}}\exp\!\big(\mathrm{IBL}_{\theta}(x,y^{\prime})\big)\,dy^{\prime}}.
$$

如方程 [38](https://arxiv.org/html/2602.20152v1#A2.E38) 所定义，在对称商 $\bar{\Theta}$ 上定义总体风险（Population risk）。
假设参数空间 $\Theta$ 是紧致的。那么，在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，以下结论成立：

- (i)
  如果 $\gamma_{c}>0$，则风险泛函 $\mathcal{M}$ 在 $\bar{\Theta}$ 中存在唯一最小化元。此外，
  $\mathcal{M}(\bar{\theta}_{1})=\mathcal{M}(\bar{\theta}_{2})\;\;\Longrightarrow\;\;\bar{\theta}_{1}=\bar{\theta}_{2}.$
- (ii)
  如果 $\gamma_{c}=0$，则风险泛函 $\mathcal{M}$ 在尺度不变商 $\widetilde{\Theta}$ 中存在唯一最小化元。此外，
  $\mathcal{M}(\widetilde{\theta}_{1})=\mathcal{M}(\widetilde{\theta}_{2})\;\;\Longrightarrow\;\;\widetilde{\theta}_{1}=\widetilde{\theta}_{2}.$

###### 证明。

在假设 [B.1](https://arxiv.org/html/2602.20152v1#A2.Thmassumption1) 下，如定理 [2.4](https://arxiv.org/html/2602.20152v1#S2.Thmtheorem4) 所确立， **IBL** 架构在由 $\bar{\Theta}$ 定义的对称群（Symmetry group）意义下是可识别的。令 $\theta^{\bullet}\in\arg\min_{\theta\in\Theta}\mathcal{M}(\theta)$ 并设 $p^{\star}(\cdot\mid x):=p_{\theta^{\bullet}}(\cdot\mid x)$。由于 $\Theta$ 是紧致的且损失 $\mathcal{M}$ 是连续的，因此存在全局最小化元。我们证明它在所述的商中是唯一的。

_情况 $\gamma_{c}>0$_。
在任何最小化元处，我们同时有 $p_{\theta}(\cdot\mid x)=p^{\star}(\cdot\mid x)$ 以及几乎处处（a.e.）成立 $\nabla_{y}\log p_{\theta}(\cdot\mid x)=\nabla_{y}\log p^{\star}(\cdot\mid x)$。
由于

$$
\nabla_{y}\log p_{\theta}(y\mid x)=\nabla_{y}\mathrm{IBL}_{\theta}(x,y)-\nabla_{y}\log Z_{\theta}(x)=\nabla_{y}\mathrm{IBL}_{\theta}(x,y),
$$

（配分函数（Partition function） $Z_{\theta}(x)$ 与 $y$ 无关），得分等式（Score equality）得出几乎处处成立 $\nabla_{y}\big(\mathrm{IBL}_{\theta}-\mathrm{IBL}_{\theta^{\bullet}}\big)(y;x)=0$。 **IBL** 不包含与 $y$ 无关的项。因此，

$$
\mathrm{IBL}_{\theta}(x,y)=\mathrm{IBL}_{\theta^{\bullet}}(x,y)\quad\text{几乎处处成立。}
$$

由定理 [2.4](https://arxiv.org/html/2602.20152v1#S2.Thmtheorem4)（在 $\bar{\Theta}$ 中的可识别性），最小化元在 $\bar{\Theta}$ 中是唯一的；特别地，

$$
\mathcal{M}(\bar{\theta}_{1})=\mathcal{M}(\bar{\theta}_{2})\ \Longrightarrow\ \bar{\theta}_{1}=\bar{\theta}_{2}.
$$

_情况 $\gamma_{c}=0$_。
此时，$\mathcal{M}$ 简化为交叉熵风险（Cross-entropy risk），当且仅当几乎处处成立 $p_{\theta}(\cdot\mid x)=p^{\star}(\cdot\mid x)$ 时，该风险被最小化。交叉熵损失仅通过 $\mathrm{IBL}_{\theta}(x,y)$ 在不同 $y$ 之间的相对值来依赖它，并且对组合效用（Compositional utility）的加性平移和正缩放具有不变性。因此，损失仅依赖于等价类 $\widetilde{\theta}\in\widetilde{\Theta}$。因此，

$$
\mathcal{M}(\widetilde{\theta}_{1})=\mathcal{M}(\widetilde{\theta}_{2})\quad\Longrightarrow\quad\widetilde{\theta}_{1}=\widetilde{\theta}_{2}.
$$

即，最小化元在 $\widetilde{\Theta}$ 中是唯一的。

因此，最小化元在所述的商空间中是唯一的。证明完成。
∎

###### **定理 B.4（Uniform M-estimation consistency (Newey & McFadden, 1994 , Theorem 2.1)）** 。

令 $(\mathcal{A},d)$ 为一个紧致度量空间（Compact metric space），并令 $\widehat{L}_{n}:\mathcal{A}\to\mathbb{R}$ 为一列随机目标函数（Random objective functions），其总体目标函数（Population objective function）为 $L:\mathcal{A}\to\mathbb{R}$，满足：

1.  $L(\alpha)$ 在 $\alpha^{\star}\in\mathcal{A}$ 处被唯一最小化；
2.  $\mathcal{A}$ 是紧致的；
3.  $L(\alpha)$ 是连续的；
4.  $\widehat{L}_{n}(\alpha)\xrightarrow{p}L(\alpha)$ 在 $\alpha\in\mathcal{A}$ 上一致成立。

那么，任何序列 $\hat{\alpha}_{n}\in\arg\min_{\alpha\in\mathcal{A}}\widehat{L}_{n}(\alpha)$ 都满足 $\hat{\alpha}_{n}\xrightarrow{p}\alpha^{\star}$。

###### **定理 B.5（IBL 的一致性（Consistency of IBL））** 。

令 $\mathcal{M}$ 为方程 [38] 中定义的 **总体风险（Population risk）** ，并令 $\mathcal{M}_{n}$ 表示其 **经验对应项（Empirical analogue）** 。
假设：

1.  $\{(X_{i},Y_{i})\}_{i=1}^{n}$ 是独立同分布（Independent and Identically Distributed, i.i.d.）样本；
2.  $\Theta$ 是 **紧集（Compact set）** ；
3.  $\theta\mapsto\mathcal{M}(\theta)$ 是连续的，且损失类允许一个 **可积包络（Integrable envelope）** ，使得
    $\sup_{\theta\in\Theta}\big|\mathcal{M}_{n}(\theta)-\mathcal{M}(\theta)\big|\;\xrightarrow{p}\;0;$

令 $\Xi$ 表示相关的 **商空间（Quotient space）** （若 $\gamma_{c}>0$ 则为 $\bar{\Theta}$，若 $\gamma_{c}=0$ 则为 $\widetilde{\Theta}$），并令 $\hat{\theta}_{n}\in\arg\min_{\theta\in\Theta}\mathcal{M}_{n}(\theta)$ 和 $\theta^{\bullet}\in\arg\min_{\theta\in\Theta}\mathcal{M}(\theta)$。那么

$$
\hat{\theta}_{n}\;\xrightarrow{p}\;\theta^{\bullet}\quad\text{在 }\Xi \text{ 中},\qquad\mathcal{M}(\hat{\theta}_{n})\;\xrightarrow{p}\;\mathcal{M}(\theta^{\bullet}).
$$

如果模型是 **正确设定的（Correctly specified）** （数据分布由某个 $\theta^{\star}\in\Theta$ 实现），那么在 $\Xi$ 中 $\theta^{\bullet}=\theta^{\star}$，因此 $\hat{\theta}_{n}\xrightarrow{p}\theta^{\star}$。

###### **证明** 。

令 $\Xi$ 表示相关的商空间：若 $\gamma_{c}>0$ 则 $\Xi=\bar{\Theta}$，若 $\gamma_{c}=0$ 则 $\Xi=\widetilde{\Theta}$。令 $\pi:\Theta\to\Xi$ 为 **典范商映射（Canonical quotient map）** 。由于 $\Theta$ 是紧集且 $\pi$ 是连续且满射的，$\Xi$ 是紧集。根据假设，$\mathcal{M}$ 和 $\mathcal{M}_{n}$ 在相应的对称性下是 **不变的（Invariant）** ，因此它们可通过 $\pi$ 进行 **分解（Factor through）** ：

$$
\widetilde{\mathcal{M}}(\xi):=\mathcal{M}(\theta),\qquad\widetilde{\mathcal{M}}_{n}(\xi):=\mathcal{M}_{n}(\theta)\quad(\text{任意 }\theta\in\pi^{-1}(\xi)).
$$

由于 $\mathcal{M}$ 在 $\Theta$ 上连续，这些定义在 $\Xi$ 上是 **良定义的（Well-defined）** 且连续的。此外，

$$
\sup_{\xi\in\Xi}\big|\widetilde{\mathcal{M}}_{n}(\xi)-\widetilde{\mathcal{M}}(\xi)\big|\;\leq\;\sup_{\theta\in\Theta}\big|\mathcal{M}_{n}(\theta)-\mathcal{M}(\theta)\big|\;\xrightarrow{p}\;0,
$$

因此 **依概率一致收敛（Uniform convergence in probability）** 在 $\Xi$ 上成立。

根据 IBL 的 **损失可识别性（Loss Identifiability）** （定理 [2.5]），$\widetilde{\mathcal{M}}$ 具有唯一的 **最小化点（Minimizer）** $\xi^{\bullet}\in\Xi$。令 $\hat{\xi}_{n}\in\arg\min_{\xi\in\Xi}\widetilde{\mathcal{M}}_{n}(\xi)$（等价地，选择 $\hat{\theta}_{n}\in\arg\min_{\theta\in\Theta}\mathcal{M}_{n}(\theta)$ 并设 $\hat{\xi}_{n}=\pi(\hat{\theta}_{n})$）。那么定理 [B.4] 的条件在紧度量空间 $(\Xi,d)$ 上成立，因此

$$
\hat{\xi}_{n}\xrightarrow{p}\xi^{\bullet}.
$$

由于 $\widetilde{\mathcal{M}}$ 在 $\Xi$ 上连续，且 $\widetilde{\mathcal{M}}(\hat{\xi}_{n})=\mathcal{M}(\hat{\theta}_{n})$，对于任意代表元 $\theta^{\bullet}\in\pi^{-1}(\xi^{\bullet})$ 有 $\widetilde{\mathcal{M}}(\xi^{\bullet})=\mathcal{M}(\theta^{\bullet})$，我们也得到

$$
\mathcal{M}(\hat{\theta}_{n})=\widetilde{\mathcal{M}}(\hat{\xi}_{n})\xrightarrow{p}\widetilde{\mathcal{M}}(\xi^{\bullet})=\mathcal{M}(\theta^{\bullet}).
$$

如果模型是正确设定的（存在 $\theta^{\star}\in\Theta$ 诱导出数据分布），则 CE/DSM 项的 **严格适当性（Strict propriety）** 意味着商空间中的唯一最小化点是 $\theta^{\star}$ 所在的类；因此 $\hat{\theta}_{n}$ 在相应的商空间中依概率收敛于 $\theta^{\star}$。
∎

###### **定理 B.6（IBL 的通用逼近性（Universal Approximation of IBL））** 。

令 $\mathcal{X}\subset\mathbb{R}^{d}$ 和 $\mathcal{Y}\subset\mathbb{R}^{m}$ 为紧集，并令 $p^{\star}(y\mid x)$ 为任意连续条件密度，使得对所有 $(x,y)\in\mathcal{X}\times\mathcal{Y}$ 有 $p^{\star}(y\mid x)>0$。那么对于任意 $\tau>0$ 和 $\varepsilon>0$，存在一个有限的 IBL 架构（其深度和宽度取决于 $\varepsilon$）和一个参数 $\theta^{\star}$，使得 **吉布斯分布（Gibbs distribution）**

$$
p_{\tau}(y\mid x;\theta^{\star})=\frac{\exp\bigl(\text{IBL}_{\theta^{\star}}(x,y)/\tau\bigr)}{\int_{\mathcal{Y}}\exp\bigl(\text{IBL}_{\theta^{\star}}(x,y^{\prime})/\tau\bigr)\mathrm{d}y^{\prime}}(42)
$$

满足

$$
\sup_{x\in\mathcal{X}}\mathrm{KL}\bigl(p^{\star}(\cdot\mid x)\,\|\,p_{\tau}(\cdot\mid x;\theta^{\star})\bigr)<\varepsilon.(43)
$$

###### **证明** 。

论证遵循与定理 [2.3] 证明中相同的构造，仅因 IBL 参数化而进行符号上的修改。
为简洁起见，省略细节。
∎

###### **引理 B.2（筛近似引理，Sieve Approximation Lemma）**

设 $\mathcal{C}:\Theta\to[0,\infty)$ 是参数空间上的一个复杂度度量，并设 $(c_{n})_{n\geq 1}$ 是一个非递减序列，满足 $c_{n}\uparrow\infty$。
定义筛（Sieve）

$$
\Theta_{n}:=\{\theta\in\Theta:\mathcal{C}(\theta)\leq c_{n}\},
$$

对于一个固定的数据生成分布 $p^{\dagger}$，令

$$
\delta_{n}(p^{\dagger})\ :=\ \inf_{\theta\in\Theta_{n}}\ \sup_{x\in\mathcal{X}}\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid x)\,\|\,p_{\theta}(\cdot\mid x)\big).
$$

则以下陈述等价：

- 1.  **筛的通用近似性（Sieve universal approximation）** ：对于每个 $\varepsilon>0$，存在一个常数 $C_{\varepsilon}<\infty$，使得
      $\inf_{\theta:\,\mathcal{C}(\theta)\leq C_{\varepsilon}}\ \sup_{x\in\mathcal{X}}\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid x)\,\|\,p_{\theta}(\cdot\mid x)\big)\ <\ \varepsilon$。
- 2.  **逼近误差趋于零（Vanishing approximation error）** ：当 $n\to\infty$ 时，$\ \delta_{n}(p^{\dagger})\downarrow 0$。

此外，如果每个 $\Theta_{n}$ 是紧致的，且映射 $\theta\mapsto\sup_{x}\mathrm{KL}(p^{\dagger}\|p_{\theta})$ 在 $\Theta_{n}$ 上连续，那么对于每个 $n$，$\delta_{n}(p^{\dagger})$ 中的下确界都能达到。

###### **证明**

_(1) $\Rightarrow$ (2)_。
固定 $\varepsilon>0$，并令 $C_{\varepsilon}(p^{\dagger})$ 如 (i) 中所述。
由于 $c_{n}\uparrow\infty$，存在 $N$ 使得对所有 $n\geq N$ 有 $c_{n}\geq C_{\varepsilon}(p^{\dagger})$。
因此，对所有 $n\geq N$，有 $\Theta_{n}\supseteq\{\theta:\mathcal{C}(\theta)\leq C_{\varepsilon}(p^{\dagger})\}$，从而

$$
\delta_{n}(p^{\dagger})\ =\ \inf_{\theta\in\Theta_{n}}\ \sup_{x}\mathrm{KL}(p^{\dagger}\|p_{\theta})\ \leq\ \inf_{\theta:\,\mathcal{C}(\theta)\leq C_{\varepsilon}(p^{\dagger})}\ \sup_{x}\mathrm{KL}(p^{\dagger}\|p_{\theta})\ <\ \varepsilon,
$$

对所有 $n\geq N$ 成立。由于 $(\delta_{n})$ 关于 $n$ 是非递增的（因为 $\Theta_{n}\uparrow$），因此有 $\delta_{n}(p^{\dagger})\downarrow 0$。

_(2) $\Rightarrow$ (1)_。
固定 $\varepsilon>0$。根据 (ii)，选择 $N$ 使得 $\delta_{N}(p^{\dagger})<\varepsilon$。
令 $C_{\varepsilon}(p^{\dagger}):=c_{N}$。则有

$$
\inf_{\theta:\,\mathcal{C}(\theta)\leq C_{\varepsilon}(p^{\dagger})}\ \sup_{x}\mathrm{KL}(p^{\dagger}\|p_{\theta})\ \leq\ \inf_{\theta\in\Theta_{N}}\ \sup_{x}\mathrm{KL}(p^{\dagger}\|p_{\theta})\ =\ \delta_{N}(p^{\dagger})\ <\ \varepsilon,
$$

此即 (i)。

可达性陈述直接由 $\Theta_{n}$ 的紧致性以及映射 $\theta\mapsto\sup_{x}\mathrm{KL}(p^{\dagger}\|p_{\theta})$ 在 $\Theta_{n}$ 上的连续性得出。
∎

###### **定理 B.7（IBL 的通用一致性，Universal Consistency of IBL）**

考虑一类 IBL（Instance-Based Learning）模型的参数空间 $\Theta$，并令 $\mathcal{C}:\Theta\to[0,\infty)$ 为一个下半连续的复杂度度量（例如，网络深度、宽度或参数范数）。
令 $(c_{n})_{n\geq 1}$ 为一个非递减序列，满足 $c_{n}\uparrow\infty$，并定义筛

$$
\Theta_{n}:=\{\theta\in\Theta:\mathcal{C}(\theta)\leq c_{n}\}.
$$

假设：

- 1.  映射 $\theta\mapsto\sup_{x}\mathrm{KL}(p^{\dagger}\|p_{\theta})$ 在每个紧致的 $\Theta_{n}$ 上连续。
- 2.  经验最小化序列 $\{\hat{\theta}_{n}\}$ 在 $\bigcup_{n}\Theta_{n}$ 中是相对紧致的，这由一致大数定律（Uniform Law of Large Numbers, LLN）连同紧致性和连续性所保证。

那么，对于任何满足定理 [B.6](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem6) 正则性假设的容许数据生成分布 $p^{\dagger}$，IBL 后验序列 $\{p_{\hat{\theta}_{n}}\}$ 满足

$$
\sup_{x\in\mathcal{X}}\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid x)\,\|\,p_{\hat{\theta}_{n}}(\cdot\mid x)\big)\xrightarrow{p}0,
$$

即 $\{p_{\hat{\theta}_{n}}\}$ 在 $x$ 上（以 KL 散度）一致收敛于 $p^{\dagger}$。

###### **证明**

固定一个容许的数据分布 $p^{\dagger}$（满足定理 [B.6](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem6) 的正则性）。
对于 $\theta\in\bigcup_{n}\Theta_{n}$，定义

$$
F(\theta)\;:=\;\sup_{x\in\mathcal{X}}\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid x)\,\|\,p_{\theta}(\cdot\mid x)\big),\qquad\delta_{n}\;:=\;\inf_{\theta\in\Theta_{n}}F(\theta).
$$

那么， **定理 B.6** 和 **引理 B.2** 共同意味着 $\delta_{n}\downarrow 0$。根据假设 1，$F$ 在每个紧集 $\Theta_{n}$ 上是连续的。

令 $\hat{\theta}_{n}\in\arg\min_{\theta\in\Theta_{n}}\mathcal{M}_{n}(\theta)$ 为任意一个 **经验风险最小化（Empirical Risk Minimization, ERM）** 解的序列。我们将证明 $F(\hat{\theta}_{n})\xrightarrow{p}0$。

_步骤 1（子序列约简与预紧性）。_
取任意子序列 $(\hat{\theta}_{n_{k}})_{k}$。根据假设 2，存在一个进一步的子序列（仍记为 $(\hat{\theta}_{n_{k}})_{k}$），以及一个（可能依赖于 $k$ 的）指标集 $N_{k}\leq n_{k}$ 和一个参数极限 $\theta_{\infty}\in\Theta_{N}$（对于某个有限的 $N$），使得 $\hat{\theta}_{n_{k}}\to\theta_{\infty}$ 依概率成立。如果需要，可以再取一个子序列，我们可以假设 $N_{k}\equiv N$。

_步骤 2（对 $\Theta_{N}$ 逼近元的风险控制）。_
对于每个 $k$，选取 $\theta_{k}\in\Theta*{N}$ 使得 $F(\theta*{k})\leq\delta*{N}+1/k$（可达性由 $F$ 在 $\Theta*{N}$ 上的紧性和连续性保证）。根据 ERM 的性质以及在 $\Theta_{N}$ 上的一致 **大数定律（Law of Large Numbers, LLN）** ，有

$$
\mathcal{M}(\hat{\theta}_{n_{k}})\;\leq\;\mathcal{M}(\theta_{k})+o_{p}(1)\qquad(k\to\infty).
$$

不失一般性地假设 **交叉熵（Cross-Entropy, CE）** 分量以正权重存在，则总体风险可分解为

$$
\mathcal{M}(\theta)\;=\;\text{const}+\gamma_{d}\,\mathbb{E}_{X}\!\left[\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid X)\,\|\,p_{\theta}(\cdot\mid X)\big)\right]\;+\;\gamma_{c}\,\mathcal{L}^{\rm DSM}(\theta),
$$

其中 $\gamma_{d}>0$（仅含 **去噪分数匹配（Denoising Score Matching, DSM）** 的情况可通过将 KL 散度替换为 **费希尔散度（Fisher Divergence）** 类似处理）。利用 $\mathbb{E}_{X}[\mathrm{KL}(\cdot\|\cdot)]\leq F(\cdot)$，我们得到

$$
\limsup_{k\to\infty}\mathbb{E}_{X}\!\left[\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid X)\,\|\,p_{\hat{\theta}_{n_{k}}}(\cdot\mid X)\big)\right]\;\leq\;\limsup_{k\to\infty}F(\theta_{k})\;\leq\;\delta_{N}.
$$

因此，沿着该子序列，

$$
\mathbb{E}_{X}\!\left[\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid X)\,\|\,p_{\hat{\theta}_{n_{k}}}(\cdot\mid X)\big)\right]\xrightarrow{p}0.
$$

_步骤 3（子序列极限的识别）。_
根据模型映射 $\theta\mapsto p_{\theta}(\cdot\mid x)$ 的连续性（来自 **定理 B.6** 的正则性）以及有界收敛定理，有

$$
\mathbb{E}_{X}\!\left[\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid X)\,\|\,p_{\theta_{\infty}}(\cdot\mid X)\big)\right]=0.
$$

因此，函数 $f(x):=\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid x)\,\|\,p_{\theta_{\infty}}(\cdot\mid x)\big)$ 对于 $P_{X}$-几乎处处的 $x$ 等于 0。由于 $f$ 在紧集 $\mathcal{X}$ 上连续（由相同的正则性保证）且 $P_{X}$ 具有 **满支撑（Full Support）** （容许分布），我们得出结论：在 $\mathcal{X}$ 上 $f(x)\equiv 0$，即

$$
F(\theta_{\infty})=\sup_{x\in\mathcal{X}}f(x)=0.
$$

_步骤 4（得出 $F(\hat{\theta}_{n*{k}})\to 0$ 依概率成立，因此 $F(\hat{\theta}*{n})\to 0$ 依概率成立）。_
根据假设 1 中 $F$ 在 $\Theta_{N}$ 上的连续性以及 $\hat{\theta}_{n_{k}}\to\theta_{\infty}$ 依概率成立，我们有 $F(\hat{\theta}_{n_{k}})\xrightarrow{p}F(\theta_{\infty})=0$。由于原始子序列是任意的，且每个子序列都允许一个进一步的子序列满足 $F(\hat{\theta}_{n_{k}})\xrightarrow{p}0$，因此整个序列满足 $F(\hat{\theta}_{n})\xrightarrow{p}0$。

因此，

$$
\sup_{x\in\mathcal{X}}\mathrm{KL}\!\big(p^{\dagger}(\cdot\mid x)\,\|\,p_{\hat{\theta}_{n}}(\cdot\mid x)\big)\xrightarrow{p}0,
$$

即在 KL 散度意义下，$p_{\hat{\theta}_{n}}(\cdot\mid x)\to p^{\dagger}(\cdot\mid x)$ 关于 $x$ 一致成立。
∎

###### **定理 B.8（极值估计量的渐近正态性（Newey & McFadden, 1994，定理 3.1））。**

假设估计量 $\hat{\theta}_{n}$ 满足 $\hat{\theta}_{n}\xrightarrow{p}\theta_{0}$，且：

1.  $\theta_{0}$ 位于参数空间 $\Theta$ 的内部；
2.  准则函数 $\hat{Q}_{n}(\theta)$ 在 $\theta_{0}$ 的一个邻域 $\mathcal{N}$ 内二次连续可微；
3.  得分函数满足
    $\sqrt{n}\,\nabla_{\theta}\hat{Q}_{n}(\theta_{0})\;\xrightarrow{d}\;\mathcal{N}(0,\Sigma);$
4.  存在一个在 $\theta_{0}$ 处连续的函数 $H(\theta)$，使得
    $\sup_{\theta\in\mathcal{N}}\;\bigl\|\,\nabla^{2}_{\theta}\hat{Q}_{n}(\theta)-H(\theta)\,\bigr\|\;\xrightarrow{p}\;0;$
5.  极限 **海森矩阵（Hessian）** $H:=H(\theta_{0})$ 是非奇异的。

那么该估计量是渐近正态的：

$$
\sqrt{n}\,(\hat{\theta}_{n}-\theta_{0})\;\xrightarrow{d}\;\mathcal{N}\!\bigl(0,\,H^{-1}\Sigma H^{-1}\bigr).
$$

###### **定理 B.9（IBL 的渐近正态性， Asymptotic Normality of IBL）**

考虑 **IBL 族（IBL family）** $p_{\theta}(y\mid x)\propto\exp(\mathrm{IBL}_{\theta}(x,y))$，其经验准则如方程 [41] 所示。假设 $(X_{i},Y_{i})_{i=1}^{n}$ 是来自一个可容许数据律的独立同分布样本，且真实参数 $\theta_{0}$ 是一个局部可识别图（locally identifiable chart）的内点。对于每个观测 $Z=(X,Y)$，令 $\ell(\theta;Z)$ 表示方程 [40] 中定义的 **单样本损失（per-sample loss）** ，因此 $\hat{Q}_{n}(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(\theta;Z_{i})$ 且 $Q(\theta):=\mathbb{E}[\ell(\theta;Z)]$。

此外，假设：

1.  **得分矩（Score moments）** 。$s(Z):=\nabla_{\theta}\ell(\theta_{0};Z)$ 满足 $\mathbb{E}[s(Z)]=0$，$\Sigma:=\mathrm{Var}(s(Z))<\infty$，且 $\frac{1}{\sqrt{n}}\sum_{i=1}^{n}s(Z_{i})\Rightarrow\mathcal{N}(0,\Sigma)$。
2.  **导数包络（Derivative envelopes）** 。存在 $\theta_{0}$ 的一个邻域 $\mathcal{N}$ 以及包络函数 $G_{1},G_{2}$，使得 $\sup_{\theta\in\mathcal{N}}\|\nabla_{\theta}\ell(\theta;Z)\|\leq G_{1}(Z)$，$\sup_{\theta\in\mathcal{N}}\|\nabla_{\theta}^{2}\ell(\theta;Z)\|\leq G_{2}(Z)$，且 $\mathbb{E}[G_{1}^{2}]+\mathbb{E}[G_{2}]<\infty$。
3.  **非退化曲率（Nondegenerate curvature）** 。$H:=\nabla_{\theta}^{2}Q(\theta_{0})$ 存在，在 $\theta_{0}$ 处连续，且是正定的，其中 $Q(\theta):=\mathbb{E}[\hat{Q}_{n}(\theta)]$。

那么，在定理 [2.7] 的条件下，

$$
\sqrt{n}\,(\hat{\theta}_{n}-\theta_{0})\ \Rightarrow\ \mathcal{N}\!\bigl(0,\ H^{-1}\Sigma H^{-1}\bigr).
$$

###### **证明**

我们用上述的 $\hat{Q}_{n}$ 来验证定理 [B.8] 的假设。

_(i) 内点性与相合性（Interior & consistency）_。根据商可识别性（quotient identifiability），固定一个局部图，使得总体极小值点有一个唯一的内点代表 $\theta_{0}$。相合性 $\hat{\theta}_{n}\xrightarrow{p}\theta_{0}$ 源于 IBL 的一致 M 估计相合性（Theorem [B.5]）。

_(ii) $C^{2}$ 准则（$C^{2}$ criterion）_。由于 $\mathrm{IBL}_{\theta}$ 关于 $\theta$ 是 $C^{2}$ 的，损失函数 $\ell(\theta;Z)$ 在 $\theta_{0}$ 的一个邻域 $\mathcal{N}$ 内是二次连续可微的，$\hat{Q}_{n}$ 亦然。

_(iii) 得分中心极限定理（Score CLT）_。根据*得分矩*，

$$
\sqrt{n}\,\nabla_{\theta}\hat{Q}_{n}(\theta_{0})\;=\;\frac{1}{\sqrt{n}}\sum_{i=1}^{n}s(Z_{i})\ \Rightarrow\ \mathcal{N}(0,\Sigma).
$$

_(iv) 海森极限（Hessian limit）_。根据*导数包络*和 **控制收敛定理（Dominated Convergence Theorem）** ，

$$
\sup_{\theta\in\mathcal{N}}\big\|\nabla_{\theta}^{2}\hat{Q}_{n}(\theta)-\nabla_{\theta}^{2}Q(\theta)\big\|\xrightarrow{p}0,
$$

因此定理 [B.8] 的假设 4 成立，其中 $H(\theta):=\nabla_{\theta}^{2}Q(\theta)$ 在 $\theta_{0}$ 处连续。

_(v) 非奇异性（Nonsingularity）_。根据*非退化曲率*，$H:=H(\theta_{0})$ 是正定的。

因此，定理 [B.8] 的所有假设均得到验证；从而有，
$\sqrt{n}(\hat{\theta}_{n}-\theta_{0})\Rightarrow\mathcal{N}(0,H^{-1}\Sigma H^{-1})$。
∎

###### **定理 B.10（IBL 估计量的有效性， Efficiency of IBL Estimators）**

在定理 [B.9] 的正则性条件下，考虑与单样本损失方程 [40] 相关联的 **估计函数（estimating function）** ：

$$
\psi_{\theta}(Z)\;:=\;\nabla_{\theta}\ell(\theta;Z),\qquad Z=(X,Y).
$$

在任何总体极小值点 $\theta^{\star}$ 处，矩条件 $\mathbb{E}[\psi_{\theta^{\star}}(Z)]=0$ 成立。定义 **敏感度矩阵（sensitivity matrix）** 和 **变异性矩阵（variability matrix）** ：

$$
J\;:=\;\mathbb{E}\!\big[\nabla_{\theta}\psi_{\theta}(Z)\big]\Big|_{\theta=\theta^{\star}},\qquad K\;:=\;\mathrm{Var}\!\big(\psi_{\theta^{\star}}(Z)\big).
$$

那么 $\hat{\theta}_{n}$ 的渐近协方差由 **戈达姆信息矩阵（Godambe information matrix）** （三明治形式）给出：

$$
\sqrt{n}\,(\hat{\theta}_{n}-\theta^{\star})\;\Rightarrow\;\mathcal{N}\!\big(0,\ J^{-1}KJ^{-1}\big).
$$

具体而言：

1.  **仅交叉熵（CE-only）** 。
    如果 $\gamma_{c}=0$（纯交叉熵）且模型是正确设定且正则的，那么 $\psi_{\theta}(Z)$（至多相差一个符号）与 **对数似然得分（log-likelihood score）** $s_{\theta}(Z)$ 重合。因此 $J=-I(\theta^{\star})$ 且 $K=I(\theta^{\star})$，其中 $I(\theta^{\star})$ 表示 **费希尔信息矩阵（Fisher information matrix）** 。由此可得
    $\sqrt{n}(\hat{\theta}_{n}-\theta^{\star})\;\Rightarrow\;\mathcal{N}\!\big(0,\,I(\theta^{\star})^{-1}\big),$
    因此该估计量是 **渐近有效的（asymptotically efficient）** ，达到了 **克拉默-拉奥下界（Cramér–Rao lower bound）** 。
2.  **交叉熵+去噪得分匹配或仅去噪得分匹配（CE+DSM or DSM-only）** 。
    假设存在一个非奇异矩阵 $R$（在 $\theta^{\star}$ 的邻域内为常数），使得
    $\psi_{\theta^{\star}}(Z)\;=\;R\,s_{\theta^{\star}}(Z)\quad\text{a.s.},$
    其中 $s_{\theta}(Z)=\nabla_{\theta}\log p_{\theta}(Z)$ 表示局部图中的参数化得分。那么 $J=RI(\theta^{\star})R^{\top}$ 且 $K=RI(\theta^{\star})R^{\top}$，因此三明治协方差再次简化为 $I(\theta^{\star})^{-1}$。因此该估计量仍然是渐近有效的。

###### 证明。

经验一阶条件为

$$
0\;=\;\frac{1}{n}\sum_{i=1}^{n}\psi_{\hat{\theta}_{n}}(Z_{i}),\qquad\psi_{\theta}(Z):=\nabla_{\theta}\ell(\theta;Z).
$$

围绕总体最小化器（Population Minimizer）$\theta^{\star}$ 进行均值展开（Mean–value Expansion）可得

$$
0\;=\;S_{n}+G_{n}(\hat{\theta}_{n}-\theta^{\star}),
$$

其中

$$
S_{n}:=\frac{1}{n}\sum_{i=1}^{n}\psi_{\theta^{\star}}(Z_{i}),\qquad G_{n}:=\frac{1}{n}\sum_{i=1}^{n}\nabla_{\theta}\psi_{\tilde{\theta}}(Z_{i}),
$$

此处 $\tilde{\theta}$ 是位于 $\hat{\theta}_{n}$ 和 $\theta^{\star}$ 之间线段上的某个中间点。

在定理 [B.9](https://arxiv.org/html/2602.20152v1#A2.Thmtheorem9) 的正则性条件下，我们有

$$
G_{n}\xrightarrow{p}J:=\mathbb{E}[\nabla_{\theta}\psi_{\theta^{\star}}(Z)],\qquad\sqrt{n}\,S_{n}\;\Rightarrow\;\mathcal{N}(0,K),\quad K:=\mathrm{Var}(\psi_{\theta^{\star}}(Z)).
$$

由于 $J$ 是非奇异的，$G_{n}$ 以概率趋于 1 可逆，因此

$$
\sqrt{n}(\hat{\theta}_{n}-\theta^{\star})=-\,G_{n}^{-1}\,\sqrt{n}\,S_{n}\;\Rightarrow\;\mathcal{N}\!\bigl(0,\;J^{-1}K(J^{-1})^{\top}\bigr).
$$

因为此处 $\psi_{\theta}=\nabla_{\theta}\ell(\theta;\cdot)$，矩阵 $J$ 与损失的期望海森矩阵（Expected Hessian）一致，且是对称的。因此，渐近协方差可以等价地写为 $J^{-1}KJ^{-1}$。

_(i) 仅交叉熵（CE-only）。_
当 $\gamma_{c}=0$ 时，每样本损失简化为 $\ell(\theta;Z)=-\log p_{\theta}(Z)$，因此 $\psi_{\theta}(Z)=-s_{\theta}(Z)$，其中 $s_{\theta}(Z)=\nabla_{\theta}\log p_{\theta}(Z)$ 表示似然得分（Likelihood Score）。
在模型设定正确（Correct Specification）和标准似然正则性条件下，信息恒等式（Information Identities）成立：

$$
\mathbb{E}[s_{\theta^{\star}}(Z)]=0,\qquad\mathrm{Var}(s_{\theta^{\star}}(Z))=I(\theta^{\star}),\qquad-\mathbb{E}[\nabla_{\theta}s_{\theta^{\star}}(Z)]=I(\theta^{\star}).
$$

因此，

$$
K=\mathrm{Var}(\psi_{\theta^{\star}}(Z))=I(\theta^{\star}),\qquad J=\mathbb{E}[\nabla_{\theta}\psi_{\theta^{\star}}(Z)]=I(\theta^{\star}),
$$

且渐近协方差简化为 $I(\theta^{\star})^{-1}$。
因此，该估计量是渐近有效的，达到了克拉默-拉奥下界（Cramér–Rao Lower Bound）（另见 Van der Vaart, 2000, 定理 5.39）。

_(ii) 在得分张成（Score-span）下的交叉熵+去噪得分匹配（CE+DSM）或仅去噪得分匹配（DSM-only）。_
假设存在一个非奇异矩阵 $R$（在 $\theta^{\star}$ 的邻域内为常数），使得

$$
\psi_{\theta^{\star}}(Z)\;=\;R\,s_{\theta^{\star}}(Z)\quad\text{a.s.},
$$

其中 $s_{\theta}(Z)$ 同样是参数得分。
在这种情况下，

$$
K=\mathrm{Var}(\psi_{\theta^{\star}}(Z))=R\,I(\theta^{\star})\,R^{\top},\qquad J=\mathbb{E}[\nabla_{\theta}\psi_{\theta^{\star}}(Z)]=-R\,I(\theta^{\star}).
$$

因此，

$$
J^{-1}K(J^{-1})^{\top}=\big(-RI(\theta^{\star})\big)^{-1}\big(RI(\theta^{\star})R^{\top}\big)\big(-RI(\theta^{\star})\big)^{-\top}=I(\theta^{\star})^{-1}.
$$

因此，三明治协方差（Sandwich Covariance）简化为费雪信息界（Fisher Information Bound），且估计量是渐近有效的。
这对应于最小距离（Minimum-distance）或广义矩方法（Generalized Method of Moments, GMM）估计量的一般效率条件（见 Newey & McFadden, 1994, 第 5 节）：条件 $\psi_{\theta^{\star}}=R\,s_{\theta^{\star}}$ 等价于他们的矩张成条件（Moment–span Condition）$G^{\prime}W=C\,G^{\prime}\Omega^{-1}$（Newey & McFadden, 1994, 方程 5.4），在此条件下戈达姆信息（Godambe Information）坍缩为费雪界。

由此，两个主张得证。
∎

###### 引理 B.1（线性组合的可识别性（Identifiability of Linear Combinations））。

###### 证明。

###### 定理 B.1（IBL(Single) 的可识别性（Identifiability of IBL(Single)））。

###### 证明。

###### 定理 B.2（IBL(Shallow) 的可识别性（Identifiability of IBL(Shallow)））。

###### 证明。

###### 定理 B.3（IBL(Deep) 的可识别性（Identifiability of IBL(Deep)））。

###### 证明。

###### 证明。

###### 证明。

###### 定理 B.4（一致 M 估计（Uniform M-estimation consistency）(Newey & McFadden, 1994, 定理 2.1)）。

###### 定理 B.5（IBL 的一致性（Consistency of IBL））。

###### 证明。

###### 定理 B.6（IBL 的通用逼近性（Universal Approximation of IBL））。

###### 证明。

###### 引理 B.2（筛逼近引理（Sieve Approximation Lemma））。

###### 证明。

###### 定理 B.7（IBL 的通用一致性（Universal Consistency of IBL））。

###### 证明。

###### 定理 B.8（极值估计量的渐近正态性（Asymptotic normality of extremum estimators）(Newey & McFadden, 1994, 定理 3.1)）。

###### 定理 B.9（IBL 的渐近正态性（Asymptotic Normality of IBL））。

###### 证明。

###### 定理 B.10（IBL 估计量的效率性（Efficiency of IBL Estimators））。

###### 证明。

<a id="appendix-c"></a>

## 附录 C 实验细节

### C.1 硬件

大多数实验在单个 NVIDIA L40S GPU 上进行。少量运行在一台配备 NVIDIA GeForce RTX 2050 GPU 和 Intel Core i7–12700H CPU 的笔记本电脑上完成。

### C.2 标准预测任务

##### 数据集（Datasets）。

在 **标准预测任务（Standard Prediction Task）** 中，我们使用了涵盖不同应用领域的 10 个 OpenML 数据集。详细信息见表 [4](#table-4)。

<a id="table-4"></a>
表 4 | 本任务中使用的标准 OpenML 数据集。#Features 表示输入变量（不包括目标和 ID）的数量。

| 名称                  | 样本量 | #特征数 | 任务类型 | 领域       |
| :-------------------- | :----: | :-----: | :------- | :--------- |
| German Credit         | 1,000  |   20    | 二分类   | 金融       |
| Adult Income          | 48,842 |   14    | 二分类   | 经济学     |
| COMPAS (two-years)    | 5,278  |   13    | 二分类   | 法律与社会 |
| Bank Marketing        | 45,211 |   16    | 二分类   | 市场营销   |
| Planning Relax        |  182   |   12    | 二分类   | 心理学     |
| EEG Eye State         | 14,980 |   14    | 二分类   | 神经科学   |
| MAGIC Gamma Telescope | 19,020 |   10    | 二分类   | 物理学     |
| Electricity           | 45,312 |    8    | 二分类   | 电气工程   |
| Wine Quality (Red)    | 1,599  |   11    | 多分类   | 化学       |
| Steel Plates Faults   | 1,941  |   27    | 多分类   | 工业工程   |

##### 基线模型（Baseline Models）。

为了进行比较，我们纳入了以下基线模型： **多层感知机（Multilayer Perceptron, MLP）** 、 **神经加法模型（Neural Additive Model, NAM）** (Agarwal et al., 2020; Kayid et al., 2020)、 **弹性网络（ElasticNet）** 、 **随机森林（Random Forest）** 、 **随机变分高斯过程（Stochastic Variational Gaussian Process, SVGP）** (Gardner et al., 2018)、 **逻辑回归（Logistic Regression）** 、 **决策树（Decision Tree）** 、 **TabNet** (Arik & Pfister, 2021)、 **多项式逻辑回归（Polynomial Logistic Regression）** 以及 **LightGBM** (Ke et al., 2017)。

<a id="table-5"></a>
表 5 | 标准预测任务中的基线模型概览

| 方法族       | 模型名称                 |
| :----------- | :----------------------- |
| 神经网络     | 标准 MLP                 |
|              | 神经加法模型（NAM）      |
|              | TabNet                   |
| 线性回归器   | 弹性网络                 |
|              | 逻辑回归                 |
|              | 多项式逻辑回归           |
| 基于树的模型 | 随机森林                 |
|              | 决策树                   |
| 梯度提升方法 | LightGBM                 |
| 贝叶斯方法   | 随机变分高斯过程（SVGP） |

##### 数据预处理（Data preprocessing）。

对于所有十个数据集，我们采用一致的预处理策略。
有序分类变量（Ordinal categorical variables）被映射为整数级别以保留其固有顺序。没有自然顺序的名义分类变量（Nominal categorical variables）使用独热编码（One-hot encoding）进行转换。连续变量被标准化为零均值和单位方差。每个数据集按 7:1:2 的比例随机划分为训练集/验证集/测试集。

##### **超参数调优协议（Hyperparameter Tuning Protocol）** 。

我们使用 Optuna 包 [1] 中的 TPE 采样器（TPE sampler）对大多数模型进行超参数优化（Hyperparameter optimization），每个数据集进行 50 次试验。对于每个模型和数据集，调优后的配置会在 8 个随机种子下进行评估。

##### **BL 模型超参数空间（BL Model Hyperparameter Space）** 。

对于 BL(Single) 和 BL(Shallow)，我们优化分类任务的交叉熵损失（Cross-entropy loss）。同时考虑了 Adam [2] 和 AdamW [3] 优化器，并在每个数据集上报告性能更优的变体。未应用数据增强（Data augmentation）。批大小（Batch size）的选择因数据集而异。

- **BL(Single)** ：所有实验报告的统一设置为：
  $\mathrm{degree}_{U}=[2],\quad\mathrm{degree}_{C}=[2,2,2],\quad\mathrm{degree}_{T}=[2,2],$
  $\sigma_{\text{params}}=0.01,\quad\sigma_{\lambda_{0}}=0.01,\quad\sigma_{\lambda_{1}}=0.01,\quad\sigma_{\lambda_{2}}=0.01.$
  其中，$\mathrm{degree}_{U}$、$\mathrm{degree}_{C}$ 和 $\mathrm{degree}_{T}$ 分别表示参数化 $U(x,y)$、$C(x,y)$ 和 $T(x,y)$ 的模块的多项式次数。
  列表同时表示模块的数量和每个模块的次数：$\mathrm{degree}_{U}=[2]$ 表示 $U$ 使用单个二次模块，$\mathrm{degree}_{C}=[2,2,2]$ 表示三个二次约束模块，$\mathrm{degree}_{T}=[2,2]$ 表示两个二次信念模块。
  $\sigma_{\text{params}}$ 初始化所有多项式模块的系数，而 $\sigma_{\lambda_{0}}$、$\sigma_{\lambda_{1}}$ 和 $\sigma_{\lambda_{2}}$ 初始化 UMP 权重 $(\lambda_{0},\lambda_{1},\lambda_{2})$。搜索网格详见表 [6](#table-6)。

- **BL(Shallow)** ：我们使用全局梯度裁剪（Global gradient clipping）值为 1.0，以及验证集性能在 20 个周期内无提升时的早停（Early stopping）耐心值。考虑深度 $L\leq 3$ 的浅层架构。搜索网格详见表 [7](#table-7)。

**参考文献**

1.  Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In _Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining_ (pp. 2623–2631).
2.  Kingma, D. P. (2014). Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_.
3.  Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_.

##### **基线模型超参数空间（Baseline Model Hyperparameter Spaces）**

对于基线模型，我们同样为基于神经网络的变体考虑了 **Adam** 和 **AdamW** 两种优化器，并在每个数据集上报告性能更优的优化器的结果。 **批量大小（batch size）** 为每个数据集单独调整。详细的超参数搜索空间总结在表 LABEL:tab:tuning_space 中。

<a id="table-6"></a>

表 6 | BL(Single) 的超参数调优空间。

| 模型       | 参数          | 搜索空间            |
| :--------- | :------------ | :------------------ |
| BL(Single) | learning_rate | {1e$-3$, 1e$-1$}    |
|            | batch_size    | {64, 128, 256, 512} |
|            | max_grad_norm | {1.0, 2.0, 5.0}     |

<a id="table-7"></a>

表 7 | BL(Shallow) 的超参数调优空间。

| 模型        | 参数           | 搜索空间                   |
| :---------- | :------------- | :------------------------- |
| BL(Shallow) | learning_rate  | LogUniform{5e$-5$, 5e$-3$} |
|             | batch_size     | {64, 128, 256, 512}        |
|             | n_layers       | UniformInt{1, 3}           |
|             | n_first_layer  | {24, 30, 36, 40}           |
|             | n_middle_layer | {8, 6, 4}                  |
|             | n_last_layer   | {2, 4, 6}                  |
|             | weight_decay   | LogUniform{1e$-4$, 1e$-1$} |

<a id="table-8"></a>
表 8 | 标准预测任务中使用的基线模型的超参数调优空间。

| 模型                  | 参数                     | 搜索空间                                    |
| :-------------------- | :----------------------- | :------------------------------------------ |
| MLP                   | learning_rate            | LogUniform{1e$-5$, 1e$-1$}                  |
|                       | batch_size               | {32, 64, 128, 256}                          |
|                       | n_layers                 | UniformInt{2, 4}                            |
|                       | hidden_size              | UniformInt{32, 256}                         |
|                       | weight_decay             | LogUniform{1e$-6$, 1e$-2$}                  |
| NAM                   | learning_rate            | LogUniform{1e$-3$, 1e$-1$}                  |
|                       | batch_size               | {128, 256, 512, 1024}                       |
|                       | patience                 | UniformInt{10, 30}                          |
| ElasticNet (SGD)      | alpha                    | LogUniform{1e$-4$, 1e$+2$}                  |
|                       | l1_ratio                 | Uniform{0.0, 1.0}                           |
|                       | max_iter                 | UniformInt{100, 2000}                       |
|                       | tol                      | LogUniform{1e$-6$, 1e$-2$}                  |
|                       | fit_intercept            | {true, false}                               |
|                       | learning_rate            | {optimal, constant, invscaling, adaptive}   |
|                       | eta0                     | LogUniform{1e$-4$, 1e$-1$}                  |
|                       | validation_fraction      | Uniform{0.05, 0.30}                         |
|                       | n_iter_no_change         | UniformInt{3, 20}                           |
| PolyLogistic          | degree                   | {2, 3}                                      |
|                       | penalty                  | {$\ell*{2}$, $\ell_{1}$, ”elasticnet”}      |
|                       | C                        | LogUniform{1e$-3$, 1e$+2$}                  |
|                       | l1_ratio                 | Uniform{0.1, 0.9}                           |
|                       | solver                   | {”liblinear”, ”lbfgs”, “newton-cg”, “saga”} |
|                       | max_iter                 | UniformInt{500, 2000}                       |
|                       | tol                      | LogUniform{1e$-5$, 1e$-3$}                  |
| Logistic (ElasticNet) | C                        | LogUniform{1e$-3$, 1e$+2$}                  |
|                       | l1_ratio                 | Uniform{0.0, 1.0}                           |
|                       | max_iter                 | UniformInt{100, 2000}                       |
|                       | tol                      | LogUniform{1e$-6$, 1e$-2$}                  |
|                       | fit_intercept            | {true, false}                               |
| LogisticRegression    | solver                   | {”liblinear”, ”lbfgs”, ”sag”}               |
|                       | C                        | LogUniform{1e$-4$, 1e$+2$}                  |
|                       | max_iter                 | UniformInt{100, 2000}                       |
|                       | tol                      | LogUniform{1e$-6$, 1e$-2$}                  |
|                       | fit_intercept            | {True, False}                               |
|                       | intercept_scaling        | Uniform{0.1, 10.0}                          |
| TabNet                | learning_rate            | LogUniform{1e$-4$, 3e$-2$}                  |
|                       | batch_size               | {128, 256, 512, 1024}                       |
|                       | virtual_batch_size       | {64, 128}                                   |
|                       | n_d=n_a                  | UniformInt{16, 64}                          |
|                       | n_steps                  | UniformInt{3, 7}                            |
|                       | gamma                    | Uniform{1.2, 1.7}                           |
|                       | lambda_sparse            | LogUniform{1e$-6$, 1e$-3$}                  |
| DecisionTree          | criterion                | {”gini”, ”entropy”, ”log_loss”}             |
|                       | max_depth                | UniformInt{3, 20}                           |
|                       | min_samples_split        | UniformInt{2, 20}                           |
|                       | min_samples_leaf         | UniformInt{1, 10}                           |
|                       | min_weight_fraction_leaf | Uniform{0.0, 0.5}                           |
|                       | max_features             | {”sqrt”, ”log2”}                            |
|                       | max_leaf_nodes           | UniformInt{10, 1000}                        |
|                       | min_impurity_decrease    | Uniform{0.0, 0.1}                           |
|                       | ccp_alpha                | Uniform{0.0, 0.1}                           |
| GP (SVGP)             | kernel                   | {rbf, matern, rational_quadratic}           |
|                       | lengthscale              | LogUniform{0.1, 10.0}                       |
|                       | rq_alpha                 | LogUniform{0.1, 5.0}                        |
|                       | num_inducing             | UniformInt{100, 500}                        |
|                       | learning_rate            | LogUniform{1e$-2$, 5e$-1$}                  |
|                       | training_iters           | UniformInt{50, 200}                         |
| RandomForest          | n_estimators             | UniformInt{100, 500}                        |
|                       | max_depth                | UniformInt{3, 30}                           |
|                       | max_features             | {”sqrt”, ”log2”}                            |
|                       | min_samples_leaf         | UniformInt{1, 10}                           |
|                       | min_samples_split        | UniformInt{2, 20}                           |

### **C.3 解读 BL：一个案例研究（Interpreting BL: A Case Study）**

#### C.3.1 解释 BL(Deep)：高层概览

**BL（Bottleneck Layer）** 的更深层变体是通过将多个 **BL(Single)** 模块堆叠成分层结构，最后加上一个仿射变换（Affine transformation）来构建的。这形成了一个相互作用的 **效用最大化问题（Utility Maximization Problem, UMP）** 系统（每个 UMP 可视为一个智能体），其中每个内部块 $\mathcal{B}$ 代表一个单一的可解释 UMP。如图 [4](#figure-4) 所示，第一层模块对应于独立的 UMP，而第二层模块则通过聚合或分配其输出来执行最优协调。这种分层结构为更深的 BL 模型提供了一种组合式解释，即将其视为由相互作用的、可解释的 UMP 组成的系统。

#### C.3.2 案例研究：补充细节

<a id="table-9"></a>
**表 9：波士顿住房数据集变量及描述。**

| 变量    | 描述                                                    |
| :------ | :------------------------------------------------------ |
| CRIM    | 城镇人均犯罪率                                          |
| ZN      | 划定为超过 25,000 平方英尺地块的住宅用地比例            |
| INDUS   | 城镇非零售商业用地比例                                  |
| CHAS    | 查尔斯河虚拟变量（若区域临河则为 1）                    |
| NOX     | 一氧化氮浓度（千万分之几）                              |
| RM      | 每套住宅的平均房间数                                    |
| AGE     | 1940 年以前建造的自有住房比例                           |
| DIS     | 到五个波士顿就业中心的加权距离                          |
| RAD     | 放射状高速公路可达性指数                                |
| TAX     | 每 10,000 美元的全额财产税率                            |
| PTRATIO | 城镇学生-教师比例                                       |
| B       | $1000(B_{k}-0.63)^{2}$，其中 $B_{k}$ 为城镇黑人居民比例 |
| LSTAT   | 低社会地位人口百分比                                    |
| MEDV    | 自有住房中位价值（单位：千美元）                        |

<a id="table-10"></a>
**表 10：深度 BL 架构中各块的语义角色。**

| 层                 | 块                                               | 代表性偏好                                 |
| :----------------- | :----------------------------------------------- | :----------------------------------------- |
| 第 1 层            | 位置敏感型买家                                   | 重视河流可达性、交通便利性和社区便利设施。 |
| 风险敏感型买家     | 厌恶本地不良因素，如污染和环境风险。             |
| 经济敏感型买家     | 对学校质量和社区社会经济构成敏感。               |
| 分区对比型买家     | 对塑造本地住房供应的分区和土地利用模式做出反应。 |
| 偏好可负担性型买家 | 强烈偏好更可负担的住房，不喜欢高价格。           |
| 第 2 层            | 综合位置-经济型买家                              | 以综合方式联合评估位置和社会经济属性。     |
| 预算冲突型买家     | 对理想位置表现出强烈偏好，但面临严格的预算约束。 |
| 平衡权衡型买家     | 以平衡的方式联合考虑多个住房属性。               |
| 第 3 层            | 代表性综合型买家                                 | 将所有低层偏好成分聚合成一个代表性家庭。   |

<a id="table-11"></a>
**表 11：深度 BL 架构中的每个块都与经济学文献中记载的经典偏好机制相对应。**

| 层 / 块                      | 代表性参考文献           |
| :--------------------------- | :----------------------- |
| 第 1 层：位置敏感型买家      | Gibbons & Machin (2005)  |
| 第 1 层：风险敏感型买家      | Chay & Greenstone (2005) |
| 第 1 层：经济敏感型买家      | Black (1999)             |
| 第 1 层：分区对比型买家      | Glaeser & Gyourko (2002) |
| 第 1 层：偏好可负担性型买家  | McFadden (1977)          |
| 第 2 层：综合位置-经济型买家 | Bayer et al. (2007)      |
| 第 2 层：预算冲突型买家      | Balseiro et al. (2019)   |
| 第 2 层：平衡权衡型买家      | Rosen (1974)             |

### C.4 高维输入预测

##### 数据集描述与预处理。

对于图像数据集，我们使用 **MNIST** 和 **Fashion-MNIST** 的官方训练/测试划分：输入被转换为单通道图像，缩放至 $[0,1]$ 范围，并使用数据集特定的统计量进行标准化。未应用任何尺寸调整或数据增强。训练使用大小为 $64$ 的随机打乱小批量（Mini-batch）。
对于文本数据集，我们应用以下流程：

1.  **数据源与官方划分。**
    我们使用 **AG News** 和 **Yelp Review Polarity** 的官方训练和测试划分，未进行任何自定义重新划分。两个数据集在标签上都是类别平衡的，我们未执行任何重采样。
2.  **数据集规模。**

    - AG News：120,000 个训练样本 / 7,600 个测试样本，包含四个平衡类别。
    - Yelp Review Polarity：560,000 个训练样本 / 38,000 个测试样本，包含两个平衡类别。

3.  **标签映射。**

    - AG News：标签 1–4 映射为 0–3。
    - Yelp Review Polarity：标签 1–2 映射为 0–1。

4.  **文本预处理与特征表示。**
    所有文本均转换为小写并进行词级分词（Tokenization）。词汇表由一元组（Unigram）和二元组（Bigram）构建，丢弃在训练语料库中出现少于两次的词。词汇表大小设限（AG News：200,000；Yelp：100,000）。我们在训练集上计算 **TF–IDF（Term Frequency–Inverse Document Frequency）** 权重，并将学习到的权重应用于测试集。使用截断奇异值分解（Truncated Singular Value Decomposition, SVD）将维度降至 128 个潜在成分。特征被标准化为零均值和单位方差，最后进行 $\ell_{2}$ 归一化。我们固定随机种子以保证可复现性，并在多次运行中复用学习到的预处理组件。

##### 额外的 OOD 检测结果。

除了准确率和 **AUROC（Area Under the Receiver Operating Characteristic Curve）** ，我们还报告了图像和文本数据集的 **AUPR（Area Under the Precision-Recall Curve）** 和 **FPR@95（False Positive Rate at 95% True Positive Rate）** ；结果如表 [12](#table-12) 所示。在图像数据集上，BL（深度=1）实现了最佳的整体平衡：它在 Fashion-MNIST 的 AUPR 上排名第一，在 Fashion-MNIST 的 FPR@95 上排名第二。在 MNIST 上，其 AUPR 排名第二，但在 FPR@95 上表现不及 E-MLP（深度=2）。这些结果表明，BL 产生了可分离的分数分布，尤其是在 Fashion-MNIST 上，尽管在相同的召回率下，其 95% FPR 阈值比 E-MLP 允许更多的 OOD 样本。在文本数据集上，OOD 检测性能依赖于具体数据集：E-MLP 在 AG News 上表现更好，而 BL 在 Yelp 上取得了更强的 OOD 性能。

<a id="table-12"></a>
**表 12：图像和文本数据集上的 OOD AUPR 和 FPR@95（%）。BL 和 E-MLP 在深度 1–3 下进行评估，参数量匹配，均不使用跳跃连接。每列前两名分别用蓝色和红色标出。**

| 模型            | MNIST             | Fashion-MNIST     |                  |                   |
| :-------------- | :---------------- | :---------------- | :--------------- | :---------------- | --- |
|                 | AUPR              | FPR@95            | AUPR             | FPR@95            |     |
| E-MLP (depth=1) | 89.37 $\pm$ 1.52  | 35.57 $\pm$ 5.87  | 91.35 $\pm$ 1.25 | 28.24 $\pm$ 4.37  |
| BL (depth=1)    | 91.57 $\pm$ 2.39  | 47.81 $\pm$ 11.29 | 91.79 $\pm$ 0.90 | 38.86 $\pm$ 2.57  |
| E-MLP (depth=2) | 91.52 $\pm$ 1.27  | 28.89 $\pm$ 2.85  | 86.19 $\pm$ 2.27 | 47.72 $\pm$ 4.79  |
| BL (depth=2)    | 91.20 $\pm$ 1.22  | 52.71 $\pm$ 18.66 | 89.30 $\pm$ 2.47 | 42.65 $\pm$ 9.53  |
| E-MLP (depth=3) | 90.04 $\pm$ 1.89  | 31.92 $\pm$ 5.76  | 84.30 $\pm$ 1.50 | 54.49 $\pm$ 2.74  |
| BL (depth=3)    | 92.36 $\pm$ 2.03  | 32.32 $\pm$ 5.76  | 88.41 $\pm$ 4.04 | 41.19 $\pm$ 13.36 |
| 模型            | AG News           | Yelp              |                  |                   |
|                 | AUPR              | FPR@95            | AUPR             | FPR@95            |     |
| E-MLP (depth=1) | 44.52 $\pm$ 15.10 | 33.82 $\pm$ 4.89  | 3.31 $\pm$ 1.60  | 54.21 $\pm$ 2.11  |
| BL (depth=1)    | 18.68 $\pm$ 16.48 | 42.03 $\pm$ 5.99  | 12.70 $\pm$ 2.29 | 40.95 $\pm$ 1.56  |
| E-MLP (depth=2) | 31.48 $\pm$ 23.94 | 40.20 $\pm$ 9.82  | 1.47 $\pm$ 2.31  | 57.80 $\pm$ 4.88  |
| BL (depth=2)    | 10.76 $\pm$ 15.94 | 53.71 $\pm$ 9.68  | 6.73 $\pm$ 2.61  | 46.54 $\pm$ 1.86  |
| E-MLP (depth=3) | 51.24 $\pm$ 9.13  | 32.96 $\pm$ 3.23  | 3.14 $\pm$ 2.22  | 55.24 $\pm$ 5.33  |
| BL (depth=3)    | 16.99 $\pm$ 17.12 | 45.24 $\pm$ 6.11  | 10.96 $\pm$ 1.10 | 42.27 $\pm$ 1.94  |

##### 参数量（Number of Parameters）

为确保 **E-MLP（Energy-based MLP）** 与 **BL（Baseline）** 之间的公平比较，我们尽可能匹配了相同深度模型的 **可训练参数量（trainable parameters）** （参见表 [13](#table-13)）。

##### 运行时间（Running Time）

为评估计算成本，我们比较了 BL 和 **基于能量的多层感知机（Energy-based MLP）** 在图像和文本数据集上的训练时间（表 [3](#table-3)）。
在可比的参数量预算下，BL 在所有数据集上通常比 E-MLP 需要稍长的训练时间。具体而言，BL 在图像数据集和 AG News 上速度稍慢，而在 Yelp 数据集上表现出与 E-MLP 相当的运行时间。

##### 校准（Calibration）

我们报告了 **期望校准误差（Expected Calibration Error, ECE）** 和 **负对数似然（Negative Log-Likelihood, NLL）** 指标来评估校准质量，结果呈现在表 [2] 中。在图像数据集上， **BL** 模型提供了显著更好的校准效果，其模型在每个指标列中都占据了前两位。在文本数据集上，校准性能大体相当，其中在 Yelp 数据集上 BL 的 NLL 略低。总体而言，这些结果表明 **BL 在提供强大预测性能的同时，也输出了可靠的概率估计** 。

<a id="table-13"></a>

**表 13：高维数据集上 E-MLP 与 BL 模型的可训练参数量（Number of trainable parameters for E-MLP and BL models across high-dimension datasets）。**

| 数据集（Dataset）    | 模型（Model）   | 参数量（# Parameters） |
| :------------------- | :-------------- | :--------------------- |
| MNIST & FashionMNIST | E-MLP (depth=1) | 203,530                |
|                      | BL (depth=1)    | 208,384                |
|                      | E-MLP (depth=2) | 235,146                |
|                      | BL (depth=2)    | 219,264                |
|                      | E-MLP (depth=3) | 238,314                |
|                      | BL (depth=3)    | 221,684                |
| AGNews               | E-MLP (depth=1) | 136,196                |
|                      | BL (depth=1)    | 149,720                |
|                      | E-MLP (depth=2) | 386,284                |
|                      | BL (depth=2)    | 397,568                |
|                      | E-MLP (depth=3) | 230,788                |
|                      | BL (depth=3)    | 224,128                |
| Yelp                 | E-MLP (depth=1) | 134,146                |
|                      | BL (depth=1)    | 148,960                |
|                      | E-MLP (depth=2) | 385,770                |
|                      | BL (depth=2)    | 397,312                |
|                      | E-MLP (depth=3) | 230,530                |
|                      | BL (depth=3)    | 224,000                |

### C.5 案例研究：BL 在波士顿住房数据集上的估计结果（Case Study: Estimation Results of BL on the Boston Housing Dataset）

<a id="table-14"></a>

**表 14：BL 模型（layer = [2, 1]）在波士顿住房数据集上学习到的估计 UMP 模块参数（Estimated UMP block parameters learned by the BL model (layer = [2, 1]) on the Boston Housing dataset）。对于每个模块，$U$ 表示效用（Utility）分量，$C$ 表示不等式约束（Inequality-Constraint）分量，$T$ 表示等式约束（Equality-Constraint）分量。**

|                                                          | Block 11 |          |          | Block 12 |          |          |
| :------------------------------------------------------- | :------: | :------: | :------: | :------: | :------: | :------: |
| 变量（Variable）                                         | $U_{11}$ | $C_{11}$ | $T_{11}$ | $U_{12}$ | $C_{12}$ | $T_{12}$ |
| $\lambda$                                                |  1.003   |  0.997   |  0.999   |  0.997   |  1.003   |  1.000   |
| 人均犯罪率（per capita crime rate, CRIM）                |   0.21   |   0.14   |   0.03   |   0.12   |   0.09   |   0.25   |
| 住宅用地比例（residential land proportion, ZN）          |   0.23   |  -0.04   |  -0.27   |   0.25   |   0.00   |   0.09   |
| 非零售商业用地面积（non-retail business acreage, INDUS） |  -0.06   |   0.21   |   0.25   |   0.16   |   0.22   |   0.27   |
| 查尔斯河虚拟变量（Charles River dummy, CHAS）            |   0.25   |   0.04   |  -0.24   |  -0.12   |  -0.20   |  -0.23   |
| 氮氧化物浓度（nitric oxide concentration, NOX）          |  -0.06   |  -0.13   |   0.21   |   0.16   |   0.02   |  -0.28   |
| 平均每户房间数（average rooms per dwelling, RM）         |   0.06   |   0.07   |   0.05   |   0.05   |  -0.19   |  -0.22   |
| 老旧单元比例（proportion of older units, AGE）           |  -0.13   |  -0.12   |  -0.09   |   0.14   |   0.08   |  -0.18   |
| 距就业中心距离（distance to employment centres, DIS）    |   0.16   |  -0.03   |   0.17   |  -0.17   |  -0.09   |   0.11   |
| 放射状公路可达性（radial highway accessibility, RAD）    |   0.24   |  -0.11   |   0.04   |  -0.28   |   0.09   |   0.10   |
| 房产税率（property tax rate, TAX）                       |  -0.20   |   0.18   |   0.22   |  -0.11   |  -0.06   |   0.23   |
| 低收入人口比例（low-income population, LSTAT）           |   0.05   |  -0.12   |  -0.09   |   0.23   |  -0.16   |  -0.19   |
| 房屋价值中位数（median home value, MEDV）                |   0.21   |  -0.08   |   0.07   |   0.08   |  -0.17   |   0.15   |
| 常数项（Constant term, C）                               |   0.03   |  -0.17   |  -0.07   |   0.11   |  -0.16   |  -0.12   |

<a id="table-15"></a>

**表 15：在波士顿住房数据集上训练的 BL 模型（layer = [5, 3, 1]）第一层模块的估计 UMP 参数（Estimated UMP parameters for the Layer 1 blocks of the BL model (layer = [5, 3, 1]) trained on the Boston Housing dataset）。此处，$U$ 表示效用（Utility）分量，$C$ 表示不等式约束（Inequality-Constraint）分量，$T$ 表示等式约束（Equality-Constraint）分量。**

| 变量（Variable）                                         | $U_{11}$ | $U_{12}$ | $U_{13}$ | $U_{14}$ | $U_{15}$ |
| :------------------------------------------------------- | :------- | :------- | :------- | :------- | :------- |
| $\lambda$                                                | 1.000    | 0.998    | 1.003    | 1.002    | 1.000    |
| 人均犯罪率（per capita crime rate, CRIM）                | 0.21     | 0.12     | 0.17     | -0.09    | 0.06     |
| 住宅用地比例（residential land proportion, ZN）          | 0.23     | 0.25     | -0.07    | -0.22    | -0.16    |
| 非零售商业用地面积（non-retail business acreage, INDUS） | -0.06    | 0.16     | 0.16     | 0.23     | -0.14    |
| 查尔斯河虚拟变量（Charles River dummy, CHAS）            | 0.25     | -0.12    | -0.22    | -0.05    | -0.01    |
| 氮氧化物浓度（nitric oxide concentration, NOX）          | -0.06    | 0.16     | -0.14    | 0.24     | 0.16     |
| 平均每户房间数（average rooms per dwelling, RM）         | 0.05     | 0.05     | 0.08     | 0.09     | -0.07    |
| 老旧单元比例（proportion of older units, AGE）           | -0.13    | 0.14     | 0.06     | -0.23    | -0.15    |
| 距就业中心距离（distance to employment centres, DIS）    | 0.16     | -0.17    | -0.07    | 0.19     | -0.10    |
| 放射状公路可达性（radial highway accessibility, RAD）    | 0.24     | -0.27    | 0.17     | -0.08    | -0.20    |
| 房产税率（property tax rate, TAX）                       | -0.20    | -0.11    | 0.19     | -0.11    | 0.10     |
| 低收入人口比例（low-income population, LSTAT）           | 0.05     | 0.23     | -0.15    | -0.28    | -0.26    |
| 房屋价值中位数（median home value, MEDV）                | 0.21     | -0.08    | 0.07     | 0.08     | -0.17    |
| 常数项（Constant term, C）                               | 0.03     | 0.12     | -0.09    | -0.06    | 0.15     |

<a id="table-16"></a>

**Table 16: Layer 2 and Layer 3 UMP parameters ($U$, $C$, $T$) for Blocks in the BL model (layer = [5, 3, 1]).**

|                             | Block 21 | Block 22 | Block 23 |          |          |          |          |          |          |
| --------------------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Variable                    | $U_{21}$ | $C_{21}$ | $T_{21}$ | $U_{22}$ | $C_{22}$ | $T_{22}$ | $U_{23}$ | $C_{23}$ | $T_{23}$ |
| $\lambda$                   | 1.000    | 1.000    | 1.000    | 0.999    | 1.003    | 1.002    | 1.001    | 1.002    | 0.999    |
| Block 11 output ($b_{1,1}$) | 0.28     | 0.06     | -0.20    | -0.31    | 0.24     | 0.18     | -0.29    | -0.08    | 0.22     |
| Block 12 output ($b_{1,2}$) | 0.21     | -0.11    | -0.09    | -0.44    | 0.12     | -0.22    | 0.15     | -0.22    | 0.20     |
| Block 13 output ($b_{1,3}$) | -0.40    | 0.18     | -0.44    | -0.36    | -0.01    | -0.09    | -0.13    | -0.14    | 0.32     |
| Block 14 output ($b_{1,4}$) | -0.27    | -0.17    | 0.30     | 0.33     | -0.34    | -0.26    | 0.28     | -0.42    | -0.34    |
| Block 15 output ($b_{1,5}$) | -0.07    | -0.29    | 0.34     | 0.22     | -0.38    | -0.08    | -0.14    | 0.25     | 0.32     |
| Constant term (C)           | 0.43     | 0.33     | 0.16     | 0.38     | -0.42    | -0.32    | -0.33    | -0.31    | -0.21    |
