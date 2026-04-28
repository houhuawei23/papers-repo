# Title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
- ArXiv: 1312.6120
- Authors: Andrew M. Saxe, James L. McClelland, Surya Ganguli
- Sections: 20
- Estimated tokens: 20.6k

## Contents
- [Abstract](#abstract)
- [1 General learning dynamics of gradient descent](#1-general-learning-dynamics-of-gradient-descent)
  - [1.1 Learning dynamics with orthogonal inputs](#11-learning-dynamics-with-orthogonal-inputs)
  - [1.2 The final outcome of learning](#12-the-final-outcome-of-learning)
  - [1.3 The time course of learning](#13-the-time-course-of-learning)
- [2 Deeper multilayer dynamics](#2-deeper-multilayer-dynamics)
- [3 Finding good weight initializations: on greediness and randomness](#3-finding-good-weight-initializations-on-greediness-and-randomness)
- [4 Achieving approximate dynamical isometry in nonlinear networks](#4-achieving-approximate-dynamical-isometry-in-nonlinear-networks)
- [5 Discussion](#5-discussion)

## Abstract

Despite the widespread practical success of deep learning methods, our theoretical understanding of the dynamics of learning in deep neural networks remains quite sparse. We attempt to bridge the gap between the theory and practice of deep learning by systematically analyzing learning dynamics for the restricted case of deep linear neural networks. Despite the linearity of their input-output map, such networks have nonlinear gradient descent dynamics on weights that change with the addition of each new hidden layer. We show that deep linear networks exhibit nonlinear learning phenomena similar to those seen in simulations of nonlinear networks, including long plateaus followed by rapid transitions to lower error solutions, and faster convergence from greedy unsupervised pretraining initial conditions than from random initial conditions. We provide an analytical description of these phenomena by finding new exact solutions to the nonlinear dynamics of deep learning. Our theoretical analysis also reveals the surprising finding that as the depth of a network approaches infinity, learning speed can nevertheless remain finite: for a special class of initial conditions on the weights, very deep networks incur only a finite, depth independent, delay in learning speed relative to shallow networks. We show that, under certain conditions on the training data, unsupervised pretraining can find this special class of initial conditions, while scaled random Gaussian initializations cannot. We further exhibit a new class of random orthogonal initial conditions on weights that, like unsupervised pre-training, enjoys depth independent learning times. We further show that these initial conditions also lead to faithful propagation of gradients even in deep nonlinear networks, as long as they operate in a special regime known as the edge of chaos.

Deep learning methods have realized impressive performance in a range of applications, from visual object classification [[1](#ref-1), [2](#ref-2), [3](#ref-3)] to speech recognition [[4](#ref-4)] and natural language processing [[5](#ref-5), [6](#ref-6)]. These successes have been achieved despite the noted difficulty of training such deep architectures [[7](#ref-7), [8](#ref-8), [9](#ref-9), [10](#ref-10), [11](#ref-11)]. Indeed, many explanations for the difficulty of deep learning have been advanced in the literature, including the presence of many local minima, low curvature regions due to saturating nonlinearities, and exponential growth or decay of back-propagated gradients [[12](#ref-12), [13](#ref-13), [14](#ref-14), [15](#ref-15)]. Furthermore, many neural network simulations have observed strikingly nonlinear learning dynamics, including long plateaus of little apparent improvement followed by almost stage-like transitions to better performance. However, a quantitative, analytical understanding of the rich dynamics of deep learning remains elusive. For example, what determines the time scales over which deep learning unfolds? How does training speed retard with depth? Under what conditions will greedy unsupervised pretraining speed up learning? And how do the final learned internal representations depend on the statistical regularities inherent in the training data?

Here we provide an exact analytical theory of learning in deep linear neural networks that quantitatively answers these questions for this restricted setting. Because of its linearity, the input-output map of a deep linear network can always be rewritten as a shallow network. In this sense, a linear network does not gain expressive power from depth, and hence will underfit and perform poorly on complex real world problems. But while it lacks this important aspect of practical deep learning systems, a deep linear network can nonetheless exhibit highly nonlinear learning dynamics, and these dynamics change with increasing depth. Indeed, the training error, as a function of the network weights, is non-convex, and gradient descent dynamics on this non-convex error surface exhibits a subtle interplay between different weights across multiple layers of the network. Hence deep linear networks provide an important starting point for understanding deep learning dynamics.

To answer these questions, we derive and analyze a set of nonlinear coupled differential equations describing learning dynamics on weight space as a function of the statistical structure of the inputs and outputs. We find exact time-dependent solutions to these nonlinear equations, as well as find conserved quantities in the weight dynamics arising from symmetries in the error function. These solutions provide intuition into how a deep network successively builds up information about the statistical structure of the training data and embeds this information into its weights and internal representations. Moreover, we compare our analytical solutions of learning dynamics in deep linear networks to numerical simulations of learning dynamics in deep non-linear networks, and find that our analytical solutions provide a reasonable approximation. Our solutions also reflect nonlinear phenomena seen in simulations, including alternating plateaus and sharp periods of rapid improvement. Indeed, it has been shown previously [[16](#ref-16)] that this nonlinear learning dynamics in deep linear networks is sufficient to qualitatively capture aspects of the progressive, hierarchical differentiation of conceptual structure seen in infant development. Next, we apply these solutions to investigate the commonly used greedy layer-wise pretraining strategy for training deep networks [[17](#ref-17), [18](#ref-18)], and recover conditions under which such pretraining speeds learning. We show that these conditions are approximately satisfied for the MNIST dataset, and that unsupervised pretraining therefore confers an optimization advantage for deep linear networks applied to MNIST. Finally, we exhibit a new class of random orthogonal initial conditions on weights that, in linear networks, provide depth independent learning times, and we show that these initial conditions also lead to faithful propagation of gradients even in deep nonlinear networks. We further show that these initial conditions also lead to faithful propagation of gradients even in deep nonlinear networks, as long as they operate in a special regime known as the edge of chaos. In this regime, synaptic gains are tuned so that linear amplification due to propagation of neural activity through weight matrices exactly balances dampening of activity due to saturating nonlinearities. In particular, we show that even in nonlinear networks, operating in this special regime, Jacobians that are involved in backpropagating error signals act like near isometries.

<a id="section-1"></a>

## 1 General learning dynamics of gradient descent

We begin by analyzing learning in a three layer network (input, hidden, and output) with linear activation functions (Fig [1](#figure-1)). We let $N_{i}$ be the number of neurons in layer $i$. The input-output map of the network is $y={W^{32}}{W^{21}}x$. We wish to train the network to learn a particular input-output map from a set of $P$ training examples $\left\{x^{\mu},y^{\mu}\right\},\mu=1,\ldots,P$. Training is accomplished via gradient descent on the squared error $\sum_{\mu=1}^{P}\left\|y^{\mu}-{W^{32}}{W^{21}}x^{\mu}\right\|^{2}$ between the desired feature output, and the network’s feature output. This gradient descent procedure yields the batch learning rule

<a id="figure-1"></a>

![network](images/network.png)

> Figure 1: The three layer network analyzed in this section.

$$
\Delta{W^{21}}=\lambda\sum_{\mu=1}^{P}{W^{32}}^{T}\left(y^{\mu}x^{\mu T}-{W^{32}}{W^{21}}x^{\mu}x^{\mu T}\right),\qquad\Delta{W^{32}}=\lambda\sum_{\mu=1}^{P}\left(y^{\mu}x^{\mu T}-{W^{32}}{W^{21}}x^{\mu}x^{\mu T}\right){W^{21}}^{T},(1)
$$

where $\lambda$ is a small learning rate. As long as $\lambda$ is sufficiently small, we can take a continuous time limit to obtain the dynamics,

$$
\tau\frac{d}{dt}{W^{21}}={W^{32}}^{T}\left({\Sigma^{31}}-{W^{32}}{W^{21}}\Sigma^{11}\right),\qquad\tau\frac{d}{dt}{W^{32}}=\left({\Sigma^{31}}-{W^{32}}{W^{21}}\Sigma^{11}\right){W^{21}}^{T},(2)
$$

where $\Sigma^{11}\equiv\sum_{\mu=1}^{P}x^{\mu}x^{\mu T}$ is an $N_{1}\times N_{1}$ input correlation matrix, ${\Sigma^{31}}\equiv\sum_{\mu=1}^{P}y^{\mu}x^{\mu T}$ is an $N_{3}\times N_{1}$ input-output correlation matrix, and $\tau\equiv\frac{1}{\lambda}.$ Here $t$ measures time in units of iterations; as $t$ varies from 0 to 1, the network has seen $P$ examples corresponding to one iteration. Despite the linearity of the network’s input-output map, the gradient descent learning dynamics given in Eqn ([2](#S1.E2)) constitutes a complex set of coupled nonlinear differential equations with up to cubic interactions in the weights.

- [1.1 Learning dynamics with orthogonal inputs](#11-learning-dynamics-with-orthogonal-inputs)
- [1.2 The final outcome of learning](#12-the-final-outcome-of-learning)
- [1.3 The time course of learning](#13-the-time-course-of-learning)

<a id="section-1-1"></a>

### 1.1 Learning dynamics with orthogonal inputs

Our fundamental goal is to understand the dynamics of learning in ([2](#S1.E2)) as a function of the input statistics $\Sigma^{11}$ and input-output statistics ${\Sigma^{31}}$. In general, the outcome of learning will reflect an interplay between input correlations, described by $\Sigma^{11}$, and the input-output correlations described by ${\Sigma^{31}}$. To begin, though, we further simplify the analysis by focusing on the case of orthogonal input representations where $\Sigma^{11}=I$. This assumption will hold exactly for whitened input data, a widely used preprocessing step.

Because we have assumed orthogonal input representations ($\Sigma^{11}=I$), the input-output correlation matrix contains all of the information about the dataset used in learning, and it plays a pivotal role in the learning dynamics. We consider its singular value decomposition (SVD)

$$
{\Sigma^{31}}={U^{33}}{S^{31}}{V^{11}}^{T}=\textstyle\sum_{\alpha=1}^{N_{1}}s_{\alpha}u_{\alpha}v_{\alpha}^{T},(3)
$$

which will be central in our analysis. Here ${V^{11}}$ is an $N_{1}\times N_{1}$ orthogonal matrix whose columns contain input-analyzing singular vectors $v_{\alpha}$ that reflect independent modes of variation in the input, ${U^{33}}$ is an $N_{3}\times N_{3}$ orthogonal matrix whose columns contain output-analyzing singular vectors $u_{\alpha}$ that reflect independent modes of variation in the output, and ${S^{31}}$ is an $N_{3}\times N_{1}$ matrix whose only nonzero elements are on the diagonal; these elements are the singular values $s_{\alpha},\alpha=1,\ldots,N_{1}$ ordered so that $s_{1}\geq s_{2}\geq\cdots\geq s_{N_{1}}$.

Now, performing the change of variables on synaptic weight space, ${W^{21}}={\overline{W}^{21}}{V^{11}}^{T}$, ${W^{32}}={U^{33}}{\overline{W}^{32}}$, the dynamics in ([2](#S1.E2)) simplify to

$$
\tau\frac{d}{dt}{\overline{W}^{21}}={\overline{W}^{32}}^{T}({S^{31}}-{\overline{W}^{32}}{\overline{W}^{21}}),\qquad\tau\frac{d}{dt}{\overline{W}^{32}}=({S^{31}}-{\overline{W}^{32}}{\overline{W}^{21}}){\overline{W}^{21}}^{T}.(4)
$$

To gain intuition for these equations, note that while the matrix elements of ${W^{21}}$ and ${W^{32}}$ connected neurons in one layer to neurons in the next layer, we can think of the matrix element ${\overline{W}^{21}}_{i\alpha}$ as connecting input mode $v_{\alpha}$ to hidden neuron $i$, and the matrix element ${\overline{W}^{32}}_{\alpha i}$ as connecting hidden neuron $i$ to output mode $u_{\alpha}$. Let $a^{\alpha}$ be the $\alpha^{\textrm{th}}$ column of ${\overline{W}^{21}}$, and let $b^{\alpha T}$ be the $\alpha^{\textrm{th}}$ row of ${\overline{W}^{32}}$. Intuitively, $a^{\alpha}$ is a column vector of $N_{2}$ synaptic weights presynaptic to the hidden layer coming from input mode $\alpha$, and $b^{\alpha}$ is a column vector of $N_{2}$ synaptic weights postsynaptic to the hidden layer going to output mode $\alpha$. In terms of these variables, or connectivity modes, the learning dynamics in ([4](#S1.E4)) become

$$
\tau\frac{d}{dt}a^{\alpha}=(s_{\alpha}-a^{\alpha}\cdot b^{\alpha})\,b^{\alpha}-\sum_{\gamma\neq\alpha}b^{\gamma}\,(a^{\alpha}\cdot b^{\gamma}),\qquad\tau\frac{d}{dt}b^{\alpha}=(s_{\alpha}-a^{\alpha}\cdot b^{\alpha})\,a^{\alpha}-\sum_{\gamma\neq\alpha}a^{\gamma}\,(b^{\alpha}\cdot a^{\gamma}).(5)
$$

Note that $s_{\alpha}=0$ for $\alpha>N_{1}$. These dynamics arise from gradient descent on the energy function

$$
E=\frac{1}{2\tau}\sum_{\alpha}(s_{\alpha}-a^{\alpha}\cdot b^{\alpha})^{2}+\frac{1}{2\tau}\sum_{\alpha\neq\beta}(a^{\alpha}\cdot b^{\beta})^{2},(6)
$$

and display an interesting combination of cooperative and competitive interactions. Consider the first terms in each equation. In these terms, the connectivity modes from the two layers, $a^{\alpha}$ and $b^{\alpha}$ associated with the same input-output mode of strength $s^{\alpha}$, cooperate with each other to drive each other to larger magnitudes as well as point in similar directions in the space of hidden units; in this fashion these terms drive the product of connectivity modes $a^{\alpha}\cdot b^{\alpha}$ to reflect the input-output mode strength $s^{\alpha}$. The second terms describe competition between the connectivity modes in the first ($a^{\alpha}$) and second ($b^{\beta}$) layers associated with different input modes $\alpha$ and $\beta$. This yields a symmetric, pairwise repulsive force between all distinct pairs of first and second layer connectivity modes, driving the network to a decoupled regime in which the different connectivity modes become orthogonal.

<a id="section-1-2"></a>

### 1.2 The final outcome of learning

The fixed point structure of gradient descent learning in linear networks was worked out in [[19](#ref-19)]. In the language of the connectivity modes, a necessary condition for a fixed point is $a^{\alpha}\cdot b^{\beta}=s_{\alpha}\delta_{\alpha\beta}$, while $a^{\alpha}$ and $b^{\alpha}$ are zero whenever $s_{\alpha}=0$. To satisfy these relations for undercomplete hidden layers ($N_{2}<N_{1},N_{2}<N_{3}$), $a^{\alpha}$ and $b^{\alpha}$ can be nonzero for at most $N_{2}$ values of $\alpha$. Since there are $\textrm{rank}({\Sigma^{31}})\equiv r$ nonzero values of $s_{\alpha}$, there are $\begin{pmatrix}r\\ N_{2}\end{pmatrix}$ families of fixed points. However, all of these fixed points are unstable, except for the one in which only the first $N_{2}$ strongest modes, i.e. $a^{\alpha}$ and $b^{\alpha}$ for $\alpha=1,\ldots,N_{2}$ are active. Thus remarkably, the dynamics in ([5](#S1.E5)) has only saddle points and no non-global local minima [[19](#ref-19)]. In terms of the original synaptic variables ${W^{21}}$ and ${W^{32}}$, all globally stable fixed points satisfy

$$
{W^{32}}{W^{21}}=\textstyle\sum_{\alpha=1}^{N_{2}}s_{\alpha}u_{\alpha}v_{\alpha}^{T}.(7)
$$

Hence when learning has converged, the network will represent the closest rank $N_{2}$ approximation to the true input-output correlation matrix. In this work, we are interested in understanding the dynamical weight trajectories and learning time scales that lead to this final fixed point.

<a id="section-1-3"></a>

### 1.3 The time course of learning

It is difficult though to exactly solve ([5](#S1.E5)) starting from arbitrary initial conditions because of the competitive interactions between different input-output modes. Therefore, to gain intuition for the general dynamics, we restrict our attention to a special class of initial conditions of the form $a^{\alpha}$ and $b^{\alpha}\propto r^{\alpha}$ for $\alpha=1,\ldots,N_{2}$, where $r^{\alpha}\cdot r^{\beta}=\delta_{\alpha\beta}$, with all other connectivity modes $a^{\alpha}$ and $b^{\alpha}$ set to zero (see [[20](#ref-20)] for solutions to a partially overlapping but distinct set of initial conditions, further discussed in Supplementary Appendix [A](#appendix-a)).

Here $r^{\alpha}$ is a fixed collection of $N_{2}$ vectors that form an orthonormal basis for synaptic connections from an input or output mode onto the set of hidden units. Thus for this set of initial conditions, $a^{\alpha}$ and $b^{\alpha}$ point in the same direction for each alpha and differ only in their scalar magnitudes, and are orthogonal to all other connectivity modes. Such an initialization can be obtained by computing the SVD of ${\Sigma^{31}}$ and taking ${W^{32}}={U^{33}}D_{a}R^{T},{W^{21}}=RD_{b}{V^{11}}^{T}$ where $D_{a},D_{b}$ are diagonal, and $R$ is an arbitrary orthogonal matrix; however, as we show

in subsequent experiments, the solutions we find are also excellent approximations to trajectories from small random initial conditions. It is straightforward to verify that starting from these initial conditions, $a^{\alpha}$ and $b^{\alpha}$ will remain parallel to $r^{\alpha}$ for all future time. Furthermore, because the different active modes are orthogonal to each other, they do not compete, or even interact with each other (all dot products in the second terms of ([5](#S1.E5))-([6](#S1.E6)) are 0 0).

Thus this class of conditions defines an invariant manifold in weight space where the modes evolve independently of each other.

If we let $a=a^{\alpha}\cdot r^{\alpha}$, $b=b^{\alpha}\cdot r^{\alpha}$, and $s=s^{\alpha}$, then the dynamics of the scalar projections $(a,b)$ obeys,

$$
\tau\frac{d}{dt}a=b\,(s-ab),\qquad\tau\frac{d}{dt}b=a\,(s-ab).(8)
$$

Thus our ability to decouple the connectivity modes yields a dramatically simplified two dimensional nonlinear system. These equations can by solved by noting that they arise from gradient descent on the error,

$$
E(a,b)=\textstyle\frac{1}{2\tau}(s-ab)^{2}.(9)
$$

This implies that the product $ab$ monotonically approaches the fixed point $s$ from its initial value. Moreover, $E(a,b)$ satisfies a symmetry under the one parameter family of scaling transformations $a\rightarrow\lambda a$, $b\rightarrow\frac{b}{\lambda}$. This symmetry implies, through Noether’s theorem, the existence of a conserved quantity, namely $a^{2}-b^{2}$, which is a constant of motion. Thus the dynamics simply follows hyperbolas of constant $a^{2}-b^{2}$ in the $(a,b)$ plane until it approaches the hyperbolic manifold of fixed points, $ab=s$. The origin $a=0,b=0$ is also a fixed point, but is unstable. Fig. [2](#figure-2) shows a typical phase portrait for these dynamics.

<a id="figure-2"></a>

![network](images/network.png)

> Figure 2: Vector field (blue), stable manifold (red) and two solution trajectories (green) for the two dimensional dynamics of $a$ and $b$ in ([8](#S1.E8)), with $\tau=1,s=1$.

As a measure of the timescale of learning, we are interested in how long it takes for $ab$ to approach $s$ from any given initial condition. The case of unequal $a$ and $b$ is treated in the Supplementary Appendix [A](#appendix-a) due to space constraints. Here we pursue an explicit solution with the assumption that $a=b$, a reasonable limit when starting with small random initial conditions. We can then track the dynamics of $u\equiv ab$, which from ([8](#S1.E8)) obeys

$$
\tau\frac{d}{dt}u=2u(s-u).(10)
$$

This equation is separable and can be integrated to yield

$$
t=\tau\int_{u_{0}}^{u_{f}}\frac{du}{2u(s-u)}=\frac{\tau}{2s}\ln\frac{u_{f}(s-u_{0})}{u_{0}(s-u_{f})}.(11)
$$

Here $t$ is the time it takes for $u$ to travel from $u_{0}$ to $u_{f}$. If we assume a small initial condition $u_{0}=\epsilon$, and ask when $u_{f}$ is within $\epsilon$ of the fixed point $s$, i.e. $u_{f}=s-\epsilon$, then the learning timescale in the limit $\epsilon\rightarrow 0$ is $t=\tau/s\ln\left(s/\epsilon\right)=O(\tau/s)$ (with a weak logarithmic dependence on the cutoff). This yields a key result: the timescale of learning of each input-output mode $\alpha$ of the correlation matrix ${\Sigma^{31}}$ is inversely proportional to the correlation strength $s_{\alpha}$ of the mode. Thus the stronger an input-output relationship, the quicker it is learned.

We can also find the entire time course of learning by inverting ([11](#S1.E11)) to obtain

$$
u_{f}(t)=\frac{se^{2st/\tau}}{e^{2st/\tau}-1+s/u_{0}}.(12)
$$

This time course describes the temporal evolution of the product of the magnitudes of all weights from an input mode (with correlation strength $s$) into the hidden layers, and from the hidden layers to the same output mode. If this product starts at a small value $u_{0}<s$, then it displays a sigmoidal rise which asymptotes to $s$ as $t\rightarrow\infty$. This sigmoid can exhibit sharp transitions from a state of no learning to full learning. This analytical sigmoid learning curve is shown in Fig. [3](#figure-3) to yield a reasonable approximation to learning curves in linear networks that start from random initial conditions that are not on the orthogonal, decoupled invariant manifold–and that therefore exhibit competitive dynamics between connectivity modes–as well as in nonlinear networks solving the same task. We note that though the nonlinear networks behaved similarly to the linear case for this particular task, this is likely to be problem dependent.

<a id="figure-3"></a>

<div align="center">
  <img src="images/network.png" width="45%" alt="network" />
  <img src="images/vector_field.png" width="45%" alt="vector_field" />
</div>

> Figure 3: Left: Dynamics of learning in a three layer neural network. Curves show the strength of the network’s representation of seven modes of the input-output correlation matrix over the course of learning. Red traces show analytical curves from Eqn. [12](#S1.E12). Blue traces show simulation of full dynamics of a linear network (Eqn. ([2](#S1.E2))) from small random initial conditions. Green traces show simulation of a nonlinear three layer network with tanh activation functions. To generate mode strengths for the nonlinear network, we computed the nonlinear network’s evolving input-output correlation matrix, and plotted the diagonal elements of ${U^{33}}^{T}{\Sigma^{31}}_{tanh}{V^{11}}$ over time. The training set consists of 32 orthogonal input patterns, each associated with a 1000-dimensional feature vector generated by a hierarchical diffusion process described in [[16](#ref-16)] with a five level binary tree and flip probability of 0.1. Modes 1, 2, 3, 5, 12, 18, and 31 are plotted with the rest excluded for clarity. Network training parameters were $\lambda=0.5e^{-3},N_{2}=32,u_{0}=1e^{-6}$. Right: Delay in learning due to competitive dynamics and sigmoidal nonlinearities. Vertical axis shows the difference between simulated time of half learning and the analytical time of half learning, as a fraction of the analytical time of half learning. Error bars show standard deviation from 100 simulations with random initializations.

<a id="section-2"></a>

## 2 Deeper multilayer dynamics

The network analyzed in Section [1](#section-1) is the minimal example of a multilayer net, with just a single layer of hidden units. How does gradient descent act in much deeper networks? We make an initial attempt in this direction based on initial conditions that yield particularly simple gradient descent dynamics.

In a linear neural network with $N_{l}$ layers and hence $N_{l}-1$ weight matrices indexed by $W^{l},l=1,\cdots,N_{l}-1$, the gradient descent dynamics can be written as

$$
\tau\frac{d}{dt}W^{l}=\left(\prod_{i=l+1}^{N_{l}-1}W^{i}\right)^{T}\left[{\Sigma^{31}}-\left(\prod_{i=1}^{N_{l}-1}W^{i}\right)\Sigma^{11}\right]\left(\prod_{i=1}^{l-1}W^{i}\right)^{T},(13)
$$

where $\prod_{i=a}^{b}W^{i}=W^{b}W^{(b-1)}\cdots W^{(a-1)}W^{a}$ with the special case that $\prod_{i=a}^{b}W^{i}=I$, the identity, if $a>b$.

To describe the initial conditions, we suppose that there are $N_{l}$ orthogonal matrices $R_{l}$ that diagonalize the starting weight matrices, that is, $R_{l+1}^{T}W_{l}(0)R_{l}=D_{l}$ for all $l$, with the special case that $R_{1}={V^{11}}$ and $R_{N_{l}}={U^{33}}$. This requirement essentially demands that the output singular vectors of layer $l$ be the input singular vectors of the next layer $l+1$, so that a change in mode strength at any layer propagates to the output without mixing into other modes. We note that this formulation does not restrict hidden layer size; each hidden layer can be of a different size, and may be undercomplete or overcomplete. Making the change of variables $W_{l}=R_{l+1}\overline{W}_{l}R_{l}^{T}$ along with the assumption that $\Sigma^{11}=I$ leads to a set of decoupled connectivity modes that evolve independently of each other. In analogy to the simplification occurring in the three layer network from ([2](#S1.E2)) to ([8](#S1.E8)), each connectivity mode in the $N_{l}$ layered network can be described by $N_{l}-1$ scalars $a^{1},\dots,a^{N_{l}-1}$, whose dynamics obeys gradient descent on the energy function (the analog of ([9](#S1.E9))),

$$
E(a_{1},\cdots,a_{N_{l}-1})=\frac{1}{2\tau}\left(s-\prod_{i=1}^{N_{l}-1}a_{i}\right)^{2}.(14)
$$

This dynamics also has a set of conserved quantities $a_{i}^{2}-a_{j}^{2}$ arising from the energetic symmetry w.r.t. the transformation $a_{i}\rightarrow\lambda a_{i}$, $a_{j}\rightarrow\frac{a_{j}}{\lambda}$, and hence can be solved exactly. We focus on the invariant submanifold in which $a_{i}(t=0)=a_{0}$ for all $i$, and track the dynamics of $u=\textstyle\prod_{i=1}^{N_{l}-1}a_{i}$, the overall strength of this mode, which obeys (i.e. the generalization of ([10](#S1.E10))),

$$
\displaystyle\tau\frac{d}{dt}u=(N_{l}-1)u^{2-2/(N_{l}-1)}(s-u).(15)
$$

This can be integrated for any positive integer $N_{l}$, though the expression is complicated. Once the overall strength increases sufficiently, learning explodes rapidly.

Eqn. ([15](#S2.E15)) lets us study the dynamics of learning as depth limits to infinity. In particular, as $N_{l}\rightarrow\infty$ we have the dynamics

$$
\tau\frac{d}{dt}u=N_{l}u^{2}(s-u)(16)
$$

which can be integrated to obtain

$$
t=\frac{\tau}{N_{l}}\left[\frac{1}{s^{2}}\log\left(\frac{u_{f}(u_{0}-s)}{u_{0}(u_{f}-s)}\right)+\frac{1}{su_{0}}-\frac{1}{su_{f}}\right].(17)
$$

Remarkably this implies that, for a fixed learning rate, the learning time as measured by the number of iterations required tends to zero as $N_{l}$ goes to infinity. This result depends on the continuous time formulation, however. Any implementation will operate in discrete time and must choose a finite learning rate that yields stable dynamics. An estimate of the optimal learning rate can be derived from the maximum eigenvalue of the Hessian over the region of interest.

For linear networks with $a_{i}=a_{j}=a$, this optimal learning rate $\alpha_{opt}$ decays with depth as $O\left(\frac{1}{N_{l}s^{2}}\right)$ for large $N_{l}$ (see Supplementary Appendix [B](#appendix-b)). Incorporating this dependence of the learning rate on depth, the learning time as depth approaches infinity still surprisingly remains finite: with the optimal learning rate, the difference between learning times for an $N_{l}=3$ network and an $N_{l}=\infty$ network is $t_{\infty}-t_{3}\sim O\left(s/\epsilon\right)$ for small $\epsilon$ (see Supplementary Appendix [B.1](#A2.SS1)). We emphasize that our analysis of learning speed is based on the number of iterations required, not the amount of computation–computing one iteration of a deep network will require more time than doing so in a shallow network.

To verify these predictions, we trained deep linear networks on the MNIST classification task with depths ranging from $N_{l}=3$ to $N_{l}=100$. We used hidden layers of size 1000, and calculated the iteration at which training error fell below a fixed threshold corresponding to nearly complete learning. We optimized the learning rate separately for each depth by training each network with twenty rates logarithmically spaced between $10^{-4}$ and $10^{-7}$ and picking the fastest. See Supplementary Appendix [C](#appendix-c) for full experimental details. Networks were initialized with decoupled initial conditions and starting initial mode strength $u_{0}=0.001$. Fig. [4](#figure-4) shows the resulting learning times, which saturate, and the empirically optimal learning rates, which scale like $O(1/N_{l})$ as predicted.

<a id="figure-4"></a>

![network](images/network.png)

> Figure 4: Left: Learning time as a function of depth on MNIST. Right: Empirically optimal learning rates as a function of depth.

Thus learning times in deep linear networks that start with decoupled initial conditions are only a finite amount slower than a shallow network regardless of depth. Moreover, the delay incurred by depth scales inversely with the size of the initial strength of the association. Hence finding a way to initialize the mode strengths to large values is crucial for fast deep learning.

<a id="section-3"></a>

## 3 Finding good weight initializations: on greediness and randomness

The previous subsection revealed the existence of a decoupled submanifold in weight space in which connectivity modes evolve independently of each other during learning, and learning times can be independent of depth, even for arbitrarily deep networks, as long as the initial composite, end to end mode strength, denoted by $u$ above, of every connectivity mode is $O(1)$. What numerical weight initilization procedures can get us close to this weight manifold, so that we can exploit its rapid learning properties?

A breakthrough in training deep neural networks started with the discovery that greedy layer-wise unsupervised pretraining could substantially speed up and improve the generalization performance of standard gradient descent [[17](#ref-17), [18](#ref-18)]. Unsupervised pretraining has been shown to speed the optimization of deep networks, and also to act as a special regularizer towards solutions with better generalization performance [[18](#ref-18), [12](#ref-12), [13](#ref-13), [14](#ref-14)]. At the same time, recent results have obtained excellent performance starting from carefully-scaled random initializations, though interestingly, pretrained initializations still exhibit faster convergence [[21](#ref-21), [13](#ref-13), [22](#ref-22), [3](#ref-3), [4](#ref-4), [1](#ref-1), [23](#ref-23)] (see Supplementary Appendix [D](#appendix-d) for discussion). Here we examine analytically how unsupervised pretraining achieves an optimization advantage, at least in deep linear networks, by finding the special class of orthogonalized, decoupled initial conditions in the previous section that allow for rapid supervised deep learning, for input-output tasks with a certain precise structure. Subsequently, we analyze the properties of random initilizations.

We consider the following pretraining and finetuning procedure: First, using autoencoders as the unsupervised pretraining module [[18](#ref-18), [12](#ref-12)], the network is trained to produce its input as its output ($y_{\textrm{pre}}^{\mu}=x^{\mu}$). Subsequently, the network is finetuned on the ultimate input-output task of interest (e.g., a classification task). In the following we consider the case $N_{2}=N_{1}$ for simplicity.

During the pretraining phase, the input-output correlation matrix ${\Sigma^{31}}_{\textrm{pre}}$ is simply the input correlation matrix $\Sigma^{11}$. Hence the SVD of ${\Sigma^{31}}_{\textrm{pre}}$ is PCA on the input correlation matrix, since ${\Sigma^{31}}_{\textrm{pre}}=\Sigma^{11}=Q\Lambda Q^{T}$, where $Q$ are eigenvectors of $\Sigma^{11}$ and $\Lambda$ is a diagonal matrix of variances. Our analysis of the learning dynamics in Section [1.1](#section-1-1) does not directly apply, because here the input correlation matrix is not white. In Supplementary Appendix [E](#appendix-e) we generalize our results to handle this case. During pretraining, the weights approach ${W^{32}}{W^{21}}={\Sigma^{31}}({\Sigma^{31}})^{-1},$ but since they do not reach the fixed point in finite time, they will end at ${W^{32}}{W^{21}}=QMQ^{T}$ where $M$ is a diagonal matrix that is approaching the identity matrix during learning. Hence in general, ${W^{32}}=QM^{1/2}C^{-1}$ and ${W^{21}}=CM^{1/2}Q^{T}$ where $C$ is any invertible matrix. When starting from small random weights, though, each weight matrix will end up with a roughly balanced contribution to the overall map. This corresponds to having $C\approx R_{2}$ where $R_{2}$ is orthogonal. Hence at the end of the pretraining phase, the input-to-hidden mapping will be ${W^{21}}=R_{2}M^{1/2}Q^{T}$ where $R_{2}$ is an arbitrary orthogonal matrix.

Now consider the fine-tuning phase. Here the weights are trained on the ultimate task of interest with input-output correlations ${\Sigma^{31}}={U^{33}}{S^{31}}{V^{11}}$. The matrix ${W^{21}}$ begins from the pretrained initial condition ${W^{21}}=R_{2}M^{1/2}Q^{T}$. For the fine-tuning task, a decoupled initial condition for ${W^{21}}$ is one that can be written as ${W^{21}}=R_{2}D_{1}{V^{11}}^{T}$ (see Section [2](#section-2)). Clearly, this will be possible only if

$$
Q={V^{11}}.(18)
$$

Then the initial condition obtained from pretraining will also be a decoupled initial condition for the finetuning phase, with initial mode strengths $D_{1}=M^{1/2}$ near one. Hence we can state the underlying condition required for successful greedy pretraining in deep linear networks: the right singular vectors of the ultimate input-ouput task of interest ${V^{11}}$ must be similar to the principal components of the input data $Q$. This is a quantitatively precise instantiation of the intuitive idea that unsupervised pretraining can help in a subsequent supervised learning task if (and only if) the statistical structure of the input is consistent with the structure of input-output map to be learned. Moreover, this quantitative instantiation of this intuitive idea gives a simple empirical criterion that can be evaluated on any new dataset: given the input-output correlation ${\Sigma^{31}}$ and input correlation $\Sigma^{11}$, compute the right singular vectors ${V^{11}}$ of ${\Sigma^{31}}$ and check that ${V^{11}}\Sigma^{11}{V^{11}}^{T}$ is approximately diagonal. If the condition in Eqn. ([18](#S3.E18)) holds, autoencoder pretraining will have properly set up decoupled initial conditions for ${W^{21}}$, with an appreciable initial association strength near $1$. This argument also goes through straightforwardly for layer-wise pretraining of deeper networks. Fig. [5](#figure-5) shows that this consistency condition empirically holds on MNIST, and that a pretrained deep linear neural network learns faster than one started from small random initial conditions, even accounting for pretraining time (see Supplementary Appendix [F](#appendix-f) for experimental details). We note that this analysis is unlikely to carry over completely to nonlinear networks. Some nonlinear networks are approximately linear (e.g., tanh nonlinearities) after initialization with small random initializations, and hence our solutions may describe these dynamics well early in learning. However as the network enters its nonlinear regime, our solutions should not be expected to remain accurate.

<a id="figure-5"></a>

![network](images/network.png)

> Figure 5: MNIST satisfies the consistency condition for greedy pretraining. Left: Submatrix from the raw MNIST input correlation matrix $\Sigma^{11}$. Center: Submatrix of ${V^{11}}\Sigma^{11}{V^{11}}^{T}$ which is approximately diagonal as required. Right: Learning curves on MNIST for a five layer linear network starting from random (black) and pretrained (red) initial conditions. Pretrained curve starts with a delay due to pretraining time. The small random initial conditions correspond to all weights chosen i.i.d. from a zero mean Gaussian with standard deviation 0.01.

As an alternative to greedy layerwise pre-training, [[13](#ref-13)] proposed choosing appropriately scaled initial conditions on weights that would preserve the norm of typical error vectors as they were backpropagated through the deep network. In our context, the appropriate norm-preserving scaling for the initial condition of an $N$ by $N$ connectivity matrix $W$ between any two layers corresponds to choosing each weight i.i.d. from a zero mean Gaussian with standard deviation $1/\sqrt{N}$. With this choice, $\langle v^{T}W^{T}Wv\rangle_{W}=v^{T}v$, where $\langle\cdot\rangle_{W}$ denotes an average over distribution of the random matrix $W$. Moreover, the distribution of $v^{T}W^{T}Wv$ concentrates about its mean for large $N$. Thus with this scaling, in linear networks, both the forward propagation of activity, and backpropagation of gradients is typically norm-preserving. However, with this initialization, the learning time with depth on linear networks trained on MNIST grows with depth (Fig. [6](#figure-6)A, left, blue). This growth is in distinct contradiction with the theoretical prediction, made above, of depth independent learning times starting from the decoupled submanifold of weights with composite mode strength $O(1)$. This suggests that the scaled random initialization scheme, despite its norm-preserving nature, does not find this submanifold in weight space. In contrast, learning times with greedy layerwise pre-training do not grow with depth (Fig. [6](#figure-6)A, left, green curve hiding under red curve), consistent with the predictions of our theory (as a technical point: note that learning times under greedy pre-training initialization in Fig. [6](#figure-6)A are faster than those obtained in Fig. [4](#figure-4) by explicitly choosing a point on the decoupled submanifold, because there the initial mode strength was chosen to be small ($u=0.001$) whereas greedy pre-training finds a composite mode strength closer to $1$).

<a id="figure-6"></a>

![vector_field](images/vector_field.png)

> Figure 6: A Left: Learning time (on MNIST using the same architecture and parameters as in Fig. [4](#figure-4)) as a function of depth for different initial conditions on weights (scaled i.i.d. uniform weights chosen to preserve the norm of propagated gradients as proposed in [[13](#ref-13)] (blue), greedy unsupervised pre-training (green) and random orthogonal matrices (red). The red curve lies on top of the green curve. Middle: Optimal learning rates as a function of depth for different weight initilizations. Right: The eigenvalue spectrum, in the complex plane, of a random $100$ by $100$ orthogonal matrix. B Histograms of the singular values of products of $N_{l}-1$ independent random Gaussian $N$ by $N$ matrices whose elements themselves are chosen i.i.d. from a zero mean Gaussian with standard deviation $1/\sqrt{N}$. In all cases, $N=1000$, and histograms are taken over $500$ realizations of such random product matrices, yielding a total $5\cdot 10^{5}$ singular values in each histogram. C Histograms of the eigenvalue distributions on the complex plane of the same product matrices in B. The bin width is 0.1, and, for visualization purposes, the bin containing the origin has been removed in each case; this bin would otherwise dominate the histogram in the middle and right plots, as it contains $32\%$ and $94\%$ of the eigenvalues respectively.

Is there a simple random initialization scheme that does enjoy the rapid learning properties of greedy-layerwise pre-training? We empirically show (Fig. [6](#figure-6)A, left, red curve) that if we choose the initial weights in each layer to be a random orthogonal matrix (satisifying $W^{T}W=I$), instead of a scaled random Gaussian matrix, then this orthogonal random initialization condition yields depth independent learning times just like greedy layerwise pre-training (indeed the red and green curves are indistinguishable). Theoretically, why do random orthogonal initializations yield depth independent learning times, but not scaled random Gaussian initializations, despite their norm preserving nature?

The answer lies in the eigenvalue and singular value spectra of products of Gaussian versus orthgonal random matrices. While a single random orthogonal matrix has eigenvalue spectra lying exactly on the unit circle in the complex plane (Fig. [6](#figure-6)A right), the eigenvalue spectra of random Gaussian matrices, whose elements have variance $1/N$, form a uniform distribution on a solid disk of radius 1 the complex plane (Fig. [6](#figure-6)C left). Moreover the singular values of an orthogonal matrix are all exactly $1$, while the squared singular values of a scaled Gaussian random matrix have the well known Marcenko-Pasteur distribution, with a nontrivial spread even as $N\rightarrow\infty$, (Fig. [6](#figure-6)B left shows the distribution of singular values themselves). Now consider a product of these matrices across all $N_{l}$ layers, representing the total end to end propagation of activity across a deep linear network:

$$
W_{\text{Tot}}=\prod_{i=1}^{N_{l}-1}W^{(i+1,i)}.(19)
$$

Due to the random choice of weights in each layer, $W_{\text{Tot}}$ is itself a random matrix. On average, it preserves the norm of a typical vector $v$ no matter whether the matrices in each layer are Gaussian or orthogonal. However, the singular value spectra of $W_{\text{Tot}}$ differ markedly in the two cases. Under random orthogonal initilization in each layer, $W_{\text{Tot}}$ is itself an orthogonal matrix and therefore has all singular values equal to $1$. However, under random Gaussian initialization in each layer, there is as of yet no complete theoretical characterization of the singular value distribution of $W_{\text{Tot}}$. We have computed it numerically as a function of different depths in Fig. [6](#figure-6)B, and we find that it develops a highly kurtotic nature as the depth increases. Most of the singular values become vanishingly small, while a long tail of very large singular values remain. Thus $W_{\text{Tot}}$ preserves the norm of a typical, randomly chosen vector $v$, but in a highly anisotropic manner, by strongly amplifying the projection of $v$ onto a very small subset of singular vectors and attenuating $v$ in all other directions. Intuitively $W_{\text{Tot}}$, as well as the linear operator $W_{\text{Tot}}^{T}$ that would be closely related to backpropagation of gradients to early layers, act as amplifying projection operators at large depth $N_{l}$. In contrast, all of the eigenvalues of $W_{\text{Tot}}$ in the scaled Gaussian case concentrate closer to the origin as depth increases. This discrepancy between the behavior of the eigenvalues and singular values of $W_{\text{Tot}}$, a phenomenon that could occur only if the eigenvectors of $W_{\text{Tot}}$ are highly non-orthogonal, reflects the highly non-normal nature of products of random Gaussian matrices (a non-normal matrix is by definition a matrix whose eigenvectors are non-orthogonal).

While the combination of amplification and projection in $W_{\text{Tot}}$ can preserve norm, it is clear that it is not a good way to backpropagate errors; the projection of error vectors onto a high dimensional subspace corresponding to small singular values would be strongly attenuated, yielding vanishingly small gradient signals corresponding to these directions in the early layers. This effect, which is not present for random orthogonal initializations or greedy pretraining, would naturally explain the long learning times starting from scaled random Gaussian initial conditions relative to the other initilizations in Fig. [6](#figure-6)A left. For both linear and nonlinear networks, a more likely appropriate condition on weights for generating fast learning times would be that of dynamical isometry. By this we mean that the product of Jacobians associated with error signal backpropagation should act as a near isometry, up to some overall global $O(1)$ scaling, on a subspace of as high a dimension as possible. This is equivalent to having as many singular values of the product of Jacobians as possible within a small range around an $O(1)$ constant, and is closely related to the notion of restricted isometry in compressed sensing and random projections. Preserving norms is a necessary but not sufficient condition for achieving dynamical isometry at large depths, as demonstrated in Fig. [6](#figure-6)B, and we have shown that for linear networks, orthogonal initializations achieve exact dynamical isometry with all singular values at $1$, while greedy pre-training achieves it approximately.

We note that the discrepancy in learning times between the scaled Gaussian initialization and the orthogonal or pre-training initializations is modest for the depths of around $6$ used in large scale applications, but is magnified at larger depths (Fig. [6](#figure-6)A left). This may explain the modest improvement in learning times with greedy pre-training versus random scaled Gaussian initializations observed in applications (see discussion in Supplementary Appendix [D](#appendix-d)). We predict that this modest improvement will be magnified at higher depths, even in nonlinear networks. Finally, we note that in recurrent networks, which can be thought of as infinitely deep feed-forward networks with tied weights, a very promising approach is a modification to the training objective that partially promotes dynamical isometry for the set of gradients currently being back-propagated [[24](#ref-24)].

<a id="section-4"></a>

## 4 Achieving approximate dynamical isometry in nonlinear networks

We have shown above that deep random orthogonal linear networks achieve perfect dynamical isometry. Here we show that nonlinear versions of these networks can also achieve good dynamical isometry properties. Consider the nonlinear feedforward dynamics

$$
x^{l+1}_{i}=\sum_{j}\,g\,W^{(l+1,l)}_{ij}\,\phi(x^{l}_{j}),(20)
$$

where $x^{l}_{i}$ denotes the activity of neuron $i$ in layer $l$, $W^{(l+1,l)}_{ij}$ is a random orthogonal connectivity matrix from layer $l$ to $l+1$, $g$ is a scalar gain factor, and $\phi(x)$ is any nonlinearity that saturates as $x\rightarrow\pm\infty$. We show in Supplementary appendix [G](#appendix-g) that there exists a critical value $g_{c}$ of the gain $g$ such that if $g<g_{c}$, activity will decay away to zero as it propagates through the layers, while if $g>g_{c}$, the strong linear positive gain will combat the damping due to the saturating nonlinearity, and activity will propagate indefinitely without decay, no matter how deep the network is. When the nonlinearity is odd ($\phi(x)=-\phi(-x)$), so that the mean activity in each layer is approximately 0 0, these dynamical properties can be quantitatively captured by the neural population variance in layer $l$,

$$
q^{l}\equiv\frac{1}{N}\sum_{i=1}^{N}(x^{l}_{i})^{2}.(21)
$$

Thus $\lim_{l\rightarrow\infty}q^{l}\rightarrow 0$ for $g<g_{c}$ and $\lim_{l\rightarrow\infty}q^{l}\rightarrow q^{\infty}(g)>0$ for $g>g_{c}$. When $\phi(x)=\tanh(x)$, we compute $g_{c}=1$ and numerically compute $q^{\infty}(g)$ in Fig. [8](#A7.F8) in Supplementary appendix [G](#appendix-g). Thus these nonlinear feedforward networks exhibit a phase-transition at the critical gain; above the critical gain, infinitely deep networks exhibit chaotic percolating activity propagation, so we call the critical gain $g_{c}$ the edge of chaos, in analogy with terminology for recurrent networks.

Now we are interested in how errors at the final layer $N_{l}$ backpropagate back to earlier layers, and whether or not these gradients explode or decay with depth. To quantify this, for simplicity we consider the end to end Jacobian

$$
J^{N_{l},1}_{ij}(x^{N_{l}})\equiv\frac{\partial x^{N_{l}}_{i}}{\partial x^{1}_{j}}\bigg{|}_{x^{N_{l}}},(22)
$$

which captures how input perturbations propagate to the output. If the singular value distribution of this Jacobian is well-behaved, with few extremely large or small singular values, then the backpropagation of gradients will also be well-behaved, and exhibit little explosion or decay. The Jacobian is evaluated at a particular point $x^{N_{l}}$ in the space of output layer activations, and this point is in turn obtained by iterating ([20](#S4.E20)) starting from an initial input layer activation vector $x^{1}$. Thus the singular value distribution of the Jacobian will depend not only on the gain $g$, but also on the initial condition $x^{1}$. By rotational symmetry, we expect this distribution to depend on $x^{1}$, only through its population variance $q^{1}$. Thus for large $N$, the singular value distribution of the end-to-end Jacobian in ([22](#S4.E22)) (the analog of $W_{\text{Tot}}$ in ([19](#S3.E19)) in the linear case), depends on only two parameters: gain $g$ and input population variance $q^{1}$.

We have numerically computed this singular value distribution as a function of these two parameters in Fig. [7](#figure-7), for a single random orthogonal nonlinear network with $N=1000$ and $N_{l}=100$. These results are typical; replotting the results for different random networks and different initial conditions (with the same input variance) yield very similar results. We see that below the edge of chaos, when $g<1$, the linear dampening over many layers yields extremely small singular values. Above the edge of chaos, when $g>1$, the combination of positive linear amplification, and saturating nonlinear dampening yields an anisotropic distribution of singular values. At the edge of chaos, $g=1$, an $O(1)$ fraction of the singular value distribution is concentrated in a range that remains $O(1)$ despite $100$ layers of propagation, reflecting appoximate dynamical isometry. Moreover, this nice property at $g=1$ remains valid even as the input variance $q^{1}$ is increased far beyond $1$, where the $\tanh$ function enters its nonlinear regime. Thus the right column of Fig. [7](#figure-7) at $g$ near $1$ indicates that the useful dynamical isometry properties of random orthogonal linear networks described above survives in nonlinear networks, even when activity patterns enter deeply into the nonlinear regime in the input layers. Interestingly, the singular value spectrum is more robust to perturbations that increase $g$ from $1$ relative to those that decrease $g$. Indeed, the anisotropy in the singular value distribution at $g=1.1$ is relatively mild compared to that of random linear networks with scaled Gaussian initial conditions (compare the bottom row of Fig. [7](#figure-7) with the right column of panel B in Fig. [6](#figure-6)). Thus overall, these numerical results suggest that being just beyond the edge of orthogonal chaos may be a good regime for learning in deep nonlinear networks.

<a id="figure-7"></a>

![network](images/network.png)

> Figure 7: Singular value distribution of the end to end Jacobian, defined in ([22](#S4.E22)), for various values of the gain $g$ in ([20](#S4.E20)) and the input layer population variance $q=q^{1}$ in ([21](#S4.E21)). The network architecture consists of $N_{l}=100$ layers with $N=1000$ neurons per layer, as in the linear case in Fig. [6](#figure-6)B.

<a id="section-5"></a>

## 5 Discussion

In summary, despite the simplicity of their input-output map, the dynamics of learning in deep linear networks reveals a surprising amount of rich mathematical structure, including nonlinear hyperbolic dynamics, plateaus and sudden performance transitions, a proliferation of saddle points, symmetries and conserved quantities, invariant submanifolds of independently evolving connectivity modes subserving rapid learning, and most importantly, a sensitive but computable dependence of learning time scales on input statistics, initial weight conditions, and network depth. With the right initial conditions, deep linear networks can be only a finite amount slower than shallow networks, and unsupervised pretraining can find these initial conditions for tasks with the right structure. Moreover, we introduce a mathematical condition for faithful backpropagation of error signals, namely dynamical isometry, and show, surprisingly that random scaled Gaussian initializations cannot achieve this condition despite their norm-preserving nature, while greedy pre-training and random orthogonal initialization can, thereby achieving depth independent learning times. Finally, we show that the property of dynamical isometry survives to good approximation even in extremely deep nonlinear random orthogonal networks operating just beyond the edge of chaos. At the cost of expressivity, deep linear networks gain theoretical tractability and may prove fertile for addressing other phenomena in deep learning, such as the impact of carefully-scaled initializations [[13](#ref-13), [23](#ref-23)], momentum [[23](#ref-23)], dropout regularization [[1](#ref-1)], and sparsity constraints [[2](#ref-2)]. While a full analytical treatment of learning in deep nonlinear networks currently remains open, one cannot reasonably hope to move towards such a theory without first completely understanding the linear case. In this sense, our work fulfills an essential pre-requisite for progress towards a general, quantitative theory of deep learning.