## Supplementary Material

<a id="appendix-a"></a>

## Appendix A Hyperbolic dynamics of learning

In Section 1.3 of the main text we treat the dynamics of learning in three layer networks where mode strengths in each layer are equal, i.e, $a=b$, a reasonable limit when starting with small random initial conditions. More generally, though, we are interested in how long it takes for $ab$ to approach $s$ from any given initial condition. To access this, given the hyperbolic nature of the dynamics, it is useful to make the hyperbolic change of coordinates,

$$
\displaystyle a=\sqrt{c_{0}}\cosh\frac{\theta}{2}\quad b=\sqrt{c_{0}}\sinh\frac{\theta}{2}\quad\textrm{for}~{}~{}a^{2}>b^{2} (23) \displaystyle a=\sqrt{c_{0}}\sinh\frac{\theta}{2}\quad b=\sqrt{c_{0}}\cosh\frac{\theta}{2}\quad\textrm{for}~{}~{}a^{2}<b^{2}.(24)
$$

Thus $\theta$ parametrizes the dynamically invariant manifolds $a^{2}-b^{2}=\pm c_{0}$. For any $c_{0}$ and $\theta$, this coordinate system covers the region $a+b>0$, which is the basin of attraction of the upper right component of the hyperbola $ab=s$. A symmetric situation exists for $a+b<0$, which is attracted to the lower left component of $ab=s$. We use $\theta$ as a coordinate to follow the dynamics of the product $ab$, and using the relations $ab=c_{0}\sinh\theta$ and $a^{2}+b^{2}=c_{0}\cosh\theta$, we obtain

$$
\tau\frac{d\theta}{dt}=s-c_{0}\sinh\theta.(25)
$$

This differential equation is separable in $\theta$ and $t$ and can be integrated to yield

$$
t=\tau\int_{\theta_{0}}^{\theta_{f}}\frac{d\theta}{s-c_{0}\sinh\theta}=\frac{\tau}{\sqrt{c^{2}_{0}+s^{2}}}\left[\ln\frac{\sqrt{c^{2}_{0}+s^{2}}+c_{0}+s\tanh\frac{\theta}{2}}{\sqrt{c^{2}_{0}+s^{2}}-c_{0}-s\tanh\frac{\theta}{2}}\right]_{\theta_{0}}^{\theta_{f}}.(26)
$$

Here $t$ is the amount of time it takes to travel from $\theta_{0}$ to $\theta_{f}$ along the hyperbola $a^{2}-b^{2}=\pm c_{0}$. The fixed point lies at $\theta=\sinh^{-1}s/c_{0}$, but the dynamics cannot reach the fixed point in finite time. Therefore we introduce a cutoff $\epsilon$ to mark the endpoint of learning, so that $\theta_{f}$ obeys $\sinh\theta_{f}=(1-\epsilon)s/c_{0}$ (i.e. $ab$ is close to $s$ by a factor $1-\epsilon$). We can then average over the initial conditions $c_{0}$ and $\theta_{0}$ to obtain the expected learning time of an input-output relation that has a correlation strength $s$. Rather than doing this, it is easier to obtain a rough estimate of the timescale of learning under the assumption that the initial weights are small, so that $c_{0}$ and $\theta_{0}$ are close to 0 0. In this case $t=O(\tau/s)$ (with a weak logarithmic dependence on the cutoff (i.e. $\ln(1/\epsilon)$). This modestly generalizes the result given in the main text: the timescale of learning of each input-output mode $\alpha$ of the correlation matrix ${\Sigma^{31}}$ is inversely proportional to the correlation strength $s_{\alpha}$ of the mode even when $a$ and $b$ differ slightly, i.e., $c_{0}$ small. This is not an unreasonable limit for random initial conditions because $|c_{0}|=|a\cdot a-b\cdot b|$ where $a$ and $b$ are random vectors of $N_{2}$ synaptic weights into and out of the hidden units. Thus we expect the lengths of the two random vectors to be approximately equal and therefore $c_{0}$ will be small relative to the length of each vector.

These solutions are distinctly different from solutions for learning dynamics in three layer networks found in [[20](#ref-20)]. In our notation, in [[20](#ref-20)], it was shown that if the initial vectors $a^{\alpha}$ and $b^{\alpha}$ satisfy the matrix identity $\sum_{\alpha}a^{\alpha}a^{\alpha^{T}}=\sum_{\alpha}b^{\alpha}b^{\alpha^{T}}$ then the dynamics of learning becomes equivalent to a matrix Riccatti equation. However, the hyperbolic dynamics derived here arises from a set of initial conditions that do not satisfy the restrictions of [[20](#ref-20)] and therefore do not arise through a solution to a matrix Ricatti equation. Moreover, in going beyond a statement of the matrix Riccatti solution, our analysis provides intuition about the time-scales over which the learning dynamics unfolds, and crucially, our methods extend beyond the three layer case to the arbitrary $N_{l}$ layer case, not studied in [[20](#ref-20)].

<a id="appendix-b"></a>

## Appendix B Optimal discrete time learning rates

In Section 2 we state results on the optimal learning rate as a function of depth in a deep linear network, which we derive here. Starting from the decoupled initial conditions given in the main text, the dynamics arise from gradient descent on

$$
E(a_{1},\cdots,a_{N_{l}-1})=\frac{1}{2\tau}\left(s-\prod_{k=1}^{N_{l}-1}a_{k}\right).(27)
$$

Hence for each $a_{i}$ we have

$$
\frac{\partial E}{\partial a_{i}}=-\frac{1}{\tau}\left(s-\prod_{k=1}^{N_{l}-1}a_{k}\right)\left(\prod_{k\neq i}^{N_{l}-1}a_{k}\right)\equiv f(a_{i})(28)
$$

The elements of the Hessian are thus

$$
\displaystyle\frac{\partial^{2}E}{\partial a_{i}a_{j}} \displaystyle= \displaystyle\frac{1}{\tau}\left(\prod_{k\neq j}^{N_{l}-1}a_{k}\right)\left(\prod_{k\neq i}^{N_{l}-1}a_{k}\right)-\frac{1}{\tau}\left(s-\prod_{k=1}^{N_{l}-1}a_{k}\right)\left(\prod_{k\neq i,j}^{N_{l}-1}a_{k}\right) (29) \displaystyle\equiv \displaystyle g(a_{i},a_{j})(30)
$$

for $i\neq j$, and

$$
\frac{\partial^{2}E}{\partial a_{i}^{2}}=\frac{1}{\tau}\left(\prod_{k\neq i}^{N_{l}-1}a_{k}\right)^{2}\equiv h(a_{i})(31)
$$

for $i=j$.

We now assume that we start on the symmetric manifold, such that $a_{i}=a_{j}=a$ for all $i,j$. Thus we have

$$
\displaystyle E(a) \displaystyle= \displaystyle\frac{1}{2\tau}\left(s-a^{N_{l}-1}\right), (32) \displaystyle f(a) \displaystyle= \displaystyle-\frac{1}{\tau}\left(s-a^{N_{l}-1}\right)a^{N_{l}-2}, (33) \displaystyle g(a) \displaystyle= \displaystyle\frac{2}{\tau}a^{2N_{l}-4}-\frac{1}{\tau}sa^{N_{l}-3} (34) \displaystyle h(a) \displaystyle= \displaystyle\frac{1}{\tau}a^{2N_{l}-4}(35)
$$

The Hessian is

$$
H(a)=\begin{bmatrix}h&g&\cdots&g&g\\ g&h&\cdots&g&g\\ \vdots&&\ddots&&\vdots\\ g&g&\cdots&h&g\\ g&g&\cdots&g&h\end{bmatrix}.(36)
$$

One eigenvector is $v_{1}=[11\cdots 1]^{T}$ with eigenvalue $\lambda_{1}=h+(N_{l}-2)g$, or

$$
\displaystyle\lambda_{1}=(2N_{l}-3)\frac{1}{\tau}a^{2N_{l}-4}-(N_{l}-2)\frac{1}{\tau}sa^{N_{l}-3}.(37)
$$

Now consider the second order update (Newton-Raphson) (here we use $1$ to denote a vector of ones)

$$
\displaystyle a^{t+1}1 \displaystyle= \displaystyle a^{t}1-H^{-1}f(a^{t})1 (38) \displaystyle= \displaystyle a^{t}1-f(a^{t})H^{-1}1 (39) \displaystyle a^{t+1} \displaystyle= \displaystyle a^{t}-f(a^{t})/\lambda_{1}(a^{t})(40)
$$

Note that the basin of attraction does not include small initial conditions, because for small $a$ the Hessian is not positive definite.

To determine the optimal learning rate for first order gradient descent, we compute the maximum of $\lambda_{1}$ over the range of mode strengths that can be visited during learning, i.e., $a\in[0,s^{1/(N_{l}-1)}]$. This occurs at the optimum, $a_{opt}=s^{1/(N_{l}-1)}$. Hence substituting this into ([37](#A2.E37)) we have

$$
\lambda_{1}(a_{opt})=(N_{l}-1)\frac{1}{\tau}s^{\frac{2N_{l}-4}{N_{l}-1}}.(41)
$$

The optimal learning rate $\alpha$ is proportional to $1/\lambda_{1}(a_{opt})$, and hence scales as

$$
\alpha\sim O\left(\frac{1}{N_{l}s^{2}}\right)(42)
$$

for large $N_{l}$.

- [B.1 Learning speeds with optimized learning rate](#b1-learning-speeds-with-optimized-learning-rate)

### B.1 Learning speeds with optimized learning rate

How does the optimal learning rate impact learning speeds? We compare the three layer learning time to the infinite depth limit learning time, with learning rate set inversely proportional to Eqn. ([41](#A2.E41)) with proportionality constant $c$.

This yields a three layer learning time $t_{3}$ of

$$
t_{3}=c\ln\frac{u_{f}(s-u_{0})}{u_{0}(s-u_{f})}(43)
$$

and an infinite layer learning time $t_{\infty}$ of

$$
t_{\infty}=c\left[\log\left(\frac{u_{f}(u_{0}-s)}{u_{0}(u_{f}-s)}\right)+\frac{s}{u_{0}}-\frac{s}{u_{f}}\right],(44)
$$

Hence the difference is

$$
t_{\infty}-t_{3}=\frac{cs}{u_{0}}-\frac{cs}{u_{f}}\approx\frac{cs}{\epsilon}(45)
$$

where the final approximation is for $u_{0}=\epsilon,u_{f}=s-\epsilon$, and $\epsilon$ small. Thus very deep networks incur only a finite delay relative to shallow networks.

<a id="appendix-c"></a>

## Appendix C Experimental setup for MNIST depth experiment

We trained deep linear networks on the MNIST dataset with fifteen different depths $N_{l}=\{3,5,8,10,14,20,28,36,44,54,64,74,84,94,100\}$. Given a 784-dimensional input example, the network tried to predict a 10-dimensional output vector containing a 1 in the index for the correct class, and zeros elsewhere. The network was trained using batch gradient descent via Eqn. ([13](#S2.E13)) on the 50,000 sample MNIST training dataset. We note that Eqn. ([13](#S2.E13)) makes use of the linearity of the network to speed training and reduce memory requirements. Instead of forward propagating all 50,000 training examples, we precompute ${\Sigma^{31}}$ and forward propagate only it. This enables experiments on very deep networks that otherwise would be computationally infeasible. Experiments were accelerated on GPU hardware using the GPUmat package. We used overcomplete hidden layers of size 1000. Here the overcompleteness is simply to demonstrate the applicability of the theory to this case; overcompleteness does not improve the representational power of the network. Networks were initialized with decoupled initial conditions and starting initial mode strength $u_{0}=0.001$, as described in the text. The random orthogonal matrices $R_{l}$ were selected by generating random Gaussian matrices and computing a QR decomposition to obtain an orthogonal matrix. Learning times were calculated as the iteration at which training error fell below a fixed threshold of $1.3\times 10^{4}$ corresponding to nearly complete learning. Note that this level of performance is grossly inferior to what can be obtained using nonlinear networks, which reflects the limited capacity of a linear network. We optimized the learning rate $\lambda$ separately for each depth by training each network with twenty rates logarithmically spaced between $10^{-4}$ and $10^{-7}$ and picking the one that yielded the minimum learning time according to our threshold criterion. The range $10^{-4}$ and $10^{-7}$ was selected via preliminary experiments to ensure that the optimal learning rate always lay in the interior of the range for all depths.

<a id="appendix-d"></a>

## Appendix D Efficacy of unsupervised pretraining

Recently high performance has been demonstrated in deep networks trained from random initial conditions [[21](#ref-21), [13](#ref-13), [22](#ref-22), [3](#ref-3), [4](#ref-4), [1](#ref-1), [23](#ref-23)], suggesting that deep networks may not be as hard to train as previously thought. These results show that pretraining is not necessary to obtain state-of-the-art performance, and to achieve this they make use of a variety of techniques including carefully-scaled random initializations, more sophisticated second order or momentum-based optimization methods, and specialized convolutional architectures. It is therefore important to evaluate whether unsupervised pretraining is still useful, even if it is no longer necessary, for training deep networks. In particular, does pretraining still confer an optimization advantage and generalization advantage when used in conjunction with these new techniques? Here we review results from a variety of papers, which collectively show that unsupervised pretraining still confers an optimization advantage and a generalization advantage.

- [D.1 Optimization advantage](#d1-optimization-advantage)
- [D.2 Generalization advantage](#d2-generalization-advantage)

### D.1 Optimization advantage

The optimization advatage of pretraining refers to faster convergence to the local optimum (i.e., faster learning speeds) when starting from pretrained initializations as compared to random initializations. Faster learning speeds starting from pretrained initial conditions have been consistently found with Hessian free optimization [[21](#ref-21), [22](#ref-22)]. This finding holds for two carefully-chosen random initialization schemes, the sparse connectivity scheme of [[21](#ref-21)], and the dense scaled scheme of [[13](#ref-13)] (as used by [[22](#ref-22)]). Hence pretraining still confers a convergence speed advantage with second order methods. Pretrained initial conditions also result in faster convergence than carefully-chosen random initializations when optimizing with stochastic gradient descent [[22](#ref-22), [13](#ref-13)]. In light of this, it appears that pretrained initial conditions confer an optimization advantage beyond what can be obtained currently with carefully-scaled random initializations, regardless of optimization technique. If run to convergence, second order methods and well-chosen scalings can erase the discrepancy between the final objective value obtained on the training set for pretrained relative to random initializations [[21](#ref-21), [22](#ref-22)]. The optimization advantage is thus purely one of convergence speed, not of finding a better local minimum. This coincides with the situation in linear networks, where all methods will eventually attain the same global minimum, but the rate of convergence can vary. Our analysis shows why this optimization advantage due to pretraining persists over well-chosen random initializations.

Finally, we note that Sutskever et al. show that careful random initialization paired with carefully-tuned momentum can achieve excellent performance [[23](#ref-23)], but these experiments did not try pretrained initial conditions. Krizhevsky et al. used convolutional architectures and did not attempt pretraining [[1](#ref-1)]. Thus the possible utility of pretraining in combination with momentum, and in combination with convolutional architectures, dropout, and large supervised datasets, remains unclear.

### D.2 Generalization advantage

Pretraining can also act as a special regularizer, improving generalization error in certain instances. This generalization advantage appears to persist with new second order methods [[21](#ref-21), [22](#ref-22)], and in comparison to gradient descent with careful random initializations [[13](#ref-13), [22](#ref-22), [25](#ref-25), [4](#ref-4)]. An analysis of this effect in deep linear networks is out of the scope of this work, though promising tools have been developed for the three layer linear case [[20](#ref-20)].

<a id="appendix-e"></a>

## Appendix E Learning dynamics with task-aligned input correlations

In the main text we focused on orthogonal input correlations ($\Sigma^{11}=I$) for simplicity, and to draw out the main intuitions. However our analysis can be extended to input correlations with a very particular structure. Recall that we decompose the input output correlations using the SVD as ${\Sigma^{31}}={U^{33}}{S^{31}}{V^{11}}^{T}$. We can generalize our solutions to allow input correlations of the form $\Sigma^{11}={V^{11}}D{V^{11}}^{T}$. Intuitively, this condition requires the axes of variation in the input to coincide with the axes of variation in the input-output task, though the variances may differ. If we take $D=I$ then we recover the whitened case $\Sigma^{11}=I$, and if we take $D=\Lambda$, then we can treat the autoencoding case. The final fixed points of the weights are given by the best rank $N_{2}$ approximation to ${\Sigma^{31}}(\Sigma^{11})^{-1}$. Making the same change of variables as in Eqn. ([4](#S1.E4)) we now obtain

$$
\tau\frac{d}{dt}{\overline{W}^{21}}={\overline{W}^{32}}^{T}({S^{31}}-{\overline{W}^{32}}{\overline{W}^{21}}D),\qquad\tau\frac{d}{dt}{\overline{W}^{32}}=({S^{31}}-{\overline{W}^{32}}{\overline{W}^{21}}D){\overline{W}^{21}}^{T}.(46)
$$

which, again, is decoupled if ${\overline{W}^{32}}$ and ${\overline{W}^{21}}$ begin diagonal. Based on this it is straightforward to generalize our results for the learning dynamics.

<a id="appendix-f"></a>

## Appendix F MNIST pretraining experiment

We trained networks of depth 5 on the MNIST classification task with 200 hidden units per layer, starting either from small random initial conditions with each weight drawn independently from a Gaussian distribution with standard deviation 0.01, or from greedy layerwise pretrained initial conditions. For the pretrained network, each layer was trained to reconstruct the output of the next lower layer. In the finetuning stage, the network tried to predict a 10-dimensional output vector containing a 1 in the index for the correct class, and zeros elsewhere. The network was trained using batch gradient descent via Eqn. ([13](#S2.E13)) on the 50,000 sample MNIST training dataset. Since the network is linear, pretraining initializes the network with principal components of the input data, and, to the extent that the consistency condition of Eqn. ([18](#S3.E18)) holds, decouples these modes throughout the deep network, as described in the main text.

<a id="appendix-g"></a>

## Appendix G Analysis of Neural Dynamics in Nonlinear Orthogonal Networks

We can derive a simple, analytical recursion relation for the propagation of neural population variance $q^{l}$, defined in ([21](#S4.E21)), across layers $l$ under the nonlinear dynamics ([20](#S4.E20)). We have

$$
q^{l+1}=\frac{1}{N}\sum_{i=1}^{N}(x^{l+1}_{i})^{2}=g^{2}\frac{1}{N}\sum_{i=1}^{N}\phi(x^{l}_{i})^{2},(47)
$$

due to the dynamics in ([20](#S4.E20)) and the orthogonality of $W^{(l+1,l)}$. Now we know that by definition, the layer $l$ population $x^{l}_{i}$ has normalized variance $q^{l}$. If we further assume that the distribution of activity across neurons in layer $l$ is well approximated by a Gaussian distribution, we can replace the sum over neurons $i$ with an integral over a zero mean unit variance Gaussian variable $z$:

$$
q^{l+1}=g^{2}\,\int\mathcal{D}z\,\phi\big{(}\sqrt{q^{l}}z\big{)}^{2},(48)
$$

where $\mathcal{D}z\equiv\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}z^{2}}\,dz$ is the standard Gaussian measure.

<a id="figure-8"></a>

![network](images/network.png)

> Figure 8: Left: The map from variance in the input layer $q^{in}=q^{l}$ to variance in the output layer $q^{out}=q^{l+1}$ in ([48](#A7.E48)) for $g=1$ and $\phi(x)=\tanh(x)$. Right: The stable fixed points of this map, $q^{\infty}(g)$, as a function of the gain $g$. The red curve is the analytic theory obtained by numerically solving ([49](#A7.E49)). The blue points are obtained via numerical simulations of the dynamics in ([20](#S4.E20)) for networks of depth $N_{l}=30$ with $N=1000$ neurons per layer. The asymptotic population variance $q^{\infty}$ is obtained by averaging the population variance in the last $5$ layers.

This map from input to output variance is numerically computed for $g=1$ and $\phi(x)=\tanh(x)$ in Fig. [8](#A7.F8), left (other values of $g$ yield a simple multiplicative scaling of this map). This recursion relation has a stable fixed point $q^{\infty}(g)$ obtained by solving the nonlinear fixed point equation

$$
q^{\infty}=g^{2}\,\int\mathcal{D}z\,\phi\big{(}\sqrt{q^{\infty}}z\big{)}^{2}.(49)
$$

Graphically, solving this equation corresponds to scaling the curve in Fig. [8](#A7.F8) left by $g^{2}$ and looking for intersections with the line of unity. For $g<1$, the only solution is $q^{\infty}=0$. For $g>1$, this solution remains, but it is unstable under the recurrence ([48](#A7.E48)). Instead, for $g>1$, a new stable solution appears for some nonzero value of $q^{\infty}$. The entire set of stable solutions as a function of $g$ is shown as the red curve in Fig. [8](#A7.F8) right. It constitutes a theoretical prediction of the population variance at the deepest layers of a nonlinear network as the depth goes to infinity. It matches well for example, the empirical population variance obtained from numerical simulations of nonlinear networks of depth $30$ (blue points in Fig. [8](#A7.F8) right).

Overall, these results indicate a dynamical phase transition in neural activity propagation through the nonlinear network as $g$ crosses the critical value $g_{c}=1$. When $g>1$, activity propagates in a chaotic manner, and so $g=1$ constitutes the edge of chaos.