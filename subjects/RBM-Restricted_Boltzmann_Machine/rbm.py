import numpy as np
import matplotlib.pyplot as plt


class RBM:
    """
    受限玻尔兹曼机（二值单元版本）
    包含 CD-k 训练算法
    """

    def __init__(
        self, n_visible, n_hidden, learning_rate=0.1, random_state=None
    ):
        """
        参数:
            n_visible: 可见层单元数
            n_hidden:  隐藏层单元数
            learning_rate: 学习率
            random_state: 随机种子（保证可重复）
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate

        if random_state is not None:
            np.random.seed(random_state)

        # 初始化权重和偏置（小随机数）
        # 权重 W_{ij} 连接可见单元 i 和隐藏单元 j
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        # 可见层偏置 a_i
        self.a = np.zeros(n_visible)
        # 隐藏层偏置 b_j
        self.b = np.zeros(n_hidden)

    def sigmoid(self, x):
        """sigmoid 激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))  # 防止溢出

    def prob_h_given_v(self, v):
        """
        给定可见层 v，计算隐藏层每个单元为 1 的概率
        P(h_j=1 | v) = sigmoid(b_j + sum_i v_i W_{ij})
        """
        # v: (batch_size, n_visible)
        # W: (n_visible, n_hidden) -> 输入乘以权重: (batch_size, n_hidden)
        activation = np.dot(v, self.W) + self.b
        return self.sigmoid(activation)

    def sample_h_given_v(self, v):
        """
        给定可见层 v，采样隐藏层状态 h（二值）
        """
        prob_h = self.prob_h_given_v(v)
        # 伯努利采样：生成 [0,1] 均匀随机数，若小于概率则取1
        return (prob_h > np.random.rand(*prob_h.shape)).astype(np.float32)

    def prob_v_given_h(self, h):
        """
        给定隐藏层 h，计算可见层每个单元为 1 的概率
        P(v_i=1 | h) = sigmoid(a_i + sum_j W_{ij} h_j)
        """
        activation = np.dot(h, self.W.T) + self.a
        return self.sigmoid(activation)

    def sample_v_given_h(self, h):
        """
        给定隐藏层 h，采样可见层状态 v（二值）
        """
        prob_v = self.prob_v_given_h(h)
        return (prob_v > np.random.rand(*prob_v.shape)).astype(np.float32)

    def gibbs_step(self, v):
        """
        一步吉布斯采样: v -> h -> v'
        """
        h = self.sample_h_given_v(v)
        v_recon = self.sample_v_given_h(h)
        return v_recon, h

    def contrastive_divergence(self, v, k=1):
        """
        执行 CD-k 算法，返回参数梯度（用于批量更新）
        参数:
            v: 输入数据 (batch_size, n_visible)
            k: 吉布斯采样步数
        返回:
            grad_W, grad_a, grad_b
        """
        batch_size = v.shape[0]

        # ----- 正相 (positive phase) -----
        # 计算隐藏层概率及采样
        prob_h0 = self.prob_h_given_v(v)  # (batch, n_hidden)
        # 为了计算期望，通常使用概率值而不是采样（低方差）
        # 但为了与 CD 理论一致，我们使用采样或概率均可。这里使用概率值作为期望近似。
        pos_associations = np.dot(v.T, prob_h0)  # (n_visible, n_hidden)
        # 正相偏置梯度
        pos_h_sum = np.sum(prob_h0, axis=0)  # (n_hidden,)

        # ----- 负相 (negative phase) -----
        # 从 v 开始，执行 k 步吉布斯采样得到重构
        v_cur = v.copy()
        for _ in range(k):
            h_cur = self.sample_h_given_v(v_cur)
            v_cur = self.sample_v_given_h(h_cur)
        # 最后一步的隐藏层概率（用于计算关联）
        prob_h_k = self.prob_h_given_v(v_cur)
        neg_associations = np.dot(v_cur.T, prob_h_k)  # (n_visible, n_hidden)
        neg_h_sum = np.sum(prob_h_k, axis=0)  # (n_hidden,)

        # ----- 梯度计算 (平均小批量) -----
        grad_W = (pos_associations - neg_associations) / batch_size
        grad_a = np.mean(v - v_cur, axis=0)  # (n_visible,)
        grad_b = (pos_h_sum - neg_h_sum) / batch_size  # (n_hidden,)

        return grad_W, grad_a, grad_b

    def update(self, grad_W, grad_a, grad_b):
        """用梯度更新参数"""
        self.W += self.lr * grad_W
        self.a += self.lr * grad_a
        self.b += self.lr * grad_b

    def train(self, data, epochs=10, batch_size=10, k=1, verbose=True):
        """
        训练 RBM
        参数:
            data: 训练数据，形状 (n_samples, n_visible)
            epochs: 迭代次数
            batch_size: 小批量大小
            k: CD-k 中的步数
            verbose: 是否打印重构误差
        """
        n_samples = data.shape[0]
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                batch = data[indices[i : i + batch_size]]
                grad_W, grad_a, grad_b = self.contrastive_divergence(batch, k)
                self.update(grad_W, grad_a, grad_b)

                # 计算重构误差（可选，用于监控）
                # 这里简单计算当前批次的均方误差
                v_recon, _ = self.gibbs_step(batch)
                loss = np.mean((batch - v_recon) ** 2)
                epoch_loss += loss * len(batch)

            if verbose:
                avg_loss = epoch_loss / n_samples
                print(
                    f"Epoch {epoch + 1}/{epochs}, Reconstruction MSE = {avg_loss:.6f}"
                )

    def reconstruct(self, v, k=1):
        """
        对输入 v 进行重构（通过 k 步吉布斯采样）
        """
        v_cur = v.copy()
        for _ in range(k):
            h = self.sample_h_given_v(v_cur)
            v_cur = self.sample_v_given_h(h)
        return v_cur

    def energy(self, v, h):
        """
        计算能量 E(v, h)
        """
        return (
            -np.dot(v, self.a)
            - np.dot(h, self.b)
            - np.dot(np.dot(v, self.W), h)
        )

    def free_energy(self, v):
        """
        计算自由能 F(v) = -log sum_h exp(-E(v,h))
        用于评估模型（可选）
        """
        # 对于二值隐藏层，自由能可解析计算:
        # F(v) = -a^T v - sum_j log(1 + exp(b_j + sum_i v_i W_{ij}))
        wx_b = np.dot(v, self.W) + self.b
        return -np.dot(v, self.a) - np.sum(np.log(1 + np.exp(wx_b)), axis=1)


# 生成一个简单的二值数据集：包含 4 种模式（每个模式 8 维）
def generate_synthetic_data(n_samples=500):
    patterns = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
        ]
    )
    data = []
    for _ in range(n_samples):
        idx = np.random.randint(len(patterns))
        data.append(patterns[idx])
        # 添加少量噪声
        noisy = np.array(data[-1])
        if np.random.rand() < 0.1:
            flip = np.random.randint(len(noisy))
            noisy[flip] = 1 - noisy[flip]
        data[-1] = noisy
    return np.array(data)


# 生成数据
data = generate_synthetic_data(2000)
print("Data shape:", data.shape)

# 创建 RBM（8 个可见单元，4 个隐藏单元）
rbm = RBM(n_visible=8, n_hidden=4, learning_rate=0.1, random_state=42)

# 训练
rbm.train(data, epochs=50, batch_size=20, k=1, verbose=True)

# 测试重构: 取前 10 个样本
test_samples = data[:10]
recon = rbm.reconstruct(test_samples, k=1)

print("\n原始样本 vs 重构样本 (二值化阈值0.5):")
for i in range(10):
    print(f"原: {test_samples[i].astype(int)}")
    print(f"重: {(recon[i] > 0.5).astype(int)}")
    print()
