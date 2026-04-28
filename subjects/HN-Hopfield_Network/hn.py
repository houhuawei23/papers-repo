import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams["font.family"] = ["Liberation Sans", "sans-serif"]
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]


class HopfieldNetwork:
    """
    经典离散 Hopfield 网络实现
    神经元状态: +1 或 -1
    权重对称: w_ij = w_ji, 自连接 w_ii = 0
    训练: Hebbian 规则 (单次外积和)
    推理: 异步随机更新直至收敛
    """

    def __init__(self, n_neurons):
        """
        初始化网络结构
        :param n_neurons: 神经元数量 (模式维度)
        """
        self.n = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))  # 权重矩阵 N x N

    def train(self, patterns):
        """
        使用 Hebbian 规则存储记忆模式 (无监督，单次写入)
        :param patterns: 形状为 (P, N) 的列表或数组，P 为模式数量，N 为神经元数
                         每个模式元素必须是 +1 或 -1
        """
        self.weights = np.zeros((self.n, self.n))
        for p in patterns:
            # 外积: 记忆模式的自相关矩阵 (除以 N 是为了缩放防止能量爆炸)
            # 公式: w_ij += (1/N) * p_i * p_j
            self.weights += np.outer(p, p) / self.n
        # 将对角线置零 (神经元无自反馈)
        np.fill_diagonal(self.weights, 0)

    def energy(self, state):
        """
        计算当前状态的 Lyapunov 能量函数
        E = -0.5 * sum_{i,j} w_ij * s_i * s_j
        (此处忽略阈值 theta, 默认 theta=0)
        能量越低，状态越接近存储的记忆吸引子。
        """
        return -0.5 * np.dot(state.T, np.dot(self.weights, state))

    def update_async(self, state, max_iters=1000, stop_threshold=1e-5):
        """
        异步随机更新神经元状态，直到收敛到吸引子。
        :param state: 初始状态向量 (形状 N, 值为 +1 或 -1)
        :param max_iters: 最大更新迭代次数
        :param stop_threshold: 能量变化阈值，小于此值认为收敛
        :return: 收敛后的状态, 能量历史记录
        """
        state = state.copy()
        energy_history = [self.energy(state)]

        for it in range(max_iters):
            # 随机选择更新顺序 (保证异步性)
            order = np.random.permutation(self.n)
            old_state = state.copy()

            for i in order:
                # 计算神经元 i 的输入加权和 (局部场)
                h = np.dot(self.weights[i, :], state)
                # 符号函数更新：若 h >= 0 则为 +1，否则 -1
                state[i] = 1 if h >= 0 else -1

            # 计算当前能量
            e = self.energy(state)
            energy_history.append(e)

            # 若状态不再变化或能量变化微小，则提前终止
            if np.array_equal(state, old_state):
                print(f"收敛于迭代 {it+1} 步 (状态无变化)")
                break
            if abs(energy_history[-1] - energy_history[-2]) < stop_threshold:
                print(f"收敛于迭代 {it+1} 步 (能量变化 < {stop_threshold})")
                break

        return state, energy_history

    def predict(self, corrupted_state, **kwargs):
        """
        从损坏或部分线索恢复原始记忆。
        本质就是调用 update_async 进行联想回忆。
        """
        return self.update_async(corrupted_state, **kwargs)


# ----------------- 辅助函数：可视化与数据生成 -----------------
def plot_patterns(patterns, titles=None, shape=(5,5)):
    """
    将一维模式向量可视化为二维图像。
    假设输入模式长度可开方 (例如 25 -> 5x5, 64 -> 8x8)
    """
    n = len(patterns)
    side = int(np.sqrt(patterns[0].shape[0]))
    if side * side != patterns[0].shape[0]:
        raise ValueError("模式长度必须是完全平方数")
    
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, p in enumerate(patterns):
        ax = axes[i]
        ax.imshow(p.reshape(side, side), cmap='gray_r', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles:
            ax.set_title(titles[i])
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def corrupt_pattern(pattern, noise_ratio=0.2):
    """
    向原始模式添加随机噪声 (翻转部分像素)
    :param pattern: 原始模式向量 (+1/-1)
    :param noise_ratio: 噪声比例
    :return: 损坏后的模式
    """
    corrupted = pattern.copy()
    n_flip = int(noise_ratio * len(pattern))
    flip_idx = np.random.choice(len(pattern), n_flip, replace=False)
    corrupted[flip_idx] *= -1  # 翻转符号
    return corrupted


# ----------------- 演示示例 -----------------
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    np.random.seed(42)

    # 1. 定义几个简单的 5x5 二值图案作为记忆 (转化为 +1/-1)
    # 为了便于可视化，定义长度为 25 的向量
    # 图案 A: 类似字母 A 的简化形状
    A = np.array([
        -1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1
    ])
    # 图案 B: 类似字母 B
    B = np.array([
         1,  1,  1, -1, -1,
         1, -1, -1,  1, -1,
         1,  1,  1, -1, -1,
         1, -1, -1,  1, -1,
         1,  1,  1, -1, -1
    ])
    # 图案 C: 类似字母 C
    C = np.array([
        -1,  1,  1,  1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
        -1,  1,  1,  1, -1
    ])

    patterns = [A, B, C]
    pattern_names = ["Pattern A", "Pattern B", "Pattern C"]

    print("=== 原始记忆模式 (存储前) ===")
    plot_patterns(patterns, pattern_names, shape=(5,5))

    # 2. 创建 Hopfield 网络 (25 个神经元)
    net = HopfieldNetwork(n_neurons=25)

    # 3. 训练 (存储记忆)
    print("\n>>> 训练网络：使用 Hebbian 规则存储三种模式...")
    net.train(patterns)

    # 4. 演示联想回忆: 从损坏的 A 开始
    print("\n>>> 演示：损坏模式 A (添加 20% 噪声) 并尝试恢复")
    noisy_A = corrupt_pattern(A, noise_ratio=0.3)  # 30% 噪声
    print(f"损坏程度: {np.mean(noisy_A != A)*100:.1f}% 的像素被翻转")

    # 可视化损坏状态
    plot_patterns([A, noisy_A], ["原始 A", "损坏的 A (输入)"], shape=(5,5))

    # 5. 运行异步更新 (联想回忆)
    recovered_state, energy_hist = net.predict(noisy_A, max_iters=500)

    # 6. 结果展示
    print(f"\n>>> 恢复结果:")
    plot_patterns([recovered_state], ["恢复出的状态"], shape=(5,5))
    print(f"与原始 A 的差异: {np.sum(recovered_state != A)} 个像素")

    # 7. 绘制能量下降曲线 (证明动力学收敛)
    plt.figure(figsize=(8, 4))
    plt.plot(energy_hist, marker='o', linestyle='-', markersize=3)
    plt.xlabel("异步更新迭代次数")
    plt.ylabel("能量 E")
    plt.title("Hopfield 网络能量下降过程 (联想回忆)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 8. 额外测试：尝试用一个不完整的模式 (部分线索)
    print("\n>>> 测试部分线索：只给出 A 的上半部分 (下半部分置为 0 输入)")
    partial_A = A.copy()
    partial_A[15:] = 0  # 后半部分置零 (但在更新时需处理 0 值，这里简单用随机初始)
    # 对于 0 值，update_async 中符号函数会将其转为 +1 (h>=0)，可能影响结果，我们将其转为随机初始化
    partial_A[15:] = np.random.choice([-1, 1], size=10)
    plot_patterns([A, partial_A], ["原始 A", "部分线索 A (下半部分随机)"], shape=(5,5))
    
    recovered_partial, _ = net.predict(partial_A)
    print("从部分线索恢复的结果:")
    plot_patterns([recovered_partial], ["恢复的 A"], shape=(5,5))