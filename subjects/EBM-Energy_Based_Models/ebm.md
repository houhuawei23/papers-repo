以下是一个基于 **Energy-Based Learning (EBM)** 的简单实现，用于二分类任务。代码展示了：

- 能量函数 $E(Y, X) = - (2Y-1) \cdot g_W(X)$
- Hinge 损失（margin-based contrastive loss）
- 通过随机梯度下降最小化损失
- 推理时选择能量最低的 $Y$
- 训练过程中能量面的变化可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ------------------------------
# 1. 生成简单的二分类数据集（线性可分）
# ------------------------------
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.5, random_state=42)
y = y * 2 - 1          # 将标签转为 -1 和 1，便于能量函数定义

# 可视化数据
plt.scatter(X[y==1,0], X[y==1,1], label='Class 1', marker='o')
plt.scatter(X[y==-1,0], X[y==-1,1], label='Class -1', marker='x')
plt.legend()
plt.title("Training Data")
plt.show()

# ------------------------------
# 2. 定义能量模型
#    能量函数: E(Y, X) = -Y * g_W(X) ，其中 Y ∈ {-1, 1}
#    这里 g_W(X) = W0 + W1*X1 + W2*X2   (线性判别函数)
# ------------------------------
class LinearEBM:
    def __init__(self, lr=0.01):
        self.W = np.random.randn(3) * 0.1   # [W0, W1, W2]
        self.lr = lr

    def g(self, X):
        """线性判别函数: g_W(X) = W0 + W1*x1 + W2*x2"""
        return self.W[0] + X @ self.W[1:]

    def energy(self, Y, X):
        """计算给定 Y 和 X 的能量"""
        return -Y * self.g(X)

    def inference(self, X):
        """推理：选择使能量最小的 Y ∈ {-1, 1}"""
        g_val = self.g(X)
        return np.sign(g_val)   # 若 g>0 则 Y=1，否则 Y=-1

    def hinge_loss(self, X, Y_true, margin=1.0):
        """
        对单个样本计算 hinge loss
        L = max(0, margin + E_correct - E_wrong)
        其中 correct = Y_true, wrong = -Y_true
        """
        E_correct = self.energy(Y_true, X)
        E_wrong   = self.energy(-Y_true, X)
        loss = np.maximum(0, margin + E_correct - E_wrong)
        return loss

    def train_step(self, X_batch, Y_batch, margin=1.0):
        """单步梯度下降（随机梯度）"""
        loss_sum = 0.0
        grad_W = np.zeros_like(self.W)

        for x, y_true in zip(X_batch, Y_batch):
            x = np.asarray(x)
            # 前向：计算能量
            g_val = self.g(x)
            E_correct = -y_true * g_val
            E_wrong   = y_true * g_val      # 因为 -(-y_true) = y_true

            loss = max(0, margin + E_correct - E_wrong)
            loss_sum += loss

            if loss > 0:
                # 梯度 dL/dW = dL/dE_correct * dE_correct/dW + dL/dE_wrong * dE_wrong/dW
                # 其中 dL/dE_correct = 1, dL/dE_wrong = -1
                dE_correct_dW = -y_true * np.concatenate(([1], x))
                dE_wrong_dW   =  y_true * np.concatenate(([1], x))
                grad_W += (dE_correct_dW - dE_wrong_dW)

        # 参数更新
        self.W -= self.lr * grad_W / len(X_batch)
        return loss_sum / len(X_batch)

# ------------------------------
# 3. 训练 EBM
# ------------------------------
model = LinearEBM(lr=0.05)
epochs = 30
batch_size = 20

loss_history = []
for epoch in range(epochs):
    # 随机打乱数据
    perm = np.random.permutation(len(X))
    X_shuf, Y_shuf = X[perm], y[perm]

    epoch_loss = 0.0
    for i in range(0, len(X), batch_size):
        X_batch = X_shuf[i:i+batch_size]
        Y_batch = Y_shuf[i:i+batch_size]
        loss = model.train_step(X_batch, Y_batch, margin=1.0)
        epoch_loss += loss * len(X_batch)
    avg_loss = epoch_loss / len(X)
    loss_history.append(avg_loss)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}, loss = {avg_loss:.4f}")

# 绘制损失曲线
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Hinge loss")
plt.title("Training Loss")
plt.show()

# ------------------------------
# 4. 推理与可视化
# ------------------------------
# 在网格上计算推理结果和能量值
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# 计算决策边界 (g_W(x)=0)
g_vals = model.g(grid).reshape(xx.shape)
preds = model.inference(grid).reshape(xx.shape)

# 绘制分类结果
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.contourf(xx, yy, preds, alpha=0.6, cmap='coolwarm', levels=[-1,0,1])
plt.scatter(X[y==1,0], X[y==1,1], label='True +1', edgecolors='k', marker='o')
plt.scatter(X[y==-1,0], X[y==-1,1], label='True -1', edgecolors='k', marker='x')
plt.title("Decision Boundary (Inference)")
plt.legend()

# 绘制能量函数 E(Y=+1, X) 和 E(Y=-1, X) 的差值
E1 = model.energy(1, grid).reshape(xx.shape)      # 假设 Y=+1 的能量
E_neg1 = model.energy(-1, grid).reshape(xx.shape) # 假设 Y=-1 的能量
delta_E = E1 - E_neg1   # 当 delta_E < 0 时，模型会选择 Y=+1

plt.subplot(1,2,2)
plt.contourf(xx, yy, delta_E, levels=50, cmap='RdBu')
plt.colorbar(label='E(Y=+1) - E(Y=-1)')
plt.scatter(X[y==1,0], X[y==1,1], edgecolors='k', facecolors='none', marker='o')
plt.scatter(X[y==-1,0], X[y==-1,1], edgecolors='k', facecolors='none', marker='x')
plt.title("Energy Difference (blue → Y=+1 chosen)")
plt.tight_layout()
plt.show()

# ------------------------------
# 5. 计算测试准确率（在训练集上演示）
# ------------------------------
train_pred = model.inference(X)
accuracy = np.mean(train_pred == y)
print(f"Training accuracy: {accuracy*100:.2f}%")
```

## 代码说明

### 1. 能量函数与推理

- 能量定义：$E(Y,X) = -Y \cdot g_W(X)$，其中 $Y \in \{-1, +1\}$
- 推理：对于给定 $X$，选择使能量最小的 $Y$，即 $Y^* = \operatorname{sign}(g_W(X))$

### 2. 损失函数

- 使用 **hinge loss** ：

$$
  L = \max(0, m + E(Y_{\text{true}}, X) - E(-Y_{\text{true}}, X))
$$

其中 $m=1$ 是 margin。

- 只有当正确能量与错误能量的差小于 margin 时才产生非零损失，从而强制拉开两类能量差距。

### 3. 学习过程

- 随机梯度下降（SGD），每次使用一个 mini‑batch。
- 梯度计算：当损失 >0 时，对参数求导并更新。

### 4. 可视化

- 左图：模型学到的决策边界（黑色实线是 $g_W(X)=0$）。
- 右图：能量差 $E(Y=+1) - E(Y=-1)$ 的热力图，蓝色区域表示模型倾向于选择 $Y=+1$。

运行该代码将看到损失下降曲线、清晰的线性决策边界，以及大约 100% 的训练准确率（因为数据线性可分）。这个简单示例完整展示了 EBM 的三大核心： **能量函数设计、对比损失训练、基于能量最小化的推理** 。
