import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# =========================
# 1. 构造示例数据（可替换为你的真实特征）
# =========================
np.random.seed(42)

# 两份数据：100 个 256 维向量
data_1 = np.random.randn(100, 256)
data_2 = np.random.randn(100, 256) + 0.5  # 加偏移以便区分（可选）

# =========================
# 2. 合并数据并构造标签
# =========================
X = np.vstack([data_1, data_2])     # (200, 256)
labels = np.array([0] * 100 + [1] * 100)

# =========================
# 3. t-SNE 降维
# =========================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    random_state=42
)

X_2d = tsne.fit_transform(X)  # (200, 2)

# =========================
# 4. 可视化
# =========================
plt.figure(figsize=(8, 6))

plt.scatter(
    X_2d[labels == 0, 0],
    X_2d[labels == 0, 1],
    c='tab:blue',
    label='Dataset A',
    alpha=0.7,
    s=40
)

plt.scatter(
    X_2d[labels == 1, 0],
    X_2d[labels == 1, 1],
    c='tab:orange',
    label='Dataset B',
    alpha=0.7,
    s=40
)

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Two 256-D Feature Sets')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
