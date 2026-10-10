import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_features(data_1, data_2):
    # =========================
    # 2. 合并数据并构造标签
    # =========================
    X = np.vstack([data_1, data_2])
    labels = np.array([0] * len(data_1) + [1] * len(data_2))

    # =========================
    # 3. t-SNE 降维
    # =========================
    tsne = TSNE(
        n_components=2,
        perplexity=80,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )

    X_2d = tsne.fit_transform(X)

    # =========================
    # 4. 可视化
    # =========================
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_2d[labels == 0, 0],
        X_2d[labels == 0, 1],
        c='tab:blue',
        label='Source',
        alpha=0.7,
        s=40
    )

    plt.scatter(
        X_2d[labels == 1, 0],
        X_2d[labels == 1, 1],
        c='tab:orange',
        label='Target',
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
