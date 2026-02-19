import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

features  = np.load("features.npy")
filenames = np.load("filenames.npy")

reduced = PCA(2).fit_transform(StandardScaler().fit_transform(features))
labels  = KMeans(n_clusters=5, random_state=42).fit_predict(reduced)

fig, ax = plt.subplots(figsize=(10, 7))

for cluster_id in np.unique(labels):
    mask = labels == cluster_id
    ax.scatter(reduced[mask, 0], reduced[mask, 1],
               s=40, alpha=0.8, label=f"Cluster {cluster_id} ({mask.sum()} images)")
    # show 2 filenames per cluster so you can see what images are inside
    for idx in np.where(mask)[0][:2]:
        ax.annotate(filenames[idx], (reduced[idx, 0], reduced[idx, 1]),
                    fontsize=6, alpha=0.6, xytext=(4, 4), textcoords="offset points")

ax.set_title("Image Clusters (PCA 2D)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clusters.png", dpi=150)
plt.show()