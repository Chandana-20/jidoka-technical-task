import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_everything(wcss, data, labels, centroids, outliers, x, y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow Graph
    ax1.plot(range(1, 6), wcss, marker='o', color='blue')
    ax1.axvline(x=2, color='red', linestyle='--', label='Optimal K')
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("WCSS (Inertia)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Cluster Graph
    ax2.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='black', s=50)
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=250, color='red', label='Centers')
    
    
    if len(outliers) > 0:
        ax2.scatter(outliers[:, 0], outliers[:, 1], facecolors='none', edgecolors='red', s=200, label='Outliers')
        
    for i in range(len(x)):
        ax2.annotate(f"p{i+1}", (x[i], y[i]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        
    ax2.set_title("K-Means Result (K=2)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

   
    plt.tight_layout()
    plt.savefig("kmeans_result.png", dpi=300) 
    plt.show()

def main():
    
    x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
    y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
    data = np.array(list(zip(x, y)))

    
    wcss = []
    for k in range(1, 6):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        km.fit(data)
        wcss.append(km.inertia_)

    
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    
    assigned_centers = centroids[labels]
    distances = np.linalg.norm(data - assigned_centers, axis=1)
    
    
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    
    
    threshold = Q3 + (1.5 * IQR)
    outliers = data[distances > threshold]

    
    print("K-MEANS RESULTS")
    print("Centroids:\n", centroids)
    print("Outliers found:\n", outliers)

    plot_everything(wcss, data, labels, centroids, outliers, x, y)

if __name__ == '__main__':
    main()