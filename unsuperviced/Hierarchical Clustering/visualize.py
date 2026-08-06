from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


iris = load_iris()

X = iris.data


Z = linkage(
    X,
    method="ward"
)


plt.figure(figsize=(10,5))

dendrogram(Z)

plt.title(
    "Iris Hierarchical Clustering"
)

plt.xlabel("Samples")
plt.ylabel("Distance")

plt.show()