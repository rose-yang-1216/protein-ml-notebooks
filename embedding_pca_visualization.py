import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load embeddings
data = pd.read_csv("data/example_embeddings.csv")

# Dimensionality reduction
pca = PCA(n_components=2)
projection = pca.fit_transform(data)

# Plot
plt.scatter(projection[:,0], projection[:,1])
plt.title("Protein Embedding Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
