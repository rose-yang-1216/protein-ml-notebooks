import numpy as np
import pandas as pd

# Simulate protein embedding vectors
# In real workflows these could come from models like ESM

np.random.seed(0)

num_sequences = 200
embedding_dim = 64

embeddings = np.random.randn(num_sequences, embedding_dim)

df = pd.DataFrame(embeddings)

df.to_csv("data/example_embeddings.csv", index=False)

print("Example embeddings saved to data/example_embeddings.csv")
