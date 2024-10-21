import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100

data = {
    "intensity": np.random.rand(n_samples) * 100,
    "aspect_ratio": np.random.rand(n_samples) * 5,
    "perimeter": np.random.rand(n_samples) * 50,
    "elongation": np.random.rand(n_samples) * 10
}

df = pd.DataFrame(data)

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply UMAP
umap_model = umap.UMAP(random_state=42)
umap_embedding = umap_model.fit_transform(scaled_data)

# Add UMAP coordinates to dataframe
df['UMAP1'] = umap_embedding[:, 0]
df['UMAP2'] = umap_embedding[:, 1]

# Visualize UMAP
plt.figure(figsize=(8, 6))
sns.scatterplot(x='UMAP1', y='UMAP2', data=df)
plt.title('UMAP projection of synthetic cell measurements')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.savefig('umap_projection.png')
plt.show()
