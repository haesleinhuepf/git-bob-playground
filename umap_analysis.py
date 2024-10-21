import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create synthetic dataset
def create_synthetic_data(num_samples=100):
    """
    Create a synthetic dataset of cell measurements.
    
    Parameters
    ----------
    num_samples : int, optional
        Number of samples to generate, by default 100

    Returns
    -------
    pd.DataFrame
        DataFrame containing the synthetic dataset
    """
    np.random.seed(0)
    data = {
        'intensity': np.random.rand(num_samples) * 100,
        'aspect_ratio': np.random.rand(num_samples) * 10,
        'perimeter': np.random.rand(num_samples) * 50, 
        'elongation': np.random.rand(num_samples) * 5
    }
    return pd.DataFrame(data)

# Determine UMAP from parameters
def determine_umap(dataframe):
    """
    Perform UMAP dimensionality reduction on the dataset.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the dataset to be reduced using UMAP

    Returns
    -------
    pd.DataFrame
        DataFrame with UMAP dimensions added
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(scaled_data)
    
    dataframe['UMAP_1'] = umap_result[:, 0]
    dataframe['UMAP_2'] = umap_result[:, 1]
    return dataframe

# Visualize UMAP
def visualize_umap(dataframe, output_filename='umap_plot.png'):
    """
    Visualize UMAP results using seaborn.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the dataset with UMAP results
    output_filename : str, optional
        Filename for saving the UMAP plot, by default 'umap_plot.png'
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='UMAP_1', y='UMAP_2', data=dataframe)
    plt.title('UMAP Projection of Synthetic Cell Measurements')
    plt.savefig(output_filename)
    plt.show()

# Main execution
if __name__ == "__main__":
    df = create_synthetic_data()
    df_with_umap = determine_umap(df)
    visualize_umap(df_with_umap)
