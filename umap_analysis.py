import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import StandardScaler

def create_synthetic_data(n_samples=100):
    """
    Create a synthetic dataset with measurements of cells.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples in the synthetic dataset. Default is 100.
    
    Returns
    -------
    dataframe : pandas.DataFrame
        A dataframe containing synthetic measurements.
    """
    np.random.seed(42)
    
    intensity = np.random.normal(loc=100, scale=15, size=n_samples)
    aspect_ratio = np.random.uniform(1.0, 3.0, size=n_samples)
    perimeter = np.random.normal(loc=50, scale=10, size=n_samples)
    elongation = np.random.uniform(0.5, 1.5, size=n_samples)
    
    data = {
        'intensity': intensity,
        'aspect_ratio': aspect_ratio,
        'perimeter': perimeter,
        'elongation': elongation
    }
    
    dataframe = pd.DataFrame(data)
    return dataframe

def perform_umap(dataframe, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Perform UMAP on the synthetic dataset and add the UMAP parameters to the dataframe.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data containing cell measurements.
    n_neighbors : int, optional
        The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
        Default is 15.
    min_dist : float, optional
        The effective minimum distance between embedded points. Default is 0.1.
    n_components : int, optional
        The desired dimensionality of the embedded space. Default is 2.
    
    Returns
    -------
    dataframe_umap : pandas.DataFrame
        Dataframe with UMAP components added.
    """
    features = dataframe[['intensity', 'aspect_ratio', 'perimeter', 'elongation']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    umap_components = umap_model.fit_transform(features_scaled)
    
    dataframe['UMAP_1'] = umap_components[:, 0]
    dataframe['UMAP_2'] = umap_components[:, 1]
    
    return dataframe

def visualize_umap(dataframe, filename='umap_plot.png'):
    """
    Visualize the UMAP results using seaborn and save the plot as a PNG file.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data containing UMAP components.
    filename : str, optional
        The name of the file where the plot will be saved. Default is 'umap_plot.png'.
    """
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=dataframe, x='UMAP_1', y='UMAP_2', palette='viridis')
    plt.title('UMAP Projection of Synthetic Cell Measurements')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    synthetic_data = create_synthetic_data()
    umap_results = perform_umap(synthetic_data)
    visualize_umap(umap_results)
