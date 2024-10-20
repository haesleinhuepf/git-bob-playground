import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

def create_mixture_data(means, stds, sizes):
    """
    Generate a 1D numeric dataset composed of several normal distributions.

    Parameters
    ----------
    means : list of float
        List of means for the normal distributions.
    stds : list of float
        List of standard deviations for the normal distributions.
    sizes : list of int
        List of sizes for the data points for each distribution.

    Returns
    -------
    data : np.ndarray
        The combined dataset from the given normal distributions.
    """
    datasets = [np.random.normal(loc=mean, scale=std, size=size) 
                for mean, std, size in zip(means, stds, sizes)]
    return np.concatenate(datasets)

def plot_violin(data1, data2):
    """
    Plot a violin plot for two datasets.

    Parameters
    ----------
    data1 : np.ndarray
        The first dataset for the violin plot.
    data2 : np.ndarray
        The second dataset for the violin plot.
    """
    sns.violinplot(data=[data1, data2])
    plt.xlabel('Dataset')
    plt.ylabel('Value')
    plt.title('Violin Plot of Two Mixture Datasets')
    plt.xticks(ticks=[0, 1], labels=['Dataset 1', 'Dataset 2'])
    plt.grid(True)
    plt.show()

# Generate two datasets from mixture of normal distributions
data1 = create_mixture_data(means=[0, 5, 10], stds=[1, 2, 1.5], sizes=[100, 150, 200])
data2 = create_mixture_data(means=[-2, 4, 8], stds=[1.5, 2.5, 1], sizes=[120, 130, 180])

# Plot the violin plot for these datasets
plot_violin(data1, data2)
