import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_mixed_normal_data(size, means, std_devs):
    """
    Generate a dataset composed of multiple normal distributions.
    
    Parameters
    ----------
    size : int
        Total number of samples in the dataset.
    means : list of float
        Means of the normal distributions.
    std_devs : list of float
        Standard deviations of the normal distributions.
    
    Returns
    -------
    np.ndarray
        Array of samples drawn from the mixed normal distributions.
    """
    data = []
    num_distributions = len(means)
    size_per_distribution = size // num_distributions
    
    for mean, std in zip(means, std_devs):
        data.append(norm.rvs(loc=mean, scale=std, size=size_per_distribution))
    
    # Flatten the list of arrays into a single array
    return np.concatenate(data)

# Generate two datasets, each a mixture of three normal distributions
data1 = generate_mixed_normal_data(300, [0, 5, 10], [1, 2, 1.5])
data2 = generate_mixed_normal_data(300, [2, 7, 12], [2, 1.5, 1])

# Prepare data for plotting
data = [data1, data2]

# Create a violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=data)
plt.title('Violin Plot of Mixed Normal Distributions')
plt.xlabel('Dataset')
plt.ylabel('Value')
plt.show()
