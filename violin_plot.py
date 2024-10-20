import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_mixed_distribution(size, means, std_devs):
    """
    Generate a 1D dataset consisting of multiple normal distributions.

    Parameters
    ----------
    size : int
        The total number of samples in the dataset.
    means : list of float
        A list containing the means of the normal distributions.
    std_devs : list of float
        A list containing the standard deviations of the normal distributions.

    Returns
    -------
    np.ndarray
        An array containing the generated dataset.
    """
    assert len(means) == len(std_devs), "Means and standard deviations lists must have the same length"
    
    num_distributions = len(means)
    samples_per_distribution = size // num_distributions
    data = np.concatenate([np.random.normal(loc=m, scale=s, size=samples_per_distribution) 
                           for m, s in zip(means, std_devs)])
    return data

# Generate two datasets
dataset1 = generate_mixed_distribution(300, [0, 3, 6], [0.5, 1.0, 1.5])
dataset2 = generate_mixed_distribution(300, [1, 4, 7], [0.6, 1.2, 1.0])

# Draw a violin plot
data = [dataset1, dataset2]
sns.violinplot(data=data)
plt.xticks([0, 1], ['Dataset 1', 'Dataset 2'])
plt.title('Violin Plot of Two Mixed Normal Distributions')
plt.xlabel('Datasets')
plt.ylabel('Values')
plt.show()
