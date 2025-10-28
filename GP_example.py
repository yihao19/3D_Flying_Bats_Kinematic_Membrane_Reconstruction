import numpy as np
import matplotlib.pyplot as plt

def squared_exponential_kernel(X1, X2, length_scale=1.0, output_scale=1.0):
    """
    Calculates the Squared Exponential (aka RBF, Gaussian) kernel between two sets of points.

    Args:
        X1: Input data points, shape (n_samples1, n_features).
        X2: Input data points, shape (n_samples2, n_features).
        length_scale: Length scale parameter.
        output_scale: Output scale parameter.

    Returns:
        Kernel matrix, shape (n_samples1, n_samples2).
    """
    sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return output_scale**2 * np.exp(-0.5 / length_scale**2 * sq_dist)


def sample_gaussian_process(X, kernel_func, kernel_params, n_samples=1):
    """
    Samples from a Gaussian Process prior.

    Args:
        X: Input locations, shape (n_samples, n_features).
        kernel_func: Kernel function.
        kernel_params: Dictionary of kernel parameters.
        n_samples: Number of samples to draw.

    Returns:
         Samples from the GP, shape (n_samples, len(X)).
    """
    mean = np.zeros(len(X))
    cov = kernel_func(X, X, **kernel_params)
    print(cov.shape)
    print(cov)
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    return samples

if __name__ == '__main__':
    # Define the input space
    X = np.linspace(0, 100, 100).reshape(-1, 1)

    # Define kernel parameters
    kernel_parameters = {'length_scale': 10, 'output_scale': 1}

    # Number of samples to draw
    num_samples = 15

    # Generate samples
    samples = sample_gaussian_process(X, squared_exponential_kernel, kernel_parameters, num_samples)

    # Plot the samples
    plt.figure(figsize=(8, 6))
    for i in range(num_samples):
        plt.plot(X, samples[i, :], label=f'Sample {i+1}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Samples from Gaussian Process with Squared Exponential Kernel')
    plt.legend()
    plt.grid(True)
    plt.show()