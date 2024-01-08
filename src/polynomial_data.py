import numpy as np

true_function = lambda x: 10 * np.sin(3.3 * np.pi * x) + 20 * x + 5
noise = lambda x: x + np.random.normal(0, 3, size=len(x))


def get_polynomial_training_data(
    number_of_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(0)
    X = np.linspace(0, 1, number_of_points)
    y = true_function(X)
    return X, noise(y)


def get_polynomial_test_data(number_of_points: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(0)
    X = np.sort(np.random.rand(number_of_points))
    y = true_function(X)
    return X, noise(y)
