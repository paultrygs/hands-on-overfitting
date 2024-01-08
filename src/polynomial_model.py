import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from src.polynomial_data import true_function
from src.polynomial_visualization import visualize_model


def get_polynomial_model(degree: int) -> Pipeline:
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    return pipeline


def get_polynomial_model_with_regularization(
    degree: int, regularization: float = 0.0000001
) -> Pipeline:
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = Ridge(alpha=regularization)
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    return pipeline


def fit_and_plot_polynomial_from_degree(
    degree: int, X: np.ndarray, y: np.ndarray, regularization=False
):
    # We initialize the polynomial model
    if regularization:
        model = get_polynomial_model_with_regularization(degree=degree)
    else:
        model = get_polynomial_model(degree=degree)

    # We train the model
    model.fit(X[:, np.newaxis], y)

    # We plot the trained model together with the raw data and the true function/trend
    visualize_model(X, y, model=model, true_function=true_function)
