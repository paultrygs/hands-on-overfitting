from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def visualize_model(
    X: np.ndarray,
    y: np.ndarray,
    model: Pipeline | None = None,
    true_function: Callable | None = None,
):
    plt.scatter(X, y, s=15, label="Raw data")
    plt.ylim((np.min(y) - 5, np.max(y) + 5))
    x_axis = np.linspace(0, 1, 100)
    if model:
        plt.plot(
            x_axis,
            model.predict(x_axis[:, np.newaxis]),
            color="orange",
            label="Trained model",
        )
        plt.scatter(
            X,
            model.predict(X[:, np.newaxis]),
            s=15,
            marker="x",
            color="red",
            label="Predicted data",
        )
    if true_function:
        plt.plot(x_axis, true_function(x_axis), color="green", label="True function")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.show()
