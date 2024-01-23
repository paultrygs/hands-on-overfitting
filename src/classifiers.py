from abc import ABC, abstractmethod
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss


class Classifier(ABC):
    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train/fit the model to the input training data.

        Warning: Applying fit changes the model's weights.

        Args:
            X_train (np.ndarray): A NumPy array of inputs to the model, e.g. images.
            y_train (np.ndarray): A NumPy array of outputs/labels. That is, y_train[0] is the true output/label of X_train[0].
        """
        self.model.fit(self._flatten(X_train), y_train)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Percentage (0 worst to 1 best) of correct predictions when using the model on all entries in X. 
        The array y consists of the correct answers.

        Args:
            X (np.ndarray): A NumPy array of prediction inputs.
            y (np.ndarray): A NumPy array with the correct labels/predictions to X.

        Returns:
            float: The percentage of correct predictions
        """
        return self.model.score(self._flatten(X), y)

    def error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate a mean error when using the model on all entries in X. 
        The array y consists of the correct answers.

        Args:
            X (np.ndarray): A NumPy array of prediction inputs.
            y (np.ndarray): A NumPy array with the correct labels/predictions to X.

        Returns:
            float: A number measuring the error when predicting y from X.
        """
        return log_loss(y, self.model.predict_proba(self._flatten(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Use the trained model on all entries in X.

        Args:
            X (np.ndarray): A NumPy array of objects the model will be applied on.

        Returns:
            np.ndarray: A NumPy array of predictions.
        """
        return self.model.predict(self._flatten(X))

    def _flatten(self, X) -> np.ndarray:
        return np.array([data.flatten() for data in X])


class PolynomialClassifier(Classifier):
    def __init__(self, degree: int, regularization: float = 1):
        """Initialize an untrained PolynomialClassifier model.

        Args:
            degree (int): The degree of polynomials to be used for classification.
            regularization (float, optional): A term for adjusting regularization. Defaults to 1.
        """
        self._model = SVC(
            degree=degree,
            C=regularization,
            max_iter=10000,
            kernel="poly",
            probability=True,
            tol=0.000001,
        )

    @property
    def model(self):
        return self._model


class NeuralNetworkClassifier(Classifier):
    def __init__(self, hidden_layer_sizes: tuple, regularization: float = 0.0001):
        """Initialize an untrained NeuralNetwork model.

        Args:
            hidden_layer_sizes (tuple): A tuple of integers representing the size of the hidden layers in the model. Ex: (10, ) gives one layer of size 10, (5, 10) gives two layers of size 5 and 10.
            regularization (float, optional): A term for adjusting regularization. Defaults to 0.0001.
        """
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=regularization,
            activation="relu",
            solver="lbfgs",
            early_stopping=False,
            max_iter=10000,
            tol=0.000001,
            max_fun=1000000,
        )

    @property
    def model(self):
        return self._model

