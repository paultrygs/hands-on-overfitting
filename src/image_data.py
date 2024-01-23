import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def get_image_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get 8x8 images of integers between 0 and 9, together with labels (revealing the depicted numbers),
    The images and labels are split into training and test datasets.

    Usage:

        X_train, X_test, y_train, y_test = get_image_data()

    where
        - X_train: training images
        - X_test: test images
        - y_train: training labels
        - y_test: test labels
    """
    _data = load_digits()
    _images = _data.images
    _labels = _data.target

    RESOLUTION = 8
    SCALE = int(RESOLUTION / 8)

    _images = np.array(
        [
            np.array(
                [
                    [image[int(i / SCALE), int(j / SCALE)] for j in range(RESOLUTION)]
                    for i in range(RESOLUTION)
                ]
            )
            for image in _images
        ]
    )

    images_1d = np.array([image.flatten() for image in _images])

    noise = lambda x: x + np.random.normal(0, 1, size=len(x))
    noisy_images_1d = np.array([noise(image) for image in images_1d])

    noisy_images = np.array(
        [image.reshape((RESOLUTION, RESOLUTION)) for image in noisy_images_1d]
    )

    X, _, y, _ = train_test_split(noisy_images, _labels, train_size=0.2)
    return train_test_split(X, y, train_size=0.33)


def plot_image(image: np.ndarray, label: int):
    """
    Plot an image (2x2 NumPy array) together with its label.
    """
    _, ax = plt.subplots(figsize=(2, 1.5))
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label {label}")
    plt.show()
