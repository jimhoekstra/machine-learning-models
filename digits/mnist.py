import numpy as np
import tensorflow as tf
from .constants import NUM_CLASSES


class MNIST:

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes

        self.x_train: None | np.ndarray = None
        self.y_train: None | np.ndarray = None
        self.x_test: None | np.ndarray = None
        self.y_test: None | np.ndarray = None

        self.load_mnist()
        self.preprocess_data()

    def load_mnist(self) -> None:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def preprocess_data(self) -> None:
        # Make sure that self.load_mnist() has been called first
        if self.x_train is None or self.y_train is None or self.x_test is None or self.y_test is None:
            raise ValueError('MNIST data must be loaded before preprocessing')
        
        # Make sure that the input pixel values are between 0 and 1
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

        self.x_train = np.expand_dims(self.x_train, -1)
        self.x_test = np.expand_dims(self.x_test, -1)

        # Convert the integer labels to a one-hot encoded 2-dim array
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)

    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.x_train is None or self.y_train is None:
            raise ValueError('MNIST data must be loaded')
        return self.x_train, self.y_train

    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self.x_test is None or self.y_test is None:
            raise ValueError('MNIST data must be loaded')
        return self.x_test, self.y_test
