from pathlib import Path
import numpy as np
import tensorflow as tf
from .constants import NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE, EPOCHS, MODELS_DIR, DIGITS_MODEL_NAME


MODEL_PATH: str = str(Path.cwd() / MODELS_DIR / DIGITS_MODEL_NAME)


class DigitRecognizer:

    def __init__(
        self, 
        load_model: bool = False,
        input_shape: tuple[int, int, int] = INPUT_SHAPE, 
        batch_size: int = BATCH_SIZE
    ) -> None:
        self.input_shape = input_shape
        self.batch_size = batch_size

        if load_model:
            model = tf.keras.models.load_model(filepath=MODEL_PATH)
            if not isinstance(model, tf.keras.Model):
                raise ValueError('something went wrong with loading the model')
            self.model: tf.keras.Model = model
            self.is_trained: bool = True
        else:
            self.model: tf.keras.Model = self.init_model()
            self.is_trained: bool = False

    def init_model(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = EPOCHS) -> None:
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=epochs, validation_split=0.1)
        self.is_trained = True

    def print_test_accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        pass

    def predict(self):
        pass

    def save(self) -> None:
        print('saved model at:', MODEL_PATH)
        tf.keras.models.save_model(model=self.model, filepath=MODEL_PATH)
