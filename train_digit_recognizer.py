from digits.digit_recognizer import DigitRecognizer
from digits.mnist import MNIST


if __name__ == '__main__':
    digit_recognizer = DigitRecognizer()
    mnist_dataset = MNIST()

    x_train, y_train = mnist_dataset.get_train_data()
    digit_recognizer.train(x_train=x_train, y_train=y_train)
    digit_recognizer.save()
