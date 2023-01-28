from flask import Flask
from digits.digit_recognizer import DigitRecognizer


app = Flask(__name__)


@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'


@app.route('/digits')
def digit_recognizer():
    digit_recognizer: DigitRecognizer = DigitRecognizer(load_model=True)
    return {'result': 0}
