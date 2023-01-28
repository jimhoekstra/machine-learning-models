import os
import numpy as np
from flask import Flask, request, make_response
from digits.digit_recognizer import DigitRecognizer


app = Flask(__name__)


@app.route('/')
def hello_world():
    return {'status': 'OK'}


def bool_to_int(value: bool):
    if value:
        return 1
    else:
        return 0


@app.route('/digits', methods=['POST', 'OPTIONS'])
def digit_recognizer():
    if request.method == 'OPTIONS':
        cors_response = make_response()
        cors_response.headers['Access-Control-Allow-Origin'] = os.environ['DIGITS_WEBSITE_URL']
        cors_response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return cors_response

    digit_recognizer: DigitRecognizer = DigitRecognizer(load_model=True)
    request_data = request.get_json()

    if request_data is None:
        return {'status': 'error'}
    
    try:
        digit = request_data['digit']
    except KeyError:
        return {'status': 'error', 'message': '"digit" parameter required in request body'}
    
    digit = list(map(bool_to_int, digit))
    digit_as_array = np.array(digit).reshape(1, 28, 28, 1)

    predictions = digit_recognizer.predict(digit_as_array)
    response = make_response({'status': 'OK', 'predictions': predictions})
    response.headers['Access-Control-Allow-Origin'] = os.environ['DIGITS_WEBSITE_URL']
    return response
