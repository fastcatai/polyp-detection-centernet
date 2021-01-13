from flask import Flask, request
from PIL import Image
from detection import PolypDetection
import numpy as np
import json

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # convert to PIL Image
    image = Image.open(request.files['image'].stream)
    # get numpy array
    image = np.array(image)
    # detect box
    predictions = detector.predict(image)
    # convert all prediction to json
    json_predictions = []
    for xmin, ymin, xmax, ymax, score, class_id in predictions:
        json_predictions.append({'xmin': xmin, 'ymin': ymin,
                                 'xmax': xmax, 'ymax': ymax,
                                 'label': labels[int(class_id)], 'score': score})
    return json.dumps(json_predictions)


@app.route('/predict', methods=['GET'])
def predict_info():
    return 'Send image via POST request to this address and get bounding box coordinates as return'


@app.route('/')
def home():
    return 'Polyp Detection Server'


if __name__ == '__main__':
    labels = ['polyp']
    detector = PolypDetection()
    app.run(host='0.0.0.0', port=1234)
