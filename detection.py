import numpy as np
from resnet import centernet
from utils import preprocess_image, get_affine_transform, affine_transform
import tensorflow as tf

# Flask web request creates a new thread with its own Tensorflow session
# Therefore we need to access our default session.
#   https://stackoverflow.com/a/61560102/1964855
#   https://kobkrit.com/tensor-something-is-ot-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1
# Alternatively we could run flask as a single threaded application.
#   https://github.com/keras-team/keras/issues/10585#issuecomment-646609593


class PolypDetection:
    def __init__(self) -> None:
        self.flip_test = True
        num_classes = 1
        self.score_threshold = 0.1
        backbone = 'resnet101'  # available backbones ['resnet50', 'resnet101', 'resnet152']
        # change local path if run locally
        # model_path = '{path/to/h5}'
        # path to run inside of docker
        model_path = '/app/centernet-resnet101-frozen-e200_b16_lr0.00001_csv_e199_l0.9415_vl1.0991.h5'
        self.sess = tf.Session()
        self.prediction_model = centernet(num_classes=num_classes,
                                          score_threshold=self.score_threshold,
                                          backbone=backbone,
                                          flip_test=self.flip_test, nms=True, freeze_bn=True)
        self.prediction_model.load_weights(model_path, by_name=True, skip_mismatch=True)
        self.sess.graph.as_default()

    def predict(self, image: np.ndarray) -> list:
        src_image = image.copy()
        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        s = max(image.shape[0], image.shape[1]) * 1.0
        tgt_w = 512
        tgt_h = 512
        image = preprocess_image(image, c, s, tgt_w=tgt_w, tgt_h=tgt_h)
        if self.flip_test:
            flipped_image = image[:, ::-1]
            input = np.stack([image, flipped_image], axis=0)
        else:
            input = np.expand_dims(image, axis=0)

        with self.sess.graph.as_default():
            predictions = self.prediction_model.predict_on_batch(input)[0]

        scores = predictions[:, 4]
        indices = np.where(scores > self.score_threshold)[0]  # select indices which have a score above the threshold

        # select those detections
        predictions = predictions[indices]
        predictions = predictions.astype(np.float64)
        trans = get_affine_transform(c, s, (tgt_w // 4, tgt_h // 4), inv=1)

        for j in range(predictions.shape[0]):
            predictions[j, 0:2] = affine_transform(predictions[j, 0:2], trans)
            predictions[j, 2:4] = affine_transform(predictions[j, 2:4], trans)

        predictions[:, [0, 2]] = np.clip(predictions[:, [0, 2]], 0, src_image.shape[1])
        predictions[:, [1, 3]] = np.clip(predictions[:, [1, 3]], 0, src_image.shape[0])

        return predictions


if __name__ == '__main__':
    from keras.preprocessing.image import img_to_array, load_img
    import json

    detector = PolypDetection()
    # image path if run locally
    image = img_to_array(load_img('D:/Users/kevin/Nextcloud/polyp-datasets/ETIS-LaribPolypDB/ETIS-LaribPolypDB-png/17.png'))
    # image = img_to_array(load_img('D:/Users/kevin/Nextcloud/polyp-datasets/CVC-VideoClinicDB-test/1/1-1.png'))
    # image = img_to_array(load_img('{path/to/image}'))
    predictions = detector.predict(image)
    json_predictions = []
    labels = ['polyp']
    for xmin, ymin, xmax, ymax, score, class_id in predictions:
        json_predictions.append({'xmin': xmin, 'ymin': ymin,
                                 'xmax': xmax, 'ymax': ymax,
                                 'label': labels[int(class_id)], 'score': score})
    print(json.dumps(json_predictions))
