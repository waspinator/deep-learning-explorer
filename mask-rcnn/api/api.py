#!/usr/bin/python3

from flask import Flask
from flask_restplus import Api, Resource, fields, reqparse
from werkzeug.contrib.fixers import ProxyFix
from werkzeug.datastructures import FileStorage
from tempfile import NamedTemporaryFile
import io
from PIL import Image
import time
import os
import numpy as np

from detector import ObjectDetector
global detector
detector = ObjectDetector()

ALLOWED_MIMETYPES = set(['image/png', 'image/jpeg'])

app = Flask(__name__)
app.config['SWAGGER_UI_JSONEDITOR'] = True
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='0.1', title='Objects Detector API',
          description='Detect objects in images')

ns = api.namespace('detector', description='Object detection operations')

image_upload = reqparse.RequestParser()
image_upload.add_argument('image', location='files',
                          type=FileStorage, required=True,
                          help='PNG or JPG file')
image_upload.add_argument('tolerance', required=False, type=int, default=2,
                          help='Fidelity of detection polygons. \
                           Higest fidelity when 0, lower with increasing values.')


@ns.route('/detect')
class Detection(Resource):

    @ns.expect(image_upload)
    def post(self):
        args = image_upload.parse_args()
        image = args['image']
        tolerance = args['tolerance']

        if tolerance is None:
            tolerance = 2

        if image.mimetype not in ALLOWED_MIMETYPES:
            return {'message': 'Only png or jpeg files accepted'}, 415

        pil_image = Image.open(io.BytesIO(image.read()))

        start_time = time.time()
        detections = detector.detect(pil_image, tolerance)
        detection_time = time.time() - start_time

        return {
            'seconds_to_detect': '{:.1f}'.format(detection_time),
            'coco': detections
        }, 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='6006', use_reloader=False)
