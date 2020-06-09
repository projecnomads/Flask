from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
from keras.models import load_model
from faceDetector import FaceDetector
from faceNetKeras import FaceNet
import numpy as np

np.set_printoptions(suppress=True)
import os

os.environ['KERAS_BACKEND'] = 'theano'
faceDetector = FaceDetector()
faceNet = FaceNet()
model_path = os.path.join(os.getcwd(),"static" + "\" + "feedNet.h5")
model = load_model(model_path)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
upload_path = "uploads"
app.config['UPLOAD_FOLDER'] = upload_path

user_dict = {0: 'gayathri', 1: 'gal', 2: 'krishnendhu', 3: 'modi', 4: 'obama', 5: 'chrispine'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    present = []

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(f"{app.config['UPLOAD_FOLDER']}", filename))
            path = f"{upload_path}/{filename}"
            croppedImages, _ = faceDetector.detectFaces(path)
            present = []
            for i in range(len(croppedImages)):
                image = croppedImages[i]
                image = cv2.resize(image, (160, 160))
                image = image.astype('float') / 255.0
                image = np.expand_dims(image, axis=0)
                embedVal = faceNet.predictEmbedding(image)
                embedVal = np.expand_dims(embedVal, axis=0)
                result = model.predict(embedVal)[0]
                present.append(user_dict[int(np.argmax(result))])
    if (len(present) > 0):
        return jsonify(present=present)
    else:
        return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)
