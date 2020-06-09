from keras.models import load_model
import os


class FaceNet:
    def __init__(self):
        self.faceNetModel = load_model(os.path.join(os.getcwd(), "static" + "\" + "facenet_keras.h5"))

    def predictEmbedding(self, image):
        return self.faceNetModel.predict(image)[0]
