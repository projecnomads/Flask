import cv2
import os


class FaceDetector:
    def __init__(self):
        self.cascadePath = os.path.join(os.getcwd(), "static" + "\" + "haarcascade_frontalface_default.xml")
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.cord = []
        self.croppedImages = []

    def detectFaces(self, image):
        '''

        :param image: takes image for face detection
        :return:
            croppedImages: Images of faces are cropped
            cord : Coordinates of cropped image
        '''
        image = cv2.imread(image)
        grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(grayScale, 1.3, 5)

        for (x, y, w, h) in faces:
            self.cord.append([x, y, w, h])
            self.croppedImages.append(image[y:y + h, x:x + w])

        return self.croppedImages, self.cord
