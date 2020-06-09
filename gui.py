from tkinter import *
from keras.models import load_model

# loading Python Imaging Library 
from PIL import ImageTk, Image

# To get the dialog box to open when required  
from tkinter import filedialog
from faceDetector import FaceDetector
from faceNetKeras import FaceNet

import numpy as np
import cv2

np.set_printoptions(suppress=True)
import os

model_path = os.path.join(os.getcwd(), '/static/feedNet.h5')
model = load_model(model_path)

user_dict = {
    0: "Arjun MS",
    1: "Asish",
    2: "Roshan",
    3: "Modi",
    4: "Obama"
}

os.environ['KERAS_BACKEND'] = 'theano'
faceDetector = FaceDetector()
faceNet = FaceNet()
# Create a windoe
root = Tk()

# Set Title as Image Loader
root.title("Image Loader")
root.geometry("500x500")

present = []


# Set the resolution of window
def open_img():
    # Select the Imagename  from a folder
    x = openfilename()
    croppedImages, _ = faceDetector.detectFaces(x)
    for i in range(len(croppedImages)):
        image = croppedImages[i]
        image = cv2.resize(image, (160, 160))
        image = image.astype('float') / 255.0
        image = np.expand_dims(image, axis=0)
        embedVal = faceNet.predictEmbedding(image)
        embedVal = np.expand_dims(embedVal, axis=0)
        result = model.predict(embedVal)[0]
        present.append(user_dict[int(np.argmax(result))])
    # opens the image
    img = Image.open(x)
    print(present)

    text = Text(root)
    text.insert(INSERT, present)
    text.grid(row=2)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    panel.grid(row=1)


def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='"pen')
    return filename


# Allow Window to be resizable
root.resizable(width=True, height=True)

# Create a button and place it into the window using grid layout
btn = Button(root, text='open image', command=open_img).grid(
    row=1, columnspan=4)

root.mainloop()
