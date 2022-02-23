import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps
import os, ssl, time

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

print(pd.Series(y).value_counts())
classes = ['A', 'B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses  = len(classes)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 500, train_size= 3500, random_state = 9)
xTrainScaled = xTrain/255
xTestScaled = xTest/255

lr = LogisticRegression(solver='saga', multi_class = 'multinomial').fit(xTrainScaled, yTrain)

lr.fit(xTrainScaled, yTrain)

def getPred(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
    test_pred = lr.predict(test_sample)
    return test_pred[0]