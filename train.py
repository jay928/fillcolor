import cv2
import numpy as np
import os

SOURCE_PATH = '/Users/jwp928/Documents/deeplearning/data/fillcolor/'


def getImages():
    x_data = []
    y_data = []

    list = os.listdir(SOURCE_PATH)
    for file in list:
        color_image = cv2.imread(file, cv2.IMREAD_COLOR)
        # grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)
        l = l / 255
        a = a / 255
        b = b / 255

        x_data.append(l)
        y_data.append([a, b])

    return np.array(x_data), np.array(y_data)


x_data, y_data = getImages()
print(x_data.shape)
print(y_data.shape)