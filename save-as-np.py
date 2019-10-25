import numpy as np
import glob, os, cv2
from os.path import isfile, join

SOURCE_PATH = '/Users/jwp928/Documents/deeplearning/data/fillcolor/'




def get_images():
    x_data = []
    y_data = []

    files = glob.glob(SOURCE_PATH + "*/*.jpg")
    for file in files:
        if file == '.DS_Store':
            continue

        print(file)
        try:
            color_image = cv2.imread(file)
            resized_image = cv2.resize(color_image, (256, 256), interpolation = cv2.INTER_LINEAR)
            lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab_image)
            l = l / 255
            a = a / 255
            b = b / 255

            x_data.append(l)
            y_data.append([a, b])
        except:
            continue


    return np.array(x_data), np.array(y_data)


x_data, y_data = get_images()

print(x_data.shape)
print(y_data.shape)

np.save(SOURCE_PATH + "x", x_data, allow_pickle=True)
np.save(SOURCE_PATH + "y", y_data, allow_pickle=True)
print("COMPLETED!")