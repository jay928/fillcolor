import cv2

SOURCE_PATH = '/Users/jwp928/Documents/deeplearning/data/fillcolor/people/1.jpg'

color_image = cv2.imread(SOURCE_PATH)
lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_image)
l = l / 255
a = a / 255
b = b / 255

print(l)