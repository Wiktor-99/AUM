from skimage.io import imread_collection
from skimage.transform import resize
import numpy as np
from joblib import load

def load_images(class_name, values, labels):
    path_to_image = 'raw-img/'
    postfix = '/*.jp*'
    image_size = (64,64,3)
    i = 0
    for image in imread_collection(path_to_image + class_name + postfix):

        img = resize(image, image_size)
        values.append(img)
        labels.append(class_name)


def create_data_set():
    class_names = ['spider', 'horse', 'elephant', 'chicken', 'sheep']
    values = []
    labels = []
    for class_name in class_names:
        load_images(class_name, values, labels)

    return np.array(values), np.array(labels)
