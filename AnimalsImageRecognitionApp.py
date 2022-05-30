from skimage.io import imread_collection
from skimage.transform import resize
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score
import sys


def load_images(path_to_images, class_name, values, labels):
    postfix = '/*.jp*'
    image_size = (64,64,3)
    for image in imread_collection(path_to_images + class_name + postfix):
        img = resize(image, image_size)
        values.append(img)
        labels.append(class_name)


def create_data_set(path_to_images):
    class_names = ['spider', 'horse', 'elephant', 'chicken', 'sheep']
    values = []
    labels = []
    for class_name in class_names:
        load_images(class_name, values, labels)

    return np.array(values), np.array(labels)

def load_models():
    return { "svm" : load("svm.sav"), "mlp" : load("mlp.sav"), "knn" : load("knn.sav")  }

def main():
    x, y = create_data_set(sys.argv[1])
    models = load_models()
    pred = models[sys.argv[2]].predict(x)
    print('Percentage correct for model: ', 100*accuracy_score(pred, y))

if __name__ == '__main__':
    main()