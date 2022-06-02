from skimage.io import imread_collection
from skimage.transform import resize
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score
import sys
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.exposure import histogram
from skimage.color import rgb2gray, rgb2hsv



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
        load_images(path_to_images, class_name, values, labels)

    return np.array(values), np.array(labels)

def load_models():
    folder = "models/"
    return { "svm" : load(folder + "svm.sav"),
             "mlp" : load(folder + "mlp.sav"),
             "knn" : load(folder + "knn.sav")  }

def get_histogram(image, bins, channel):
    hist, _ = histogram(rgb2hsv(image)[:, :, channel], nbins=bins)
    return hist

def get_all_colors_histograms(image, bins = 110):
    return np.reshape((get_histogram(image, bins, 0), get_histogram(image, bins, 1), get_histogram(image, bins, 2)), (bins*3))

def get_hog_transform(image):
    return hog(rgb2gray(image), orientations = 5, pixels_per_cell = (8,8), cells_per_block = (8,8))

def extract_features(image):
    return np.concatenate((get_hog_transform(image), get_all_colors_histograms(image)))

def extract_hog_features(values):
    scalify = StandardScaler()
    extracted_features = np.array([ extract_features(img) for img in values ])
    return scalify.fit_transform(extracted_features)

def main():
    x, y = create_data_set(sys.argv[1])
    models = load_models()
    x = extract_hog_features(x)
    pred = models[sys.argv[2]].predict(x)
    print('Percentage correct for model: ', 100*accuracy_score(pred, y))

if __name__ == '__main__':
    main()