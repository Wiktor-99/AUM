from os.path import isdir
from skimage.io import imread_collection
from skimage.transform import resize
import numpy as np
from joblib import load
import sys
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.exposure import histogram
from skimage.color import rgb2gray, rgb2hsv


def crate_path_to_directory(folder_name):
    return folder_name + '/*'

def create_path_to_data(path):
    if isdir(path):
        return crate_path_to_directory(path)

    return path

def load_images_from_directory(path_to_images):
    image_size = (64,64,3)
    values = []
    for image in imread_collection(path_to_images):
        img = resize(image, image_size)
        values.append(img)

    return np.array(values)

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

def print_help():
    print("Image based animal classification app")
    print("Usage of script python3 AnimalsImageRecognitionApp.py path_to_image/s algorithm_1 [algorithm_2] [algorithm_3]")
    print("Available ML algorithms:")
    print(" -svm")
    print(" -mlp")
    print(" -knn")
    print("App as results return print list of predicted classes")

def main():
    path = create_path_to_data(sys.argv[1])

    x = load_images_from_directory(path)

    models = load_models()
    x = extract_hog_features(x)

    pred = models[sys.argv[2]].predict(x)
    print('Predicted values', pred)

if __name__ == '__main__':
    main()