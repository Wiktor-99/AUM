from os.path import isdir, isfile
from skimage.io import imread_collection
import sys
from skimage.transform import resize
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skimage.feature import hog
from skimage.exposure import histogram
from skimage.color import rgb2gray, rgb2hsv
from sklearn.base import BaseEstimator, ClassifierMixin

def crate_path_to_directory(folder_name):
    return folder_name + '/*'

def create_path_to_data(path):
    if isdir(path):
        return crate_path_to_directory(path)
    elif isfile(path):
        return path

    return None

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
    print("Usage of script:")
    print("python3 AnimalsImageRecognitionApp.py -f path_to_image/s -a algorithm_1 [algorithm_2] [algorithm_3] [-w] [w1] [w2] [w3]")
    print("-f file")
    print("-a algorithms list")
    print("-w weights for algorithms")
    print("Available ML algorithms:")
    print(" -svm")
    print(" -mlp")
    print(" -knn")
    print("App as results return print list of predicted classes")

class VoteClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classifiers, weights=None):
        self.classifiers = classifiers
        self.weights = weights
        self.label_encoder = LabelEncoder()


    def fit(self, x_train, y_train):
        return self

    def predict(self, x_test):
        result = np.asarray([clf.predict(x_test) for clf in self.classifiers])

        result = self.transform_labels(result)

        final_result = np.apply_along_axis(self.get_argmax_class, axis=0, arr = result)

        return self.label_encoder.inverse_transform(final_result)

    def transform_labels(self, predictions):
        fitted_labels = []
        for i in range(len(self.classifiers)):
            self.label_encoder.fit(predictions[i])
            fitted_labels.append(self.label_encoder.transform(predictions[i]))

        return fitted_labels

    def get_argmax_class(self,y_value):
         return np.argmax(np.bincount(y_value, weights=self.weights))


def get_classifiers(models, classifiers_names):
    classifiers = []
    for algorithm_name in classifiers_names:
        if algorithm_name in models:
            classifiers.append(models[algorithm_name])
        else:
            print(f"{algorithm_name} algorithm is unavailable")
            print("Pleas use on of knn, svm, mlp.")
            return None

    return classifiers

def get_weights(weights_strings):
    weights = []
    for weight in weights_strings:
        try:
            weights.append(float(weight))
        except ValueError:
            print("One of weights is not float, will not apply any of weights")
            return None

    return weights

def parse_classifiers_and_weights(models, argv):
    if argv[3] == "-a":
        if "-w" in argv:
            index_of_w = argv.index("-w")
            classifiers = get_classifiers(models, argv[4:index_of_w])
            weights = get_weights(argv[index_of_w+1:])
        else:
            classifiers = get_classifiers(models, argv[4:])
            weights = None
    return classifiers,weights

def make_prediction(path, classifiers, weights):
    classifier = VoteClassifier(classifiers, weights)

    data = extract_hog_features(load_images_from_directory(path))

    prediction = classifier.predict(data)
    print('Predicted values', prediction)

def prase_input_file():
    if sys.argv[1] == "-f":
        return create_path_to_data(sys.argv[2])
    return None

def main():
    if sys.argv[1] == "-h":
        print_help()
        return

    path = prase_input_file()

    if not path:
        print("First argument is not file or directory, for help call with -h")
        return

    classifiers, weights = parse_classifiers_and_weights(load_models(), sys.argv)

    if not classifiers:
        return

    if weights:
        if len(weights) != len(classifiers):
            print("weights and classifier must have same length")
            return

    make_prediction(path, classifiers, weights)

if __name__ == '__main__':
    main()