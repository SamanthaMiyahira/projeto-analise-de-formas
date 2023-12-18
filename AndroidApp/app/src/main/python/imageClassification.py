import cv2
import os
from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
from skimage import data
# from google.colab import drive

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import struct

def extract_images(images):
        root_directory = 'data/EP - Analise de Formas/'
        root_directory_name = join(dirname(__file__), root_directory) # converte o nome no formato de chaquopy
        for class_folder in os.listdir(root_directory_name):
            class_path = os.path.join(root_directory, class_folder)
            class_path_name = join(dirname(__file__), class_path)
            if os.path.isdir(class_path_name):
                for image_file in os.listdir(class_path_name):
                    file_path = os.path.join(class_path, image_file)
                    file_path_name = join(dirname(__file__), file_path)
                    if os.path.isfile(file_path_name) and image_file.lower().endswith('.jpg'):
                        image = mpimg.imread(file_path_name)
                        image = cv2.resize(image, (450, 350))
                        images.append({
                            'class': class_folder,
                            'path': file_path_name,
                            'image': image
                        })
        images = np.array(images)
        return images

def RGB2GRAY(image):
    original_image = image['image']
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': gray_image
    }
    return final_image

# Apply data augmentation function to the grayscale image
def add_gradient_background(image):
    # Generate a gradient image with the same dimensions as the input image
    gradient = np.zeros_like(image['image'])
    rows, cols = gradient.shape
    # Gradient: left to right
    for i in range(cols):
        gradient[:, i] = i * (255/cols)
    # Add the gradient to the original image
    augmented_image = cv2.addWeighted(image['image'], 0.7, gradient, 0.3, 0)

    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': augmented_image
    }
    return final_image

def convert_to_log(image):
    # avoids division by zero
    image['image'] = image['image'] + 0.1
    # applies the logarithmic transformation.
    c = 255 / np.log(1 + np.max(image['image']))
    log_image = c * (np.log(image['image'] + 1))
    # converts the values of the float matrix to integers
    log_image = np.array(log_image, dtype = np.uint8)
    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': log_image
    }
    return final_image

def convert_to_exp(image):
    # evita divisao por zero
    image['image'] = image['image'] + 0.1
    # aplica a transformacao logaritmica
    c = 255 / np.log(1 + np.max(image['image']))
    exp_image = np.exp(image['image']) ** (1/c) - 1
    exp_image = np.array(exp_image, dtype = np.uint8)
    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': exp_image
    }
    return final_image
def media_filter_augmentation(image):
    kernel_size = 3
    blurred_image = cv2.blur(image['image'], (kernel_size, kernel_size))
    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': blurred_image
    }
    return final_image
def equalize_histogram(image):
    equalized_image = cv2.equalizeHist(image['image'].astype(np.uint8))
    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': equalized_image
    }
    return final_image

def automatic_segmentation(image):
    # Apply Otsu's Thresholding
    _, segmented_image = cv2.threshold(image['image'].astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': segmented_image
    }
    return final_image

def feret_box(image):
    binary_image = image['image']
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contour of max area
    max_contour = max(contours, key=cv2.contourArea)

    # ferex box delimitation
    x, y, w, h = cv2.boundingRect(max_contour)
    feret_box = binary_image[y:y+h, x:x+w]

    image_with_feret_box = {
        'class': image['class'],
        'path': image['path'],
        'image': binary_image,
        'feret_box': feret_box,
        'bounding_box': (x, y, w, h)

    }

    return image_with_feret_box

def feature_vector(image):
    img = copy.deepcopy(image['image'])
    x, y = img.shape
    features = np.reshape(img, (x*y))

    final_image = {
        'class': image['class'],
        'path': image['path'],
        'image': image['image'],
        'features': features
    }

    return final_image

classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM": SVC(gamma=2, C=1),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression()
}
def get_better_classifier(dataset):
    X = np.array([img['features'] for img in dataset])
    y = np.array([img['class'] for img in dataset])

    # Set of train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    better_classifier = ""
    max_score = -1

    # train and test each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        if(score != 1 and score > max_score): # max accuracy but not overfitting
            max_score = score
            better_classifier = name
        print(f"\nClassifier: {name}")
        print("Accuracy:", score)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return better_classifier

def process_image(image):
    processes = [RGB2GRAY, add_gradient_background, convert_to_log, convert_to_exp,
                 media_filter_augmentation, equalize_histogram, automatic_segmentation,
                 feret_box, feature_vector]

    for process in processes:
        image = process(image)

    return image

def train_model():
    images = []
    images = extract_images(images)
    total_images = len(images)
    print(total_images)

    ################## gray images ##################

    print("comecou gray image")

    gray_images = []

    for i in range(total_images):
        gray_img = RGB2GRAY(images[i])
        gray_images.append(gray_img)

    gray_images = np.array(gray_images)

    print("executou gray image")

    ################# images gradient #################

    print("comecou gradient image")

    gradient_images = []

    for i in range(total_images):
        augmented_image = add_gradient_background(gray_images[i])
        gradient_images.append(augmented_image)

    gradient_images = np.array(gradient_images)

    print("acabou gradient image")

    ################## log images ##################

    print("comecou log image")

    log_images = []

    for i in range(total_images):
        augmented_image = convert_to_log(gray_images[i])
        log_images.append(augmented_image)

    log_images = np.array(log_images)

    print("acabou log image")

    ################## exp images ##################

    print("comecou exp image")

    exp_images = []

    for i in range(total_images):
        augmented_image = convert_to_exp(gray_images[i])
        exp_images.append(augmented_image)

    exp_images = np.array(exp_images)

    print("acabou exp image")

    ################## media images ##################

    print("comecou media image")

    convolution_images = []

    for i in range(total_images):
        augmented_image = media_filter_augmentation(gray_images[i])
        convolution_images.append(augmented_image)

    convolution_images = np.array(convolution_images)

    print("comecou media image")

    ################## normal images ##################

    print("comecou normal image")

    normal_images = []

    for i in range(total_images):
        augmented_image = equalize_histogram(gray_images[i])
        normal_images.append(augmented_image)

    normal_images = np.array(normal_images)

    print("acabou normal image")
    #
    # ### SEGMENTACAO MANUAL FAZER !!!!!!
    #
    ################## segmented images ##################

    print("comecou segmented image")

    segmented_images = []

    for i in range(total_images):
        segmented_image = automatic_segmentation(gray_images[i])
        segmented_images.append(segmented_image)

    segmented_images = np.array(segmented_images)

    print("acabou segmented image")

    ################## feret box images ##################

    print("comecou feret box image")

    feret_box_images = []

    for i in range(len(segmented_images)):
        augmented_image = feret_box(segmented_images[i])
        feret_box_images.append(augmented_image)

    feret_box_images = np.array(feret_box_images)

    print("acabou feret box image")

    # ################ features images ##########################

    print("comecou feature image")

    image_features = []

    for i in range(len(feret_box_images)):
        img_feature = feature_vector(feret_box_images[i])
        image_features.append(img_feature)

    image_features = np.array(image_features)

    print("acabou feature image")

    # ################ get better classifier ##########################

    better_classifer_name = get_better_classifier(image_features)
    return better_classifer_name

def recognize_image(picture, better_classifier):
    # Convert the byte array to a NumPy array
    np_array = np.frombuffer(picture, dtype=np.uint8)

    # Decode the image from NumPy array (assuming it's a color image)
    picture = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    picture = cv2.resize(picture, (450, 350))
    picture = {
        'class': "known",
        'path': "none",
        'image': picture,
    }

    picture = process_image(picture)
    picture['features'] = picture['features'].reshape(1, -1)
    image_predict = classifiers[better_classifier].predict(picture['features'])
    print(f"The image class is: {image_predict[0]}")

    return image_predict[0]
