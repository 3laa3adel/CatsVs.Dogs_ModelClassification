from cv2 import imshow
import matplotlib.image as mpimg
import numpy as np
import timeit
import os
import cv2
import pyttsx3
from scipy import rand
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from skimage.feature import hog
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
###########################################################
# 1- Read The Data.
g = pyttsx3.init()
dir = 'images'
category = ['Cat', 'Dog']
data = []
for c in category:
    folder = os.path.join(dir, c)
    label = category.index(c)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (128, 64))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(2, 2), visualize=True)
        data.append([fd, label])
###########################################################
# 2- Store Data To Train & Test It.
X = []
Y = []
for feat, label in data:
    X.append(feat)
    Y.append(label)
###########################################################
# 3- Create an Instance of SVM (Poly) and Fit out Data.
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, test_size=0.09, random_state=10, shuffle=True)
poly_svc = SVC(kernel='poly', gamma=0.8, C=0.01).fit(xtrain, ytrain)
prediction = poly_svc.predict(xtest)
accuracy = np.mean(prediction == ytest)
print("Testing Accuracy : ", "%.2f" % accuracy)
print("Training Accuracy : ", "%.2f" % poly_svc.score(xtrain, ytrain))
############################################################
