import os
import cv2
import numpy as np
from scipy.cluster import vq
from scipy.cluster.vq import kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#Takes the descriptors returned by OpenCV's detectAndCompute function and stacks them vertically
def stackDesc(desc):

    temp = desc[0][1]
    for path, des in desc[1:]:
        temp = np.vstack((temp, des))

    return temp

#Performs K-means clustering on the given descriptors
#We only need the codebook for the purpose of this project
def performKMeans(descriptors, k):

    codebook, distortion = kmeans(descriptors, k, 1)
    return codebook

#Calculates the Histogram, given the descriptors and vocabulary
def calcFeatures(paths, descriptors, vocabulary, k):

    path_len = len(paths)
    temp = np.zeros((path_len, k), "float32")

    #Using vector quantization to match the codebook
    for i in range(path_len):
        code, dist = vq.vq(descriptors[i][1], vocabulary)
        for j in code:
            temp[i][j] += 1

    return temp

print("Starting....")

#Current training path
training_path = 'images/train'

print('The current training path is set to ', training_path)
print('Starting training')

#The number of classes we have, or 'labels'. In our case it is just positive and negative.
labels = os.listdir(training_path)

#Reading the data from the training path
paths = []
classes = []
id = 0

for label in labels:
    folder = os.path.join(training_path, label)
    class_path = [os.path.join(folder, x) for x in os.listdir(folder)]
    paths += class_path
    classes += [id] * (len(class_path))
    id += 1

#OpenCV2/s sift object
sift = cv2.xfeatures2d.SIFT_create()

#List of descriptors (Not vertically stacked yet)
desc = []

#We read each image (and convert it to grayscale for cv2's SIFT)
for img_path in paths:
    print('Currently computing features from ', img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Computing the descriptors
    k, des = sift.detectAndCompute(img, None)
    desc.append((img_path, des))

print('Finished computing features, stacking descriptors now')
#We need the descriptors stacked vertically in an array
descriptors = stackDesc(desc)

print('Performing KMeans clustering')
#The rule of thumb for k is usually 10 * (number of classes)
#We only have 2 classes, so k = 20 should be ok
k = 20
vocabulary = performKMeans(descriptors, k)

#Calculating the Histogram of features
print('Calculating Histogram')
features = calcFeatures(paths, desc, vocabulary, k)

#Scaling the features
print('Scaling features')
scaler = StandardScaler().fit(features)
features = scaler.transform(features)

#We then use the features to train the SVM classifier
print('Fitting the SVM classifier')
svm = LinearSVC()
svm.fit(features, np.array(classes))

#We're done with training
print('Training done')

#Current testing path
testing_path = 'images/test'
print('Testing path is currently set to ', testing_path)

#The labels found in the testing folder (We only use this for the os.join function below)
test_labels = os.listdir(testing_path)

test_paths = []

for label in test_labels:
    folder = os.path.join(testing_path, label)
    class_path = [os.path.join(folder, x) for x in os.listdir(folder)]
    test_paths += class_path

#List of descriptors for the testing data
test_desc = []

#Again, we convert the images to grayscale and compute the features
#This time for test data
for img_path in test_paths:
    print('Currently computing features from ', img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, des = sift.detectAndCompute(img, None)
    test_desc.append((img_path, des))

print('Finished computing features, stacking descriptors now')
#Stack the test descriptors vertically
test_descriptors = stackDesc(test_desc)

print('Calculating Histogram')
#Calculate the Histogram for the features of test data
test_features = calcFeatures(test_paths, test_desc, vocabulary, k)

print('Scaling features')
#Scale (normalize) the test features
test_features = scaler.transform(test_features)

#Finally, we use the SVM classifier we trained earlier to predict the labels 
predictions =  [test_labels[i] for i in svm.predict(test_features)]

#We now have the predictions and the corresponding image_paths, we can visalize the results
for img_path, pred in zip(test_paths, predictions):
        img = cv2.imread(img_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        #Properties for the text on top of the image
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (0, 1440)
        scale = 4
        color = (255, 0, 0)
        line = 10
        cv2.putText(img, pred, origin, font, scale, color, line)
        cv2.imshow("Image", img)

        #Wait for 2 seconds before displaying the next image
        cv2.waitKey(1000)