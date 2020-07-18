import os
import numpy as np
import cv2
from tqdm import tqdm
import random

train_dir = r'glasses_image_data\training_data' #train_dir contains the directory of the training images
test_dir = r'glasses_image_data\testing_data' #test_dir contains the directory of the test images
img_size = 100


def label_image(img): #label_image looks at the name of the any given images and assigns the image a label
    word_label = img[0]
    if word_label == '0':
        return 0
    elif word_label == '1':
        return 1

#-------------------------CONVERTING TEST IMAGES INTO ARRAY FORM-----------------------#

test_data = []
test_label_data = []
for img in tqdm(os.listdir(test_dir)): #this for-loop converts all images inside test_dir into arrays and assigns each image a label.
    label = label_image(img)
    path = os.path.join(test_dir, img)
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (100, 100))
    test_data.append(img)
    test_label_data.append(label)
y = np.array(test_label_data)
c = list(zip(y, test_data))
random.shuffle(c)
y, test_data = zip(*c)
np.save('test_data.npy', test_data) #saves the arrays of each image in the file test_data.npy
np.save('test_data_label.npy', y) #saves the labels (whether the person in the image is wearing glasses or not) of each image in the file test_data_label.npy

#-------------------------------------------------------------------------------------------#


#-------------------------CONVERTING TRAIN IMAGES INTO ARRAY FORM-----------------------#

test_data = []
train_data = []
train_label_data = []
for img in tqdm(os.listdir(train_dir)): #this for-loop converts all images inside train_dir into arrays and assigns each image a label.
    label = label_image(img)
    label = label_image(img)
    path = os.path.join(train_dir, img)
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (100, 100))
    train_data.append(img)
    train_label_data.append(label)
x = np.array(train_label_data)
c = list(zip(x, train_data))
random.shuffle(c)
x, train_data = zip(*c)
np.save('train_data.npy', train_data) #saves the arrays of each image in the file train_data.npy
np.save('train_data_label.npy', x) #saves the labels (whether the person in the image is wearing glasses or not) of each image in the file train_data_label.npy

#-------------------------------------------------------------------------------------------#