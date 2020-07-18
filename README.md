# Glasses-Neural-Network

# INTRODUCTION
This is my first machine learning project that I have built.
This project- Glasses Neural Netowrk - creates a neural network which can detect if a person inside an image is wearing glasses or not.
You input an image inside the neural network and it will output whether or not the person is wearing glasses. (0 means no glasses and 1 means the person is wearing glasses)

# LIBRARIES I'VE USED:
Here is a list of the libraries I've used in the project:
 1. os
 2. numpy
 3. cv2
 4. tqdm
 5. random
 6. tensorflow
 7. matplotlib

# DATA USED FOR TRAINING AND EVALUATING THE MODEL:
I have used the MeGlass dataset. The dataset is available here: https://github.com/cleardusk/MeGlass

# DESCRIPTION OF FILES AND FOLDERS:
The project contains 3 main python files:
 1. gather_data.py: This file contains code for rearranging the image data into a form that can be processed by the neural network. It transforms all the images in our dataset      into array form and also assigns a label to each data (0 = no glassses, 1 = wearing glasses)
 2. create_and_evaluate_neural_network.py: This file contains code that creates the neural network and trains the neural network on our training data. After training, it also evaluates our trained neural network on our test data.
 3. predictions.py: This file contains code that uses our trained neural network to predict whether the person in an image is wearing glasses data. It uses images from the 'predciton_data' folder inside the 'glasses_image_data'. So if you want to upload an image to be predicted by the network, please put your iamge inside the prediction_data folder.
 
The project also contains many folder:
 1. glasses_image_data: This folder contains our entire dataset in .jpg form. The 'training_data' subfolder contains the images we've used in training our model. Similarly the 'testing_data' and 'prediciton_data' subfolder contains images that we've used for testing and predicting.
 2. glasses_array_data: This folder also contains our entire dataset except that the dataset has been converted to array form (.npy) and can be used by our neural network. It also contains 'training_data_label.npy' and 'testing_data_label.npy' which contains the labels for the arrays (whether the array represents an image in whcih the person is wearing glasses or not).
 3. glasse_neural_network: This folder contains a model of our neural_network that I trained preivously. It has a an accuracy of around 92% on our test data.
 
 # RESULTS AND ISSUES WITH OUR NEURAL NETWORK
 The neural network has an accuracy rating of 92% on our testing data. While this is a very high percentage, the model does run into some errors which I have listed below:
 1. When training the neural network, the accuracy of the model randomly falls down and gets stuck at 0.5140 and does not go back up. This occurence seems essentially random as sometimes the accuracy goes as high as 0.93 but at times falls down at 0.5140. To solve this I repeated the training multiple times to get a better accuracy.
 2. I tested the neural network with my images of myself to see how accurate it was. After inputting around 50 images, I came to the conclusion that the neural network will only detect if you're wearing glasses if the glasses are at the centre of the image. Otherwise it will say you are not wearing glasses. This is probably because of the dataset I've used and the fact that all the people wearing glasses are usually in the middle of the image.
