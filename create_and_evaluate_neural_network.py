import numpy as np
import tensorflow as tf
from tensorflow import keras

#---------------LOADING OUR DATA-----------------#
train_data = np.load('glasses_array_data/train_data.npy')
train_data_label = np.load('glasses_array_data/train_data_label.npy')
test_data = np.load('glasses_array_data/test_data.npy')
test_data_label = np.load('glasses_array_data/test_data_label.npy')
#-------------------------------------------------#


#------CREATING OUR MODEL------#
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#------------------------------#


#--------------TRAINING AND TESTING OUR MODEL------------------#
model.fit(train_data, train_data_label, epochs=10) #this line is used to train our model

test_loss, test_acc = model.evaluate(test_data, test_data_label, verbose=2) #this line is used to test our model on our testing data

print('\nTest accuracy:', test_acc)
#----------------------------------------------------------------#


#model.save('saved_model/my_model')
#Use the above line to save your trained model to any directory of your choice
