import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

model = keras.models.load_model('glasses_neural_network')

# test_data = np.load('glasses_array_data/test_data.npy')
# test_data_label = np.load('glasses_array_data/test_data_label.npy')
# test_loss, test_acc = model.evaluate(test_data, test_data_label, verbose=2)
# print('\nTest accuracy:', test_acc)
# Uncomment the lines above if you want to evaluate your loaded model

prediction_dir = r'glasses_image_data\prediction_data'
img_data = []
for img in tqdm(os.listdir(prediction_dir)):
    path = os.path.join(prediction_dir, img)
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (100, 100))
    img_data.append(img)

np.save('img_data.npy', img_data)
test = np.load('img_data.npy')

# All the code below is used to predict the value of an image. To change what image you are viewing, change the value
# of 'm' (0 is the first image) The code below will also display the image you are predicting to see if the
# prediction is accurate.

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
m = 2

predictions = probability_model.predict(test)
print(predictions[m])
print('The output is:')
print(np.argmax(predictions[m]))

plt.figure()
plt.imshow(test[m])
plt.colorbar()
plt.grid(False)
plt.show()
