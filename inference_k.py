import keras
import numpy as np
import os
from PIL import Image
from keras.models import Model, load_model

print("Loading Model...")
model = load_model("flower_keras.h5")
print("Model Loaded")

path = "./test_dataset/"
test_imgs_array = []
test_labels = []
print("Loading Test Dataset")

for root, dirs, files in os.walk(path):
    for d in dirs:
        for r,_,fs in os.walk(path+str(d)):
            for f in fs:
                img = Image.open(path+str(d)+"/"+f)
                img = img.resize((150,150))
                img = np.asarray(img)
                img = img/255
                test_imgs_array.append(img)
                test_labels.append(int(d))


print("Test Dataset Loaded\n")

print("Preprocessing  Dataset\n")

test_imgs_array = np.array(test_imgs_array)


test_labels = np.array(test_labels)
from keras.utils.np_utils import to_categorical
test_labels_oneHot = to_categorical(test_labels)

from sklearn.utils import shuffle
test_imgs_array, test_labels_oneHot = shuffle(test_imgs_array, test_labels_oneHot)

print("Preprocessing completed\n")


print("Predicting...\n")

pred = model.evaluate(test_imgs_array,test_labels_oneHot)

print("Loss:- ", pred[0])
print("Accuracy:- ", pred[1]*100,"%")
