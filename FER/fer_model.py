import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers


"""#**Building a two class classification model**

### Helper function to analyze number of images in each folder
"""

def count_exp(path, name):
    d = {}
    for emotion in os.listdir(path):
        direc = path + emotion
        d[emotion] = len(os.listdir(direc))
    df = pd.DataFrame(d, index=[name])
    return df

train_path = '/content/drive/MyDrive/facial_expression/data/train/'
test_path = '/content/drive/MyDrive/facial_expression/data/test/'

train_img_count = count_exp(train_path, 'train')
test_img_count = count_exp(test_path, 'test')

print(train_img_count)
print(test_img_count)

"""### Visualizing Images in Train and Test directories"""

plt.figure(figsize=(14,22))
i = 1
for e in os.listdir(train_path):
    img = load_img((train_path + e +'/'+ os.listdir(train_path + e)[5]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(e)
    plt.axis('off')
    i += 1
plt.show()

plt.figure(figsize=(14,22))
i = 1
for e in os.listdir(test_path):
    img = load_img((test_path + e +'/'+ os.listdir(test_path + e)[5]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(e)
    plt.axis('off')
    i += 1
plt.show()

train_data_gen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

train_set = train_data_gen.flow_from_directory(train_path, 
                                                 batch_size=64, 
                                                 target_size=(48,48), 
                                                 shuffle=True, 
                                                 color_mode='grayscale', 
                                                 class_mode='categorical')

test_data_gen = ImageDataGenerator(rescale=1./255)

test_set = test_data_gen.flow_from_directory(test_path, 
                                                 batch_size=64, 
                                                 target_size=(48,48), 
                                                 shuffle=True, 
                                                 color_mode='grayscale', 
                                                 class_mode='categorical')

train_set.class_indices

test_set.class_indices

def model_cnn(input_size, classes):
    model = tf.keras.models.Sequential()   

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape =input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classes, activation='softmax'))

    #Compliling the model
    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

row, col = 48, 48
classes = 2
FER_model = model_cnn((row,col,1), classes)

hist = FER_model.fit(x=train_set,
                 validation_data=test_set,
                 epochs=20)

"""### Visualizing Model Accuracy"""

plt.figure(figsize=(14,5))
plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

train_loss, train_accu = FER_model.evaluate(train_set)
test_loss, test_accu = FER_model.evaluate(test_set)
print("Train accuracy = {:.2f} , Validation accuracy = {:.2f}".format(train_accu*100, test_accu*100))

FER_model.save_weights('fer_model_best.h5')
