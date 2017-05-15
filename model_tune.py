import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import load_model
import os




import sklearn
from random import shuffle

model = load_model('./Working/model-Base-Epoch03.h5'
                   )

samples = []
def grab_csv(direc):
    with open("C:/Users/M0J0/Desktop/data/"+direc+"/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

#grab_csv("New_track")
#grab_csv("Stock")
grab_csv("Last_Turn")
grab_csv("Avoidance")


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

steering_correction = .2
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path = "C:/Users/M0J0/Desktop/data/All_Pics/IMG/"

                #Uncomment at your own hazzard.
                #print(path+batch_sample[0].split('\\')[-1])
                center_image = cv2.imread(path+batch_sample[0].split('\\')[-1])
                left_image = cv2.imread(path+batch_sample[1].split('\\')[-1])
                right_image = cv2.imread(path+batch_sample[2].split('\\')[-1])

                center_angle = float(batch_sample[3])
                left_angle = center_angle + steering_correction
                right_angle = center_angle - steering_correction

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)

                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

                #append flipped images
                images.append(cv2.flip(center_image,1))
                images.append(cv2.flip(left_image,1))
                images.append(cv2.flip(right_image,1))

                #append flipped angles
                angles.append(center_angle*-1)
                angles.append(left_angle*-1)
                angles.append(right_angle*-1)


            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

print("Data prepared to Load")


#Define Callbacks for our model; create a checkpoint after each epoch, and an early stop for when the validation loss becomes stagnant
model_checkpoint = ModelCheckpoint('./checkpoint/Epoch{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
#reduce_LROP = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)


#Train our Model.
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=20, callbacks = [ model_checkpoint, early_stop])


model.save('Updated_model.h5')