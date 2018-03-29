import numpy as np
import matplotlib as plt
import pandas as pd

data = pd.read_csv("data/merged_train.csv")
train_y = data['Weekly_Sales']
train_x = data.drop('Weekly_Sales', 1)


from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=14)

import keras 
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 32
epochs = 150

#Add network
model = Sequential()
model.add(Dense(100 ,input_shape=(train_x.shape[1],), activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()

#Train network
model_train = model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_y))
model.save("model.h5py")

accuracy = model_train.history['acc']
val_accuracy = model_train.history['val_acc']
loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
epochs = range(len(accuracy))
plt.subplot(epochs, accuracy, 'bo', label='Training accuracy')
plt.subplot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()