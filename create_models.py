import numpy as np
import matplotlib as plt
import pandas as pd

data = pd.read_csv("data/merged_train.csv")
train_y = data['Weekly_Sales']
train_x = data.drop('Weekly_Sales', 1)


from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=14)

import keras 
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 10

#Add network
model = Sequential()
model.add(Dense(100 ,input_shape=(140,), activation='linear'))
model.add(LeakyReLU(alpha=0.0))
model.add(Dense(50, activation='linear'))
model.add(LeakyReLU(alpha=0.0))
model.add(Dense(10, activation='linear'))
model.add(LeakyReLU(alpha=0.0))
model.add(Dense(1, activation='linear'))

model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(lr=0.01),metrics=['accuracy'])

model.summary()

#Train network
model_train = model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_y))
model.save("model.h5py")