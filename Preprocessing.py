import pandas as pd
from keras.utils import to_categorical

features = pd.read_csv("data/features.csv")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
stores = pd.read_csv("data/stores.csv")

merged_train = pd.merge(train, stores)
merged_train = pd.merge(merged_train , features)
merged_train = merged_train.fillna(0.)
merged_train = merged_train.drop('Date', 1)
merged_train = pd.get_dummies(merged_train, columns=['Store', 'Dept', 'Type'])
merged_train.to_csv("data/merged_train.csv", index =False)

merged_test = pd.merge(test, stores)
merged_test = pd.merge(merged_test, features)
merged_test = merged_test.fillna(0.)
merged_test = merged_test.drop('Date', 1)
merged_test = pd.get_dummies(merged_test, columns=['Store', 'Dept', 'Type'])
merged_test.to_csv("data/merged_train.csv", index =False)
