import pandas as pd
from keras.utils import to_categorical

def add_month_col(row):
      return row['Date'][0]

features = pd.read_csv("data/features.csv")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
stores = pd.read_csv("data/stores.csv")

merged_train = pd.merge(train, stores)
merged_train = pd.merge(merged_train , features)
merged_train['Month'] = merged_train.apply(add_month_col, axis=1)
merged_train[['CPI','Unemployment']] = merged_train[['CPI','Unemployment']].fillna(method='ffill')
merged_train = merged_train.fillna(0.)
merged_train = merged_train.drop('Date', 1)
merged_train = pd.get_dummies(merged_train, columns=['Store', 'Dept', 'Type', 'Month'])
merged_train.to_csv("data/merged_train.csv", index =False)

merged_test = test.merge(test.merge(stores, how='left', sort=False))
merged_test = merged_test.merge(merged_test.merge(features, how='left', sort=False))
merged_test['Month'] = merged_test.apply(add_month_col, axis=1)
merged_test[['CPI','Unemployment']] = merged_test[['CPI','Unemployment']].fillna(method='ffill')
merged_test = merged_test.fillna(0.)
merged_test['Month'] = merged_test.apply(add_month_col, axis=1)
merged_test = merged_test.drop('Date', 1)
merged_test = pd.get_dummies(merged_test, columns=['Store', 'Dept', 'Type', 'Month'])
merged_test.to_csv("data/merged_test.csv", index =False)

