import pandas as pd

features = pd.read_csv("data/features.csv")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
stores = pd.read_csv("data/stores.csv")

merged_train = pd.merge(train, stores)
merged_train = pd.merge(merged_train , features)

merged_test = pd.merge(test, stores)
merged_test = pd.merge(merged_train, features)

merged_train.to_csv("data/merged_train.csv")
merged_test.to_csv("data/merged_test.csv")
