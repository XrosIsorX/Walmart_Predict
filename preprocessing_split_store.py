import pandas as pd

month = ['Month_1_1', 'Month_1_2', 'Month_2_1', 'Month_2_2', 'Month_3_1', 'Month_3_2', 'Month_4_1', 'Month_4_2', 'Month_5_1', 'Month_5_2', 'Month_6_1', 'Month_6_2', 'Month_7_1', 'Month_7_2', 'Month_8_1', 'Month_8_2', 'Month_9_1', 'Month_9_2', 'Month_10_1', 'Month_10_2', 'Month_11_1', 'Month_11_2', 'Month_12_1', 'Month_12_2']

def add_month_col(row):
      if int(row['Date'].split('/')[1]) <= 15:
            half = '1'
      else:
            half = '2'
      return row['Date'].split('/')[0] + '_' + half

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
merged_train = pd.get_dummies(merged_train, columns=['Dept', 'Type', 'Month'])
split_store = {}
for s in merged_train['Store'].unique():
      split_store[s] = merged_train[merged_train['Store'] == s]
      split_store[s] = split_store[s].drop('Store', 1)
      split_store[s].to_csv("data/merged_train_split_store_" + str(s) + ".csv", index =False)

merged_test = test.merge(test.merge(stores, how='left', sort=False))
merged_test = merged_test.merge(merged_test.merge(features, how='left', sort=False))
merged_test['Month'] = merged_test.apply(add_month_col, axis=1)
merged_test[['CPI','Unemployment']] = merged_test[['CPI','Unemployment']].fillna(method='ffill')
merged_test = merged_test.fillna(0.)
merged_test = merged_test.drop('Date', 1)
merged_test = pd.get_dummies(merged_test, columns=['Dept', 'Type', 'Month'])
for m in month:
      if m not in merged_test.index:
            merged_test[m] = 0
split_store = {}
for s in merged_test['Store'].unique():
      split_store[s] = merged_test[merged_test['Store'] == s]
      split_store[s] = split_store[s].drop('Store', 1)
      split_store[s].to_csv("data/merged_test_split_store_" + str(s) + ".csv", index =False)

