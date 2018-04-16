import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

test_x = pd.read_csv("data/merged_test.csv")
train = pd.read_csv('data/merged_train.csv')

train_x = np.array(train.drop('Weekly_Sales', 1))
train_y = np.array(train['Weekly_Sales'])

from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=15)

from sklearn.externals import joblib

rf = RandomForestRegressor()
#rf = joblib.load('models/model_random_forest.pkl')
rf.fit(train_x, train_y)
print(rf.score(valid_x, valid_y))

joblib.dump(rf, 'model_random_forest.pkl')
print("Target", valid_y[:10])
print("Prediction", rf.predict(valid_x)[:10])

test_result = rf.predict(test_x)
submission = pd.read_csv("data/sampleSubmission.csv")
submission['Weekly_Sales'] = np.array(test_result)

submission.to_csv("data/submission.csv", index =False)
