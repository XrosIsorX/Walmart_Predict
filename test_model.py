import numpy as np
import matplotlib as plt
import pandas as pd

train_x = pd.read_csv("data/merged_test.csv")

# data = pd.read_csv("data/merged_train.csv")
# train_y = data['Weekly_Sales']
# train_x = data.drop('Weekly_Sales', 1)

from keras.models import load_model

model = load_model("model.h5py")

print("test data : ")
print(train_x[:10])

result = model.predict(train_x)
print("result : ")
print(result[:10])

submission = pd.read_csv("data/sampleSubmission.csv")
submission['Weekly_Sales'] = np.array(result)

submission.to_csv("data/submission.csv", index =False)

