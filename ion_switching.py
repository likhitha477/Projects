# Importing Libraries

import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from lightgbm import LGBMClassifier

# Initializing variables

split = 0.7
STD = 0.01

# Importing train data. Using cleaned data obtained from this link: https://www.kaggle.com/cdeotte/data-without-drift
data = pd.read_csv('train_clean.csv')

# Establishing rolling features function with variable window sizes per feature. These were chosen post-model fitting based on a trial basis and comparing with feature importances obtained below
def add_rolling_features(data):
    data['r_mean_3000'] = data['signal'].rolling(3000).mean()
    data['r_stdev_5'] = data['signal'].rolling(5).std()
    data['r_ema_20'] = data['signal'].ewm(span=20, adjust=False).mean()
    data['r_ema_5'] = data['signal'].ewm(span=5, adjust=False).mean()
    data['signal'] = data['signal'] + np.random.normal(0,STD,size=len(data['signal']))
    return data

# Adding rolling features to dataset
data = add_rolling_features(data)

# Plotting signal vs. time to get a visual representation of the data
fig = plt.figure(figsize = (25, 10), dpi = 50, facecolor ='w', edgecolor ='k')
plt.plot(data['time'], data['signal'])

# Seperating data into features and labels based on headers and converting into an np.array
labels = np.array(data['open_channels'])
feature_cols = ['signal',
                'r_mean_3000',
                'r_stdev_5',
                'r_ema_20',
                'r_ema_5',
               ]

features= data[feature_cols]
feature_list = list(data.columns)
features = np.array(features)

# Using test_train_split to using an 80:20 split between train and test respectively
train_features, test_features, train_labels, test_labels = \
train_test_split(features, labels, test_size = 0.2, random_state = 42)

# Scaling data to remove any potential bias when fitting
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

# Using Light Gradient Boosting Model classification with a maximum tree depth of 4
model = LGBMClassifier(max_depth = 4)

# Fitting the model
model.fit(train_features, train_labels)

# Extract feature importances
fi = pd.DataFrame({'feature': list(feature_cols),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)

fi.head(20)

# Predicting the Test set results
predictions = model.predict(test_features)

# Making the Confusion Matrix
pd.crosstab(test_labels, predictions, rownames = ['Actual'], colnames = ['Predicted'])

# Accuracy Score
accuracy_score(test_labels, predictions)
