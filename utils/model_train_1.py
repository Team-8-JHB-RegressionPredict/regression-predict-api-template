"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Team_8_JHB

    Description: This script is responsible for training a simple linear
    regression model which is used within the API.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')

train_data = train.select_dtypes(include=['int64']).copy()

y_train = train_data[['Time from Pickup to Arrival']]
X_train = train_data.drop(['Arrival at Destination - Day of Month', 'Arrival at Destination - Weekday (Mo = 1)', 'Time from Pickup to Arrival'], axis =1)

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
