#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install geopy
# !pip install feature_engine


# In[1]:


# import packages
import pprint
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from datetime import datetime, date

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

get_ipython().run_line_magic('matplotlib', 'inline')

pp = pprint.PrettyPrinter()
from geopy import distance


# In[2]:


# load data
train_data = pd.read_csv('https://raw.githubusercontent.com/Team-8-JHB-RegressionPredict/regression-predict-api-template/master/predict%20deliverable/data/Train.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/Team-8-JHB-RegressionPredict/regression-predict-api-template/master/predict%20deliverable/data/Test.csv')
riders_data = pd.read_csv('https://raw.githubusercontent.com/Team-8-JHB-RegressionPredict/regression-predict-api-template/master/predict%20deliverable/data/Riders.csv')


# In[3]:


predictors = ['Distance (KM)',
 'rider_id_bins',
 'rider_speed',
 'No_of_Ratings',
 'No_Of_Orders',
 'Time from Arrival at Pickup to Actual Pickup',
 'Time from Confirmation to Arrival at Pickup',
 'Pickup_hour',
 'add_hour_minute',
 'sin_pickup_time'
]
target = 'Time from Pickup to Arrival'
pp.pprint(predictors + [target])


# In[4]:


# merge the train & test data with the riders data.
def mergeRiders(df):
    df = pd.merge(left = df, right = riders_data, how = 'left')
    return df

train_data = mergeRiders(train_data)
test_data = mergeRiders(test_data)


# #### RIDER ID

# In[5]:


from feature_engine.categorical_encoders import OrdinalCategoricalEncoder
X = train_data.loc[:, ['Rider Id']]
y = train_data.loc[:, 'Time from Pickup to Arrival']

enc = OrdinalCategoricalEncoder(encoding_method='ordered')
enc.fit(X, y)
train_data['rider_id_enc'] = enc.transform(X)['Rider Id']


# In[6]:


from feature_engine.discretisers import DecisionTreeDiscretiser
DT_disc = DecisionTreeDiscretiser(
                        cv=10,
                        scoring='neg_root_mean_squared_error',
                        variables=['rider_id_enc'],
                        regression=True,
                        param_grid={'max_depth': [1, 2, 3, 4, 5, 6, 10],
                                    'min_samples_leaf': [10, 4, 2, 1]},
                        random_state=1)
X_train = train_data.loc[:, ['rider_id_enc']]
y_train = train_data.loc[:, 'Time from Pickup to Arrival']
DT_disc.fit(X_train, y_train)
DT_discr = DT_disc.transform(X_train)
train_data['rider_id_bins'] = DT_discr['rider_id_enc']


# ### RIDER SPEED

# In[7]:


def getSpeedPerOrder(df):
    df = df.copy()
    pick_up_to_arr_minutes = df['Time from Pickup to Arrival'].div(60)
    df['Speed_per_order'] = df['Distance (KM)'].div(pick_up_to_arr_minutes)
    return df


# In[8]:


# apply the function to train_data
train_data = getSpeedPerOrder(df = train_data)

#generate summary stats
speed_summary_stats = train_data['Speed_per_order'].describe()
speed_summary_stats


# In[9]:


# find bounds
def findBounds(summary_stats, std = 2.5):
    upper_bound = summary_stats['std'] * std
    lower_bound = summary_stats['std'] * -std
    return lower_bound, upper_bound


# In[10]:


def findPercentile(x, i = 15):
    ith_percentile_lower_bound = np.percentile(x, i)
    return ith_percentile_lower_bound


# In[11]:


rider_avg_speed = train_data.groupby('Rider Id')['Speed_per_order'].mean()

rider_speed_data = pd.DataFrame(rider_avg_speed).reset_index().rename({'Speed_per_order':'rider_avg_speed'})
rider_speed_data.rename(columns = {'Speed_per_order':'rider_avg_speed'}, inplace = True
)
avg_speed_summary_stats = rider_speed_data['rider_avg_speed'].describe()
avg_speed_summary_stats 


# In[12]:


train_data = pd.merge(
    left = train_data,
    right = rider_speed_data,
    how='left',
    on = 'Rider Id'
)
train_data.head()


# In[13]:


lower, upper = findBounds(avg_speed_summary_stats)
lower, upper


# In[14]:


ith_percentile_lower_bound = findPercentile(train_data['rider_avg_speed'], i = 2.5)
ith_percentile_lower_bound


# In[15]:


high_avg_speed_riders = train_data[train_data['rider_avg_speed'] > upper]['Rider Id'].unique()
print(high_avg_speed_riders[:5])
print(len(high_avg_speed_riders))


# In[16]:


low_avg_speed_riders = train_data[train_data['rider_avg_speed'] < ith_percentile_lower_bound]['Rider Id'].unique()
print(low_avg_speed_riders[:5])
print(len(low_avg_speed_riders))


# In[17]:


def riderChar(x):
    if x in high_avg_speed_riders:
        value = 'fast'
    elif x in low_avg_speed_riders:
        value = 'slow'
    else:
        value = 'reasonable'
    return value

train_data['rider_speed'] = train_data['Rider Id'].map(riderChar)


# In[18]:


new_vars = ['rider_speed', 'rider_id_bins']
test_data = pd.merge(
    left = train_data.loc[:, new_vars + ['Rider Id']].drop_duplicates(),
    right = test_data,
    how = 'right',
    on = 'Rider Id'
)
test_data.head()


# In[19]:


test_data['rider_speed'].fillna(train_data['rider_speed'].mode()[0], inplace = True)
test_data['rider_id_bins'].fillna(train_data['rider_id_bins'].mode()[0], inplace = True)


# ### TIME VARIABLES

# In[20]:


month_day_vars = [
    'Placement - Day of Month', 
    'Confirmation - Day of Month', 
    'Arrival at Pickup - Day of Month', 
    'Pickup - Day of Month']
time_vars = [
    'Placement - Time',
    'Confirmation - Time',
    'Arrival at Pickup - Time',
    'Pickup - Time'
]


# In[21]:


def getTimeObjects(df):
    datetime_vars = list()
    df = df.copy()
    for month_col, time_col in zip(month_day_vars, time_vars):
        new_col_name = '{}'.format(time_col.split('-')[0].replace(' ', ''))
        datetime_vars.append(new_col_name)
        print(new_col_name)
        
        values = list()
        Dates = list()
        for row in df.index.values:
            value = '2020' + '-' + '1' + '-' + str(df[month_col][row]) 
            values.append(value)
            
            date_string = values[row]
            time_string = df[time_col][row]

            Datetime = pd.to_datetime(date_string + ' ' + str(time_string))
    #         print(Datetime)
            Dates.append(Datetime)
        df[new_col_name] = Dates
    return df, datetime_vars


# In[22]:


train_data, datetime_vars = getTimeObjects(train_data)
test_data, datetime_vars = getTimeObjects(test_data)
train_data.loc[:, datetime_vars].head()


# In[23]:


iter_dict = {
    'Time from Placement to Confirmation': ['Confirmation', 'Placement'],
    'Time from Confirmation to Arrival at Pickup': ['ArrivalatPickup', 'Confirmation'],
    'Time from Arrival at Pickup to Actual Pickup': ['Pickup', 'ArrivalatPickup'],
    'Time from Placement to Actual Pickup': ['Pickup', 'Placement'],
    'Time from Placement to Arrival at Pickup': ['ArrivalatPickup', 'Placement'],
    'Time from Confirmation to Actual Pickup': ['Pickup', 'Confirmation']
}
def getTimeDifferences(df, iter_dict):
    df = df.copy()
    numeric_time_vars_sub = list()
    for new_column, inputs in iter_dict.items():
        numeric_time_vars_sub.append(new_column)
        col1 = inputs[0]
        col2 = inputs[1]
        df[new_column] = df[col1] - df[col2]
        df[new_column] = df[new_column].map(lambda timedelt: timedelt.total_seconds())
    return df, numeric_time_vars_sub


# In[24]:


# apply function to both train and test data
train_data, numeric_time_vars_sub  = getTimeDifferences(train_data, iter_dict)
test_data, numeric_time_vars_sub  = getTimeDifferences(test_data, iter_dict)
train_data.loc[:, numeric_time_vars_sub].head()


# In[25]:


def getHourMinute(df):
    df = df.copy()
    hour_vars = list()
    minute_vars = list()
    for col in datetime_vars:
        hours = list()
        minutes = list()
        for order in df.index.values:
            hour = df.loc[:, col][order].hour
            minute = df.loc[:, col][order].minute
            hours.append(hour)
            minutes.append(minute)
    #     print(col)
    #     print(hours, minutes)
        new_column_name_hour = '{}_hour'.format(col)
        new_column_name_minute = '{}_minute'.format(col)
        hour_vars.append(new_column_name_hour)
        minute_vars.append(new_column_name_minute)
        
        df[new_column_name_hour] = hours
        df[new_column_name_minute] = minutes
    return df, hour_vars, minute_vars


# In[26]:


# apply function to both train and test data
train_data, hour_vars, minute_vars = getHourMinute(train_data)
test_data, hour_vars, minute_vars = getHourMinute(test_data)
train_data.loc[:, hour_vars + minute_vars].head()


# In[27]:


# relacing the outlier value with mean. Dos so fror the test data as well
ind_to_replace = train_data[train_data['Pickup_hour'] == 0].loc[:, 'Pickup_hour'].index
train_data.loc[ind_to_replace, 'Pickup_hour'] = np.mean(train_data['Pickup_hour'])
ind_to_replace = test_data[test_data['Pickup_hour'] == 0].loc[:, 'Pickup_hour'].index
test_data.loc[ind_to_replace, 'Pickup_hour'] = np.mean(train_data['Pickup_hour'])


# In[28]:


def generateInteractionHourMinute(df):
    df = df.copy()
    df['add_hour_minute'] = df['Pickup_minute'].div(60).add(df['Pickup_hour'])
    return df


# In[29]:


# apply function to both train and test data
train_data = generateInteractionHourMinute(train_data)
test_data = generateInteractionHourMinute(test_data)


# In[30]:


def getSecondsPastMidnight(x):
    date = x.date()
    twelve = pd.to_datetime(str(date) + ' ' + '00:00:00 AM')
    diff = x - twelve
    return diff.total_seconds()


# In[31]:


# applying the function to both train and test data
train_data['seconds_past_midnight'] = train_data['Pickup'].map(getSecondsPastMidnight)
test_data['seconds_past_midnight'] = test_data['Pickup'].map(getSecondsPastMidnight)


# In[32]:


def generateSinCosTime(df):
    seconds_in_day = 24*60*60
    df = df.copy()
    df['sin_pickup_time'] = np.sin(2*np.pi*df['seconds_past_midnight']/seconds_in_day)
    df['cos_pickup_time'] = np.cos(2*np.pi*df['seconds_past_midnight']/seconds_in_day)
    return df


# In[33]:


# apply function to both train and test data
train_data = generateSinCosTime(train_data)
test_data = generateSinCosTime(test_data)


# ### Preprocessing

# In[35]:


def getDummies(df):
    df = df.copy()
    df = pd.get_dummies(df, drop_first=True)
    return df


# In[38]:


X_train = getDummies(train_data.loc[:, predictors])
X_test = getDummies(test_data.loc[:, predictors])


# In[39]:


y_train = train_data.loc[:, target]


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)


# In[42]:


X_train.head()


# In[62]:


feature_vector = X_train.copy()

