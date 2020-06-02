# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, date

def preProc(feature_vector_df):
    rider_features = pd.read_csv('https://raw.githubusercontent.com/Team-8-JHB-RegressionPredict/regression-predict-api-template/master/predict%20deliverable/data/rider_features.csv')




#     feature_vector_df = pd.DataFrame(['Order_No_21660', 'User_Id_1329', 'Bike', 3, 'Business', 31, 5, '12:16:49 PM', 31, 5, '12:22:48 PM', 31, 5, '12:23:47 PM',
#                          31, 5, '12:38:24 PM', 4, 21.8, np.nan, -1.2795183, 36.8238089, -1.273056, 36.811298, 'Rider_Id_812', 4402, 1090, 14.3, 1301])
#     feature_vector_df = feature_vector_df.T
#     feature_vector_df.columns = ['Order No', 'User Id', 'Vehicle Type', 'Platform Type', 'Personal or Business',
#                     'Placement - Day of Month',
#                     'Placement - Weekday (Mo = 1)',
#                     'Placement - Time',
#                     'Confirmation - Day of Month',
#                     'Confirmation - Weekday (Mo = 1)',
#                     'Confirmation - Time',
#                     'Arrival at Pickup - Day of Month',
#                     'Arrival at Pickup - Weekday (Mo = 1)',
#                     'Arrival at Pickup - Time',
#                     'Pickup - Day of Month',
#                     'Pickup - Weekday (Mo = 1)',
#                     'Pickup - Time',
#                     'Distance (KM)',
#                     'Temperature',
#                     'Precipitation in millimeters',
#                     'Pickup Lat',
#                     'Pickup Long',
#                     'Destination Lat',
#                     'Destination Long',
#                     'Rider Id',
#                     'No_Of_Orders',
#                     'Age',
#                     'Average_Rating',
#                     'No_of_Ratings']




    feature_vector_df = pd.merge(feature_vector_df, rider_features, how='left', on='Rider Id')
    feature_vector_df['rider_speed_reasonable'] = rider_features['rider_speed_reasonable'].mode()[0]
    feature_vector_df['rider_speed_slow'] = rider_features['rider_speed_slow'].mode()[0]
    feature_vector_df['rider_id_bins'] = rider_features['rider_id_bins'].mode()[0]


    # In[7]:


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


    # In[9]:


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
                Dates.append(Datetime)
            df[new_col_name] = Dates
        return df, datetime_vars
    feature_vector_df, datetime_vars = getTimeObjects(feature_vector_df)


    # In[10]:


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
            df[new_column] = df[new_column].map(
                lambda timedelt: timedelt.total_seconds())
        return df, numeric_time_vars_sub

    feature_vector_df, numeric_time_vars_sub = getTimeDifferences(feature_vector_df, iter_dict)


    # In[11]:


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
            new_column_name_hour = '{}_hour'.format(col)
            new_column_name_minute = '{}_minute'.format(col)
            hour_vars.append(new_column_name_hour)
            minute_vars.append(new_column_name_minute)

            df[new_column_name_hour] = hours
            df[new_column_name_minute] = minutes
        return df, hour_vars, minute_vars
    feature_vector_df, hour_vars, minute_vars = getHourMinute(feature_vector_df)


    # In[13]:


    def generateInteractionHourMinute(df):
        df = df.copy()
        df['add_hour_minute'] = df['Pickup_minute'].div(60).add(df['Pickup_hour'])
        return df
    feature_vector_df = generateInteractionHourMinute(feature_vector_df)


    # In[14]:


    def getSecondsPastMidnight(x):
        date = x.date()
        twelve = pd.to_datetime(str(date) + ' ' + '00:00:00 AM')
        diff = x - twelve
        return diff.total_seconds()

    feature_vector_df['seconds_past_midnight'] = feature_vector_df['Pickup'].map(getSecondsPastMidnight)


    # In[15]:


    def generateSinCosTime(df):
        seconds_in_day = 24*60*60
        df = df.copy()
        df['sin_pickup_time'] = np.sin(2*np.pi*df['seconds_past_midnight']/seconds_in_day)
        df['cos_pickup_time'] = np.cos(2*np.pi*df['seconds_past_midnight']/seconds_in_day)
        return df

    feature_vector_df = generateSinCosTime(feature_vector_df)


    # In[16]:




    predictors = ['Distance (KM)',
                  'No_of_Ratings',
                  'No_Of_Orders',
                  'rider_id_bins',
                  'rider_speed_slow',
                  'rider_speed_reasonable',
                  'Time from Arrival at Pickup to Actual Pickup',
                  'Time from Confirmation to Arrival at Pickup',
                  'Pickup_hour',
                  'add_hour_minute',
                  'sin_pickup_time'
                  ]

    predictor_vector = feature_vector_df.loc[:, predictors]
    return predictor_vector

