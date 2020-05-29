"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, cross_val_score, validation_curve, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import json

#test = pd.read_csv('utils/data/test_data.csv')
#riders = pd.read_csv('utils/data/riders.csv')
#test = test.merge(riders, how='left', on='Rider Id')

def remove_outlier(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    # interquartile range
    iqr = q3 - q1

    low_bound  = q1 - 1.5 * iqr
    high_bound = q3 + 1.5 * iqr
    df_out = df.loc[(df[column_name] > low_bound) & (df[column_name] < high_bound)]
    return df_out

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """

    if(type(data) != pd.DataFrame):
        # Convert the json string to a python dictionary object
        feature_vector_dict = json.loads(data)

        # Load the dictionary as a Pandas DataFrame.
        feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

        predict_vector = feature_vector_df.copy()
        return predict_vector
    else:
        predict_vector = data
        print(1)
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    predict_vector = predict_vector.drop(['Arrival at Pickup - Day of Month', 'Arrival at Pickup - Weekday (Mo = 1)'], axis=1)
    predict_vector = predict_vector.drop(['Precipitation in millimeters'], axis=1)
    predict_vector = remove_outlier(predict_vector, 'Temperature')
    predict_vector['Temperature'].fillna((predict_vector['Temperature'].mean()), inplace=True)
    predict_vector = predict_vector.drop(['Vehicle Type'], axis=1)
    predict_vector = predict_vector.drop(
        ['Placement - Day of Month', 'Placement - Weekday (Mo = 1)', 'Placement - Time', 'Confirmation - Day of Month',
         'Confirmation - Weekday (Mo = 1)', 'Confirmation - Time'], axis=1)

    test_am_pm_split = [x.split(" ")[-1] for x in predict_vector['Pickup - Time']]
    predict_vector['Pickup - Time'] = test_am_pm_split

    train_am_pm_split = [x.split(" ")[-1] for x in predict_vector['Arrival at Pickup - Time']]
    predict_vector['Arrival at Pickup - Time'] = train_am_pm_split
    #print(predict_vector)
    predict_vector = predict_vector.drop(['Arrival at Pickup - Time'], axis=1)

    #adding weekday cat variables
    day_dict = {1: 'Mon', 2: 'Tues', 3: 'Wed', 4: 'Thurs', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
    predict_vector = predict_vector.replace({'Pickup - Weekday (Mo = 1)': day_dict})

    #creating dummmy variables
    predict_vector = pd.get_dummies(predict_vector, columns=['Pickup - Weekday (Mo = 1)'])
    predict_vector = pd.get_dummies(predict_vector, columns=['Personal or Business'])
    predict_vector = pd.get_dummies(predict_vector, columns=['Platform Type'])
    predict_vector = pd.get_dummies(predict_vector, columns=['Pickup - Time'])
    predict_vector = predict_vector.drop(
        ['Pickup - Weekday (Mo = 1)_Sat', 'Platform Type_4', 'Pickup - Time_PM', 'Personal or Business_Personal'],
        axis=1)
    predict_vector = predict_vector.drop(['Order No', 'User Id', 'Rider Id'], axis=1)

    #Scaling variables
    std = StandardScaler()
    predict_vector = pd.DataFrame(std.fit_transform(predict_vector), columns=predict_vector.columns, index=predict_vector.index)
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return [round(i) for i in prediction.tolist()]
