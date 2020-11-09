import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    
    print('load is done')
    return pd.read_csv(df_path)

def divide_train_test(df, target):
    
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target,axis=1),
                                                        df[target],
                                                        test_size=0.2,
                                                        random_state=42)
    print('split is done')                                                        
    return X_train, X_test, y_train, y_test


def encode_categorical(df, var):
    # replaces strings by numbers using mappings dictionary
    le = LabelEncoder()

    le.fit(df[var]) 
    df[var] = le.transform(df[var])
    return df

def train_model(df, target, output_path):
    # initialise the model
    lin_model = LinearRegression()
    
    # train the model
    lin_model.fit(df, target)
    
    # save the model
    joblib.dump(lin_model, output_path)
    
    return None

def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)

