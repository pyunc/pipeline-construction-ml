import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error


class Pipeline:
    
    ''' When we call the FeaturePreprocessor for the first time
    we initialise it with the data set we use to train the model,
    plus the different groups of variables to which we wish to apply
    the different engineering procedures'''
    
    
    def __init__(self,
                target, 
                categorical_encode,
                random_state,
                test_size = 0.2,
                percentage = 0.01):
        
        # data sets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # models
        self.LabelEncoder = LabelEncoder()
        self.model = LinearRegression()
        
        # groups of variables to engineer
        self.target = target
        self.categorical_encode = categorical_encode
        
        # more parameters
        self.test_size = test_size
        self.random_state = random_state
    
    # ======= functions to transform data =================
    
    def encode_categorical_variables(self, df):
        
        ''' replace categories by numbers in categorical variables'''

        df = df.copy()
            
        for variable in self.categorical_encode:
            self.LabelEncoder.fit(df[variable]) 
            df[variable] = self.LabelEncoder.transform(df[variable])
        
        return df
    
    # ====   master function that orchestrates feature engineering =====

    def fit(self, data):
        '''pipeline to learn parameters from data, fit the scaler and lasso'''
        
        # setarate data sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data, data[self.target],
                test_size = self.test_size,
                random_state = self.random_state)
        
        
        # encode categorical variables
        self.X_train = self.encode_categorical_variables(self.X_train)
        self.X_test = self.encode_categorical_variables(self.X_test)          
        
        
        # train model
        self.model.fit(self.X_train, self.y_train)
        
        return self    
            
    def transform(self, data):
        ''' transforms the raw data into engineered features'''
        
        data = data.copy()
        
        # encode categorical variables
        data = self.encode_categorical_variables(data)
            
        return data
    
    def predict(self, data):
        ''' obtain predictions'''
        
        data = self.transform(data)
        
        predictions = self.model.predict(data)
        
        return predictions
    
    def evaluate_model(self):
        '''evaluates trained model on train and test sets'''
        
        pred = self.model.predict(self.X_train)
        
        print('train r2: {}'.format((r2_score(self.y_train, pred))))
        
        
        pred = self.model.predict(self.X_test)
        
        print('test r2: {}'.format((r2_score(self.y_test, pred))))            

