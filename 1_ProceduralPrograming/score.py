import preprocessing_functions as pf
from sklearn.metrics import r2_score,mean_squared_error
import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    
    # encode variables
    # encode categorical variables
    for var in config.CATEGORICAL_ENCODE:
        pf.encode_categorical(data, var)

    
    # make predictions
    predictions = pf.predict(data, config.OUTPUT_MODEL_PATH)

    print('predicted')
    
    return predictions

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
    
    from math import sqrt
    import numpy as np
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)

    print('test r2:{}'.format(r2_score(pred,y_test)))

    print('test mse:{}'.format(mean_squared_error(pred,y_test)))