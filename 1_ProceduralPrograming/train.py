import numpy as np
import preprocessing_functions as pf
import config
import warnings
warnings.simplefilter(action='ignore')

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(df = data, target = config.TARGET)

# encode categorical variables
for var in config.CATEGORICAL_ENCODE:
    pf.encode_categorical(X_train, var)

# train model and save
pf.train_model(X_train,
               y_train,
               config.OUTPUT_MODEL_PATH)

print('Finished training')