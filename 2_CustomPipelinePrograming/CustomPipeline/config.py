
# ====   PATHS ===================

PATH_TO_DATASET = "/home/pauloyun/Documentos/model-Udemy/DeploymentMlModels/medical-cost/insurance.csv"
                     
# variable groups for engineering steps
TARGET = 'charges'

# variables to transform
NUMERICAL_VAR = ['age','bmi','children']

# variables to encode
CATEGORICAL_ENCODE = ['sex', 'smoker', 'region']

# selected features for training
FEATURES = ['sex', 'smoker', 'region', 'age','bmi','children']
