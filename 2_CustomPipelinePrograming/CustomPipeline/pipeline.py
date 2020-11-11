import pandas as pd
import numpy as np

from preprocessors import Pipeline
import config

pipeline = Pipeline(target =  config.TARGET,
                    categorical_encode = config.CATEGORICAL_ENCODE,
                    random_state = 42
                    )

if __name__ == '__main__':
    
    # load data set
    data = pd.read_csv(config.PATH_TO_DATASET)
    
    pipeline.fit(data)
    print('model performance')
    pipeline.evaluate_model()
    print()
    print('Some predictions:')
    preditions = pipeline.predict(data)
    print(preditions)