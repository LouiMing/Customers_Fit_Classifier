import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

DATA_PATH = "./test.json"

## for TA's test
## you need to modify the class name to your student id.
## you also need to implement the predict function, which reads the .json file,
## calls your trained model and returns predict results as an ndarray

class PB20000000():
    def predict(self, data_path): 
        # a dummy system
        with open(data_path, "r") as f:
            data_list = json.load(f)
        size = len(data_list)
        pred = np.random.randint(1, 4, size)
        
        return pred


## for local validation
if __name__ == '__main__':
    with open(DATA_PATH, "r") as f:
        test_data_list = json.load(f)
    true = np.array([int(data["fit"]) for data in test_data_list])
    bot = PB20000000()
    pred = bot.predict(DATA_PATH)

    macro_f1 = f1_score(y_true=true, y_pred=pred, average="macro")
    print(macro_f1)