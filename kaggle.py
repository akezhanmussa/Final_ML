import os
import cv2 
import pandas as pd
import numpy as np
from test import give_label

def kaggle_test():
    path_csv = os.path.abspath("sample_submission.csv")
    path_train = os.path.abspath("static")
    train_file_name_class= pd.read_csv(path_csv, header=0)
    train_file_names=train_file_name_class['id']
    X_test = []
    for file_name in train_file_names.values:
        img=cv2.imread(path_train + '/' +  file_name)
        img1 = cv2.resize(img, (224,224))
        img1=img1/255
        X_test.append(img1)
    X_test = np.array(X_test)
    
    return X_test 

def generate_kaggle_csv():
    X_test = kaggle_test()
    Y_test = give_label(X_test)
    path_csv = os.path.abspath("predictions.csv")
    prep_file_name_class = pd.read_csv(path_csv, header = 0)
    prep_file_names = prep_file_name_class['has_cactus']
  
    for id, _ in enumerate(prep_file_names.values):
        prep_file_names.values[id] = 1 if Y_test[id][0] > 0.5 else 0

    prep_file_name_class.to_csv(os.path.abspath('results2.csv'))




