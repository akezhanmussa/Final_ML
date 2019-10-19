import cv2 
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def prep_data():
    path_csv = os.path.abspath("train.csv")
    path_train = os.path.abspath("train")
    train_file_name_class= pd.read_csv(path_csv, header=0)
    train_file_names=train_file_name_class['id']
    Y=train_file_name_class['has_cactus']
    X = []
    trainY= []
    i=-1
    counter_one = 0
    
    for file_name in train_file_names.values:
        i = i+1
        if(i==13000):
            break  
        if (Y[i] == 1):
                if (counter_one >= 6000):
                        continue 
                counter_one += 1
        img=cv2.imread(path_train + '/' +  file_name)
        img1 = cv2.resize(img, (224,224))
        img1=img1/255
        X.append(img1)
        trainY.append(Y[i])
         
    
    X = np.array(X)
    trainY = np.array(trainY)
    trainY = to_categorical(trainY)
    X_train, X_test, Y_train, Y_test = train_test_split(X, trainY, test_size=0.33, shuffle = True)

    return X_train, Y_train, X_test, Y_test




    