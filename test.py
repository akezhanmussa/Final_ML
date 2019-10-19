
from models import AlexNet
from data import prep_data
import os
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json

is_loaded = False

def create_model():
    alex = AlexNet()
    alex.create_model()
    alex_json = alex.model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(alex_json)


def test():
    print("---HERE-----")
    alex = AlexNet()
    print("\n", "----MODEL IS CREATED----")
    X_train, Y_train, X_test, Y_test = prep_data()
    print("\n", "-----DATA IS READY-----")

    if (not os.path.exists('./model.h5')):
        alex.compile()
        alex.train(X_train, Y_train, 5, 50)
        alex.save_w(os.path.abspath("model.h5"))
        print("\n", "-----TRAINING IS DONE-----")
    else:
        alex.load_w(os.path.abspath("model.h5"))
        alex.compile()
        print("\n", "-----WEIGHTS ARE LOADED-----")

    scores = alex.evaluate(X_test, Y_test)
    return scores

def give_label(X):
    global is_loaded
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()   
    alex = AlexNet()
    loaded_model = model_from_json(loaded_model_json)
    alex.model = loaded_model

    if(os.path.exists('./model.h5')):
        if not is_loaded:
            alex.load_w(os.path.abspath("model.h5"))
            alex.compile()
            print("\n", "-----WEIGHTS ARE LOADED-----")
            is_loaded = True
        Y = alex.predict(X)
        return Y
    else:
        print("\n", "-----TRAIN THE MODEL BEFORE-----")
        return None










