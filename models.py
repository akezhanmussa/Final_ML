
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import numpy as np
from keras.optimizers import SGD


class AlexNet:

    def __init__(self):
        self.model = None

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        self.model = model
        

    def train(self, X_train, Y_train, epochs, batch_size):
        self.model.fit(X_train, Y_train, epochs, batch_size) 

    def evaluate(self, X_valid, Y_valid):
        scores = self.model.evaluate(X_valid, Y_valid)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        return scores

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_w(self, path_name):
        self.model.save_weights(path_name)

    def load_w(self, path_name):
        self.model.load_weights(path_name)

    def compile(self):
        opt = SGD(lr = 0.01)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    