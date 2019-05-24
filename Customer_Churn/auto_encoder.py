import tensorflow as tf
from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras import regularizers
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import platform
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
import preprocessing
from sklearn.metrics import classification_report

predictors_train, label_train, predictors_test, label_test = preprocessing.get_data()

def connect_model():
    input_ = Input(shape=(predictors_train.shape[1],))
    dropout=Dropout(0.1)(input_)
    encoded = Dense(10, activation='tanh', activity_regularizer=regularizers.l2(10e-5))(dropout)
    decoded = Dense(predictors_train.shape[1], activation='tanh')(encoded)
    network = Model(input_, decoded)
    network.compile(optimizer='adam', loss='mean_squared_error')
    return network

network = connect_model()
history=network.fit(predictors_train[np.where(label_train == 0)],predictors_train[np.where(label_train == 0)],
               epochs=400,
                batch_size=256,
                shuffle=True,
                validation_split=0.1
                       )

def return_losses(predictors_test, preds):
    loss = np.zeros(len(predictors_test))
    for i in range(len(predictors_test)):
        loss[i] = ((preds[i] - predictors_test[i]) ** 2).mean(axis=None)
        
    return loss

threshold=history.history["loss"][-1]

testing_set_predictions = network.predict(predictors_test)
test_losses = return_losses(predictors_test,testing_set_predictions)
testing_set_predictions = np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses>threshold)] = 1

print(classification_report(label_test, testing_set_predictions))