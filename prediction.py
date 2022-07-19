# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:55:29 2022

@author: saikumar
"""
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
from pytp.utils.evaluate import get_ade,get_fde
def load_models():
    
    model1 = load_model('MultiEndec_model')
    model2 = load_model('GRU_model')
    model3 = load_model('CNNLSTM_model')
    return model1, model2, model3
def mlModel():
    SVMmodel= 'svm_multiout.sav'
    model4 = pickle.load(open(SVMmodel,'rb'))
    return model4

def svmpredict(model, x):
    m,n = x.shape
    x = x.reshape(1,m*n)
    y_hat = model.predict(x)
    return y_hat.reshape(12,2)

def predict(model, x,n_features):
    x = x.reshape(1,8,n_features)
    pred = model.predict(x)
    return pred

def model_predictions(x, y,n_features):
    ade_scores = []
    fde_scores = []
    models  = load_models()
    #DL models
    for i in models:
        pred = predict(i, x, n_features)
        ade = get_ade(pred, y)
        y = y.reshape(1,pred.shape[1],pred.shape[2])
        fde = get_fde(pred, y)
        ade_scores.append(ade)
        fde_scores.append(fde)
    # ml modell
    model4 = mlModel()
    pred = svmpredict(model4, x)
    ade = get_ade(pred, y)
    fde = get_fde(pred.reshape(1,pred.shape[0],pred.shape[1]), y.reshape(1,pred.shape[0],pred.shape[1]))
    ade_scores.append(ade)
    fde_scores.append(fde)
    return pd.DataFrame({'fde_score':fde_scores,'ade_score':ade_scores}, index=['LSTM','GRU Model','CNN LSTM','SVM_multioutput'])
    

if __name__ == '__main__':
    # this is scaled values
    x = np.array([[0.99821399, 0.05973963],
           [0.99822836, 0.05949905],
           [0.99824323, 0.05924908],
           [0.9982581 , 0.058998  ],
           [0.99827306, 0.05874426],
           [0.99828811, 0.05848796],
           [0.99830159, 0.05825754],
           [0.99831644, 0.05800247]])
    
    y_actual  = np.array([[0.9983311 , 0.05774963],
           [0.99834557, 0.05749898],
           [0.99835985, 0.05725049],
           [0.99837394, 0.05700414],
           [0.99838777, 0.05676137],
           [0.99840151, 0.05651919],
           [0.99841516, 0.05627761],
           [0.99842855, 0.05603953],
           [0.99844193, 0.05580058],
           [0.99845507, 0.05556508],
           [0.99846811, 0.05533015],
           [0.998481  , 0.05509719]])
    
    score_df = model_predictions(x, y_actual, x.shape[-1]).sort_values(by=['fde_score','ade_score'])
    print(score_df)
    
    
    
    
    