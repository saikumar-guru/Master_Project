# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:55:29 2022

@author: Saikumar
"""
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import itertools
import pickle
from GRU_implemetation import * 
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
    avg_ade, avg_fde = 0 , 0
    total_ade_scores = []
    total_fde_scores = []
    models  = load_models()
    #DL models
    models = models[1:]
    for i in models:
        for j in range(len(x)):
            pred = predict(i, x[j], n_features)
            ade = get_ade(pred, y[j])
            new_y = y[j].reshape(1,pred.shape[1],pred.shape[2])
            #print(new_y)
            fde = get_fde(pred, new_y)
            ade_scores.append(ade)
            fde_scores.append(fde)
        #print(ade_scores)
        total_ade_scores.append(sum(ade_scores)/len(ade_scores))
        total_fde_scores.append(sum(fde_scores)/len(fde_scores))
    # ml modell
    #print('total ade', total_ade_scores)
    #print('total fde',total_fde_scores)
    '''
    model4 = mlModel()
    for i in range(len(x)):
        pred = svmpredict(model4, x[i])
        ade = get_ade(pred, y[i])
        fde = get_fde(pred.reshape(1,pred.shape[0],pred.shape[1]), y[i].reshape(1,pred.shape[0],pred.shape[1]))
        ade_scores.append(ade)
        fde_scores.append(fde)
    total_ade_scores.append(sum(ade_scores)/len(ade_scores))
    total_fde_scores.append(sum(fde_scores)/len(fde_scores))
    print(total_ade_scores, total_fde_scores)
    '''
    return pd.DataFrame({'fde_score':total_fde_scores,'ade_score':total_ade_scores}, index=['GRU Model','CNN LSTM'])
    

if __name__ == '__main__':
    '''
    stack = MakeInputOutput(dataset)
    # normalized the dataset
    X, y = split_sequences(stack, n_steps_in, n_steps_out)
    X = normalize(X.reshape(X.shape[0], X.shape[1]*X.shape[2])).reshape(X.shape[0], X.shape[1],X.shape[2])
    '''

    x = X[:]
    y_actual  = y[:]
    #x = X
    #y_actual  = y
    score_df = model_predictions(x, y_actual, x.shape[-1]).sort_values(by=['fde_score','ade_score'])
    print(score_df)
    
    
    
    
    