# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:03:19 2022

@author: saikumar
"""
import pylab as plt
import numpy as np
from prediction import *
def plot(model, X_feature, y_actual):
    # checking which model it is
    if 'SVR' in str(model.__repr__):
        pred = svmpredict(model, X_feature)
    else:
        pred = predict(model, X_feature,X_feature.shape[-1])
        
    #X1_actual = y_actual[:,0]
    #X2_actual = y_actual[:,1]
    pred = pred.reshape((pred.shape[1],pred.shape[2]))
    #x1_predicted = pred[:,0]
    #x2_predited = pred[:,1]
    print(X_feature.shape)
    print(pred.shape)
    plt.figure(figsize=(30,15))
    plt.plot(y_actual.ravel(), label="Actual Path", c="Red")
    plt.plot(pred.ravel(), label="Prediction path", c = "Blue" )
    plt.tight_layout()
    plt.xlabel("X-plane")
    plt.ylabel("Y-plane")
    plt.legend()
    
    try:
        plt.savefig(f'{model.name}.png')
    except:
        print('some error is coming while saving')
    plt.show()
    
    
    
def MakePlots(x,y_actual):
    models  = load_models()
    #DL models
    for i in models:
        plt = plot(i, x, y_actual)
        #Saving the plots
        
        

if __name__ == '__main__':
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
    MakePlots(x,y_actual)
    
    
    
    
    