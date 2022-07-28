# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:03:19 2022

@author: saikumar
"""
import pylab as plt
import numpy as np
from prediction import *
'''
def plot(model, X_feature, y_actual):
    pred_ = []
    
    # checking which model it is
    for i in range(len(X_feature)):
        if 'SVR' in str(model.__repr__):
            pred = svmpredict(model, X_feature[i])
        else:
            pred = predict(model, X_feature[i],X_feature[i].shape[-1])
        pred_.append(pred)
            
    #X1_actual = y_actual[:,0]
    #X2_actual = y_actual[:,1]
    #pred = pred.reshape((pred.shape[1],pred.shape[2]))
    pred_ = np.array([i.reshape((pred.shape[1],pred.shape[2])) for i in pred_])
    #x1_predicted = pred[:,0]
    #x2_predited = pred[:,1]
    print(pred_.shape)
    print(X_feature.shape)
    print(pred.shape)
    print(y_actual.shape)
    plt.figure(figsize=(30,15))
    #f, axes = plt.subplots(len(y_actual), 1, figsize=(30,15))
    #f.tight_layout()

    #for i in range(len(y_actual)):
    #    #print(y_actual[i])
    #    axes[i].plot(y_actual[i][0],label="Actual path", c = "Blue", linestyle='--')
    #    axes[i].plot(pred_[i][0],label="Prediction Path", c="Red")
    #plt.plot(y_actual[0].ravel(), label="Actual Path", c="Red")
    #plt.xlabel("X-plane")
    #plt.ylabel("Y-plane")
    #plt.legend()
    y_actual_x = y_actual.reshape((y_actual.shape[0]*y_actual.shape[1],2))[:,0]
    y_actual_y = y_actual.reshape((y_actual.shape[0]*y_actual.shape[1],2))[:,1]
    y_predicted_x = pred_.reshape((pred_.shape[0]*pred_.shape[1],2))[:,0]
    y_predicted_y = pred_.reshape((pred_.shape[0]*pred_.shape[1],2))[:,1]
    plt.plot(y_actual_x.ravel(), label="Actual X plane Path", c="Red")
    plt.plot(y_actual_y.ravel(), label="Actual Y plane Path", c="orange")
    plt.plot(y_predicted_x.ravel(), label="predicted X plane Path", c="green")
    plt.plot(y_predicted_y.ravel(), label="predicted Y plane Path", c="olive")
    
    #plt.plot(y_actual[j].ravel(), label="Actual Path", c="Red")
    #plt.plot(pred_[j].ravel(), label="Prediction path", c = "Blue" )
    #plt.tight_layout()
    plt.xlabel("X-plane")
    plt.ylabel("Y-plane")
    plt.legend()
    
    try:
        plt.savefig(f'{model.name}.png')
    except:
        print('some error is coming while saving')
    plt.show()
'''
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

    
    
def MakePlots(x,y_actual):
    models  = load_models()
    #DL models
    pred_ = []
    for i in models:
        #plt = plot(i, x, y_actual)
        #Saving the plots
        # checking which model it is
        if 'SVR' in str(i.__repr__):
            pred = svmpredict(i, X_feature[i])
        else:
            pred = predict(i, x,x.shape[-1])
        pred_.append(pred.ravel())
    print(pred_)
    plt.figure(figsize=(30,15))
    plt.tight_layout()
    #plt.plot(x.ravel(), label="Actual X values", c="red")
    plt.ylim([0,1])
    #print(np.mean(y_actual.ravel()))
    plt.plot(smooth(x.ravel(),10), label="Actual X values", c="red", linewidth=7.0)
    plt.plot(smooth(y_actual.ravel(),10),'--', label="Actual Y values", c="red", linewidth=7.0)
   # plt.plot(smooth(pred_[0],20),'o-', label=f"{models[0].name} path", c = "cyan" ,linewidth=3.0)
    plt.plot(smooth(pred_[1],10),'o-', label=f"{models[1].name} path", c = "Blue" ,linewidth=2.0)
    plt.plot(smooth(pred_[2],10), 'o--',label=f"{models[2].name} path", c = "green",linewidth=1.0 )
    plt.xlabel("X-plane")
    plt.ylabel("Y-plane")
    plt.legend()
    plt.savefig('output.png')
        

if __name__ == '__main__':
    x = X[0]
    y_actual  = y[0]
    pred_ = MakePlots(x,y_actual)

    
    
    
    