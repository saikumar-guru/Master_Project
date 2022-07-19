# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:02:43 2022

@author: saikumar
"""

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from math import sqrt
import itertools
from CNN_LSTM import *

model = load_model('CNNLSTM_model')
dataset = pd.read_csv("features/features.csv")[:80000]

def buildDataset(df):
    X_pred,Y_pred = [],[]
    def predict(model, x,n_features):
        x = x.reshape(1,8,n_features)
        pred = model.predict(x)
        return pred[:,:,0][0][0],pred[:,:,1][0][0]
    
    stack = MakeInputOutput(dataset)
    X, Y = split_sequences(stack, n_steps_in, n_steps_out)
    #X = addNewFeature(X,['ObstED'],'features/features.csv')
    #X_valid = X[200:400]
    
    
    for i in range(len(X)):
        x,y = predict(model, X[i], X.shape[-1])
        X_pred.append(x)
        Y_pred.append(y)
        
    X_pred = X_pred + [0]*(len(df) - len(X_pred))
    Y_pred = Y_pred + [0]*(len(df) - len(Y_pred))
    dataset['predicted_X'] = X_pred
    dataset['predicted_Y'] = Y_pred
    
    return dataset

df_new = buildDataset(dataset)

def CalculateED(df, x , y):
    df['nearestNeighbourDist'] = ''
    df['Collision'] = ''
    def Eculidean_distance(x,y):
        return np.linalg.norm(x-y)

    def SingleED(df, id_=1):
        ALLED, ED = [],[]
        try:
            following_id = df.loc[(df['id']==id_)]['followingId']
            precedingId = df.loc[(df['id']==id_)]['precedingId']
            leftPrecedingId = df.loc[(df['id']==id_)]['leftPrecedingId']
            leftAlongsideId = df.loc[(df['id']==id_)]['leftAlongsideId']
            leftFollowingId = df.loc[(df['id']==id_)]['leftFollowingId']
            rightPrecedingId = df.loc[(df['id']==id_)]['rightPrecedingId']
            rightAlongsideId = df.loc[(df['id']==id_)]['rightAlongsideId']
            rightFollowingId = df.loc[(df['id']==id_)]['rightFollowingId']
            Filterid = [following_id, precedingId, leftPrecedingId, leftAlongsideId, leftFollowingId,rightPrecedingId,rightAlongsideId, rightFollowingId ]
            frame_size = len(following_id)
            for i in range(len(following_id)):
                p1 = np.array(df.loc[(df['id']==id_) & (df['frame']==i+1)][[x,y]])
                for filter_ in Filterid:
                    p2 = np.array(df.loc[(df['id']==filter_[filter_.index[i]]) & (df['frame']==i+1)][[x, y]])
                    ed = Eculidean_distance(p1, p2)
                    ED.append(ed)
                min_ = np.min(np.array(ED)[np.nonzero(np.array(ED))])
                ALLED.append(min_)
                ED = []
        except :
            pass
        return ALLED

    ids = list(df['id'].unique())
    ALL_ED = []
    for i in ids:
        ED = SingleED(df, id_ = i)
        ALL_ED.append(ED)
    ED =  list(itertools.chain(*ALL_ED))
    df['nearestNeighbourDist'] =  ED + [0]*(len(df)- len(ED)) # padding if the array having unequal length 
    df['Collision'] = df['nearestNeighbourDist']<0.1
    return df


df = CalculateED(df_new,'predicted_X','predicted_Y')
print(df)
Collision = np.array(df['Collision'].value_counts())
print(f'Percentage of Collision = {round((Collision[1]/Collision[0])* 100,2)}%')
