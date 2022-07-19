# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:55:40 2022

@author: saikumar
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 21:29:15 2022

@author: saikumar
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from sklearn.preprocessing import normalize
early_stopping = EarlyStopping()

dataset = pd.read_csv('features/features.csv')
n_steps_in, n_steps_out = 8, 12 # number of input / number of outputs

def MakeInputOutput(dataset):
    # this method will make the base line model dataset which is basically x and y value itself.
    unique_ids = list(np.unique(dataset.id))
    X_values = dataset['x']
    Y_values = dataset['y']
    X_values = np.array(X_values).reshape((len(X_values), 1))
    Y_values = np.array(Y_values).reshape((len(Y_values), 1))
    stack = np.hstack((X_values,Y_values))
    return stack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    # this function will make the sequence feature and output. in this method I am basically trying to convert time series into supervised structure where the input of first 8 sequence in time series would be the predicted 12 points
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def addNewFeature(X_old, columns, feature_csv):
    # this function i have made in order to add more feature and correspondly train and predict the model
    df = pd.read_csv(feature_csv)
    new_X = [] # current feature
    All_feature = [] # whole feature dataset
    for i in columns:
        new_x = np.array(df[i])
        if sum(np.isnan(new_x))>0:
            new_x = np.nan_to_num(new_x)
        # converting to the n_setpes in features
        for i in range(len(new_x)):
            end_idx = i + n_steps_in
            seqx = new_x[i:end_idx]
            new_X.append(seqx)
    for i in range(len(X_old)):
        x = new_X[i].reshape(n_steps_in,1)
        new_feature_array   = np.append(X_old[i],x,axis=1)
        All_feature.append(new_feature_array)
    print("INFO: allnew features has been added")
    return np.array(All_feature)


stack = MakeInputOutput(dataset)
# normalized the dataset
stack = normalize(stack)
X, y = split_sequences(stack, n_steps_in, n_steps_out)
#X = normalize(X.reshape(X.shape[0], X.shape[1]*X.shape[2])).reshape(X.shape[0], X.shape[1],X.shape[2])


# adding new feature obstED to the base model
#X = addNewFeature(X,['ObstED'],'features/features.csv')
X_train,y_train = X[:40000],y[:40000]
X_valid,y_valid = X[80000:82000],y[80000:82000]
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# SVR - multioutput model

def buildmodel(x_train,y_train):
    # Create the SVR regressor
    n_sample,nx, ny = X_train.shape
    trainx = X_train.reshape((n_sample, nx*ny))
    
    n_sample,nx, ny = y_train.shape
    trainy = y_train.reshape((n_sample, nx*ny))
    
    svr = SVR(epsilon=0.2)
    # Create the Multioutput Regressor
    mor = MultiOutputRegressor(svr)
    
    # Train the regressor
    mor = mor.fit(trainx, trainy)
    
    return mor


def svmpredict(model, x):
    m,n = x.shape
    x = x.reshape(1,m*n)
    y_hat = model.predict(x)
    return y_hat
# fit model
if __name__ == "__main__":
   mor = buildmodel(X_train,y_train )
   pred = svmpredict(mor, X[0]).reshape(12,2)
   # saving the model
   model= 'svm_multiout.sav'
   pickle.dump(mor,open(model,'wb'))
   print('model saved successfully')
   