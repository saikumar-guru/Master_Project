# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:45:16 2022

@author: saikumar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 21:29:15 2022

@author: saikumar
"""
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Conv1D,MaxPooling1D, Flatten, Dropout
from keras.layers import RepeatVector, Bidirectional, GRU
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from keras import Input
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
X_train,y_train = X[:],y[:]
X_valid,y_valid = X[:],y[:]
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define CNN-lstm
def buildModel(X_train):
    ##### Step 3 - Specify the structure of a Neural Network
    model = Sequential(name="GRU_Model") # Model
    model.add(Input(shape=(X_train.shape[1],X_train.shape[2]), name='Input-Layer')) # Input Layer - need to speicfy the shape of inputs
    model.add(Bidirectional(GRU(units=64, activation='tanh', recurrent_activation='sigmoid', stateful=False), name='Hidden-GRU-Encoder-Layer')) # Encoder Layer
    model.add(RepeatVector(n_steps_out, name='Repeat-Vector-Layer')) # Repeat Vector
    model.add(Bidirectional(GRU(units=32, activation='tanh', recurrent_activation='sigmoid', stateful=False, return_sequences=True), name='Hidden-GRU-Decoder-Layer')) # Decoder Layer
    #ouput layer
    model.add(TimeDistributed(Dense(units=2, activation='linear'), name='Output-Layer')) # Output Layer, Linear(x) = x

    model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['MeanSquaredError', 'MeanAbsoluteError','accuracy'], 
              loss_weights=None, 
              weighted_metrics=None,
              run_eagerly=None, 
              steps_per_execution=None 
             )
    
    return model
    

def predict(model, x, n_features):
    x = x.reshape(1,8,n_features)
    y_hat = model.predict(x)
    print(y_hat)
# fit model
if __name__ == "__main__":
    model = buildModel(X_train) 
    history2 = model.fit(X_train, y_train, epochs=300, verbose=1, callbacks=[early_stopping], validation_data=(X_valid,y_valid))
    model.save('GRU_model')
    try:
        GRU_model = open('GRU_model.txt', 'wt')
        GRU_model.write(str(history2.history))
        GRU_model.close()
        print('GRU_model model summary has been created successfully')
    except:
        print("Unable to write to file")
    