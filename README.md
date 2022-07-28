# Master_Project
Group 3 Masters Project Trajectory Prediction of Vehicle


##Table of contents

* [Project Summary](#project-Summary)
* [Project Implementation](#project-Implementation)
* [Code Explanation](#Code-explanation)
* [Model Plot](#Model-Plot)

# Project Summary

Around the world, car accidents result in a lot of injuries and fatalities each year.
Through the recent use of driver-assistance systems, this number has somewhat decreased.
Reducing this number can be greatly helped by the development of driver-assistance systems, sometimes known as automated driving systems.
Estimating and predicting the movement of surrounding vehicles is critical for an automated vehicles and advanced safety systems.
Furthermore, predicting the trajectory is influenced by a variety of factors, including driver behavior during accidents, the history of the vehicle's movement and the surrounding vehicles, and their location on the traffic scene.
The vehicle must navigate a safe path through traffic while reacting quickly to other drivers' unpredictable behaviors.
Herein, to predict automated vehicles' paths, we used LSTM and GRU models.
Deep learning models were used as a tool for extracting the features of these images. 
The models we used can predict the vehicle's future path on any freeway only by viewing the images related to the history of the target vehicle's movement and its neighbors. 
along with that For our data set, we have used a  plot, which compares the predicted data with real data.

# Project Implementation

 Execute the tasks one by by accordingly, Since the dataset is huge we could not be able to keep it in whole in one file.

Perform the tasks in below format and the results are saved in the diffirent files.

1.Task wise execution (Task 1.ppt)

Task 1 : Prepared a presentation on datasets

Task 2 : Calcuated the eculidean distances as per the feature selction and  Extracted the required features.

2. Feature Selection (2A,2B,2C,2D).py files


Task 3. a trajectory prediction model (CNN_lstm_Model, GRU_Model).py files

Bulid the models using the algorithms and Calculated the accuracy values for the models.

Task 4. (prediction.py)

Calculated the ade, fde values of different models.

Task 5. (collision_percentage.py)

Calculated the percentage of collisions of the models pass the model according to the requirement.

Task 6.  (plot.py)

Plotted a graphs of prediction and compared with actual plot.



# Code Explanation
#### Importing all necessary libraries

```
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Conv1D,MaxPooling1D, Flatten, Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import normalize
early_stopping = EarlyStopping()
```

#### Loading our dataset using pandas library.
```
dataset = pd.read_csv('C:\\Users\\akhil\\Desktop\\Final_code\\Final_code\\features.csv')
```

#### We are calculating Euclidean distance
```
def euclidean_distance():
    print("Finding Euclidean distance for a pedestrian from the nearest obstacle point")
    print("For each pedestrian, id of the leading pedestrian is given. we can treat this vehicle as the obstacle.\
        \n from this we can calsulate the Eculidian distance")
    df = pandas.read_csv("features.csv")
    df["ObstED"] = ''
    for frame_no in df['frame']:
        followingid = df['followingId'][df['frame']==frame_no].tolist()
        if followingid[0] == 0:
            print("No obstacle ahead")
            df["ObstED"][frame_no] = -1
        else:
            x1 = df['x'][(df['id']==followingid[0]) & (df['frame']==frame_no)].tolist()[0]
            y1 = df['y'][(df['id']==followingid[0]) & (df['frame']==frame_no)].tolist()[0]
            x2 = df['x'][df['frame']==frame_no].tolist()[0]
            y2 = df['y'][df['frame']==frame_no].tolist()[0]
            ED = sqrt((x1-x2)**2 + (y1-y2)**2)
            print(ED)
            df["ObstED"][frame_no] = ED
    df.to_csv("features.csv", index=False)

euclidean_distance()
```

#### This function we have made in order to add more feature and correspondly train and predict the model
```
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
```

#### Specifying the structure of Neural network CNN_LSTM
```
def buildModel():
    model = Sequential(name="CNN_lstm_Model")
    model.add(Conv1D(filters=24, kernel_size=3, activation='relu', input_shape=(n_steps_in, n_features))) # inputlayer
    # hidden layer
    model.add(Conv1D(filters=24, kernel_size=3, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(MaxPooling1D(pool_size=2))
    # --------------------------------------------    
    model.add(Flatten())
    
    model.add(RepeatVector(n_steps_out))
    
    model.add(LSTM(200, activation='relu', return_sequences=True))
    #  output layer
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(2)))
    model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['MeanSquaredError', 'MeanAbsoluteError','accuracy'],
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None, 
              steps_per_execution=None 
             )  
    return model
```

#### Here in the below we are defining GRU model
```
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
```


#### Here in the below plotting of the graph

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
    
    plt.ylim([0,1])
    #print(np.mean(y_actual.ravel()))
    plt.plot(smooth(x.ravel(),10), label="Actual X values", c="red", linewidth=7.0)
    plt.plot(smooth(y_actual.ravel(),10),'--', label="Actual Y values", c="red", linewidth=7.0)
  
    plt.plot(smooth(pred_[1],10),'o-', label=f"{models[1].name} path", c = "Blue" ,linewidth=2.0)
    plt.plot(smooth(pred_[2],10), 'o--',label=f"{models[2].name} path", c = "green",linewidth=1.0 )
    plt.xlabel("X-plane")
    plt.ylabel("Y-plane")
    plt.legend()
    plt.savefig('output.png')





