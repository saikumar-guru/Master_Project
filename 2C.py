import pandas
from math import sqrt

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