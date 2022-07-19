import pandas
from math import sqrt

def euclidean_distance():
    print("Finding Euclidean distance for a pedestrian from the nearest nighbour point")
    print("For each pedestrian, ids of the other pedestrians are given. we can find nearest neighbour from that.")
    df = pandas.read_csv("features/features.csv")
    df['nearestNeighbourDist'] = ''
    for frame_no in df['frame']:
        distances = {}
        id_list = ['followingId','precedingId','leftPrecedingId',
                'leftAlongsideId','leftFollowingId','rightPrecedingId','rightAlongsideId',
                'rightFollowingId']
        for id in id_list:
            xid = df[id][df['frame']==frame_no].tolist()
            if xid[0] != 0:
                x1 = df['x'][(df['id']==xid[0]) & (df['frame']==frame_no)].tolist()[0]
                y1 = df['y'][(df['id']==xid[0]) & (df['frame']==frame_no)].tolist()[0]
                x2 = df['x'][df['frame']==frame_no].tolist()[0]
                y2 = df['y'][df['frame']==frame_no].tolist()[0]
                ED = sqrt((x1-x2)**2 + (y1-y2)**2)
                distances[xid[0]] = ED
        minv = min(distances.values())
        neighbour = [key for key in distances if distances[key] == minv]
        print("Distance from nearest neighbour is ",minv , "for neighbour with id ",neighbour, "at frame no",frame_no)
        #df['nearestNeighbourDist'][frame_no] = minv
        break
    return df['nearestNeighbourDist']


euclidean_distance()