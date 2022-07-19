import pandas
from math import sqrt

def euclidean_distance():
    print("Finding Euclidean distance for a pedestrian from the destination point")
    print("For each pedestrian considering the end of highway section as the destination.\
        \n point for the pedestrian at each time step")
    df = pandas.read_csv("C:\\Users\\Accesskey\\OneDrive\\Desktop\\Master_Project\\Master_Project\\Columns\\01_tracks.csv")
    df['DestED'] = ''
    for frame in df.index:
            x1 = 0
            y1 = df['y'][frame] #taking y1 similar to the y2 assuming that the pedestrian is in the same lane
            x2 = df['x'][frame]
            y2 = df['y'][frame]
            ED = sqrt((x2-x1)**2 + (y2-y1)**2)
            print(ED)
            df['DestED'][frame] = ED
    df.to_csv("features.csv", index=False)


euclidean_distance()