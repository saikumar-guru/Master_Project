import pandas

def speed(): 
    df = pandas.read_csv("Columns/01_tracks.csv")
    try:
        id = int(input("plese enter id of the pedestrain to extract speed at each time step: "))
        print("Printing speed of the pedestrian with id ",id," at each time step\n")
        print(df["xVelocity"][df["id"] == id])
    except:
        print("Incorrect input")

speed()