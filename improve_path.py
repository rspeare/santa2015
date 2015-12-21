import pandas as pd
import numpy as np
from haversine import haversine

north_pole = [90.,0.]
weight_limit = 1000.
sleigh_weight = 10.
best_score=12395765387.87850

def weighted_trip_length(stops, weights): 
    """
    Calculated the weighted weariness associated with a single trip.
    Called with arguments:
         trip[['Latitude','Longitude']], trip.Weight.tolist()
    """
    global north_pole
    global weight_limit
    global sleigh_weight
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)
    
    dist = 0.0
    prev_stop = north_pole
    prev_weight = np.sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    if (np.sum(weights)> weight_limit-10):
        return np.inf
    else:
        return dist

def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        print("One of the sleighs over weight limit!")
        return np.inf
 
    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())
    return dist    

def swap(i,j,df):
    #df.iloc[i-1:j+1]
    df.T[[i, j]] = df.T[[j, i]]
    # Swap Trip Id's
    tmp=df.T[i].TripId
    df.T[i].TripId=df.T[j].TripId
    df.T[j].TripId=tmp
    return df

def swap2(i,j,dfc):
    """
    Swap Everything but the TripId
    """
    for attr in ['GiftId','Latitude','Longitude','Weight']:
        tmpi=dfc.iloc[i][attr]
        tmpj=dfc.iloc[j][attr]
        dfc.iloc[i][attr]=tmpj
        dfc.iloc[j][attr]=tmpi

def merge2(i,j,df):
    tmp1=df.iloc[[i]]
    tmp2=df.iloc[[j]]
    

def propose_swap(dfc,Temp,lbound,hbound):
    """
    Propose a random Swap of two cities in the traveling salesmen problem
    """
    i1,i2=np.random.randint(lbound,high=hbound,size=2)
    trip0ID=dfc.iloc[i1]['TripId']
    trip1ID=dfc.iloc[i2]['TripId']
    trip0=dfc[dfc['TripId']==trip0ID]
    trip1=dfc[dfc['TripId']==trip1ID]

    dist1=weighted_trip_length(trip0[['Latitude','Longitude']], trip0.Weight.tolist())+weighted_trip_length(trip1[['Latitude','Longitude']], trip1.Weight.tolist())
    
    swap(i1,i2,dfc)
    trip0ID=dfc.iloc[i1]['TripId']
    trip1ID=dfc.iloc[i2]['TripId']
    trip0=dfc[dfc['TripId']==trip0ID]
    trip1=dfc[dfc['TripId']==trip1ID]

    dist2=weighted_trip_length(trip0[['Latitude','Longitude']], trip0.Weight.tolist())+weighted_trip_length(trip1[['Latitude','Longitude']], trip1.Weight.tolist())

    if (dist2 < dist1):
#        print('accepted')
#        print(dist2-dist1)
        return (dist2 - dist1)
    else:
        prob=np.exp((dist1-dist2)/Temp)
        sample=np.random.rand()
        # Accept Swap with probability exp(-deltaD/T)
        if (sample < prob):
#            print('accepted with probability :',prob)
#            print(dist2-dist1)
            return (dist2 - dist1)
        else:
#            print('rejected with probability :',1.-prob)
            swap(i1,i2,dfc)
            return 0.
    # should never get here
    return (dist2 - dist1)

def propose_merge(dfc,Temp,lbound,hbound):
    """
    Propose a random Swap of two cities in the traveling salesmen problem
    """
    i1,i2=np.random.randint(lbound,high=hbound,size=2)

    trip0ID=dfc.iloc[i1]['TripId']
    trip1ID=dfc.iloc[i2]['TripId']
    trip0=dfc[dfc['TripId']==trip0ID]
    trip1=dfc[dfc['TripId']==trip1ID]

    dist1=weighted_trip_length(trip0[['Latitude','Longitude']], trip0.Weight.tolist())+weighted_trip_length(trip1[['Latitude','Longitude']], trip1.Weight.tolist())

    tmp=dfc.TripId[i1]
    dfc.TripId[i1]=dfc.TripId[i2]

    trip0ID=dfc.iloc[i1]['TripId']
    trip1ID=dfc.iloc[i2]['TripId']
    trip0=dfc[dfc['TripId']==trip0ID]
    trip1=dfc[dfc['TripId']==trip1ID]

    dist2=weighted_trip_length(trip0[['Latitude','Longitude']], trip0.Weight.tolist())+weighted_trip_length(trip1[['Latitude','Longitude']], trip1.Weight.tolist())

    if (dist2 < dist1):
#        print('accepted merge')
#        print(dist2-dist1)
        return (dist2 - dist1)
    else:
        prob=np.exp((dist1-dist2)/Temp)
        sample=np.random.rand()
        # Accept Swap with probability exp(-deltaD/T)
        if (sample < prob):
#            print('accepted merge with probability :',prob)
            return (dist2 - dist1)
        else:
#            print('rejected merge with probability :',1.-prob)
            dfc.TripId[i1]=tmp
            return 0.
    # should never get here
    return (dist2 - dist1)

def running_mean(x,N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def burn_swaps(Temp,m,df,lbound,hbound):
    c=[]
    for i in np.arange(m):
        delta=propose_swap(df,Temp,lbound,hbound)
        c.append(delta)
    return np.array(c)

def burn_merges(Temp,m,df,lbound,hbound):
    c=[]
    for i in np.arange(m):
        delta=propose_merge(df,Temp,lbound,hbound)
        c.append(delta)
    return np.array(c)


