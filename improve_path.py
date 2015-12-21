import pandas as pd
import numpy as np
from haversine import haversine

north_pole = [90.,0.]
weight_limit = 1000.
sleigh_weight = 10.

def weighted_trip_length(stops, weights): 
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)
    
    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    if (np.sum(weights)> weight_limit):
        return np.inf
    else:
        return dist

def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()
    
    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")
 
    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())
    
    return dist    

best_score=12395765387.87850
def swap2(i,j,dfc):
    for attr in ['GiftId','Latitude','Longitude','Weight']:
        tmpi=dfc.iloc[i][attr]
        tmpj=dfc.iloc[j][attr]
        dfc.iloc[i][attr]=tmpj
        dfc.iloc[j][attr]=tmpi

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
    
    swap2(i1,i2,dfc)
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
            swap2(i1,i2,dfc)
            return 0.
    # should never get here
    return (dist2 - dist1)
def running_mean(x,N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')

def burn_in(T,m,df,lbound,hbound):
    c=[]
    for i in np.arange(m):
        delta=propose_swap(df,Temp,lbound,hbound)
        c.append(delta)
    return np.array(c)

print('reading old model file')
df0=pd.read_csv('santas_route_2.csv')
print(df0.head())
m0=10000
Temp=10**4.50182022654
var=5.
count=0
print('log(T): '+str(np.log(Temp)/np.log(10.))+' var: '+str(var))
for n in np.arange(100):
    m0=np.amax([var,1.])**2.5*400
    mu=burn_in(Temp,m0,df0,0,len(df0))
    var=np.std(mu/Temp)
    print('        equilibriation('+str(count)+')  var: '+str(var))
    score2=weighted_reindeer_weariness(df0[all_trips.columns])
    print(score2)
    if (var < 5.):
        df0.to_csv('santas_route_4.csv')
    else:
        count+=m0

