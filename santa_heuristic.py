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

gifts = pd.read_csv('gifts.csv')
sample_sub = pd.read_csv('sample_submission.csv')

all_trips = sample_sub.merge(gifts, on='GiftId')


best_score=12395765387.87850

# Add lots of 'north pole' stops, concatenate them to the original gifts data frame
def initialize(frac):
    notValid=True
    while (notValid):
        seed1=np.insert(north_pole,0,-1)
        seed1=np.insert(seed1,3,0.)
        s=pd.DataFrame(seed1,index=gifts.columns.values).T
        s.head()
        s=pd.DataFrame(seed1,index=gifts.columns.values).T
    
        for i in np.arange(np.log(len(gifts)*frac)/np.log(2)):
            s=pd.concat([s,s])
        print(len(s))
    
        dfc=pd.concat([gifts,s])
        dfc.head()
        
        # Now randomly distribute the stops
        dfc=dfc.iloc[np.random.permutation(len(dfc))]
        dfc.head()
    
        stops=np.where(dfc['GiftId']==-1)[0]
        
        dfc['tripW']=np.zeros(len(dfc))
        dfc['TripId']=np.zeros(len(dfc))

        ###### CHECK IF A VALID SET OF STOPS
        cumWeights=[]
    
        np.insert(stops,0,0)
        np.insert(stops,len(stops),len(dfc)+1)
    
        tripWeight=np.sum(dfc['Weight'].values[:stops[0]])
        cumWeights.append(tripWeight)
        dfc['tripW'].values[:stops[0]]=tripWeight
        dfc['TripId'].values[:stops[0]]=0
        
        for i in np.arange(len(stops)-1):
    #    print(i)
            tripWeight=np.sum(dfc['Weight'].values[stops[i]:stops[i+1]])
            cumWeights.append(tripWeight)
            dfc['tripW'].values[stops[i]:stops[i+1]]=tripWeight
            dfc['TripId'].values[stops[i]:stops[i+1]]=i
        
        tripWeight=np.sum(dfc['Weight'].values[stops[-1]:])
        cumWeights.append(tripWeight)
        dfc['tripW'].values[stops[-1]:]=tripWeight
        dfc['TripId'].values[stops[-1]:]=i+1

        cumWeights=np.array(cumWeights)
        
        if np.any(dfc['tripW'].values > 1000.-10.):
            print('Too much weight in the sleigh!')
            frac*=1.1
        else:
            print('legal set of stops')
            notValid=False
    if np.any(np.isnan(my_trips)):
        print('WARNING THERE ARE NAN TRIP IDS')
    #print('calculating initial score fraction...relative to Naive')
    return dfc
   
def permute_init(df):
    notValid=True
    dfc=df
    while (notValid):    
            # Now randomly distribute the stops
        dfc=dfc.iloc[np.random.permutation(len(df))]
        dfc.head()
    
        stops=np.where(dfc['GiftId']==-1)[0]
        
        dfc['tripW']=np.zeros(len(dfc))
        dfc['TripId']=np.zeros(len(dfc))

        ###### CHECK IF A VALID SET OF STOPS
        cumWeights=[]
    
        np.insert(stops,0,0)
        np.insert(stops,len(stops),len(dfc)+1)
    
        tripWeight=np.sum(dfc['Weight'].values[:stops[0]])
        cumWeights.append(tripWeight)
        dfc['tripW'].values[:stops[0]]=tripWeight
        dfc['TripId'].values[:stops[0]]=0
        
        for i in np.arange(len(stops)-1):
    #    print(i)
            tripWeight=np.sum(dfc['Weight'].values[stops[i]:stops[i+1]])
            cumWeights.append(tripWeight)
            dfc['tripW'].values[stops[i]:stops[i+1]]=tripWeight
            dfc['TripId'].values[stops[i]:stops[i+1]]=i
        
        tripWeight=np.sum(dfc['Weight'].values[stops[-1]:])
        cumWeights.append(tripWeight)
        dfc['tripW'].values[stops[-1]:]=tripWeight
        dfc['TripId'].values[stops[-1]:]=i+1

        cumWeights=np.array(cumWeights)
        
        if np.any(dfc['tripW'].values > 1000.-10.):
            print('Too much weight in the sleigh!')
        else:
            print('legal set of stops')
            notValid=False
    if np.any(np.isnan(my_trips)):
        print('WARNING THERE ARE NAN TRIP IDS')
    score1=weighted_reindeer_weariness(df[all_trips.columns])
    score2=weighted_reindeer_weariness(dfc[all_trips.columns])
    print('score1 :',score1)
    print('score2 :',score2)

    if (score2 < score1):
        return dfc
    else:
        return df

    #print('calculating initial score fraction...relative to Naive')
