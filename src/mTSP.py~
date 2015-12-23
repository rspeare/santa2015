import numpy as np
import pickle
from haversine import haversine
import pandas as pd

class mission:
    def __init__(self,Ntrips):
        """
        Creates an m-Traveling Salesmen object. Used for the Holiday
        FICO kaggle competition, where Santa must deliver all packages
        with as little distance -- or in this case, a specified metric
        called reindeer weariness -- as possible. 

        ATTRIBUTES:
        dmap: a dictionary lookup table, where, given a giftID, you are
             returned a latitude, longitude pair
        wmap: a dictionary lookup table, where, given a giftID, your are
             given the corresponding package weight
        tmap: a dictionary lookup table, where, given a giftID, your are
             given the corresponding tripID and position within that trip
        gifts: all unique gift identifiers
        trips: a list of lists, containing all gifts and their relative 
             position within santas delivery schedule
        north_pole: lon,lat position of the north pole
        sleigh_weight: the "package" weight of the sleigh itself, which
             must be included in the evaluate of our distance metric
        limit: design weight limit for the vehicle. No trip can carry more
             than this specified amount
        
        """
        self.dmap=pickle.load(open("../data/xmap.pickle","rb"))
        self.wmap=pickle.load(open("../data/wmap.pickle","rb"))
#        self.tmap=pickle.load(open("../data/tmap.pickle","rb"))
        self.tmap={}
        self.gifts=pickle.load(open("../data/GiftIds.pickle","rb"))
        self.trips=[[] for i in np.arange(Ntrips)]
#        self.gifts=[item for sublist in self.trips for item in sublist]
#        self.gifts=[i for i in np.arange(1,len(self.dmap.keys())+1)]

        # Sleigh Weight and North Pole Location, set identified as (-1)
        self.north_pole=[90.,0.]
        self.sleigh_weight=10.
        self.limit=1000.
        self.dmap.update({-1:self.north_pole})
        self.wmap.update({-1:self.sleigh_weight})

    def init_trips(self,size):
        """
        Initialize the list of lists, trips,
        into trips of size 'size'. The lower the size,
        the more likely it is that you stumble upon a feasible
        solution. (Due to weight Constraints)
        """
        while (True):
            print('trying initial trips of size, ',size)
            data=[i for i in np.arange(1,len(self.gifts)+1)]
            np.random.shuffle(data)
            self.trips=[data[x:x+size] for x in np.arange(0, len(data), size)]
            msg=[[self.tmap.update({gift:[i,self.trips[i].index(gift)]}) for gift in self.trips[i]] for i in np.arange(len(self.trips))]
            self.check_loads()
            break
        return self.loss()

    def init_small_trips(self,N):
        """
        Try to fit all packges in to N trips.
        Requires gift Id's and mapped weights.
        """
        weights=[self.wmap[g] for g in self.gifts]
        gifts=self.gifts
        self.trips=[[] for n in np.arange(N)]
        a=np.vstack([gifts,weights]).T
        sorted_gifts=a[a[:,1].argsort()] # sorted gift,weight array
        gifts=sorted_gifts[:,0]
        i=0
        for g in gifts:
#            print(i)
            load=self.get_load(self.trips[i])
            if (load+self.wmap[g] < self. limit):
                self.trips[i % N].append(g.astype(int))
            i+=1
            i=i%N

        for i in np.arange(len(self.trips)):
            self.reindex_trip(i)
        self.check_loads()
        return self.loss()

    
    def dist(self,g1,g2):
        """
        Compute the Haversine distance between two gifts
        """
        return haversine(self.dmap[g1],self.dmap[g2])
    
    def check_loads(self):
        loads=[]
        for i in np.arange(len(self.trips)):
            loads.append(self.check_trip_load(self.trips[i],i))
        return loads
            
    def check_trip_load(self,trip,i):
        s=0
        for stop in trip:
            s+=self.wmap[stop]
            if (s > self.limit):
                raise ValueError(str(s)+' of mass in trip '+str(i)+' with packages '+str(trip)+' and weight ')
        return s

    def get_load(self,trip):
        s=0
        for stop in trip:
            s+=self.wmap[stop]
        return s
        
    def trip_weariness2(self,gifts,ii):
        """
        DEPRECATED
        """
        if (len(gifts)==0):
            return 0
        self.check_trip_load(gifts,ii)

        tuples=[self.dmap[g] for g in gifts[::-1]]
        tuples.append(self.north_pole)
        weights=[self.wmap[g] for g in gifts[::-1]]
        weights.append(self.sleigh_weight)

        dist=0.0
        prev= self.north_pole
        prev_weight=sum(weights)

        for location,weight in zip(tuples,weights):
            dist=dist+haversine(location,prev)*prev_weight
            prev=location
            prev_weight-=weight
        return dist

    def trip_weariness(self,gifts,ii):
        """
        Compute the weighted reindeer weariness
        of a trip, consisting of a sequence of Gifts.
        If the trip is infeasible, return infinity
        """
        if (len(gifts)==0):
            return 0
        self.check_trip_load(gifts,ii)

        dist=0.0
        prev=-1 # start at the north_pole
        load=self.wmap[prev] #sleigh_weight

        for g in gifts[::-1]:
            next=g
            dist+=load*self.dist(prev,next)
            load+=self.wmap[next]
            prev=next

        dist+=load*self.dist(next,-1)
        return dist

    def weighted_trip_length(self,trip): 
        """
        DEPRECATED
        """
        tuples = [self.dmap[g] for g in trip]
        tuples.append(self.north_pole)
        weights = [self.wmap[g] for g in trip]
        weights.append(self.sleigh_weight)
    
        dist = 0.0
        prev_stop = self.north_pole
        prev_weight = sum(weights)
        for location, weight in zip(tuples, weights):
            dist = dist + haversine(location, prev_stop) * prev_weight
#            print(weight,location,dist)
            prev_stop = location
            prev_weight = prev_weight - weight
        return dist

    def WRW3(self):
        dist=0.0
        for trip in self.trips:
#            stops=[self.dmap[g] for g in trip]
#            weights=[self.wmap[g] for g in trip]
            dist +=self.weighted_trip_length(trip)
    
        return dist    
    def loss(self):
        wrw=0.
        for i in np.arange(len(self.trips)):
            wrw+=self.trip_weariness(self.trips[i],i)
        return wrw
    def WRW2(self):
        wrw=0.
        for i in np.arange(len(self.trips)):
            wrw+=self.trip_weariness2(self.trips[i],i)
        return wrw

    def swap(self,g1,g2):
        """
        Swap two gift locations for their
        order in the chain.
        """
        try:
            t1,i1=self.tmap[g1]
            t2,i2=self.tmap[g2]
        except TypeError:
            print('bad lookup at',g1,g2)
        
#        print('g,t,i,w: ',[g1,t1,i1],[g2,t2,i2])
        try:
            w1=self.wmap[g1]
            w2=self.wmap[g2]
        except TypeError:
            print('bad lookup at',[g1,g2])

        self.trips[t1][i1]=g2
        self.trips[t2][i2]=g1
        self.tmap[g1]=[t2,i2]
        self.tmap[g2]=[t1,i1]
#        self.tmap.update({g2:[t1,i1]})
        return 0

    def swap3(self,g1,g2,g3):
        """
        Swap two gift locations for their
        order in the chain.
        """
        try:
            t1,i1=self.tmap[g1]
            t2,i2=self.tmap[g2]
            t3,i3=self.tmap[g3]
        except TypeError:
            print('bad lookup at',g1,g2,g3)

        self.trips[t1][i1]=g3
        self.trips[t2][i2]=g1
        self.trips[t3][i3]=g2

#        self.trips[t1].remove(g1)
#        self.trips[t2].remove(g2)
#        self.trips[t3].remove(g3)
#        self.trips[t1].insert(i1,g3)
#        self.trips[t2].insert(i2,g1)
#        self.trips[t3].insert(i3,g2)
        self.tmap[g1]=[t2,i2]
        self.tmap[g2]=[t3,i3]
        self.tmap[g3]=[t1,i1]
#        self.tmap.update({g1:[t2,i2]})
#        self.tmap.update({g2:[t3,i3]})
#        self.tmap.update({g3:[t1,i1]})
        return 0

    def burn_swap(self,m0,Temp):
        mu=np.zeros(m0)
        for i in np.arange(m0):
            mu[i]=self.propose_swap(Temp)
        return np.std(mu/Temp)

    def burn_swap3(self,m0,Temp):
        mu=np.zeros(m0)
        for i in np.arange(m0):
            mu[i]=self.propose_swap3(Temp)
        return np.std(mu/Temp)

    def burn_merge(self,m0,Temp):
        mu=np.zeros(m0)
        for i in np.arange(m0):
            mu[i]=self.propose_merge(Temp)
        return np.std(mu/Temp)

    def propose_swap(self,Temp):
        g1,g2=np.random.choice(self.gifts,2)
        try:
            t1,i1=self.tmap[g1]
            t2,i2=self.tmap[g2]
        except TypeError:
            print('bad lookup at',[g1,g2])

        unique_trips= list(set([t1,t2]))
        d1=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])
#        d1=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        self.swap(g1,g2)
        try:
            d2=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])
#            d2=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        except ValueError:
            # infeasible swap
            self.swap(g1,g2)
            return 0.
        if (d2 == np.inf):
            self.swap(g1,g2)
            return 0.

        # If route improvement, accept the move
        if (d2 < d1):
            return d2-d1
        else:
            prob = np.exp((d1-d2)/Temp)
            sample = np.random.rand()
            # Accept move with probability e^(-delta/T)
            if (sample < prob):
                return d2-d1
            # Reject move with probability 1-e^(-delta/T)
            else:
                self.swap(g1,g2)
                return 0.0

    def propose_swap3(self,Temp):
        g1,g2,g3=np.random.choice(self.gifts,3)
        try:
            t1,i1=self.tmap[g1]
            t2,i2=self.tmap[g2]
            t3,i3=self.tmap[g3]
        except TypeError:
            print('bad lookup at',[g1,g2,g3])

        unique_trips= list(set([t1,t2,t3]))
        d1=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])
#        d1=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        self.swap3(g1,g2,g3)
        try:
            d2=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])
#            d2=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        except ValueError:
            # infeasible swap
            self.swap3(g1,g2,g3)
            self.swap3(g1,g2,g3)
            return 0.
        if (d2 == np.inf):
            self.swap3(g1,g2,g3)
            self.swap3(g1,g2,g3)
            return 0.

        # If route improvement, accept the move
        if (d2 < d1):
            return d2-d1
        else:
            prob = np.exp((d1-d2)/Temp)
            sample = np.random.rand()
            # Accept move with probability e^(-delta/T)
            if (sample < prob):
                return d2-d1
            # Reject move with probability 1-e^(-delta/T)
            else:
                self.swap3(g1,g2,g3)
                self.swap3(g1,g2,g3)
                return 0.0

    def reindex_trip(self,t):
        for i in np.arange(len(self.trips[t])):
            g=self.trips[t][i]
            self.tmap[g]=[t,i]

    def propose_merge(self,Temp):
        g1,g2=np.random.choice(self.gifts,2)
        try:
            t1,i1=self.tmap[g1]
            t2,i2=self.tmap[g2]
        except TypeError:
            print('bad lookup at',[g1,g2])

        # Calculate initial weariness
        unique_trips= list(set([t1,t2]))
        d1=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])

        # set g2 as neighbor of g1
        self.trips[t2].remove(g2)
        self.trips[t1].insert(i1,g2)

        # Calculate new weariness
        try:
            d2=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])
        except ValueError:
            # if infeasible swap,undo
            self.trips[t1].remove(g2)
            self.trips[t2].insert(i2,g2)
            return 0.
        if (d2 == np.inf):
            self.trips[t1].remove(g2)
            self.trips[t2].insert(i2,g2)
            return 0.

        # If route improvement, accept the move and change lookuptable
        if (d2 < d1):
            self.reindex_trip(t1)
            self.reindex_trip(t2)
            return d2-d1
        else:
            prob = np.exp((d1-d2)/Temp)
            sample = np.random.rand()
            # Accept move with probability e^(-delta/T)
            # update lookuptable
            if (sample < prob):
                self.reindex_trip(t1)
                self.reindex_trip(t2)
#                self.tmap.update({g2:[t1,i1]})
                return d2-d1
            # Reject move with probability 1-e^(-delta/T)
            else:
                self.trips[t1].remove(g2)
                self.trips[t2].insert(i2,g2)
                return 0.0

    def write_submission(self,filename):
        with open(filename, 'w') as f:
            f.write('GiftId,TripId\n')
            for id in np.arange(len(self.trips)):
                for g in self.trips[id]:
                    f.writelines(str(g)+','+str(id)+'\n')
        

    def read_submission(self,filename):
        sub=pd.read_csv(filename)
        tripIds=sub['TripId'].unique()
        self.trips=[[] for t in tripIds]
        self.trips=[sub[sub['TripId']==t]['GiftId'].tolist() for t in tripIds]
        msg=[[self.tmap.update({gift:[i,self.trips[i].index(gift)]}) for gift in self.trips[i]] for i in np.arange(len(self.trips))]
        #[self.tmap.update({sub.GiftId[i]:[sub.TripId[i],]}) for i in np.arange(len(sub))]

    def check_tmap(self):
        for g in self.gifts:
#            print(g,self.tmap[g])
            t,i=self.tmap[g]
            if (self.trips[t][i]!=g):
                raise ValueError('bad look up at for gift '+str(g)+' at trip,index '+str(t)+','+str(i))
