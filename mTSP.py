import numpy as np
import pickle
from haversine import haversine
#import pickle
#dmap={all_trips.iloc[i].GiftId: [all_trips.iloc[i].Latitude,all_trips.iloc[i].Longitude] for i in np.arange(len(all_trips))}
#wmap={all_trips.iloc[i].GiftId: all_trips.iloc[i].Weight for i in np.arange(len(all_trips))}
#pickle.dump(dmap,open("dmap.pickle","wb"))
#pickle.dump(wmap,open("wmap.pickle","wb"))

class mission:
    def __init__(self,size):
        self.dmap=pickle.load(open("dmap.pickle","rb"))
        self.wmap=pickle.load(open("wmap.pickle","rb"))
        self.tmap=pickle.load(open("tmap.pickle","rb"))
        self.trips=pickle.load(open("trips0.pickle","rb"))

        self.gifts = [item for sublist in self.trips for item in sublist]

#        self.trips=[[] for i in np.arange(Ntrips)]
        self.north_pole=[90.,0.]
        self.sleigh_weight=10.
        self.limit=1000.
        self.dmap.update({0:self.north_pole})
        self.wmap.update({0:self.sleigh_weight})

    def init_trips(self,size):
        data=[i for i in np.arange(100000)]
        np.random.shuffle(data)
        self.trips=[data[x:x+size] for x in np.arange(0, len(data), size)]
        msg=[[self.tmap.update({gift:[i,self.trips[i].index(gift)]}) for gift in self.trips[i]] for i in np.arange(len(self.trips))]
        return self.WRW()

    def dist(self,g1,g2):
        """
        Compute the Haversine distance between two gifts
        """
        return haversine(self.dmap[g1],self.dmap[g2])

    def trip_weariness(self,gifts):
        """
        Compute the weighted reindeer weariness
        of a trip, consisting of a sequence of Gifts.
        If the trip is infeasible, return infinity
        """
        # if empty trip return zero
        if (len(gifts)==0):
            return 0.

        # else, go through the trip backwards
        # adding to santas sack as you
        gifts.append(0)
        load=self.sleigh_weight #self.wmap[0] # sleigh_weight
#        print('load: ',load)
        prev=0 # start at the north_pole
        weary=0.0
        for g in gifts:
            next=g
            weary+=load*self.dist(prev,next)
            load+=self.wmap[next]
            prev=next
        # Check if trip was infeasible for reindeer
        if (load > self.limit):
            print('WARNING! ILLEGAL SLEIGH WEIGHT')
            print('load: ',load)
            weary=np.inf
        return weary
            
    def WRW(self):
        wrw=0.
        for trip in self.trips:
            wrw+=self.trip_weariness(trip)
        return wrw

    def swap(self,g1,g2):
        """
        Swap two gift locations for their
        order in the chain.
        """
        [t1,i1]=self.tmap[g1]
        [t2,i2]=self.tmap[g2]
        
        print('g,t,i,w: ',[g1,t1,i1],[g2,t2,i2])
        w1=self.wmap[g1]
        w2=self.wmap[g2]

        self.trips[t1][i1]=g2
        self.trips[t2][i2]=g1
        self.tmap.update({g1:[t2,i2]})
        self.tmap.update({g2:[t1,i1]})
        return 0

    def propose_swap(self,Temp):
        g1,g2=np.random.choice(self.gifts,2)
        [t1,i1]=self.tmap[g1]
        [t2,i2]=self.tmap[g2]

        d1=self.trip_weariness(self.trips[t1])+self.trip_weariness(self.trips[t2])
        self.swap(g1,g2)
        d2=self.trip_weariness(self.trips[t1])+self.trip_weariness(self.trips[t2])
        if (d2 == np.inf):
            self.swap(g1,g2)
            return 0.
#        d2=self.WRW()
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
        

        

        


