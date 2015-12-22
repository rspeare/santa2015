import numpy as np
import pickle
from haversine import haversine

class mission:
    def __init__(self,Ntrips):
        self.dmap=pickle.load(open("../data/xmap.pickle","rb"))
        self.wmap=pickle.load(open("../data/wmap.pickle","rb"))
        self.tmap=pickle.load(open("../data/tmap.pickle","rb"))
        self.gifts=pickle.load(open("../data/GiftIds.pickle","rb"))
        self.trips=[[] for i in np.arange(Ntrips)]
#        self.gifts=[item for sublist in self.trips for item in sublist]
#        self.gifts=[i for i in np.arange(1,len(self.dmap.keys())+1)]

        # Sleigh Weight and North Pole Location
        self.north_pole=[90.,0.]
        self.sleigh_weight=10.
        self.limit=1000.
        self.dmap.update({0:self.north_pole})
        self.wmap.update({0:self.sleigh_weight})

    def init_trips(self,size):
        """
        Initialize the list of lists, trips,
        into trips of size 'size'. The lower the size,
        the safer it is that you stumble upon a feasible
        solution with weight constraints
        """
        while (True):
            print('trying initial trips of size, ',size)
            data=[i for i in np.arange(1,len(self.gifts)+1)]
            np.random.shuffle(data)
            self.trips=[data[x:x+size] for x in np.arange(0, len(data), size)]
            msg=[[self.tmap.update({gift:[i,self.trips[i].index(gift)]}) for gift in self.trips[i]] for i in np.arange(len(self.trips))]
            break
        return self.WRW()
    
    def dist(self,g1,g2):
        """
        Compute the Haversine distance between two gifts
        """
        return haversine(self.dmap[g1],self.dmap[g2])
    
    def check_loads(self):
        for i in np.arange(len(self.trips)):
            self.check_trip_load(self.trips[i],i)
            
    def check_trip_load(self,trip,i):
        s=0
        for stop in trip:
            s+=self.wmap[stop]
            if (s > self.limit):
                raise ValueError(str(s)+' of mass in trip '+str(i)+' with packages '+str(trip)+' and weight ')
        
    def trip_weariness(self,gifts,ii):
        """
        Compute the weighted reindeer weariness
        of a trip, consisting of a sequence of Gifts.
        If the trip is infeasible, return infinity
        """
        # if empty trip return zero
        if (len(gifts)==0):
            return 0
        self.check_trip_load(gifts,ii)

#        gifts.append(0)
        load=self.sleigh_weight #self.wmap[0] # sleigh_weight
#        print('load: ',load)
        prev=0 # start at the north_pole
        weary=0.0
        for g in [0]+gifts:
            next=g
            weary+=load*self.dist(prev,next)
            load+=self.wmap[next]
            prev=next
        return weary
            
    def WRW(self):
        wrw=0.
        for i in np.arange(len(self.trips)):
            wrw+=self.trip_weariness(self.trips[i],i)
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
        self.tmap.update({g1:[t2,i2]})
        self.tmap.update({g2:[t1,i1]})
        return 0

    def burn_in(self,m0,Temp):
        mu=np.zeros(m0)
        for i in np.arange(m0):
            mu[i]=self.propose_swap(Temp)
        return np.std(mu/Temp)

    def propose_swap(self,Temp):
        g1,g2=np.random.choice(self.gifts,2)
        try:
            t1,i1=self.tmap[g1]
            t2,i2=self.tmap[g2]
        except TypeError:
            print('bad lookup at',[g1,g2])
        
        d1=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        self.swap(g1,g2)
        try:
            d2=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        except ValueError:
            # infeasible swap
            self.swap(g1,g2)
            return 0.
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
        

        

        

