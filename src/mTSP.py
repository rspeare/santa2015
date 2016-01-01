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

#        self.distMatrix=np.zeros((len(self.gifts)+1,len(self.gifts)+1)).astype(np.float32)
#        self.init_dist_matrix()
        # Sleigh Weight and North Pole Location, set identified as (-1)
        self.north_pole=[90.,0.]
        self.sleigh_weight=10.
        self.limit=1000.
        self.dmap.update({-1:self.north_pole})
        self.wmap.update({-1:self.sleigh_weight})

    def init_dist_matrix(self):
        print('initializing distance matrix...')
        for g1 in self.gifts:
            for g2 in self.gifts:
                self.distMatrix[g1,g2]=haversine(self.dmap[g1],self.dmap[g2])
        for g in self.gifts:
            self.distMatrix[0,g]=haversine(self.north_pole,self.dmap[g])
            self.distMatrix[g,0]=self.distMatrix[0,g]

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

    def init_trips_by_longitude(self):
        """
        Group trips by longitude. (Sorting initial array and assigning.
        """
        df=pd.read_csv('../data/gifts.csv')
        dfs=df.sort_values(by='Longitude')
        self.trips=[]
        t=[]
        load=0
        j=0
        for i in np.arange(len(dfs)):
            load+=dfs.Weight.values[i]
            if (load < self.limit-self.sleigh_weight):
                t.append(dfs.GiftId.values[i])
            else:
                # If trip fully loaded, finish out and start
                # new one
                self.trips.append(t)
                self.reindex_trip(j)
                j+=1
                t=[]
                load=dfs.Weight.values[i]
                t.append(dfs.GiftId.values[i])
        t.append(dfs.GiftId.values[i])
        self.trips.append(t)
        self.reindex_trip(j)

        self.check_loads()
        self.check_tmap()
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
        prev=-1# start at the north_pole
#        prev=0# start at the north_pole #for matrix type
        load=self.wmap[prev] #sleigh_weight

        for g in gifts[::-1]:
            next=g
            dist+=load*self.dist(prev,next)
#            dist+=load*self.distMatrix[prev,next]
            load+=self.wmap[next]
            prev=next

        dist+=load*self.dist(next,-1)
#        dist+=load*self.distMatrix[next,0]
        return dist
   
    def loss(self):
        wrw=0.
        for i in np.arange(len(self.trips)):
            wrw+=self.trip_weariness(self.trips[i],i)
        return wrw


    def propose_Kswap(self,K,Temp,giftset):
        gifts=list(np.random.choice(giftset,2))
        try:
            index=[self.tmap[g] for g in gifts]
        except TypeError:
            print('bad lookup in Kswap')
            
        unique_trips= list(set([i[0] for i in index]))
        d1=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])
#        d1=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        self.Kswap(gifts,1)
        try:
            d2=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])
#            d2=self.trip_weariness(self.trips[t1],t1)+self.trip_weariness(self.trips[t2],t2)
        except ValueError:
            # infeasible swap
            self.Kswap(gifts,-1)
            return 0.
        if (d2 == np.inf):
            self.Kswap(gifts,-1)
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
                self.Kswap(gifts,-1)
                return 0.0 
    def Kswap(self,gifts,direction):
        """
        Swap K gift locations for their
        order in the chain.
        """
        try:
            index=[self.tmap[g] for g in gifts]
        except TypeError:
            print('bad lookup in Kswap')
            
        # Perform a cyclic permutation of the list of 
        # giftId's gifts, and update their location in the
        # lookup table
#        print(gifts)
        if (direction > 0):
            last=gifts.pop()
            gifts.insert(0,last)
        else:
            first=gifts[0]
            gifts.remove(first)
            gifts.append(first)
#        print(gifts)
        for i,g in list(zip(index,gifts)):
#            print(i,g)
            self.trips[i[0]][i[1]]=g
            self.tmap[g]=[i[0],i[1]]

        return 
    def burn_Kswap(self,k,m0,Temp,giftset):
        mu=np.zeros(m0)
        for i in np.arange(m0):
            mu[i]=self.propose_Kswap(k,Temp,giftset)
        return np.std(mu/Temp)

    def burn_join(self,m0,Temp,trials):
        mu=np.zeros(m0)
        for i in np.arange(m0):
            mu[i]=self.propose_join(Temp,trials)
        return np.std(mu/Temp)

    def burn_split(self,m0,Temp):
        mu=np.zeros(m0)
        for i in np.arange(m0):
            mu[i]=self.propose_split(Temp)
        return np.std(mu/Temp)

    def burn_merge(self,m0,Temp):
        mu=np.zeros(m0)
        
        for i in np.arange(m0):
            lengths=np.array([len(t) for t in self.trips])
            sindex=np.argsort(lengths)
            j=0
            while(len(self.trips[sindex[j]])==0):
                j+=1
#            print(self.trips[sindex[j]])
            mu[i]=self.propose_merge(Temp,self.trips[sindex[j]]+self.trips[sindex[j+1]]+self.trips[sindex[j+2]])
        return np.std(mu/Temp)

    def propose_split(self,Temp):
        g1=np.random.choice(self.gifts,1)[0]
#        print(g1)
#        print(self.tmap[g1])
        handicap=10**6
        try:
            t1,i1=self.tmap[g1]
        except:
            print('bad lookup at', g1)
        d1=self.trip_weariness(self.trips[t1],t1)

        trip1=self.trips[t1][:i1]
        trip2=self.trips[t1][i1:]
        if (len(trip1)+len(trip2) != len(self.trips[t1])):
            raise ValueError('error, l1+l2 != l during split!')

        d2=self.trip_weariness(trip1,t1)+self.trip_weariness(trip2,t1)+handicap

        if (d2 < d1 ):
            self.trips[t1]=trip1
            self.trips.append(trip2)
#            print('reindexing list',t1)
            self.reindex_trip(t1)
#            print('reindexing list',len(self.trips)-1)
            self.reindex_trip(len(self.trips)-1)
            return d2-d1
        else:
            prob = np.exp((d1-d2)/Temp)
            sample = np.random.rand()
            # Accept move with probability e^(-delta/T)
            # update lookuptable
            if (sample < prob):
                self.trips[t1]=trip1
                self.trips.append(trip2)
                self.reindex_trip(t1)
                self.reindex_trip(len(self.trips)-1)
                return d2-d1
            # Reject move with probability 1-e^(-delta/T)
            else:
                return 0.0

    def propose_join(self,Temp,trials):
        g1,g2=np.random.choice(self.gifts,2)
#        print(g1)
#        print(self.tmap[g1])
        try:
            t1,i1=self.tmap[g1]
            t2,i2=self.tmap[g2]
        except:
            print('bad lookup at', [g1,g2])
        unique_trips= list(set([t1,t2]))
        # Already joined
        if (len(unique_trips)==1):
            return 0.0

        # Check to see of loads can be added
        load1=self.check_trip_load(self.trips[t1],t1)
        load2=self.check_trip_load(self.trips[t2],t2)
        if (load1+load2 > self.limit - self.sleigh_weight):
            return 0.0

        # If valid join, calculate pre-join distance
        d1=sum([self.trip_weariness(self.trips[t],t) for t in unique_trips])

        # Append trips
        d2=np.inf
        bi=0
        for i in np.arange(len(self.trips[t1])):
            test=self.trip_weariness(self.trips[t1][:i]+self.trips[t2]+self.trips[t1][i:],t1)
            if (test < d2):
                d2=test
                bi=i # best index for insertion

        # Try 10,000 random trials, to see of the joined trip can be made cheaper
        total_trip=self.trips[t1]+self.trips[t2]
        for i in np.arange(trials):
            np.random.shuffle(total_trip)
            test=self.trip_weariness(total_trip,-1)
            if (test < d2):
                bi=-1
                d2=test
                best_trip=total_trip

        if (d2 < d1):
            if (bi < 0):
                self.trips[t1]=best_trip
            else:
                self.trips[t1]=self.trips[t1][:bi]+self.trips[t2]+self.trips[t1][bi:]
            self.trips[t2]=[]
#            print('reindexing list',t1)
            self.reindex_trip(t1)
#            print('reindexing list',len(self.trips)-1)
            self.reindex_trip(t2)
            return d2-d1
        else:
            prob = np.exp((d1-d2)/Temp)
            sample = np.random.rand()
            # Accept move with probability e^(-delta/T)
            # update lookuptable
            if (sample < prob):
                if (bi < 0):
                    self.trips[t1]=best_trip
                else:
                    self.trips[t1]=self.trips[t1][:bi]+self.trips[t2]+self.trips[t1][bi:]
                self.trips[t2]=[]
            #            print('reindexing list',t1)
                self.reindex_trip(t1)
            #            print('reindexing list',len(self.trips)-1)
                self.reindex_trip(t2)
                return d2-d1
            # Reject move with probability 1-e^(-delta/T)
            else:
                return 0.0                
               
    def reorder_trip(self,t):
        trip=self.trips[t]
        best=self.trip_weariness(trip,0)
        load1=sum([self.wmap[g] for g in self.trips[t]])
        for i in np.arange(len(self.trips[t])):
#            print(i)
            newtrip=np.concatenate([trip[i:],trip[:i]])
            newscore=self.trip_weariness(newtrip,0)
            if (newscore < best):
#                print('trip '+str(t)+' improved!',best-newscore)
                self.trips[t]=list(newtrip)
                best=newscore
        load2=sum([self.wmap[g] for g in self.trips[t]])
        if (not np.isclose(load2,load1)):
            raise ValueError("loads are unequal after reordering. Trip: "+str(t)+"load1 "+str(load1)+"load2 "+str(load2))
        self.reindex_trip(t)

    def reorder_all_trips(self):
        for t in np.arange(len(self.trips)):
            self.reorder_trip(t)
            
    def diff(self,a, b):
        """
        Compute the difference between two lists:
        
        A = [1,2,3,4]
        B = [2,5]

        A - B = [1,3,4]
        B - A = [5]
        """
        b = set(b)
        return [aa for aa in a if aa not in b]  
    def reindex_trip(self,t):
        for i in np.arange(len(self.trips[t])):
            g=self.trips[t][i]
            self.tmap[g]=[t,i]
    def index_trips(self):
        self.tmap={}
        print('indexing now...')
        for t in np.arange(len(self.trips)):
#            self.reindex_trip(t)
            for i in np.arange(len(self.trips[t])):
                g=self.trips[t][i]
                self.tmap.update({g:[t,i]})

    def propose_merge(self,Temp,gifts):
        """
        Given a set of gifts, propose to merge one of them 
        into the rest of the total gift set. 
        """
        g1=np.random.choice(gifts,1)[0]
        g2=np.random.choice(self.diff(self.gifts,gifts),1)[0]
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
        import pandas as pd
        sub=pd.read_csv(filename)
        tripIds=sub.TripId.unique()
        self.trips=[[] for t in tripIds]
        self.trips=[sub[sub['TripId']==t]['GiftId'].tolist() for t in tripIds]
        print('getting ready to index trips')
        self.index_trips()

    def check_tmap(self):
        for g in self.gifts:
            try:
                t,i=self.tmap[g]
            except:
                raise ValueError('bad look up atg= ',g)
            if (self.trips[t][i]!=g):
                raise ValueError('bad look up at for gift '+str(g)+' at trip,index '+str(t)+','+str(i))

    def anneal(self,numtrips,t,ms,mj,alpha,trials):
        """
        anneal(N,T,m)

        N is the numbe of trips, the length of list of lists 
        associated with the distribution schedule.

        T is initial temperature.

        ms is the number of swap-merge iterations per temperature (needed
        to get to equilibrium).

        mj is the number of join split proposals per T. typically should be small

        alpha is the cooling schedule, T --> T*alpha, when equilibrium is reached.

        trials is the number of random permutations to try when
        attempting to join two trips. Essentially controls the strength of the
        persuasion to join trips.
        """
        i=0
        while(True):
            print('split var',self.burn_split(mj,t))
            var2=self.burn_Kswap(2,ms,t)
            var=self.burn_merge(ms,t)
            print('join var',self.burn_join(mj,t,trials))
            print('score: '+str(self.loss())+' var2: '+str(var2))
            self.check_tmap()
            self.check_loads()
            if (var2<3.0):
                t*=alpha
                print('Log(T)',np.log(t)/np.log(10))
            i+=1
            if (i % 10 ==0):
                print('writing submission file')
                # read, write operation allows for clean up with 
                # growing number of trips
                self.write_submission('../data/'+str(numtrips)+'trips.csv')
                self.read_submission('../data/'+str(numtrips)+'trips.csv')

