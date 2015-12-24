import numpy as np
import pandas as pd
import mTSP

NN=1477
m=mTSP.mission(20)
m.init_small_trips(NN)
m.check_loads()
m.check_tmap()

def anneal(numtrips,t,m3):
    i=0
    while(True):
        var2=m.burn_swap(m3,t)
        var=m.burn_merge(m3,t)
        print('score: '+str(m.loss())+' var2: '+str(var2))
        m.check_tmap()
        m.check_loads()
        if (var2<3.0):
            t*=.99
            print('Log(T)',np.log(t)/np.log(10))
        if (i%10 ==0):
            m.write_submission('../data/'+str(numtrips)+'trips.csv')
        i+=1

