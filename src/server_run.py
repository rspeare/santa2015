import numpy as np
import pandas as pd
import mTSP
fname='../data/local_search_owl_70_1.2523_v2.csv'
m=mTSP.mission(10)
print('reading submission')
m.read_submission(fname)
print('checking loads and dictionary')
m.check_loads()
m.check_tmap()

t=10**3
m0=10**4
i=0
while(True):
    i+=1
    print('burning Kswaps...')
    var=m.burn_Kswap(4,m0,t,m.gifts)
    var=m.burn_Kswap(3,m0,t,m.gifts)
    print('burning merges...')
    m.burn_merge(m0,t)
    print('burning splits...')
    m.burn_split(m0,t)
    print('reordering trips...')
    m.reorder_all_trips()
    print('checking things...')
    m.check_tmap()
    m.check_loads()
    print('score : '+str(m.loss())+' temp: '+str(t)+' var '+str(var))
    if (var < 3.0):
        t*=0.99
    if (i %10 ==0):
        print('writing to file...')
        m.write_submission(fname)
        m.read_submission(fname)
        print('number of trips ',len(m.trips))

