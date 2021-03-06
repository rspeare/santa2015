import numpy as np
import pandas as pd
import pickle
import sys
ver=sys.version[:3]
print("creating Weight and Position Dictionaries wmap, xmap...")
df = pd.read_csv('data/sample_submission.csv')
dfG = pd.read_csv('data/gifts.csv')
xmap={}
res=[xmap.update({dfG.GiftId[i]:[dfG.Latitude[i],dfG.Longitude[i]]}) for i in np.arange(len(dfG))]
wmap={}
res=[wmap.update({dfG.GiftId[i]:dfG.Weight[i]}) for i in np.arange(len(dfG))]
tmap={}
res= [tmap.update({df.GiftId[i]:df.TripId[i]}) for i in np.arange(len(df))]
pickle.dump(xmap,open("data/xmap"+ver+".pickle","wb"))
pickle.dump(wmap,open("data/wmap"+ver+".pickle","wb"))
pickle.dump(tmap,open("data/tmap"+ver+".pickle","wb"))
pickle.dump(dfG.GiftId,open("data/GiftIds"+ver+".pickle","wb"))
