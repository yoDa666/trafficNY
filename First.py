#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:52:11 2021

@author: johanna
"""

import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians

import seaborn as sns
plt.style.use('seaborn')

import numpy as np
from scipy.stats.mstats import mquantiles
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


#read csv file
df = pd.read_csv('/Users/johanna/Desktop/creditshelf/data/Motor_Vehicle_Collisions_-_Crashes.csv', low_memory=False)

#transform column titles
df.columns = [c.replace(' ', '_') for c in df.columns]
print('Get information on the input data' + '\n\n')
print(df.info())

#find elements with no coordinate data (no longitude and latitude data) -> invalid data points
dfNoCoordinates = df[(df.LATITUDE == 0) & (df.LONGITUDE == 0)]

#get min and max values of longitude and latitude data to evaluate the correctness of the data
BBoxDf = (df.LONGITUDE.min(),   df.LONGITUDE.max(), df.LATITUDE.min(), df.LATITUDE.max())

#print number of unique boroughs eg how many accident data points occur in each borough, print number of invalid data points, print min and max coordinate values
print('number of unique boroughs and number of data points per borough')
print('number of invalid data points included'+ '\n\n')
print(df.BOROUGH.value_counts(dropna=False))

print('There are: ' + str(len(dfNoCoordinates)) + ' data points without lat and long data.')

print('valuation of min/max values of lat and long data')
print('there are data points with missing lat and long data' + '\n\n')
print(BBoxDf)

#drop elements with no borough information. do the same analysis for the new data set:

df = df.dropna(subset=['BOROUGH'])

print('We dropped data points without borough information and evaluate the first data points' + '\n\n')
print(df.head())

#find elements with no coordinate data (no longitude and latitude data) -> invalid data points
dfBoroughNoCoordinates = df[(df.LATITUDE == 0) & (df.LONGITUDE == 0) ]

#get min and max values of longitude and latitude data to evaluate the correctness of the data
BBoxTrimed = (df.LONGITUDE.min(),   df.LONGITUDE.max(), df.LATITUDE.min(), df.LATITUDE.max())

#print number of unique boroughs eg how many accident data points occur in each borough, print number of invalid data points, print min and max coordinate values
print('number of unique boroughs and number of data points per borough after tarnsformation of dataframe' + '\n\n' )
print(df.BOROUGH.value_counts())

print('There are still: ' + str(len(dfBoroughNoCoordinates)) + ' data points without lat and long data.')

print('valuation of min/max values of lat and long data')
print('there are data points with missing lat and long data' + '\n\n')
print(BBoxTrimed)

#for the rest of the exercise, we drop all the invalid data points. For further simplification, we drop every data point without an information on the corresponding borough


#we only look at the data with important information in regad to our problem. We drop the irrelevant data.
df = df[['CRASH_DATE','BOROUGH', 'LATITUDE', 'LONGITUDE', 'NUMBER_OF_CYCLIST_INJURED','NUMBER_OF_CYCLIST_KILLED']]

df = df[(df.LATITUDE != 0) | (df.LONGITUDE != 0)]
print('We dropped data points without lat/long information and evaluate the data points' + '\n\n')

print(df.describe())


#print unique values of injured/killed cyclists -> we evaluate if there are any invalid data points (NaN values) to replace the data. 
print(df.NUMBER_OF_CYCLIST_INJURED.value_counts(dropna=False))
print(df.NUMBER_OF_CYCLIST_KILLED.value_counts(dropna=False))

#I tried to get some information on the min/max data of the borough coordinates. This could be used to classify the data points without borough information. But a first analysis didn't show a simple and good method to do so. So I decided to drop the points for the sake of this exercise.
#See later visualization

dfM = df[df.BOROUGH == "MANHATTAN"]
dfBX = df[df.BOROUGH == "BRONX"]
dfBR= df[df.BOROUGH == "BROOKLYN"]
dfQ = df[df.BOROUGH == "QUEENS"]
dfS = df[df.BOROUGH == "STATEN ISLAND"]

BBoxM = (dfM.LONGITUDE.min(), dfM.LONGITUDE.max(), dfM.LATITUDE.min(), dfM.LATITUDE.max())
BBoxBX = (dfBX.LONGITUDE.min(), dfBX.LONGITUDE.max(), dfBX.LATITUDE.min(), dfBX.LATITUDE.max())
BBoxBR = (dfBR.LONGITUDE.min(), dfBR.LONGITUDE.max(), dfBR.LATITUDE.min(), dfBR.LATITUDE.max())
BBoxQ = (dfQ.LONGITUDE.min(), dfQ.LONGITUDE.max(), dfQ.LATITUDE.min(), dfQ.LATITUDE.max())
BBoxS = (dfS.LONGITUDE.min(), dfS.LONGITUDE.max(), dfS.LATITUDE.min(), dfS.LATITUDE.max())

'''
print(BBoxM)
print(BBoxBX)
print(BBoxBR)
print(BBoxQ)
print(BBoxS)
'''

#we create a new dataframe wih only the accidant data points, where cyclists were injured/killed. For complexity reasons this reduces the number of data points. 
df2 = df[((df.LATITUDE != 0) | (df.LONGITUDE != 0)) & (df.NUMBER_OF_CYCLIST_INJURED > 0 | (df.NUMBER_OF_CYCLIST_KILLED > 0))]

print(df2.info())

df2 = df2.dropna(how='any')

print(df2.describe())


print('First exercise: We search for the most dangerous borough.' + '\n\n' + 'For this we look at the number of accidents that occured in one borough.' + '\n\n' +  'We also look at the number of accidents that occured per km^2.' + '\n\n' + 'We analyse the data for all the accidents and the accidents where cyclists were injured/killed')


#get number of accidents per km^2 of borough
def getRelFrequency(borough,value):

    #Area of boroughs in km^2
    areaM = 59.1
    areaBR = 180
    areaBX = 110
    areaQ = 280
    areaS = 152 
    
    cases = {
        'MANHATTAN': round(value/areaM,0),
        'BROOKLYN': round(value/areaBR,0),
        'BRONX': round(value/areaBX,0),
        'QUEENS': round(value/areaQ,0),
        'STATEN ISLAND': round(value/areaS,0)
    }
    
    return cases[borough]


#get absolute number of accidents per borough 
dsBorough = df.BOROUGH.value_counts()
print(dsBorough)
dsBoroughRel = df.BOROUGH.value_counts(normalize=True)
print(dsBoroughRel)

#get absolute number of accidents per borough where cyclist injured/killed
dsBoroughCyclist = df2.BOROUGH.value_counts()
print(dsBoroughCyclist)
dsBoroughCyclistRel = df2.BOROUGH.value_counts(normalize=True)
print(dsBoroughCyclistRel)

#plot pie chart to visualize the distribution
plt.subplot(2, 1, 1)
dsBorough.plot.pie(figsize=(6, 6))
plt.ylabel('')

plt.subplot(2, 1, 2)
dsBoroughCyclist.plot.pie(figsize=(6, 6)) 
plt.ylabel('')
plt.show()

#get number of accidents per km^2 per borough    
dfFrequency = df.BOROUGH.value_counts().rename_axis('unique_values').reset_index(name='counts')
arrFrequency = dfFrequency.values
dictRelFrequency= {arrFrequency[index][0]: getRelFrequency(arrFrequency[index][0],arrFrequency[index][1]) for index in range(len(arrFrequency))}
dsFrequency = pd.Series(data=dictRelFrequency, index=['BROOKLYN', 'MANHATTAN', 'QUEENS','BRONX','STATEN ISLAND'])

#get number of accidents per km^2 per borough where cyclist injured/killed
df2Frequency = df2.BOROUGH.value_counts().rename_axis('unique_values').reset_index(name='counts')
arr2Frequency = df2Frequency.values
dictCyclistRelFrequency= {arr2Frequency[index][0]: getRelFrequency(arr2Frequency[index][0],arr2Frequency[index][1]) for index in range(len(arr2Frequency))}
dsCylistFrequency = pd.Series(data=dictCyclistRelFrequency, index=['BROOKLYN', 'MANHATTAN', 'QUEENS','BRONX','STATEN ISLAND'])

#plot pie chart to visualize the distribution
plt.subplot(2, 1, 1)
dsFrequency.plot.pie(figsize=(6, 6))
plt.ylabel('')

plt.subplot(2, 1, 2)
dsCylistFrequency.plot.pie(figsize=(6, 6)) 
plt.ylabel('')
plt.show()
   

'''
Second exercise: We want to analyse the loctions of the bike stations.
We search for the area with the highest accident density. For this exercise we concentrate us on the accidents, where a cyclist was injured/killed (only to reduce complexity/runtime)
We want to compare this area with the locations of the bike stations.
'''

# download raw json object
url = 'https://feeds.citibikenyc.com/stations/stations.json'
data = urllib.request.urlopen(url).read().decode()
obj = json.loads(data)
#transform usable data into dict
useDataDict = obj['stationBeanList']
#sample an dict entry to locate the usable information
print(useDataDict[0])


#load useable date into dict
stationDat = {'ID': [elem['id'] for elem in useDataDict],
              'TOTAL_DOCKS': [elem['totalDocks'] for elem in useDataDict],
              'STATUS': [elem['statusValue'] for elem in useDataDict],
              'LONGITUDE': [elem['longitude'] for elem in useDataDict],
              'LATITUDE' : [elem['latitude'] for elem in useDataDict]}
#load dict into pandas dataframe
dfStat = pd.DataFrame (stationDat, columns = ['ID','TOTAL_DOCKS','STATUS','LONGITUDE','LATITUDE'])

#valide how many stations are not in service to reduce the number of stations
print(dfStat.STATUS.value_counts(dropna=False))

#drop stations not in service
dfStat = dfStat[dfStat.STATUS == 'In Service']

#sample new data to validate that no invalid data points remain
print(dfStat.info())
dfStat.dropna(how='any')
print(dfStat.info())
print(dfStat.head())


fig, ax = plt.subplots(figsize = (8,7))
sns.kdeplot(df2.LONGITUDE, df2.LATITUDE, cmap="Reds", shade=True)
ax.scatter(dfStat.LONGITUDE, dfStat.LATITUDE, zorder=1, alpha= 0.5, c='b', s=2)
ax.set_title('Plotting Spatial Data BIKE Station on New York accidents heatmap')
ax.set_xlim(df2.LONGITUDE.min(),df2.LONGITUDE.max())
ax.set_ylim(df2.LATITUDE.min(),df2.LATITUDE.max())
plt.show()



'''
Exercise 3: We want to estimate the density of the occured accidents in New York to predicte, where an accident is most likely.
Again we use the accident data points, where an cyclist was injured/killed (to reduce complexity/runtime)
To estimate and predict we use an kernel density estimation
'''



# represent points consistently as (lat, lon)
df2Coordinates = df2[['LATITUDE', 'LONGITUDE']]
coords = df2Coordinates.values

#set up the parameters
#search for a fitted bandwidth
#metric haversine because we use longitude and latitude data 

grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.01, 1.0, 10)},
                    cv=5) # 20-fold cross-validation
grid.fit(coords)
print(grid.best_params_)


#best estimates bandwith 0.01 -> create density estimation model
kde = KernelDensity(metric='haversine', kernel='gaussian', algorithm='ball_tree', bandwidth=0.01).fit(coords)

#get best log scores for the test data eg the accident data points
logprobStat = kde.score_samples(coords)
 
#get a quantil to evaluate which points are isolated eg. outliiers
alpha_set = 0.95
tau_kde = mquantiles(logprobStat, 1. - alpha_set)

#get a quantil to evaluate which points create the most dense region
alpha_set2 = 0.05
tau_kde2 = mquantiles(logprobStat, 1. - alpha_set2)


outliers = np.argwhere(logprobStat < tau_kde)
outliers = outliers.flatten()
coords_outliers = coords[outliers]

normal_samples = np.argwhere((logprobStat >= tau_kde) & (logprobStat < tau_kde2))
normal_samples = normal_samples.flatten()
coords_valid = coords[normal_samples]

best_samples = np.argwhere(logprobStat >= tau_kde2)
best_samples = best_samples.flatten()
coords_best = coords[best_samples]

with plt.style.context(("seaborn", "ggplot")):
    plt.scatter(coords_valid[:, 1], coords_valid[:, 0], c="tab:blue", label="Valid Samples",s=1)
    plt.scatter(coords_outliers[:, 1], coords_outliers[:, 0], c="tab:red", label="Outliers <5%",s=1)
    plt.scatter(coords_best[:, 1], coords_best[:, 0], c="tab:green", label="Highest density >95%",s=1)
    plt.legend(loc="best")
    plt.show()


#evaluate data and get data point with highest estimated density
maxValue = max(logprobStat)
maxIndex = [i for i, j in enumerate(logprobStat) if j == maxValue]
maxCoords = coords[maxIndex][0]

print(maxCoords)

# approximate radius of earth in km
def getDistance(lon1,lat1, lon2,lat2):
    R = 6371.0088
    
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    
    #print("Result:", distance)
    return distance
    
#calculate distances to bike stations from data point with max density. Get bike station with min distance
stationDistDict={stationDat['ID'][index]: getDistance(stationDat['LONGITUDE'][index],stationDat['LATITUDE'][index],maxCoords[1],maxCoords[0]) for index in range(len(stationDat['ID']))}
minDist = min([item for item in stationDistDict.values()])
minId = [key for key in stationDistDict.keys() if stationDistDict[key] == minDist]

print([useDataDict[index] for index in range(len(useDataDict)) if useDataDict[index]['id'] == minId[0]])


'''
Extention Excercise 2

Sample station data and evaluate estimated density for this data. Station locations with highest density can maybe be relocated?
'''


'''
Generell extentions:
    
We can evalate on which day the accidents occured.
Overall we could get some generell traffic data and analyse when we have a high vehicle volume and correlate this data with our accident data.

We can also evaluate the exact time of the accidents and correlate this also with generell traffic data. 

We can evaluate weather data and data on the road texture to see if this data correlates.


Also we can look at the data I removed in my analysis and try to fill in the missing values.
Here I visualized the data points per borough (here Manhattan) to get a better overview of the accident data points with valid/not missing borough
'''
BBoxStat = (dfStat.LONGITUDE.min(),dfStat.LONGITUDE.max(),dfStat.LATITUDE.min(),dfStat.LATITUDE.max())
ny_m = plt.imread('data/map.png')

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(dfM.LONGITUDE, dfM.LATITUDE, zorder=1, alpha= 0.4, c='b', s=2)
ax.set_title('Plotting Spatial Data on NY Map')
ax.set_xlim(dfStat.LONGITUDE.min(),dfStat.LONGITUDE.max())
ax.set_ylim(dfStat.LATITUDE.min(),dfStat.LATITUDE.max())
ax.imshow(ny_m, zorder=0, extent = BBoxStat, aspect= 'equal')
