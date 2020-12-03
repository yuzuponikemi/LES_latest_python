# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:21:34 2018

@author: power
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv


       

namelist =[]
wallhf4mslist =[]
burn_durationdict ={}
wallhf4msdict ={}
#4ms is not always true Don't care about the number

with open('C:/Users/power/Desktop/python/allWHF/cumulative_wall_heat_flux_at4.0ms.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    readerlist = list(reader)
    readerlist = readerlist[1:]
        
    for row in readerlist:
        namelist.append(row[0])
        wallhf4mslist.append(row[1])
        name = row[0]
        wallhf4msdict[name]=row[1]
      
with open('C:/Users/power/Desktop/python/p_dpdt/burn_duration.csv', 'r', newline='') as fg:
    reader = csv.reader(fg)
    readerlist = list(reader)
    readerlist = readerlist[1:]
    burn_durations= []
    for row in readerlist:
        
        burn_durations.append(float(row[3]))
        name =row[0]
        burn_durationdict[name]=row[3]
        
#'''plot'''    --mistake cannot match right label and number
#plt.figure(figsize=(12,9))
##plt.title("p_ambient")
#i = 0
#while i < len(namelist):
#    plt.plot(burn_durations[i],float(wallhf4mslist[i]),'o',label=namelist[i])
#    i +=1
#plt.xlabel('burnDuration')
#plt.ylabel('HeatLoss')
##plt.setp(ax1.get_xticklabels(), visible=False)
#
#plt.legend()
#plt.grid()
#plt.show()

xmax = max(burn_durations)
xmin = min(burn_durations)
    
#colorlist = ['#008000', '#DC143C', '#0000FF', '#000000', '#00FFFF', '#A52A2A', '#8A2BE2', '#808080','#808000','#FF1493']
colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
'''plot'''    
plt.figure(figsize=(12,9))

#plt.title("HeatLossVSburnDuration", fontsize=30)
#plt.title("p_ambient")
for key in wallhf4msdict.keys():
    plt.plot(float(burn_durationdict[key]),float(wallhf4msdict[key]),'o',label=key,markersize=15)
plt.xlabel('burnDuration    ms ', fontsize=30)
plt.ylabel('HeatLoss        MW', fontsize=30)
#plt.setp(ax1.get_xticklabels(), visible=False)

plt.legend(prop={'size':20,})
#plt.xticks([1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8])
plt.tick_params(labelsize = 20,direction = 'in',length = 7)


#plt.show()

filename = "C:/Users/power/Desktop/python/images/allWHFvsBurnDuration.png"
plt.savefig(filename)


#bd = ['burn_duration']
#wh = ['HeatLoss']
#colums = ['versions']
with open('wallHF_allvsBurnDuration.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['version','HeatLoss','burnduration'])
    for key in wallhf4msdict.keys():
        spamwriter.writerow([key,wallhf4msdict[key],burn_durationdict[key]])
for key in wallhf4msdict.keys():
        print([key,wallhf4msdict[key],burn_durationdict[key]])
        
#
#for key in wallhf4msdict.keys():
#    bd.append(burn_durationdict[key])
#    wh.append(float(wallhf4msdict[key]))
#    colums.append(key)
#df = pd.DataFrame([bd,wh],columns = colums)
#
#
#fig, ax = plt.subplots(1, 1)
#plotting.table(ax, df, loc='center')
#ax.axis('off')
#filename = "C:/Users/power/Desktop/python/images/wallHF_allvsBurnDurationTABLE.png"
#plt.savefig(filename,dpi=200)