# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:40:07 2019

@author: power
"""


import csv
import glob
import os


print('version??? I mean folder name.')
version = str(input())

print('OK! Let\'s process '+version)

#4ms is not always true Don't care about the number

os.chdir('C:/Users/power/Desktop/python/3dplot/'+version)

namelist = []
#namelist = glob.glob('./WHFarea*.csv')
namelist = glob.glob('./3D*.csv')     
        

k = len(version) + 5
points = []

for name in namelist:

    TIME = float(name[k:-4])*1000
    with open(name, newline='') as name:

        reader = csv.reader(name)
        for row in reader:
            if 'F' in row:continue
            y = float(row[17])
            x = float(row[27])
            if x > 950:break
            points.append([x,y,TIME])

os.chdir('C:/Users/power/Desktop/python/3dplot')

with open('DataPV3_'+version+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['Distance','Velocity in x direction ','Time    ns'])
        for point in points:
            spamwriter.writerow(point)
            
print('It\'s done!')
