# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:23:24 2019

@author: power

Calc ave of some files.
It has to be one csv file for each version


Change file name for [glob] and [output csv].

Don't forget to adjust amount of arrays!!
check your source and define adequate num



"""


import os
import csv
import glob

class Version:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []
        self.x5 = []
        
        self.x0 = []
        self.x6 = []

        
        
        #define your file name form here

'''main'''

print('case?')
#case=str(input())  
case = 'WHF_area'
os.chdir("c:\\Users\\power\\Desktop\\python\\"+case)
cdic = {'xtip':'xtip_','WHF_area':'WHFarea','shapeWall':'RAVE','P_SCH':'SCHMASS','p_dpdt':'p_fuel','allWHF':'log_allWHF'}
print('version?')
version = str(input())
namelist = []
namelist = glob.glob('./*'+cdic[case]+'*'+version+'*.csv')
#LES_2017after内のWallHF_ave_time~~~.csvを読み出したい
#それはやめとく　GP4と他を区別するのが面倒
#こうやって、そのフォルダ内のｃｓｖを読み出すのは共通の機能にしておく
'''

'''
#print(namelist)
length = []
namet=[]
for name in namelist:
    namet.append(name[2:])
#print(namet)

trimedname = map(lambda x : x[2:], namelist)
#print(list(trimedname))
width = 0.5000000E-04
inslist =[]
for name in namet:
    labelname = name[19:-4]
    ins = Version(name,labelname)
    print(ins.name)
    with open(name, newline='') as name:

        reader = csv.reader(name)
        readerlist = list(reader)
        readerlist = readerlist[1:]
        
        for row in readerlist:
            ins.x0.append(float(row[0]))
            ins.x1.append(float(row[1]))
            ins.x2.append(float(row[2]))
            ins.x3.append(float(row[3]))
            ins.x4.append(float(row[4]))
            
            
#            ins.x6.append(float(row[6]))
#            ins.x7.append(float(row[7]))
            
        
#        print(ins.label)
        #print(ins.x)
#        print(ins.y)
#        print(ins.cumulative)
        inslist.append(ins)

length=[]
for ins in inslist:
    length.append(len(ins.x1))
bottom = min(length)
limit = max(length)

avex1 = [0 for x in range(limit)]
avex2 = [0 for x in range(limit)]
avex3 = [0 for x in range(limit)]
avex4 = [0 for x in range(limit)]
avex0 = [0 for x in range(limit)]
for ins in inslist:
    for i,x1 in enumerate(ins.x1):
        avex1[i] += x1
    for i,x2 in enumerate(ins.x2):
        avex2[i] += x2
    for i,x3 in enumerate(ins.x3):
        avex3[i] += x3
    for i,x4 in enumerate(ins.x4):
        avex4[i] += x4
    for i,x0 in enumerate(ins.x0):
        avex0[i] += x0
for i in range(0,len(avex1)):
    
    avex1[i] = avex1[i]/len(inslist)
    avex2[i] = avex2[i]/len(inslist)
    avex3[i] = avex3[i]/len(inslist)
    avex4[i] = avex4[i]/len(inslist)
    avex0[i] = avex0[i]/len(inslist)   
    
cdic2 = {'xtip':'xtip_','WHF_area':'WHFarea','shapeWall':'RAVE','P_SCH':'SCHMASS','p_dpdt':'p_fuel_average_','allWHF':'log_allWHF_average_'}
    

with open(cdic2[case]+version+'_ave.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['avex0','avex1','avex2','avex3','avex5'])
    for i in range(bottom):
        spamwriter.writerow([avex0[i],avex1[i],avex2[i],avex3[i],avex4[i]])

print('Done!')

