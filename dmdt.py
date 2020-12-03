# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:18:08 2018

@author: power
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import math



class Version:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.x = [] #time
        self.y = [] #dm/dt

        self.injV = []

              
        
        
        
        

'''main'''
namelist = []
namelist = glob.glob('./p_fuel*.csv')
#LES_2017after内のWallHF_ave_time~~~.csvを読み出したい
#それはやめとく　GP4と他を区別するのが面倒
#こうやって、そのフォルダ内のｃｓｖを読み出すのは共通の機能にしておく
'''
p_fuel_180118a_single_L.csvといったファイル名
同じRのファイルをフォルダに突っ込む→グラフを作成
'''
#print(namelist)

namet=[]
for name in namelist:
    namet.append(name[2:])
#print(namet)

trimedname = map(lambda x : x[2:], namelist)
#print(list(trimedname))
width = 0.5000000E-04
inslist =[]
length = []
DDRPINI = 0.104*10**(-3) #width of injection hole
DENDRP = 0.684*10**3


for name in namet:
    labelname = name[15:-4]
    ins = Version(name,labelname)
    
    with open(name, newline='') as name:

        reader = csv.reader(name)
        readerlist = list(reader)
        readerlist = readerlist[1:]
        
        for row in readerlist:
            ins.x.append(float(row[1]))
            ins.y.append(float(row[4]))
            r = DDRPINI*math.sqrt(0.8)/2
            ins.injV.append(float(row[4])/(DENDRP*math.pi*r**2))

            
            
        print(ins.name)
        print(ins.label)
#        print(ins.x[:20000])
#        print(ins.y)
        print(ins.injV[:10000:500])
        inslist.append(ins)
        length.append(len(ins.y))

with open('allInjV.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+[ins.label for ins in inslist])
    for i in range(min(length)):
        spamwriter.writerow([i*5*10**(-5)]+[ins.injV[i] for ins in inslist])

'''plot'''    
plt.figure(figsize=(10,8))
plt.xlim([0,2.0])
plt.ylim([0,0.005])
#ax1 = plt.subplot(211)
#plt.title("INJECTIONRATE", fontsize=30)
for ins in inslist:
    plt.plot(ins.x[:40000],ins.y[:40000],label=ins.label)
plt.xlabel('time[ms]', fontsize=30)
plt.ylabel('dm/dt[g/s]', fontsize=30)
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.legend(fontsize = 'small',frameon = None,prop={'size':20,},bbox_to_anchor=(1.05, 1))
plt.xticks( np.arange(0, 2.5, 0.5) )
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.grid()
#plt.show()
filename = "C:/Users/power/Desktop/python/images/dmdt.png"
plt.savefig(filename,dpi = 200, bbox_inches = 'tight', pad_inches = 0.1)
            





#ax2 = plt.subplot(212, sharex=ax1)
plt.figure(figsize=(8,8))
plt.xlim([0,2.0])
plt.ylim([0,1000])
#plt.title("dp/dt_movingAve")
for ins in inslist:
    plt.plot(ins.x[:40000],ins.injV[:40000],label=ins.label)
plt.xlabel('time      ms', fontsize=30)
plt.ylabel('InjV       m/s', fontsize=30)
plt.legend(fontsize = 'small',frameon = None,prop={'size':20,},bbox_to_anchor=(1, 1))
plt.xticks( np.arange(0, 2.01, 0.5) )
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.show()

filename = "C:/Users/power/Desktop/python/images/injV.png"
plt.savefig(filename,dpi = 200, bbox_inches = 'tight', pad_inches = 0.1)

#plt.figure(figsize=(18,9))
##plt.subplot(312)
#plt.title("dp/dt")
#for ins in inslist:
#    plt.plot(ins.x,ins.slope,label=ins.label)
#plt.xlabel('time(ms)')
#plt.ylabel('dp/dt')
#plt.legend()
#plt.grid()
#plt.show()
#燃焼期間をcsvで出力