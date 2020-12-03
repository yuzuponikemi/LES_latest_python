#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:39:20 2018

@author: fiction
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import glob



class Version:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.x = []
        self.y = []
        self.slope = []
        self.smoothslope = []
    def calc_burntime(self,start=5,end=95):
        k = 0
        while k < len(self.smoothslope):
            if self.smoothslope[k] < 0:
                self.smoothslope[k] = 0
            k = k + 1
        
        whole_dpdt = sum(self.smoothslope)
#        print(whole_dpdt)
        cumultive = 0
        cumultive2 = 0
        i = 0
        j = 0
        while cumultive < whole_dpdt*start/100:
            cumultive = cumultive + self.smoothslope[i]
            i = i + 1
        self.starttime = self.x[i]
        print('starttime='+str(self.starttime))
        while cumultive2 < whole_dpdt*end/100:
            cumultive2 = cumultive2 + self.smoothslope[j]
            j += 1
        self.endtime = self.x[j]
        print('endtime='+str(self.endtime))
        self.burntime = self.endtime - self.starttime
        print('burntime='+str(self.burntime))
              
        
        
        
        

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
length=[]
for name in namet:
    labelname = name[15:-4]
    ins = Version(name,labelname)
    
    with open(name, newline='') as name:

        reader = csv.reader(name)
        readerlist = list(reader)
        readerlist = readerlist[1:]
        
        for row in readerlist:
            ins.x.append(float(row[1]))
            ins.y.append(float(row[3]))
            
            
#        print(ins.name)
#        print(ins.label)
#        print(ins.x)
#        print(ins.y)
#        print(ins.cumulative)
        inslist.append(ins)
        length.append(len(ins.y))


for ins in inslist:
    for i in range(len(ins.y)-1):
        ins.slope.append(float(ins.y[i+1]-ins.y[i])/width)
    ins.slope.append(ins.slope[len(ins.y)-2])

#param_aveは移動平均近似のパラメータ
param_ave = 501
half_param_ave = int((param_ave - 1)/2)

c = np.array(1)

#slopeの移動平均近似を計算
for ins in inslist:
    a = np.array(list(ins.slope))
    b = np.ones(param_ave)/float(param_ave)
    c = np.convolve(a,b,'valid')
    ins.smoothslope = c.tolist()
       
  

for ins in inslist:
    print('--------------')
    print(ins.label)
    ins.calc_burntime(5,95)
    


with open('burn_duration.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['version','starttime','endtime','burnduration'])
    for ins in inslist:
        spamwriter.writerow([ins.label,ins.starttime,ins.endtime,ins.burntime])

with open('alldpdt.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+[ins.label for ins in inslist])
    for i in range(min(length))[half_param_ave:-half_param_ave-20000]:
        spamwriter.writerow([i*5*10**(-5)]+[ins.smoothslope[i] for ins in inslist])


'''plot'''    
markers1 = ["*", ",", "o", "v", "^", "<", ">", "D", "d", ]#marker = markers1[num]
plt.figure(figsize=(12,12))
ax1 = plt.subplot(211)
ax1.set_xlim(0,4)
ax1.set_ylim(7.0,7.4)
#plt.title("p_ambient", fontsize=30)
for num,ins in enumerate(inslist):
    plt.plot(ins.x,ins.y,label=ins.label)
#plt.xlabel('time(ms)')
plt.ylabel('Pressure[bar]', fontsize=30)
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.legend()
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
plt.xticks( np.arange(0, 4, 0.5) )

plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.grid()
#plt.show()

ax2 = plt.subplot(212, sharex=ax1)
ax2.set_xlim(0,4.01)
ax2.set_ylim(0.0,0.401)
#plt.title("dp/dt_movingAve", fontsize=30)
for num,ins in enumerate(inslist):
    xfixed = ins.x[half_param_ave:-half_param_ave]
    plt.plot(xfixed,ins.smoothslope,label=ins.label)
plt.xlabel('time[ms]', fontsize=30)
plt.ylabel('dp/dt[bar/s]', fontsize=30)
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
#plt.grid()
plt.xticks( np.arange(0, 4.01, 0.5) )
plt.yticks(np.arange(0,0.401, 0.1))
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.show()

filename = "C:/Users/power/Desktop/python/images/p_dpdt.png"
plt.savefig(filename)


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