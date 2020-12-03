# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:48:23 2019

@author: power
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
        self.WHF = []
        self.smoothsWHF = []
        self.cumulativeWHF = []
        self.cumulativesmoothWHF = []

        
        
        

'''main'''

#print('version?')
#version = str(input())
#namelist = []
#namelist = glob.glob('./*log_allWHF'+'*'+version+'*.csv')

namelist = []
namelist = glob.glob('./*log_allWHF*.csv')

#LES_2017after内のWallHF_ave_time~~~.csvを読み出したい
#それはやめとく　GP4と他を区別するのが面倒
#こうやって、そのフォルダ内のｃｓｖを読み出すのは共通の機能にしておく
'''
log_allWHF_180210a_inv_little.csvといったファイル名
同じRのファイルをフォルダに突っ込む→グラフを作成
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
    cumulative = 0
    with open(name, newline='') as name:

        reader = csv.reader(name)
        readerlist = list(reader)
        readerlist = readerlist[1:]
        
        for row in readerlist:
            ins.x.append(float(row[1]))
            ins.WHF.append(float(row[2]))
            cumulative = cumulative + float(row[2])*0.05
            ins.cumulativeWHF.append(cumulative)
            
            
#        print(ins.name)
#        print(ins.label)
        #print(ins.x)
#        print(ins.y)
#        print(ins.cumulative)
        inslist.append(ins)
        length.append(len(ins.x))
        
        
#for ins in inslist:
#    for i in range(len(ins.y)-1):
#        ins.slope.append(float(ins.y[i+1]-ins.y[i])/width)
#    ins.slope.append(ins.slope[len(ins.y)-2])

#param_aveは移動平均近似のパラメータ
param_ave =    1001
half_param_ave = int((param_ave - 1)/2)

c = np.array(1)

#WHFの移動平均近似を計算
lenfix = []
for ins in inslist:
    a = np.array(list(ins.WHF))
    b = np.ones(param_ave)/float(param_ave)
    c = np.convolve(a,b,'valid')
    ins.smoothWHF = c.tolist()
    cumu = 0
    for k in ins.smoothWHF:
        cumu = cumu + k*0.05
        ins.cumulativesmoothWHF.append(cumu)
    lenfix.append(len(ins.x[half_param_ave:-half_param_ave]))


for ins in inslist:
    print(ins.label)
#    print(ins.name)
#    print(ins.x)
#    print(ins.smoothWHF)
#    print(ins.x[half_param_ave:-half_param_ave])
#    print(ins.WHF)
#    print(ins.cumulativeWHF)
#    print(ins.cumulativesmoothWHF)


#？？msにおける累積熱流束の値を冷却損失の値として保存
time = 4.0 #ms
with open('cumulative_wall_heat_flux_at'+str(time)+'ms.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['version','wallHF('+str(time)+'ms)'])
    for ins in inslist:
        timeindex = ins.x.index(time)
        spamwriter.writerow([ins.label,ins.cumulativeWHF[timeindex]])
        print([ins.label,ins.cumulativeWHF[timeindex]])

with open('allIWHF.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+[ins.label for ins in inslist])
    for i in range(min(length)):
        spamwriter.writerow([i*5*10**(-5)]+[ins.cumulativeWHF[i] for ins in inslist])

with open('allIsmoothWHF.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+[ins.label for ins in inslist])
    xfixed = ins.x[half_param_ave:-half_param_ave]
    for i,x in enumerate(xfixed[:min(lenfix)]):
        spamwriter.writerow([x]+[ins.smoothWHF[i] for ins in inslist])

#raw data

'''plot'''    
plt.figure(figsize=(20,15))
ax1 = plt.subplot(211)
ax1.set_xlim(0,4.51)
ax1.set_ylim(0.0,0.0301)
plt.title("allWHF", fontsize=30)
for ins in inslist:
    plt.plot(ins.x,ins.WHF,label=ins.label)
#plt.xlabel('time(ms)')
plt.ylabel('Qwall       MW', fontsize=30)
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.legend()
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
plt.xticks( np.arange(0, 4.5, 0.4) )
plt.tick_params(labelsize = 20,direction = 'in',length = 7)

#plt.show()

ax2 = plt.subplot(212, sharex=ax1)
plt.subplot(212)
ax2.set_xlim(0,4.51)
ax2.set_ylim(0.0,50.01)
plt.title("cumultive_WHF", fontsize=30)
for ins in inslist:
    plt.plot(ins.x,ins.cumulativeWHF,label=ins.label)
plt.xlabel('time        ms', fontsize=30)
plt.ylabel('Total Heat Loss      J', fontsize=30)
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
#plt.grid()
plt.xticks( np.arange(0, 4.5, 0.5) )
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.show()
#
filename = "C:/Users/power/Desktop/python/images/allWHF.png"
plt.savefig(filename,dpi = 200, bbox_inches = 'tight', pad_inches = 0.1)
#       


#Averaged data

'''plot'''    
plt.figure(figsize=(16,12))
ax1 = plt.subplot(211)
ax1.set_xlim(0,4.51)
ax1.set_ylim(0.0,0.0301)
#plt.title("allWHF_movingAve", fontsize=30)
for ins in inslist:
    xfixed = ins.x[half_param_ave:-half_param_ave]
    plt.plot(xfixed,ins.smoothWHF,label=ins.label)
plt.xlabel('time       ms', fontsize=30)
plt.ylabel('Qwall      MW', fontsize=30)
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.legend()
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
plt.xticks( np.arange(0, 4.501, 0.5) )
plt.tick_params(labelsize = 20,direction = 'in',length = 7)

#plt.show()

ax2 = plt.subplot(212, sharex=ax1)
ax2.set_xlim(0,4.51)
ax2.set_ylim(0.0,50.01)
#plt.title("cumultive_WHF_movingAve", fontsize=30)
for ins in inslist:
    xfixed = ins.x[half_param_ave:-half_param_ave]
    plt.plot(xfixed,ins.cumulativesmoothWHF,label=ins.label)
plt.xlabel('time        ms', fontsize=30)
plt.ylabel('Total Heat Loss    J', fontsize=30)
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
#plt.grid()
plt.xticks( np.arange(0, 4.501, 0.5) )
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.show()

filename = "C:/Users/power/Desktop/python/images/allWHF_movingAve.png"
plt.savefig(filename,dpi = 200, bbox_inches = 'tight', pad_inches = 0.1)