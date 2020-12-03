# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:02:10 2019

@author: power
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
namelist = []
namelist = glob.glob('./res_temp_vwrflow_*.csv')
#LES_2017after内のWallHF_ave_time~~~.csvを読み出したい
#それはやめとく　GP4と他を区別するのが面倒
#こうやって、そのフォルダ内のｃｓｖを読み出すのは共通の機能にしておく

'''\\\\\\module\\\\\\\_'''
timeB = 0
timeL = 400
sep = 2
#VWRunit = 20          # v per 1mm
#Uunit = 20
#Tunit = 250        #temp per 1mm
#period = 'eoi'   #eoi or all
#shaping = 'single'
#scale = 1.0             # y axis scale of U and VWR
#aveDu = 0.2  #ms
#fontz = 25
'''____________________'''
#print(namelist)

namet=[]
for name in namelist:
    namet.append(name[2:])
#print(namet)

trimedname = map(lambda x : x[2:], namelist)
#print(list(trimedname))

"""
time	a_temp	b_temp	a_vflow	b_vflow
"""

class Version:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.x = []
        self.y = []
        self.cumulative = []
        self.time = []
        self.a_temp = []
        self.b_temp = []
        self.a_vflow = []
        self.b_vflow = []
        self.sa_temp = []
        self.sb_temp = []
        self.sa_vflow = []
        self.sb_vflow = []
        self.hat7mm_v = []
        self.shat7mm_v = []
        
label=[]
for name in namet:
    label.append(name[17:-4])
#print(label)

inslist =[]
for name in namet:
    labelname = name[17:-4]
    ins = Version(name,labelname)
    cumulative = 0
    with open(name, newline='') as name:

        reader = csv.reader(name)
        for row in reader:
            if 'time' in row:continue
            if 'ave' in row:continue
            ins.time.append(float(row[0]))
            ins.a_temp.append(float(row[1]))
            ins.b_temp.append(float(row[2]))
            ins.a_vflow.append(float(row[3]))
            ins.b_vflow.append(float(row[4]))
            h_v = float(row[3]) * 7 + float(row[4])
            ins.hat7mm_v.append(h_v)
#            cumulative = cumulative + float(row[1])
#            ins.cumulative.append(cumulative)
#        print(ins.name)
#        print(ins.label)
#        print(ins.x)
#        print(ins.y)
#        print(ins.cumulative)
        inslist.append(ins)
#print(inslist)
#print('The time WRP reaches 13mm is_')
#for ins in inslist:
#    time = 0
#    print(ins.label)
#    for r in ins.y:
#        if r <= 13:
#            time = ins.x[ins.y.index(r)]
#    print(str(time)+'ms')
        
#param_aveは移動平均近似のパラメータ
param_ave = 31
half_param_ave = int((param_ave - 1)/2)

c = np.array(1)

#slopeの移動平均近似を計算
for ins in inslist:
    a = np.array(list(ins.a_temp))
    b = np.ones(param_ave)/float(param_ave)
    c = np.convolve(a,b,'valid')
    ins.sa_temp = c.tolist()


    a = np.array(list(ins.b_temp))
    c = np.convolve(a,b,'valid')
    ins.sb_temp = c.tolist()         
    a = np.array(list(ins.a_vflow))
    c = np.convolve(a,b,'valid')
    ins.sa_vflow = c.tolist()         
    a = np.array(list(ins.b_vflow))
    c = np.convolve(a,b,'valid')
    ins.sb_vflow = c.tolist()      
    a = np.array(list(ins.hat7mm_v))
    c = np.convolve(a,b,'valid')
    ins.shat7mm_v = c.tolist()  

'''plot temp res'''
#plt.figure(figsize=(8,6))
#ax1 = plt.subplot(211) 
##plt.title("Rwall_f=0.01")
#for ins in inslist:
#    timefixed = ins.time[half_param_ave:-half_param_ave]
#    plt.plot(timefixed[timeB:],ins.sa_temp[timeB:],label=ins.label)
#plt.legend(fontsize = 'large',frameon = True)
#plt.xlabel('t     ms', fontsize=20)
#plt.ylabel('temp  [a]  ', fontsize=20)
#plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.setp(ax1.get_xticklabels(), visible=False)
#
#ax2 = plt.subplot(212)
#for ins in inslist:
#    timefixed = ins.time[half_param_ave:-half_param_ave] 
#    plt.plot(timefixed[timeB:],ins.sb_temp[timeB:],label=ins.label)
#plt.legend(fontsize = 'large',frameon = True)
#plt.xlabel('t     ms', fontsize=20)
#plt.ylabel('temp  [b]  ', fontsize=20)
#plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.setp(ax1.get_xticklabels(), visible=False)
#
#
#
#filename = "C:/Users/power/Desktop/python/images/res_temp.png"
#plt.savefig(filename)

'''plot flow res'''
plt.figure(figsize=(14,18))
ax1 = plt.subplot(211) 
#plt.title("Rwall_f=0.01")
for ins in inslist:
    timefixed = ins.time[half_param_ave:-half_param_ave]
    plt.plot(timefixed[timeB:],ins.sa_vflow[timeB:],label=ins.label)
    plt.scatter(ins.time[timeB::sep],ins.a_vflow[timeB::sep],label = ins.label)
plt.legend(fontsize = 'large',frameon = True)
plt.xlabel('t     ms', fontsize=20)
plt.ylabel('vflow  [a]  ', fontsize=20)
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(212)
for ins in inslist:
    timefixed = ins.time[half_param_ave:-half_param_ave] 
    plt.plot(timefixed[timeB:],ins.sb_vflow[timeB:],label=ins.label)
    plt.scatter(ins.time[timeB::sep],ins.b_vflow[timeB::sep],label = ins.label)
plt.legend(fontsize = 'large',frameon = True)
plt.xlabel('t     ms', fontsize=20)
plt.ylabel('flow  [b]  ', fontsize=20)
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
plt.setp(ax1.get_xticklabels(), visible=False)


filename = "C:/Users/power/Desktop/python/images/res_flow.png"
plt.savefig(filename)



plt.figure(figsize=(14,12))
for ins in inslist:
    timefixed = ins.time[half_param_ave:-half_param_ave] 
    plt.plot(timefixed[timeB:],ins.shat7mm_v[timeB:],label=ins.label)
    #plt.scatter(ins.time[timeB::sep],ins.hat7mm_v[timeB::sep],label = ins.label)
plt.legend(fontsize = 'large',frameon = True)
plt.xlabel('t     ms', fontsize=20)
plt.ylabel('h of vwrflow at R=7mm  ', fontsize=20)
plt.tick_params(labelsize = 20,direction = 'in',length = 7)

filename = "C:/Users/power/Desktop/python/images/vwrflow_h_flow.png"
plt.savefig(filename)