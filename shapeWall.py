# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:12:48 2018

@author: power
"""

#import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
namelist = []
namelist = glob.glob('./SHAPE_W*.csv')
#LES_2017after内のWallHF_ave_time~~~.csvを読み出したい
#それはやめとく　GP4と他を区別するのが面倒
#こうやって、そのフォルダ内のｃｓｖを読み出すのは共通の機能にしておく
'''
point00mm_single_L.csvといったファイル名
同じRのファイルをフォルダに突っ込む→グラフを作成
'''
#print(namelist)

namet=[]
for name in namelist:
    namet.append(name[2:])
#print(namet)

trimedname = map(lambda x : x[2:], namelist)
#print(list(trimedname))

glob.glob('./*/*.csv')

class Version:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.x = []
        self.y = []
        self.cumulative = []
        
label=[]
for name in namet:
    label.append(name[15:-4])
#print(label)

inslist =[]
for name in namet:
    labelname = name[15:-4]
    ins = Version(name,labelname)
    cumulative = 0
    with open(name, newline='') as name:

        reader = csv.reader(name)
        for row in reader:
            ins.x.append(float(row[0]))
            ins.y.append(float(row[1]))
#            cumulative = cumulative + float(row[1])
#            ins.cumulative.append(cumulative)
#        print(ins.name)
#        print(ins.label)
#        print(ins.x)
#        print(ins.y)
#        print(ins.cumulative)
        inslist.append(ins)
#print(inslist)
print('The time WRP reaches 13mm is_')
for ins in inslist:
    time = 0
    print(ins.label)
    for r in ins.y:
        if r <= 13:
            time = ins.x[ins.y.index(r)]
    print(str(time)+'ms')
            

#with open('cumulative_wall_heat_flux_at4ms.csv', 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile)
#    spamwriter.writerow(['version','wallHF(4ms)'])
#    for ins in inslist:
#        spamwriter.writerow([ins.label,ins.cumulative[80]])


plt.figure(figsize=(8,6))
#ax1 = plt.subplot(211) 
#plt.title("Rwall_f=0.01")
for ins in inslist:
    plt.plot(ins.x,ins.y,label=ins.label)
plt.legend(fontsize = 'large',frameon = True)
plt.xlabel('t     ms', fontsize=20)
plt.ylabel('Wall Radial Penetration  mm', fontsize=20)
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.grid()

#plt.show()
#ax2 = plt.subplot(212, sharex=ax1)
#plt.subplot(212) 
#plt.title("cumulative_wall_heat_flux_all")
#for ins in inslist:
#    plt.plot(ins.x,ins.cumulative,label=ins.label)
##plt.legend()
#plt.xlabel('time(ms)')
#plt.ylabel('Rwall')
#plt.grid()
#plt.show()

filename = "C:/Users/power/Desktop/python/images/SHAPE_wallR.png"
plt.savefig(filename)

