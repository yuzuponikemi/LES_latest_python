# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:18:13 2019

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
        self.time = []
        self.y = []
        self.frac1 = []
        self.frac2 = []
        self.Ycen_b = []
        self.Zcen_b = []
    
              
        
        
        
        

'''main'''
namelist = []
namelist = glob.glob('./Spray_centre*.csv')
#LES_2017after内のWallHF_ave_time~~~.csvを読み出したい
#それはやめとく　GP4と他を区別するのが面倒
#こうやって、そのフォルダ内のｃｓｖを読み出すのは共通の機能にしておく
'''
Spray_centre_single_200MPa.csvといったファイル名
同じRのファイルをフォルダに突っ込む→グラフを作成
'''
#print(namelist)

namet=[]
for name in namelist:
    namet.append(name[2:])
#print(namet)


'''
'MS          ','Ycen    (mm)','Zcen     (mm)','Frac     (--)','Ycen_a   (mm)','Zcen_a   (mm)','Frac_a   (--)Ycen_b   (mm)','Zcen_b   (mm)','Frac_b   (--)'
0              1               2                3               4               5               6          7               8               9
'''

trimedname = map(lambda x : x[2:], namelist)
#print(list(trimedname))
width = 0.5000000E-04
inslist =[]
for name in namet:
    labelname = name[13:-4]
    ins = Version(name,labelname)
    
    with open(name, newline='') as name:

        reader = csv.reader(name)
        readerlist = list(reader)
        readerlist = readerlist[1:]
        
        for row in readerlist:
            ins.time.append(float(row[0]))
            ins.frac1.append(float(row[9]))
            ins.frac2.append(float(row[6]))
            ins.Ycen_b.append(float(row[7]))
            ins.Zcen_b.append(float(row[8]))
            
            
#        print(ins.name)
#        print(ins.label)
#        print(ins.x)
#        print(ins.y)
#        print(ins.cumulative)
        inslist.append(ins)
#
#for ins in inslist:
#    for i in range(len(ins.y)-1):
#        ins.slope.append(float(ins.y[i+1]-ins.y[i])/width)
#    ins.slope.append(ins.slope[len(ins.y)-2])

##param_aveは移動平均近似のパラメータ
#param_ave = 501
#half_param_ave = int((param_ave - 1)/2)
#
#c = np.array(1)
#
##slopeの移動平均近似を計算
#for ins in inslist:
#    a = np.array(list(ins.slope))
#    b = np.ones(param_ave)/float(param_ave)
#    c = np.convolve(a,b,'valid')
#    ins.smoothslope = c.tolist()
#       
#  
#
#for ins in inslist:
#    print('--------------')
#    print(ins.label)
#    ins.calc_burntime(5,95)
#    
#
#
#with open('burn_duration.csv', 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile)
#    spamwriter.writerow(['version','starttime','endtime','burnduration'])
#    for ins in inslist:
#        spamwriter.writerow([ins.label,ins.starttime,ins.endtime,ins.burntime])
#



'''plot'''    
markers1 = ["*", ",", "o", "v", "^", "<", ">", "D", "d", ]#marker = markers1[num]
plt.figure(figsize=(12,12))
ax1 = plt.subplot(211)
#ax1.set_xlim(0,4)
#ax1.set_ylim(7.0,7.4)
#plt.title("p_ambient", fontsize=30)
for num,ins in enumerate(inslist):
    plt.plot(ins.time,ins.frac1,label=ins.label)
#plt.xlabel('time(ms)')
plt.ylabel('frac1 [mm]', fontsize=30)
#plt.setp(ax1.get_xticklabels(), visible=False)
#plt.legend()
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
plt.xticks( np.arange(0, 4, 0.5) )

plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.grid()
#plt.show()

ax2 = plt.subplot(212)
#ax2.set_xlim(0,4.01)
#ax2.set_ylim(0.0,0.401)
plt.title("0.25-1.5ms", fontsize=30)
"""壁に当たる→噴射終わり　をプロットすればいいか"""
for num,ins in enumerate(inslist):
    plt.scatter(ins.Ycen_b[:],ins.Zcen_b[:],label=ins.label,marker = markers1[num])
plt.xlabel('Ycen [mm]', fontsize=30)
plt.ylabel('Zcen [mm]', fontsize=30)
plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
#plt.grid()
#plt.xticks( np.arange(0, 4.01, 0.5) )
#plt.yticks(np.arange(0,0.401, 0.1))
plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#plt.show()
#
filename = "C:/Users/power/Desktop/python/images/spray_centre_frac.png"
plt.savefig(filename)

print('plot 0.25-1.5ms')