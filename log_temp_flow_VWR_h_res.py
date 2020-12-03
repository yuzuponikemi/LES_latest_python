# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:36:01 2019

@author: power
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os
print('version??? I mean just version name.')
#version = str(input())
version = 'soft3'
print('OK! Let\'s process '+version)

#4ms is not always true Don't care about the number

'''\\\\\\module\\\\\\\_'''
timebottom = 0
timelimit = int(400)  #390-920  0.005
sep = 2
vunit = 20          # v per 1mm
Tunit = 250        #temp per 1mm
#period = 'eoi'   #eoi or all
#shaping = 'group3'    # inv soft last group1 group2 or single
scale = 1.0
aveDu = 0.1 #ms      # duration of average calc    [50] means 0.25ms
'''____________________'''


"""Read shapeWallave data for calc range"""
os.chdir('C:/Users/power/Desktop/python/shapeWall')

#if 'mm' in version:
#    file = version[:13]
#    scale = float(version[14:-2])
#else:
#    file = version
#    scale = 1.0
#    folda = file + '_1.4mm'

shapedic = {}

with open('C:/Users/power/Desktop//python/shapeWall/SHAPE_WallRAVE_'+version+'.csv', 'r', newline='') as fg:
    reader = csv.reader(fg)
    readerlist = list(reader)
    burn_durations= []
    for row in readerlist:
        shapedic[float(row[0])]=float(row[1])




#0.005ms刻み　のデータにshapedicを対応させる
for t in range(1,400):
    num = round(t * 0.005,4)
    print(num)
    if num not in list(shapedic.keys()):
        for inde,key in enumerate(list(shapedic.keys())):
            if key >= num:
                shapedic[num] = shapedic[list(shapedic.keys())[inde-1]]
                break



shapedic['ave'] = 12


#os.chdir('C:/Users/power/Desktop/python/log_temp_flow/log_temp_flow_Y_'+folda)
os.chdir('C:/Users/power/Desktop/python/U_VWR_FLOW/log_temp_flow_U_' + version)


namet = []
#namelist = glob.glob('./WHFarea*.csv')
namet = glob.glob('./log_temp_flow_U_VWR*.csv') 


class Version:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.time = 0
        self.Rs = []
        self.delt = []
        self.vwrflow = []
        self.temp = []
        self.Vpeak = []
        self.Tpeak = []
        self.Vpeakdelt = []
        self.Tpeakdelt = []
        self.HF = []
'''
X(mm)	delta(mm)	R(mm)	F_ave	U_ave(m/s)	VWR_ave(m/s)	UVW_ave(m/s)	T_ave	Heat_Flux[Q/m2]  Zcen_b(mm)  Ycen_b(mm)
0           1         2       3         4             5             6            7           8             9           10
'''       
inslist = []   
fcc = 0.5
for name in namet[:timelimit]:
    labelname = str(float(name[22:-8]))
    ins = Version(name,labelname)
    ins.time = float(labelname)
    with open(name, newline='') as name:

        reader = csv.reader(name)
        readerlist = list(reader)
        readerlist = readerlist[1:]
        
        for row in readerlist:
#            if not (float(row[0])%fcc) == 0:
#                continue
            ins.Rs.append(float(row[2]))
            ins.HF.append(float(row[8]))
            ins.delt.append(float(row[1]))
            ins.vwrflow.append(float(row[5]))
            ins.temp.append(float(row[7]))           
            
#        print(ins.name)
        print(ins.label)
#        print(ins.Rs)
#        print(ins.delt)
#        print(ins.cumulative)
        inslist.append(ins)
        
for ins in inslist:
    deltls = ins.delt
    rsls = ins.Rs
    for i in range(0,len(ins.Rs)):
        R = ins.Rs[i]
        ins.vwrflow[i] = ins.vwrflow[i]/40 + R
        ins.temp[i] = ins.temp[i]/1000 +R-0.5
        rs = sorted(list(set(ins.Rs)))
    for r in rs:
        rls =[]
        for i in range(0,len(ins.Rs)):
            if ins.Rs[i] == r:
                rls.append(i)
        Tpeak = max(ins.temp[rls[0]:rls[-1]])
        Vpeak = max(ins.vwrflow[rls[0]:rls[-1]])
        ins.Vpeak.append(Vpeak)
        ins.Vpeakdelt.append(ins.delt[ins.vwrflow.index(Vpeak)])
        ins.Tpeak.append(Tpeak)
        ins.Tpeakdelt.append(ins.delt[ins.temp.index(Tpeak)])


"""-------------calc time-averaged V and Temp ------------ """
#start = 17  #10 = 0.5ms       0.5ms before EOI
enddic = {'single_200MPa':314,'single_130MPa':390,'single_270MPa':270,\
          'inv2':336,'inv1':326,'soft1':326,'soft2':336,'last1':326,'last2':336,\
          'soft3':352,'last3':352,'inv3':352}
end = enddic[version] - 40    # -40 avoid last 0.2ms of tinj
start = end - int(aveDu/0.005)
#End of injection
#shape = 23 #10 = 5mm   130,200,270共通
#   ---------------use shapedic from csv



aveV = [0 for x in rsls]
aveT = [0 for x in deltls]
aveHF = [0 for x in deltls]
for ins in inslist[start:end]:
    for i,velo in enumerate(ins.vwrflow):
        aveV[i] += velo
    for i,temp in enumerate(ins.temp):
        aveT[i] += temp
    for i,HF in enumerate(ins.HF):
        aveHF[i] += HF
leng = len(inslist[start:end])
for i in range(0,len(aveV)):
    aveV[i] = aveV[i]/leng 
    aveT[i] = aveT[i]/leng
    aveHF[i] = aveHF[i]/leng

aveVpeak = []
aveTpeak = []
avpdelt = []
atpdelt = []
rs = sorted(list(set(rsls)))
for r in rs:
    rls =[]
    for i in range(0,len(rsls)):
        if rsls[i] == r:
            rls.append(i)
    Tpeak = max(aveT[rls[0]:rls[-1]])
    Vpeak = max(aveV[rls[0]:rls[-1]])
    aveVpeak.append(Vpeak)
    avpdelt.append(deltls[aveV.index(Vpeak)])
    aveTpeak.append(Tpeak)
    atpdelt.append(deltls[aveT.index(Tpeak)])    

#



ave = Version('ave','ave')
ave.Rs = rsls
ave.time = 'ave'
ave.delt = deltls
ave.vwrflow = aveV
ave.temp = aveT
ave.Vpeak = aveVpeak
ave.Tpeak = aveTpeak
ave.Vpeakdelt = avpdelt
ave.Tpeakdelt = atpdelt
ave.HF = aveHF



inslist.append(ave)





'''-------------------plot-------------------------'''  
        
if not os.path.exists("C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version):
    os.mkdir("C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version)
  
for ins in (inslist[timebottom:timelimit]+inslist[-1:]):
    '''plot temp'''
    plt.figure(figsize=(12,16))
    ax1 = plt.subplot(311)
    ax1.set_xlim(0,20.01)
    ax1.set_ylim(0.0,scale)
    #plt.title("INJECTIONRATE", fontsize=30)
    rs = list(set(ins.Rs))

    for r in rs:
        rls =[]
        for i in range(0,len(ins.Rs)):
            if ins.Rs[i] == r:
                rls.append(i)
        ax1.plot(ins.temp[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],label=r,marker = '.')
#        plt.show()
    plt.plot(ins.Tpeak,ins.Tpeakdelt,color = 'k',linestyle= 'dotted')   
    
    "近似式の計算"
    shape = int(2*shapedic[ins.time])
    if shape <= 5:shape = 6
    res1=np.polyfit(ins.Tpeak[5:shape], ins.Tpeakdelt[5:shape], 1)
    print(res1,ins.label,'temp')
    y1 = np.poly1d(res1)(ins.Tpeak[5:shape])
    plt.plot(ins.Tpeak[5:shape],y1,color = 'k',linestyle= 'dashed')
    plt.yticks( np.arange(0, scale+0.01, scale/2) )     
#    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R', fontsize=30)
    plt.ylabel('   delt     mm', fontsize=20)
    plt.title('temp_'+ins.label+'_'+version, fontsize=15)
    #plt.setp(ax1.get_xticklabels(), visible=False)
#   plt.legend(fontsize = 'small',frameon = None,prop={'size':20,},bbox_to_anchor=(1.05, 1))
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = 20,direction = 'in',length = 7)
    #plt.grid()
    #plt.show()
    
    '''plot HF'''
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.set_xlim(0,20)
    #plt.title("dp/dt_movingAve")

    plt.plot(ins.Rs,ins.HF,color = 'k')

    
#   ax2.scatter(ins.vwrflow,ins.delt,label=ins.label,marker ='.')
#    plt.yticks( np.arange(0, scale+0.01, scale/2) )
#    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R      mm', fontsize=20)
    plt.ylabel('   HF    MW/mm2', fontsize=20)
#    plt.title('vwrflow_'+ins.label+'_'+version, fontsize=15)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#   plt.show()
    #plt.grid()
    
    '''plot vwrflow'''
    ax3 = plt.subplot(313, sharex=ax1)
    ax3.set_xlim(0,20)
    ax3.set_ylim(0.0,scale)
    #plt.title("dp/dt_movingAve")
    rs = list(set(ins.Rs))

    for r in rs:
        rls =[]
        for i in range(0,len(ins.Rs)):
            if ins.Rs[i] == r:
                rls.append(i)
        ax3.plot(ins.vwrflow[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],label=ins.label,marker = '.')
    plt.plot(ins.Vpeak,ins.Vpeakdelt,color = 'k',linestyle= 'dotted')
    
    "近似式の計算"
    shape = int(2*shapedic[ins.time])
    if shape <= 5:shape = 6
    res2=np.polyfit(ins.Vpeak[5:shape], ins.Vpeakdelt[5:shape], 1)
    ins.res = list(res1)+list(res2)
    print(res2,ins.label,'vwrflow')
    print('1mm=40m/s,  1mm = 1000K')
    y2 = np.poly1d(res2)(ins.Vpeak[5:shape])
    plt.plot(ins.Vpeak[5:shape],y2,color = 'k',linestyle= 'dashed')
    
#   ax2.scatter(ins.vwrflow,ins.delt,label=ins.label,marker ='.')
    plt.yticks( np.arange(0, scale+0.01, scale/2) )
#    plt.yticks( np.arange(0, 4.01, 1) )
    plt.xlabel('    r            mm', fontsize=20)
    plt.ylabel('   delt        mm', fontsize=20)
    plt.title('vwrflow_'+ins.label+'_'+version, fontsize=15)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#   plt.show()
    #plt.grid()
    filename = "C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version+"/temp_vwrflow_HF"+version+'_'+ins.label+"ms.png"
    plt.savefig(filename)
    plt.show()
    if ins.time == 'ave':print('average of '+str(aveDu)+' ms before EOI')
    
    
#   近似式情報の保存
#restime = []
#a_temp = []
#b_temp = []
#a_vwrflow = []
#b_vwrflow = [] 
with open('../res_temp_vwrflow_'+version+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['time','a_temp','b_temp','a_vwrflow','b_vwrflow'])
        for ins in (inslist[timebottom:timelimit]+inslist[-1:]):
            ike = [ins.time]+ins.res
            spamwriter.writerow(ike)
#            restime.append(ins.time)
            