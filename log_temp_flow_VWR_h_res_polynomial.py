# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:25:12 2019

@author: power

各ｒでの多項式によるカーブフィッティング
numpy.polyfit を使った最小二乗法によるもの

内挿により得た最大値をつなぎ、境界層とする

"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os
print('version??? I mean just version name.')
#version = str(input())
version = 'single_200MPa'
print('OK! Let\'s process '+version)

#4ms is not always true Don't care about the number

'''\\\\\\module\\\\\\\_'''
timebottom = 0
timelimit = int(800)  #390-920  0.005
#sep = 1
vunit = 20          # v[m/s] per 1mm
Tunit = 250        #temp[K] per 1mm
#period = 'eoi'   #eoi or all
#shaping = 'group3'    # inv soft last group1 group2 or single
scale = 1.0 #[mm]
aveDu = 0.1 #ms      # duration of average calc    [50] means 0.25ms
Tdelta = 0.03 #[K]   #constant?
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

#shapedic = {}
#
#with open('C:/Users/power/Desktop//python/shapeWall/SHAPE_WallRAVE_soft3.csv', 'r', newline='') as fg:
#    reader = csv.reader(fg)
#    readerlist = list(reader)
#    burn_durations= []
#    for row in readerlist:
#        shapedic[float(row[0])]=float(row[1])
#
##0.005ms刻み　のデータにshapedicを対応させる
#for t in range(1,int(800) ):
#    num = round(t * 0.005,4)
#    print(num)
#    if num not in list(shapedic.keys()):
#        for inde,key in enumerate(list(shapedic.keys())):
#            if key >= num:
#                shapedic[num] = shapedic[list(shapedic.keys())[inde-1]]
#                break
#shapedic['ave'] = 20




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
        self.Tdeltadtdx = [] # size = len(list(set(ins.Rs)))
'''
Read data
X(mm)	delta(mm)	R(mm)	F_ave	U_ave(m/s)	VWR_ave(m/s)	UVW_ave(m/s)	T_ave	Heat_Flux[Q/m2]  Zcen_b(mm)  Ycen_b(mm)
0           1         2       3         4             5             6            7           8             9           10
'''     
print('Reading data...')  
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
        #print(ins.label)
#        print(ins.Rs)
#        print(ins.delt)
#        print(ins.cumulative)
        inslist.append(ins)
        

# delt に合わせて　ｙ軸を１０００分割する
y_latent = np.linspace(0,1.0,1000)
print('adjusting data')
print('calculating Velocity(+temp) boundary layer')
errorcounter = 0
for ins in inslist:
    deltls = ins.delt
    rsls = ins.Rs
    for i in range(0,len(ins.Rs)):
        R = ins.Rs[i]
        ins.vwrflow[i] = ins.vwrflow[i]/40 + R
        ins.temp[i] = ins.temp[i]/1000 +R-0.5
        rs = sorted(list(set(ins.Rs))) # rs has no overlaping
    for r in rs:
        rls =[] # index list
        for i in range(0,len(ins.Rs)):
            if ins.Rs[i] == r:
                rls.append(i)
        "各rでの壁面近傍3点の温度勾配により算出する温度境界層厚さの計算"
        a,b=np.polyfit(ins.temp[rls[0]:rls[3]],ins.delt[rls[0]:rls[3]], 1)
        if 0<a<100000:
            ins.Tdeltadtdx.append(Tdelta/a)
        else:
            ins.Tdeltadtdx.append(0)
            errorcounter +=1
                
        "各ｒでの内挿された最大値の包絡線による境界層近似式の計算"
#        shape = int(2*shapedic[ins.time])
#        if shape <= 5:shape = 6
        wt=np.polyfit(ins.delt[rls[0]:rls[-1]],ins.temp[rls[0]:rls[-1]], 5)
        wv=np.polyfit(ins.delt[rls[0]:rls[-1]],ins.vwrflow[rls[0]:rls[-1]], 5)
        
        y1 = np.poly1d(wt)(y_latent)
        y2 = np.poly1d(wv)(y_latent)
        ins.Tpeak.append(max(y1))
        ins.Tpeakdelt.append(y_latent[np.argmax(y1)])
        ins.Vpeak.append(max(y2))
        ins.Vpeakdelt.append(y_latent[np.argmax(y2)])
    ins.Vpeak[0],ins.Vpeakdelt[0] = 0,0


#        Tpeak = max(ins.temp[rls[0]:rls[-1]])
#        Vpeak = max(ins.vwrflow[rls[0]:rls[-1]])
#        ins.Vpeak.append(Vpeak)
#        ins.Vpeakdelt.append(ins.delt[ins.vwrflow.index(Vpeak)])
#        ins.Tpeak.append(Tpeak)
#        ins.Tpeakdelt.append(ins.delt[ins.temp.index(Tpeak)])


"""-------------calc time-averaged V and Temp ------------ """
#start = 17  #10 = 0.5ms       0.5ms before EOI
enddic = {'single_200MPa':314,'single_130MPa':390,'single_270MPa':270,\
          'inv2':336,'inv1':326,'soft1':326,'soft2':336,'last1':326,'last2':336,\
          'soft3':352,'last3':352,'inv3':352,'soft3_ave':352,'last3_ave':352,'inv3_ave':352}
end = enddic[version] - 40    # -40 avoid last 0.2ms of tinj
start = end - int(aveDu/0.005)
#End of injection
#shape = 23 #10 = 5mm   130,200,270共通
#   ---------------use shapedic from csv

print('calculating time-average')

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
aveTdeltadtdx = []
rs = sorted(list(set(rsls)))
for r in rs:
    rls =[]
    for i in range(0,len(rsls)):
        if rsls[i] == r:
            rls.append(i)
#    Tpeak = max(aveT[rls[0]:rls[-1]])
#    Vpeak = max(aveV[rls[0]:rls[-1]])
#    aveVpeak.append(Vpeak)
#    avpdelt.append(deltls[aveV.index(Vpeak)])
#    aveTpeak.append(Tpeak)
#    atpdelt.append(deltls[aveT.index(Tpeak)]) 
            
    "各rでの壁面近傍3点の温度勾配により算出する温度境界層厚さの計算"
    a,b=np.polyfit(aveT[rls[0]:rls[3]],deltls[rls[0]:rls[3]], 1)
    if 0<a<100000:
        aveTdeltadtdx.append(Tdelta/a)
    else:
        aveTdeltadtdx.append(0)
    
    wta=np.polyfit(deltls[rls[0]:rls[-1]],aveT[rls[0]:rls[-1]], 5)
    wva=np.polyfit(deltls[rls[0]:rls[-1]],aveV[rls[0]:rls[-1]], 5)
        
    y1a = np.poly1d(wta)(y_latent)
    y2a = np.poly1d(wva)(y_latent)
    aveTpeak.append(max(y1a))
    atpdelt.append(y_latent[np.argmax(y1a)])
    aveVpeak.append(max(y2a))
    avpdelt.append(y_latent[np.argmax(y2a)])


ave = Version('ave','ave_of_'+str(aveDu))
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
ave.Tdeltadtdx = aveTdeltadtdx



inslist.append(ave)





'''-------------------plot-------------------------'''  
        
if not os.path.exists("C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version):
    os.mkdir("C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version)
  
for ins in (inslist[timebottom:timelimit]+inslist[-1:]):
    print(ins.Tdeltadtdx)
    print(rs)
    '''plot temp'''
    plt.figure(figsize=(12,16))
    ax1 = plt.subplot(311)
    ax1.set_xlim(0,20.01)
    ax1.set_ylim(0.0,scale)
    #plt.title("INJECTIONRATE", fontsize=30)
    rs = sorted(list(set(ins.Rs)))

    for r in rs:
        rls =[]
        for i in range(0,len(ins.Rs)):
            if ins.Rs[i] == r:
                rls.append(i)
        ax1.plot(ins.temp[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],label=r,marker = '.')
        
        
    plt.plot(rs,ins.Tdeltadtdx,color = 'b',linestyle= 'dashdot')
    #plt.plot(ins.Tpeak,ins.Tpeakdelt,color = 'k',linestyle= 'dotted')   
    "近似式の計算"
    #shape = int(2*shapedic[ins.time])
    shape = 41
    #if shape <= 5:shape = 6
    res1=np.polyfit(ins.Tpeak[:shape], ins.Tpeakdelt[:shape], 5)
    res3=np.polyfit(rs[:shape], ins.Tdeltadtdx[:shape], 5)
    #print(res1,ins.label,'temp')
    y1 = np.poly1d(res1)(ins.Tpeak[:shape])
    #plt.plot(ins.Tpeak[:shape],y1,color = 'k',linestyle= 'dashed')
    y3 = np.poly1d(res3)(rs[:shape])
    plt.plot(rs[:shape],y3,color = 'b',linestyle= 'solid')

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
    #shape = int(2*shapedic[ins.time])
    shape = 41
    #if shape <= 5:shape = 6
    res2=np.polyfit(ins.Vpeak[:shape], ins.Vpeakdelt[:shape], 9)
    ins.res = list(res1)+list(res2)
    #print(res2,ins.label,'vwrflow')
    print('1mm='+str(vunit)+'m/s,  1mm = '+str(Tunit)+'K')
    y2 = np.poly1d(res2)(ins.Vpeak[:shape])
    plt.plot(ins.Vpeak[:shape],y2,color = 'k',linestyle= 'dashed')
    
#   ax2.scatter(ins.vwrflow,ins.delt,label=ins.label,marker ='.')
    plt.yticks( np.arange(0, scale+0.01, scale/2) )
#    plt.yticks( np.arange(0, 4.01, 1) )
    plt.xlabel('    r            mm', fontsize=20)
    plt.ylabel('   delt        mm', fontsize=20)
    plt.title('vwrflow_'+ins.label+'ms_'+version, fontsize=15)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#   plt.show()
    #plt.grid()
    filename = "C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version+"/temp_vwrflow_HF"+version+'_'+ins.label+"ms.png"
    plt.savefig(filename)
    plt.show()
    if ins.time == 'ave':print('average of '+str(aveDu)+' ms before EOI')
    print(ins.Vpeakdelt)
    print(ins.Vpeak)
    
    
#   近似式情報の保存
#restime = []
#a_temp = []
#b_temp = []
#a_vwrflow = []
#b_vwrflow = [] 
    with open('../Tdeltadtdx_temp_'+version+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['time(ms)','0.0','0.5','1.0','1.5','2.0',\
                                 '2.5','3.0','3.5','4.0','4.5','5.0','5.5',\
                                 '6.0','6.5','7.0', '7.5','8.0','8.5','9.0',\
                                 '9.5','10.0','10.5','11.0','11.5','12.0',\
                                 '12.5', '13.0','13.5','14.0','14.5','15.0',\
                                 '15.5','16.0','16.5','17.0','17.5','18.0',\
                                 '18.5','19.0','19.5','20.0'])
            for ins in (inslist[timebottom:timelimit]+inslist[-1:]):
                ike = [ins.time]+ins.Tdeltadtdx
                spamwriter.writerow(ike)
    #            restime.append(ins.time)
    '''毎時，境界層を作る点の座標を保存，各R上での値に補正しない'''
    with open('../Boundary_Layer_delt_vwrflow_'+version+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['time(ms)','0.0','0.5','1.0','1.5','2.0',\
                                 '2.5','3.0','3.5','4.0','4.5','5.0','5.5',\
                                 '6.0','6.5','7.0', '7.5','8.0','8.5','9.0',\
                                 '9.5','10.0','10.5','11.0','11.5','12.0',\
                                 '12.5', '13.0','13.5','14.0','14.5','15.0',\
                                 '15.5','16.0','16.5','17.0','17.5','18.0',\
                                 '18.5','19.0','19.5','20.0'])
            for ins in (inslist[timebottom:timelimit]+inslist[-1:]):
                ike = [ins.time]+ins.Vpeakdelt
                spamwriter.writerow(ike)
    #            restime.append(ins.time)
    
    with open('../Boundary_Layer_r_vwrflow_'+version+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['time(ms)','R=0','R=0.5'])
            for ins in (inslist[timebottom:timelimit]+inslist[-1:]):
                ike = [ins.time]+ins.Vpeak
                spamwriter.writerow(ike)
    #            restime.append(ins.time)