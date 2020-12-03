# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:47:48 2019

@author: power

各ｒでの多項式によるカーブフィッティング

scipy.interpolate により二次スプライン補間を行う

VWRは内挿により得た最大値をつなぎ、境界層とする

カラープロットによる可視化も

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import glob
import os
from scipy import interpolate
print('version??? I mean just version name.')
#version = str(input())
version = 'soft3_ave'
print('OK! Let\'s process '+version)

#4ms is not always true Don't care about the number

'''\\\\\\module\\\\\\\_'''
timebottom = 0
timelimit = int(800)  #390-920  0.005
imaging = [0,0]
#sep = 1
vunit = 20          # v[m/s] per 1mm
Tunit = 250        #temp[K] per 1mm
#period = 'eoi'   #eoi or all
#shaping = 'group3'    # inv soft last group1 group2 or single
scale = 1.0 #[mm]
aveDu = 0.1 #ms      # duration of average calc    [50] means 0.25ms
Tdelta = 0.03 #[K]   #constant?
method = interpolate.interp1d#"2次スプライン補間"
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

#shapetime = []
#shapeR = []
#
#with open('C:/Users/power/Desktop//python/shapeWall/SHAPE_WallRAVE_'+version[:-4]+'.csv', 'r', newline='') as fg:
#    reader = csv.reader(fg)
#    readerlist = list(reader)
#    for row in readerlist:
#        shapetime.append(float(row[0]))
#        shapeR.append(float(row[1]))
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

#"""Read WHF_area data for visualisation"""
#os.chdir('C:/Users/power/Desktop/python/WHF_area')
#timeareaR = []
#areaR = []
#with open('C:/Users/power/Desktop//python/WHF_area/all'+version+'smoothR.csv', 'r', newline='') as fg:
#    reader = csv.reader(fg)
#    readerlist = list(reader)
#    burn_durations= []
#    for row in readerlist[::800]:
#        timeareaR.append(row[0])
#        areaR.append(row[1])


        

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
        self.HFSr = []
        self.HFr = []
        self.Vpeak = []
        self.Tpeak = []
        self.Vpeakdelt = []
        self.Tpeakdelt = []
        self.HF = []
        self.Tdeltadtdx = [] # size = len(list(set(ins.Rs)))
        self.fitted_curve_v = []

'''
Read data
X(mm)	delta(mm)	R(mm)	F_ave	U_ave(m/s)	VWR_ave(m/s)	UVW_ave(m/s)	T_ave	Heat_Flux[Q/m2]  Zcen_b(mm)  Ycen_b(mm)
0           1         2       3         4             5             6            7           8             9           10
'''     
print('Reading data...')  

lssr = []
lsr = [x*0.5 for x in range(0,42)]
for r in lsr:
    sr = math.pi*(r+0.25)*(r+0.25) - sum(lssr)
    lssr.append(sr)
    
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
        
        for p,row in enumerate(readerlist):
#            if not (float(row[0])%fcc) == 0:
#                continue
            ins.Rs.append(float(row[2]))
            ins.HF.append(float(row[8]))
            ins.HFr.append(float(row[2])*float(row[8])*10**(-3))
            if (p%16 == 0):
                ins.HFSr.append(float(row[8])*(lssr[(p//16)]))
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
        ins.vwrflow[i] = ins.vwrflow[i]/vunit + R
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


ave = Version('ave','EOI_ave_of_'+str(aveDu))
ave.Rs = rsls
ave.time = 'ave'
ave.delt = deltls
ave.vwrflow = aveV
ave.temp = aveT
ave.Vpeak = aveVpeak
ave.Tpeak = aveTpeak
ave.Vpeakdelt = [0]+avpdelt[1:]
ave.Tpeakdelt = atpdelt
ave.HF = aveHF
for num, hf in enumerate(aveHF):
    ave.HFr.append(hf*rsls[num])
ave.Tdeltadtdx = aveTdeltadtdx
for p,hf in enumerate(aveHF):
    if (p%16 == 0):
        ave.HFSr.append(hf*(lssr[(p//16)]))


inslist.append(ave)





'''-------------------plot-------------------------'''  
rs_latent = np.linspace(0.1, 19.8, 1000)
        
if not os.path.exists("C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version):
    os.mkdir("C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version)
  
for ins in (inslist[imaging[0]:imaging[1]]+inslist[-1:]):
#    print('Tdeltdtdx',ins.Tdeltadtdx)
#    print('rs',rs)
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
    #res1=np.polyfit(ins.Tpeak[:shape], ins.Tpeakdelt[:shape], 5)
    #print(res1,ins.label,'temp')
    #y1 = np.poly1d(res1)(ins.Tpeak[:shape])
    #plt.plot(ins.Tpeak[:shape],y1,color = 'k',linestyle= 'dashed')
    
    res3=np.polyfit(rs[:shape], ins.Tdeltadtdx[:shape], 5)
    y3 = np.poly1d(res3)(rs[:shape])
    plt.plot(rs[:shape],y3,color = 'b',linestyle= 'solid')
    
    fitted_curve = method(rs,ins.Tdeltadtdx)
    plt.scatter(rs,ins.Tdeltadtdx, label="observed")
    plt.plot(rs_latent, fitted_curve(rs_latent), c="red", label="fitted")

    plt.yticks( np.arange(0, scale+0.01, scale/2) )     
#    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R', fontsize=30)
    plt.ylabel('   delt     mm', fontsize=20)
    plt.title('temp_'+ins.label+'ms_'+version, fontsize=15)
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
    
    ax5 = ax2.twinx()
    #plt.title("Wall_Heat_Flux"+ins.label, fontsize=fontz)
    plt.title('Wall Heat Flux', fontsize=20)

    ax5.plot(rs,[x for x in ins.HFSr],linestyle= 'dotted')

    ax2.set_ylabel('   HF        MW/m2', fontsize=20)
    ax5.set_ylabel('   HF * Sr      MW', fontsize=20)
#    plt.yticks( np.arange(0, scale+0.01, scale/2) )
#    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R      mm', fontsize=20)
#    plt.ylabel('   HF    MW/mm2', fontsize=20)
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
    rs = sorted(list(set(ins.Rs)))

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
#    res2=np.polyfit(ins.Vpeak[:shape], ins.Vpeakdelt[:shape], 9)
#    #print(res2,ins.label,'vwrflow')
#
#    y2 = np.poly1d(res2)(ins.Vpeak[:shape])
#    plt.plot(ins.Vpeak[:shape],y2,color = 'k',linestyle= 'dashed')
    fitted_curve = method(ins.Vpeak,ins.Vpeakdelt)
    plt.scatter(ins.Vpeak, ins.Vpeakdelt, label="observed")
    #plt.plot(ins.Vpeak, fitted_curve(ins.Vpeak), c="red", label="fitted")
    plt.scatter(rs[1:-1], fitted_curve(rs[1:-1]), c="blue", label="fittedrs")
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
    print('1mm='+str(vunit)+'m/s,  1mm = '+str(Tunit)+'K')
    #plt.grid()
    filename = "C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/"+version+"/temp_vwrflow_HF"+version+'_'+ins.label+"ms.png"
    plt.savefig(filename)
    plt.show()
    if ins.time == 'ave':print('average of '+str(aveDu)+' ms before EOI')
#    print('Vpeakdelt',ins.Vpeakdelt)
#    print('Vpeak',ins.Vpeak)
    
    
'''   補間した境界層厚さ情報の保存   '''

with open('../Tdeltadtdx_'+version+'.csv', 'w', newline='') as csvfile:
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
'''毎時，境界層を作る点の座標を保存，各R上での値に補正する'''
with open('../BdLayer_delt_linearIp_vwrflow_'+version+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['time(ms)','0.0','0.5','1.0','1.5','2.0',\
                             '2.5','3.0','3.5','4.0','4.5','5.0','5.5',\
                             '6.0','6.5','7.0', '7.5','8.0','8.5','9.0',\
                             '9.5','10.0','10.5','11.0','11.5','12.0',\
                             '12.5', '13.0','13.5','14.0','14.5','15.0',\
                             '15.5','16.0','16.5','17.0','17.5','18.0',\
                             '18.5','19.0','19.5','20.0'])
        for c,ins in enumerate(inslist[timebottom:timelimit]+inslist[-1:]):
            fitted_curve = method(ins.Vpeak,ins.Vpeakdelt)
            ike = [ins.time]+[0]+list(fitted_curve(rs[1:-1]))
            ins.fitted_curve_v = fitted_curve
            spamwriter.writerow(ike)
     
        
for ins in inslist:
    for num,Vpeak in enumerate(ins.Vpeak):
        ins.Vpeak[num] = (Vpeak-rs[num])*vunit     
        
        
'''-------------------plot 速度境界層　温度勾配　各Rドーナツ領域壁面熱損失-------------------------'''
print('Imaging BDlayer,etc')
xlim = 4
ylim = 15
laten = range(2,198,1)
plt.figure(figsize=(20,8))
#ax1 = plt.subplot(511)
#ax1.set_xlim(0,xlim)
#ax1.set_ylim(0,ylim)
#ax1.grid(c = 'w')
##plt.title(version, fontsize=20)
#for ins in (inslist[timebottom:timelimit]):
#       
#    im = plt.scatter([ins.time for i in laten], [i*0.1 for i in laten], vmin=0, vmax=1, c = ins.fitted_curve_v([i*0.1 for i in laten]), cmap = cm.nipy_spectral,label="fittedrs")
#
#cbar = plt.colorbar(im)
#cbar.set_label("  Velocity Boundary Layer   mm", size=15)
#cbar.ax.tick_params(labelsize=15)
##cbar.set_ticks(np.arange(0,1.01,0.25))
#plt.clim(0, 1)
##plt.xlabel('                                        t                                   ms', fontsize=20)
#plt.ylabel('                  R          mm', fontsize=20)
#plt.xticks( np.arange(0, xlim+0.01,1 ) )
#plt.yticks( np.arange(0, ylim+0.01,5 ) )
#plt.tick_params(labelsize = 20,direction = 'out',length = 7)

ax2 = plt.subplot(221)
ax2.set_xlim(0,xlim)
ax2.set_ylim(0,ylim)
ax2.grid(c = 'w')
for ins in (inslist[timebottom:timelimit]):
    fitted_curve = method(rs,ins.Tdeltadtdx) 
    im = plt.scatter([ins.time for i in laten], [i*0.1 for i in laten], vmin=0, vmax=1, c = fitted_curve([i*0.1 for i in laten]), cmap = cm.nipy_spectral,label="fittedrs")

cbar = plt.colorbar(im)
cbar.set_label("                       $\it{\delta_{T}}$               mm", size=15)
cbar.ax.tick_params(labelsize=15) 
#cbar.set_ticks(np.arange(0,1.01,0.25))
plt.clim(0, 1.0)
#plt.xlabel('                                        t                                   ms', fontsize=20)
plt.ylabel('                R           mm', fontsize=20)
plt.xticks( np.arange(0, xlim+0.01,1 ) )
plt.yticks( np.arange(0, ylim+0.01,5 ) )
plt.tick_params(labelsize = 20,direction = 'out',length = 7)



ax3 = plt.subplot(223)
ax3.set_xlim(0,xlim)
ax3.set_ylim(0, ylim)
ax3.grid(c = 'w')
for ins in (inslist[timebottom:timelimit]):
    fitted_curve = method(rs,ins.HFSr) 
    im = plt.scatter([ins.time for i in laten], [i*0.1 for i in laten], vmin=0, vmax=2000, c = fitted_curve([i*0.1 for i in laten]), cmap = cm.nipy_spectral,label="fittedrs")
#im = plt.scatter([float(x) for x in timeareaR],[float(y) for y in areaR], c = [1999 for i in areaR],marker = '1', cmap = cm.nipy_spectral, vmin=0, vmax=200)
#im = plt.scatter([float(x) for x in shapetime],[float(y) for y in shapeR], c = [1999 for i in shapeR],marker = '_', cmap = cm.nipy_spectral, vmin=0, vmax=200)

cbar = plt.colorbar(im)
cbar.set_label("                     $\it{q_{wall}}$×$\it{S_{R}}$            MW", size=15)
cbar.ax.tick_params(labelsize=15) 
cbar.set_ticks(np.arange(0,2000.01,500))
plt.clim(0, 2000)
#plt.xlabel('                                        t                                   ms', fontsize=20)
plt.ylabel('                 R          mm', fontsize=20)
plt.xticks( np.arange(0, xlim+0.01,1 ) )
plt.yticks( np.arange(0, ylim+0.01,5 ) )
plt.tick_params(labelsize = 20,direction = 'out',length = 7)
#filename = "C:/Users/power/Desktop/python/images//23Temporal_BDLayer_temp_HFSr_"+version+".png"
#plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)

#ax4 = plt.subplot(414)
'''HFr'''
ax4 = plt.subplot(222)

ax4.set_xlim(0,xlim)
ax4.set_ylim(0, ylim)
ax4.grid(c = 'w')
for ins in (inslist[timebottom:timelimit]):
    fitted_curve = method(ins.Rs[::16],ins.HFr[::16]) 
    im = plt.scatter([ins.time for i in laten], [i*0.1 for i in laten], vmin=0, vmax=0.6, c = fitted_curve([i*0.1 for i in laten]), cmap = cm.nipy_spectral,label="fittedrs")
#im = plt.scatter([float(x) for x in timeareaR],[float(y) for y in areaR], c = [499 for i in areaR],marker = '1', cmap = cm.nipy_spectral, vmin=0, vmax=500)
#im = plt.scatter([float(x) for x in shapetime],[float(y) for y in shapeR], c = [499 for i in shapeR],marker = '_', cmap = cm.nipy_spectral, vmin=0, vmax=200)

cbar = plt.colorbar(im)
cbar.set_label("                   $\it{q_{wall}}$× $\it{R}$      MW/m", size=15)
cbar.ax.tick_params(labelsize=15) 
cbar.set_ticks(np.arange(0,0.601,0.3))
plt.clim(0, 0.6)
#plt.xlabel('                                        t                                   ms', fontsize=20)
plt.ylabel('                R          mm', fontsize=20)
plt.xticks( np.arange(0, xlim+0.01,1 ) )
plt.yticks( np.arange(0, ylim+0.01,5 ) )
plt.tick_params(labelsize = 20,direction = 'out',length = 7)
'''MaxV'''

ax4 = plt.subplot(224)

ax4.set_xlim(0,xlim)
ax4.set_ylim(0, ylim)
ax4.grid(c = 'w')
for ins in (inslist[timebottom:timelimit]):
    fitted_curve = method(rs,ins.Vpeak) 
    im = plt.scatter([ins.time for i in laten], [i*0.1 for i in laten], vmin=0, vmax=130, c = fitted_curve([i*0.1 for i in laten]), cmap = cm.nipy_spectral,label="fittedrs")
#im = plt.scatter([float(x) for x in timeareaR],[float(y) for y in areaR], c = [499 for i in areaR],marker = '1', cmap = cm.nipy_spectral, vmin=0, vmax=500)
#im = plt.scatter([float(x) for x in shapetime],[float(y) for y in shapeR], c = [499 for i in shapeR],marker = '_', cmap = cm.nipy_spectral, vmin=0, vmax=200)

cbar = plt.colorbar(im)
cbar.set_label("                Maximum $\it{v}$        m/s", size=15)
cbar.ax.tick_params(labelsize=15) 
cbar.set_ticks(np.arange(0,130.01,65))
plt.clim(0, 130)
plt.xlabel('                                        t                                   ms', fontsize=20)
plt.ylabel('                R          mm', fontsize=20)
plt.xticks( np.arange(0, xlim+0.01,1 ) )
plt.yticks( np.arange(0, ylim+0.01,5 ) )
plt.tick_params(labelsize = 20,direction = 'out',length = 7)
filename = "C:/Users/power/Desktop/python/images//Temporal_temp_HFr_HF_maxV"+version+".png"
plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)