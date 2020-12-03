# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:31:46 2019

@author: power
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os
from scipy import interpolate
import math
from matplotlib import mathtext




'''\\\\\\module\\\\\\\_'''
first = 0
timelimit = 800  #400 - 890
imaging = [0,0]  #[start,end]  [0,0] => image just 'ave'
sep = 5
VWRunit = 40          # v per 1mm
Uunit = 20
Tunit = 1000        #temp per 1mm
period = 'eoi'   #eoi or all
shaping = 'single'
scale = 1.0             # y axis scale of U and VWR
aveDu = 0.05  #ms
fontz = 22
Tdelta = 1 #[K]   #constant?
method = lambda x, y: interpolate.interp1d(x, y, kind="nearest")#"補間"
rlim = 15
'''____________________'''

print('length??? I mean just delta length.')
#version = str(input())
length = '_'+str(scale)+'mm'
scale = float(length[1:-2])
print('OK! Let\'s process '+length)

#4ms is not always true Don't care about the number

#"""Read shapeWallave data for calc range"""
#os.chdir('C:/Users/power/Desktop/python/shapeWall')
#
#if 'mm' in version:
#    file = version[:13]
#    scale = float(version[14:-2])
##




class Version:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.time = 0
        self.Rs = []
        self.delt = []
        self.temp = []
        self.tempsonomama = []
        self.VWRpeak = []
        self.Tpeak = []
        self.VWRpeakdelt = []
        self.Tpeakdelt = []
        self.res = []
        self.HF = []
        self.HFr = []
        self.HFSr = []
        self.shapedic = {}
        self.Uflow = []
        self.VWRflow = []
        self.Upeak = []
        self.Upeakdelt = []
        self.centre = []
        self.Tdeltadtdx = []

        
        
        
        
inslist = [] 
#versions = ['single_130MPa','single_200MPa','single_270MPa']
if shaping == 'inv':
    versions = ['single_200MPa','inv1','inv2']
elif shaping == 'last':
    versions = ['single_200MPa','last1','last2']
elif shaping == 'soft':
    versions = ['single_200MPa','soft1','soft2']
elif shaping == 'group1':
    versions = ['soft1','inv1','last1']
elif shaping == 'group2':
    versions = ['soft2','inv2','last2']
elif shaping == 'group3':
    versions = ['soft3','inv3','last3']
elif shaping == 'group3_ave':
    versions = ['soft3_ave','inv3_ave','last3_ave']
elif shaping == 'single':
    versions = ['single_130MPa','single_200MPa','single_270MPa']
elif shaping == '200vali':
    versions = ['single_200MPa_kakudo2NF','single_200MPa_kakudoNF','single_200MPa_kakudoNF2']
elif shaping == 'scoreValidation':
    versions = ['single_130MPa','130_kakudoNF','130_kakudo2NF','130_kakudoNF2','130_kakudo+NF']
elif shaping == 'soft3':
    versions = ['soft3','soft3_2','soft3_3','soft3_ave']
elif shaping == 'last3':
    versions = ['last3','last3_2','last3_3','last3_ave']
elif shaping == 'inv3':
    versions = ['inv3','inv3_2','inv3_3','inv3_ave']
else:
    versions = ['single_130MPa','single_200MPa','single_270MPa']


'''
X(mm)	delta(mm)	R(mm)	F_ave	U_ave(m/s)	VWR_ave(m/s)	UVW_ave(m/s)	T_ave	Heat_Flux[Q/m2]  Zcen_b(mm)  Ycen_b(mm)
0           1         2       3         4             5             6            7           8             9           10
''' 

#Tdeltadtdx = []
#BdLayer_vwr = []
#os.chdir('C:/Users/power/Desktop/python/U_VWR_FLOW')
#for version in versions:
#    with open('C:/Users/power/Desktop//python/U_VWR_FLOW/Tdeltadtdx_'+version+'.csv', 'r', newline='') as fg:
#        reader = csv.reader(fg)
#        readerlist = list(reader)
#        for row in readerlist[1:]:
#            Tdeltadtdx.append(row)
#    with open('C:/Users/power/Desktop//python/U_VWR_FLOW/BdLayer_delt_linearIp_vwrflow_'+version+'.csv', 'r', newline='') as fg:
#        reader = csv.reader(fg)
#        readerlist = list(reader)
#        for row in readerlist[1:]:
#            BdLayer_vwr.append(row)

lssr = []
lsr = [x*0.5 for x in range(0,42)][:-1]
for r in lsr:
    sr = math.pi*(r+0.25)*(r+0.25) - sum(lssr)
    lssr.append(sr)
    
for version in versions:   
    os.chdir('C:/Users/power/Desktop/python/U_VWR_FLOW/log_temp_flow_U_'+version)
    
    namet = []
    namet = glob.glob('./log_temp_flow_U_VWR*.csv')      


    for name in namet[:timelimit]:
        labelname = str(float(name[22:-8]))+'ms'
        ins = Version(version,labelname)
        ins.time = float(labelname[:-2])
        with open(name, newline='') as name:
    
            reader = csv.reader(name)
            readerlist = list(reader)
            readerlist = readerlist[1:]
            
            if(len(readerlist[0])==12):ins.centre = [readerlist[0][9],readerlist[0][10]]
            for p,row in enumerate(readerlist):
                if not (float(row[2])%0.5) == 0:
                    continue
                ins.Rs.append(float(row[2]))
                ins.HF.append(float(row[8]))
                if (p%16 == 0):
                    ins.HFSr.append(float(row[8])*(lssr[(p//16)]))
                ins.HFr.append(float(row[2])*float(row[8]))                
                
                
                ins.delt.append(float(row[1]))
#                ins.vflow.append(float(row[3]))
                ins.temp.append(float(row[7]))  
                ins.tempsonomama.append(float(row[7]))
                ins.Uflow.append(float(row[4]))
                ins.VWRflow.append(float(row[5]))

                
    #        print(ins.name)
            print(ins.label)
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
        ins.Uflow[i] =ins.Uflow[i]/Uunit + R
        ins.VWRflow[i] = ins.VWRflow[i]/VWRunit + R
        ins.temp[i] = ins.temp[i]/Tunit +R-0.5
        rs = sorted(list(set(ins.Rs))) # rs has no overlaping
    for r in rs:
        rls =[] # index list
        for i in range(0,len(ins.Rs)):
            if ins.Rs[i] == r:
                rls.append(i)
        "各rでの壁面近傍3点の温度勾配により算出する温度境界層厚さの計算"
        #a,b=np.polyfit(ins.tempsonomama[rls[0]:rls[3]],ins.delt[rls[0]:rls[3]], 1)
        a,b=np.polyfit(ins.delt[rls[0]:rls[3]],ins.tempsonomama[rls[0]:rls[3]], 1)
        if 0<a<100000:
            #ins.Tdeltadtdx.append(Tdelta/a)
            ins.Tdeltadtdx.append(a)
        else:
            ins.Tdeltadtdx.append(0)
            errorcounter +=1
                
        "各ｒでの内挿された最大値の包絡線による擬似境界層厚さの計算"
#        shape = int(2*shapedic[ins.time])
#        if shape <= 5:shape = 6
#        wt=np.polyfit(ins.delt[rls[0]:rls[-1]],ins.temp[rls[0]:rls[-1]], 5)
        wv=np.polyfit(ins.delt[rls[0]:rls[-1]],ins.VWRflow[rls[0]:rls[-1]], 5)
        
#        y1 = np.poly1d(wt)(y_latent)
        y2 = np.poly1d(wv)(y_latent)
#        ins.Tpeak.append(max(y1))
#        ins.Tpeakdelt.append(y_latent[np.argmax(y1)])
        ins.VWRpeak.append(max(y2))
        ins.VWRpeakdelt.append(y_latent[np.argmax(y2)])
    ins.VWRpeak[0],ins.VWRpeakdelt[0] = 0,0

##        
#print('Adjusting data...')
#for ins in inslist:
#    deltls = ins.delt
#    rsls = ins.Rs
#    for i in range(0,len(ins.Rs)):
#        R = ins.Rs[i]
#        ins.Uflow[i] = ins.Uflow[i]/Uunit + R
#        ins.VWRflow[i] = ins.VWRflow[i]/VWRunit + R
#        ins.temp[i] = (ins.temp[i]-473)/Tunit + R
#        rs = sorted(list(set(ins.Rs)))
#    for r in rs:
#        rls =[]
#        for i in range(0,len(ins.Rs)):
#            if ins.Rs[i] == r:
#                rls.append(i)
#        
#        VWRpeak = max(ins.VWRflow[rls[0]:rls[-1]])
#        ins.VWRpeak.append(VWRpeak)
#        ins.VWRpeakdelt.append(ins.delt[ins.VWRflow.index(VWRpeak)])
#        Tpeak = max(ins.temp[rls[0]:rls[-1]])
#        ins.Tpeak.append(Tpeak)
#        ins.Tpeakdelt.append(ins.delt[ins.temp.index(Tpeak)])
#        Upeak = max(ins.Uflow[rls[0]:rls[-1]])
#        ins.Upeak.append(Upeak)
#        ins.Upeakdelt.append(ins.delt[ins.Uflow.index(Upeak)])


"""-------------calc time-averaged V and Temp + HF ------------ """
#start = 21  #10 = 0.5ms       0.5ms before EOI
  #130-39 200-31 270-27                       enddic has num of 0.2ms before EOI
#enddic = {'single_200MPa':31,'single_130MPa':39,'single_270MPa':27,\
#          'inv2':34,'inv1':32,'soft1':32,'soft2':34,'last1':32,'last2':34,\
#         'soft3':35,'inv3':35,'last3':35,'single_200MPa_4':31,'single_130MPa_4':39,'single_270MPa_4':27\
#         ,'130_kakudoNF':39,'130_kakudo2NF':39,'130_kakudoNF2':39,'130_kakudo+NF':39,\
#         'single_200MPa_kakudo2NF':31,'single_200MPa_kakudoNF':31,'single_200MPa_kakudoNF2':31,\
#         'single_270MPa_kakudo2NF':27,'single_270MPa_kakudoNF':27,'single_270MPa_kakudoNF2':27}
enddic = {'single_200MPa':314,'single_130MPa':390,'single_270MPa':270,\
          'inv2':336,'inv1':326,'soft1':326,'soft2':336,'last1':326,'last2':336,\
          'soft3':352,'last3':352,'inv3':352,'soft3_2':352,'soft3_3':352,'soft3_ave':352\
          ,'inv3_2':352,'inv3_3':352,'inv3_ave':352,'last3_2':352,'last3_3':352,'last3_ave':352}
#End of injection
#shape = 23 #10 = 5mm   130,200,270共通
#   ---------------use shapedic from csv
#versions = ['single_200MPa','single_130MPa','single_270MPa']
print('calclating time-averaged V and Temp + HF...')
for case,version in enumerate(versions):
    if period == 'eoi':
        end = enddic[version]+(case)*timelimit-40   # -40 avoid [last 0.2ms] of tinj
        start = (case)*timelimit + 1
        start = end - int(aveDu/0.005)
    else:
        end = (case+1)*timelimit
        start = case*timelimit + 1
    print(version,start,end)

            
    aveV = [0 for x in deltls]
    aveU = [0 for x in deltls]
    aveT = [0 for x in deltls]
    aveHF = [0 for x in deltls]
    aveHFSr = [0 for x in range(0,41)]
    for ins in inslist[start:end]:
        for i,velo in enumerate(ins.VWRflow):
            aveV[i] += velo
        for i,Uelo in enumerate(ins.Uflow):
            aveU[i] += Uelo
        for i,temp in enumerate(ins.temp):
            aveT[i] += temp
        for i,HF in enumerate(ins.HF):
            aveHF[i] += HF
            if (i%16 == 0):
                aveHFSr[i//16] += HF * lssr[i//16]
    for i in range(0,len(aveV)):
        aveV[i] = aveV[i]/len(inslist[start:end])
        aveU[i] = aveU[i]/len(inslist[start:end])
        aveT[i] = aveT[i]/len(inslist[start:end])
        aveHF[i]= aveHF[i]/len(inslist[start:end])
    for i in range(len(aveHFSr)):
        aveHFSr[i]= aveHFSr[i]/len(inslist[start:end])
    
    aveVpeak = []
    aveTpeak = []
    aveUpeak = []
    aVpdelt = []
    aUpdelt = []
    atpdelt = []
    aveTdeltadtdx = []
    aveTsonomama = []
    for num,ave in enumerate(aveT):
        aveTsonomama.append((ave+R+0.5)*Tunit)
    rs = sorted(list(set(rsls)))
    for r in rs:
        rls =[]
        for i in range(0,len(rsls)):
            if rsls[i] == r:
                rls.append(i)
        "各rでの壁面近傍3点の温度勾配により算出する温度境界層厚さの計算"
        a,b=np.polyfit(aveTsonomama[rls[0]:rls[3]],deltls[rls[0]:rls[3]], 1)
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
        aVpdelt.append(y_latent[np.argmax(y2a)])   
#        aveUpeak.append(Upeak)
#        aUpdelt.append(deltls[aveU.index(Upeak)])
    if period == 'all':
        
        ave = Version(version,'Ave_of_2ms_ASOI')
    else:
        ave = Version(version,'Ave_of_'+str(aveDu)+'ms_BEOI')
    ave.Rs = rsls
    ave.time = 'ave'
    ave.delt = deltls
    ave.VWRflow = aveV
    ave.temp = aveT
    ave.VWRpeak = aveVpeak
    ave.Tpeak = aveTpeak
    ave.VWRpeakdelt = aVpdelt
    ave.Tpeakdelt = atpdelt
#    ave.Upeak = aveUpeak
#    ave.Upeakdelt = aUpdelt
    ave.HF = aveHF
    ave.HFSr = aveHFSr
    for num, hf in enumerate(aveHF):
        ave.HFr.append(hf*rsls[num])
    ave.Uflow = aveU
    ave.Tdeltadtdx = aveTdeltadtdx
    ave.VWRpeak[0],ave.VWRpeakdelt[0] = 0,0
        
    
    inslist.append(ave)
    print('ok')

'''---------------------------------save averaged data--------------------------'''
lab = ['R(mm)','delt(mm)','U_ave(m/s)','VWR_ave(m/s)','T_ave','Heat_Flux[Q/m2]','HfSr','HFr']

os.chdir('C:/Users/power/Desktop/python/U_VWR_FLOW')   
for num,version in enumerate(versions):
    ave = inslist[-3+num]
    
    with open('log_temp_flow_U_VWR_ave_'+version+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(lab)
        for i in range(len(ave.Rs)):
            R = ave.Rs[i]
            if (i%16 == 0):
                spamwriter.writerow([ave.Rs[i],ave.delt[i],(ave.Uflow[i]-R)*Uunit,(ave.VWRflow[i]-R)*VWRunit,(ave.temp[i]-R+0.5)*Tunit,ave.HF[i],ave.HFr[i],ave.HFSr[int(i/16)]])
            else:
                spamwriter.writerow([ave.Rs[i],ave.delt[i],(ave.Uflow[i]-R)*Uunit,(ave.VWRflow[i]-R)*VWRunit,(ave.temp[i]-R+0.5)*Tunit,ave.HF[i]])
            







'''-------------------------save temporal data of each R------------------------'''
saveR = [0,5,10,15]
TempBdofR = []
VflowBdofR = []
WHFofR = []       #make 3 each lines of csv file 
HFSrofR = []
MaxVofR = []
for i,insori in enumerate(inslist[:timelimit]):
    sametime = [insori]
    for instance in inslist[inslist.index(insori)+1:]:
        if instance.time == insori.time:
            sametime.append(instance)

    caseT=[]
    caseV=[]
    caseWHF=[]
    caseHFSr = []
    casemaxV = []
    for R in saveR:
        for ins in sametime:
            caseT.append(ins.Tdeltadtdx[R*2])
            caseV.append(ins.VWRpeakdelt[R*2])
            caseWHF.append(ins.HF[R*2])
            caseHFSr.append(ins.HFSr[R*2])
            casemaxV.append((ins.VWRpeak[R*2]-R)*VWRunit)
    TempBdofR.append(caseT)
    VflowBdofR.append(caseV)
    WHFofR.append(caseWHF)
    HFSrofR.append(caseHFSr)
    MaxVofR.append(casemaxV)
    
emp = []
for colum in [[version+'_R='+str(R) for version in versions] for R in saveR]:
    emp = emp + colum
os.chdir('C:/Users/power/Desktop/python/U_VWR_FLOW')   
with open('TempBLofR.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+emp)
    for i in range(len(TempBdofR)):
        spamwriter.writerow([i*5*10**(-3)]+TempBdofR[i])
with open('VflowBLofR.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+emp)
    for i in range(len(VflowBdofR)):
        spamwriter.writerow([i*5*10**(-3)]+VflowBdofR[i])
with open('WHFofR.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+emp)
    for i in range(len(WHFofR)):
        spamwriter.writerow([i*5*10**(-3)]+WHFofR[i])
with open('HFSrofR.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+emp)
    for i in range(len(HFSrofR)):
        spamwriter.writerow([i*5*10**(-3)]+HFSrofR[i])
#with open('HFrofR.csv', 'w', newline='') as csvfile:
#    spamwriter = csv.writer(csvfile)
#    spamwriter.writerow(['ms']+emp)
#    for i in range(len(HFrofR)):
#        spamwriter.writerow([i*5*10**(-3)]+HFrofR[i])
with open('MaxVofR.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['ms']+emp)
    for i in range(len(MaxVofR)):
        spamwriter.writerow([i*5*10**(-3)]+MaxVofR[i])

'''-------------------plot-------------------------'''  
        
if not os.path.exists("C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_BdLayer_"+shaping+length):
    os.mkdir("C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_BdLayer_"+shaping+length)
    
x_latent = np.linspace(0,15,1000)     
colorlist = ["g", "r", "b", "m", "y", "k", "w"]   
#colorlist = ['xkcd:orange','xkcd:violet','xkcd:bright blue','y','k','w'] 

for i,insori in enumerate(inslist[imaging[0]:imaging[1]]+inslist[-len(versions):-len(versions)+1]):
    sametime = [insori]
    for instance in inslist[inslist.index(insori)+1:]:
        if instance.time == insori.time:
            sametime.append(instance)
    #if (sametime[0].time != 'ave'):continue    

    '''plot temp'''
    plt.figure(figsize=(12,28))
    ax1 = plt.subplot(411)
    ax1.set_xlim(0,rlim + 0.01)
    ax1.set_ylim(0.0,scale)
    
    ax6 = ax1.twinx()
    ax6.set_ylim(0,30000)
    #plt.title("INJECTIONRATE", fontsize=30)
    for r in lsr[::sep]:
        ax1.plot([r for delt in deltls],deltls,color = colorlist[-2], marker = '_',markersize=4,linestyle= 'dotted', alpha = 0.4)
    for num,ins in enumerate(sametime):
        rs = sorted(list(set(ins.Rs)))
        for r in rs[first::sep]:
            rls =[]
            for i in range(0,len(ins.Rs)):
                if ins.Rs[i] == r:
                    rls.append(i)
            ax1.plot(ins.temp[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',linestyle= 'dotted',color = colorlist[num])
    #        plt.show()
        ax6.plot(rs,ins.Tdeltadtdx,color = colorlist[num],label = ins.name)      
        #plt.plot(ins.Tpeak,ins.Tpeakdelt,color = colorlist[num],linestyle= 'dotted',label = ins.name)   
#    ax1.legend(loc='uppper right',
#           bbox_to_anchor=(1.2, 0.8, 0.2,0.1), 
#           borderaxespad=0.,prop={'size':16,})
#    "近似式の計算"
#    shape = int(2*ins.shapedic[ins.time])
#    if shape <= 5:shape = 6
#    res1=np.polyfit(ins.Tpeak[5:shape], ins.Tpeakdelt[5:shape], 1)
#    print(res1,ins.label,'temp')
#    y1 = np.poly1d(res1)(ins.Tpeak[5:shape])
#    plt.plot(ins.Tpeak[5:shape],y1,color = 'k',linestyle= 'dashed')
    ax1.set_yticks( np.arange(0, scale+0.01, 0.5) )     
    ax6.set_yticks( np.arange(0, 30000+0.01, 15000) )     
    ax1.set_xlabel('                                           R                                 [mm]', fontsize=fontz)
    #plt.ylabel('                    '+r'$\delta$'+'             [mm]', fontsize=fontz)
    ax1.set_ylabel('                    '+r'$\delta$'+'             [mm]', fontsize=fontz)
    ax6.set_ylabel('                  '+r'$dT/d\delta$'+'       [K/mm]', fontsize=fontz) 
    #plt.title('Temperature_'+ins.label, fontsize=fontz)
    plt.title(ins.label+' from SOI', fontsize=fontz)
    #plt.title('Temperature [1mm--250K]', fontsize=fontz)
    #plt.setp(ax1.get_xticklabels(), visible=False)
#   plt.legend(fontsize = 'small',frameon = None,prop={'size':20,},bbox_to_anchor=(1.05, 1))
    plt.xticks( np.arange(0, rlim + 0.01,5 ) )
    #plt.tick_params(labelsize = fontz,direction = 'in',length = 7)
    ax1.tick_params(labelsize = fontz,direction = 'in',length = 7)
    ax6.tick_params(labelsize = fontz,direction = 'in',length = 7)
    #plt.grid()
    #plt.show()
    
    
    '''plot HF'''
    mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
    ax2 = plt.subplot(412, sharex=ax1)
    ax2.set_xlim(0,rlim + 0.01)
    ax2.set_ylim(0,1000)
    
    
    ax5 = ax2.twinx()
    ax5.set_ylim(0,1000)
    #plt.title("Wall_Heat_Flux"+ins.label, fontsize=fontz)
    for num,ins in enumerate(sametime):
        ax2.plot(ins.Rs,ins.HF,color =colorlist[num],label = ins.name,linestyle= 'dotted')
#        ax5.plot(rs,ins.HFSr,color = colorlist[num],linestyle= 'dotted')
        fitted_curve = method(rs, ins.HFSr)
        ax5.plot(x_latent,fitted_curve(x_latent),color = colorlist[num])
    ax2.legend(loc='uppper right',
       bbox_to_anchor=(0.6, 0.8, 0.2,0.1), 
       borderaxespad=0.,prop={'size':16,})
#    plt.rcParams.update({'mathtext.default': 'default',
#                     'mathtext.fontset': 'stix',
#                     'font.family': 'Times New Roman',})
    ax2.set_ylabel('   Averaged Heat Flux    ['+'$MW/mm^{2}$'+']', fontsize=fontz)
    ax5.set_ylabel('          Heat Flux × '+'$S_R$'+'       [MW]', fontsize=fontz)   

    ax2.set_yticks(np.arange(0,1001,250))
    ax5.set_yticks(np.arange(0,2001,500))
#    plt.ylim(0,)
    ax2.set_xlabel('                                           R                                 [mm]', fontsize=fontz)
    
    #plt.ylabel('             HF          MW/mm2', fontsize=fontz)
    #plt.title('Wall_Heat_Flux_'+ins.label, fontsize=fontz)

    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, rlim + 0.01,5 ) )
    ax2.tick_params(labelsize = fontz,direction = 'in',length = 7)
    ax5.tick_params(labelsize = fontz,direction = 'in',length = 7)
#   plt.show()
    #plt.grid()
    
    '''plot VWRflow'''
    ax3 = plt.subplot(413, sharex=ax1)
    ax3.set_xlim(0,rlim + 0.01)
    ax3.set_ylim(0.0,scale)
    #plt.title("dp/dt_movingAve")
    for r in lsr[::sep]:
        ax3.plot([r for delt in deltls],deltls,color = colorlist[-2], marker = '_',markersize=4,linestyle= 'dotted', alpha = 0.5)
    for num,ins in enumerate(sametime):
            
        rs = sorted(list(set(ins.Rs)))
    
        for r in rs[::sep]:
            rls =[]
            for i in range(0,len(ins.Rs)):
                if ins.Rs[i] == r:
                    rls.append(i)
            ax3.plot(ins.VWRflow[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',linestyle= 'dotted',color = colorlist[num])
        plt.plot(ins.VWRpeak,ins.VWRpeakdelt,color = colorlist[num],label = ins.name)
    ax3.legend(loc='uppper right',
           bbox_to_anchor=(1.2, 0.8, 0.2,0.1), 
           borderaxespad=0.,prop={'size':16,})
#    "近似式の計算"
#    shape = int(2*ins.shapedic[ins.time])
#    if shape <= 5:shape = 6
#    res2=np.polyfit(ins.Vpeak[5:shape], ins.Vpeakdelt[5:shape], 1)
#    ins.res = list(res1)+list(res2)
#    print(res2,ins.label,'vflow')
    print('VWRflow 1mm='+str(VWRunit)+'m/s,  1mm = '+str(Tunit)+'K')
#    y2 = np.poly1d(res2)(ins.Vpeak[5:shape])
#    plt.plot(ins.Vpeak[5:shape],y2,color = 'k',linestyle= 'dashed')
    
#   ax2.scatter(ins.vflow,ins.delt,label=ins.label,marker ='.')
    plt.yticks( np.arange(0, scale+0.01, scale/2) )
#    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R      mm', fontsize=20)
    plt.ylabel('                    '+r'$\delta$'+'            [mm]', fontsize=fontz)
    plt.xlabel('                                           R                                 [mm]', fontsize=fontz)
    #plt.title('HorizonalFlow_'+ins.label, fontsize=fontz)
    #plt.title('Horizontal Flow [1mm--20m/s]', fontsize=fontz)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, rlim + 0.01,5 ) )
    plt.tick_params(labelsize = fontz,direction = 'in',length = 7)
#   plt.show()
    #plt.grid()
    
    '''plot Uflow'''
    ax4 = plt.subplot(414, sharex=ax1)
    ax4.set_xlim(0,rlim + 0.01)
    ax4.set_ylim(0.0,scale)
    #plt.title("dp/dt_movingAve")
    for r in lsr[::sep]:
        ax4.plot([r for delt in deltls],deltls,color = colorlist[-2], marker = '_',markersize=4,linestyle= 'dotted', alpha = 0.5)
    for num,ins in enumerate(sametime):
            
        rs = sorted(list(set(ins.Rs)))
    
        for r in rs[::sep]:
            rls =[]
            for i in range(0,len(ins.Rs)):
                if ins.Rs[i] == r:
                    rls.append(i)
            ax4.plot(ins.Uflow[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',color = colorlist[num],linestyle= 'dotted')
        #plt.plot(ins.Upeak,ins.Upeakdelt,linestyle= 'dotted',color = colorlist[num],label = ins.name)
    ax4.legend(loc='uppper right',
           bbox_to_anchor=(1.2, 0.8, 0.2,0.1), 
           borderaxespad=0.,prop={'size':16,})
#    "近似式の計算"
#    shape = int(2*ins.shapedic[ins.time])
#    if shape <= 5:shape = 6
#    res2=np.polyfit(ins.Vpeak[5:shape], ins.Vpeakdelt[5:shape], 1)
#    ins.res = list(res1)+list(res2)
#    print(res2,ins.label,'vflow')
    print('Uflow  1mm='+str(Uunit)+'m/s,  1mm = '+str(Tunit)+'K')
    for ins in sametime:
        if (ins.centre!=[]):print(str(ins.name)+'   Zcen_b(mm)  Ycen_b(mm)' + str(ins.centre))
#    y2 = np.poly1d(res2)(ins.Vpeak[5:shape])
#    plt.plot(ins.Vpeak[5:shape],y2,color = 'k',linestyle= 'dashed')
    
#   ax2.scatter(ins.vflow,ins.delt,label=ins.label,marker ='.')
    plt.yticks( np.arange(0, scale+0.01, scale/2) )
#    plt.yticks( np.arange(0, 4.01, 1) )
    plt.xlabel('                                           R                                 [mm]', fontsize=fontz)
    plt.ylabel('                     '+r'$\delta$'+'            [mm]', fontsize=fontz)
    #plt.title('VerticalFlow_'+ins.label, fontsize=fontz)
    #plt.title('Vertical Flow [1mm--20m/s]', fontsize=fontz)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, rlim + 0.01,5 ) )
    plt.tick_params(labelsize = fontz,direction = 'in',length = 7)
    
    filename = "C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_BdLayer_"+shaping+length+"/U_VWR_"+ins.label+"ms.svg"
    plt.savefig(filename, format="svg")
#    filename = "C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_BdLayer_"+shaping+length+"/U_VWR_"+ins.label+".png"
#    plt.savefig(filename)
    plt.show()


if period == 'all':
    print('average of '+str(0.005*timelimit)+'ms')
else:
    print('average of '+str(aveDu)+' ms before EOI')
    
    





    
