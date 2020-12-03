# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:53:42 2019

@author: power
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os





'''\\\\\\module\\\\\\\_'''
first = 0
timelimit = 400  #400 - 890
sep = 2
VWRunit = 20          # v per 1mm
Uunit = 20
Tunit = 250        #temp per 1mm
period = 'eoi'   #eoi or all
shaping = 'group3_ave'
scale = 1.0             # y axis scale of U and VWR
aveDu = 0.2  #ms
fontz = 25
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
        self.VWRpeak = []
        self.Tpeak = []
        self.VWRpeakdelt = []
        self.Tpeakdelt = []
        self.res = []
        self.HF = []
        self.shapedic = {}
        self.Uflow = []
        self.VWRflow = []
        self.Upeak = []
        self.Upeakdelt = []
        self.centre = []
        
        
        
        
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
    
for version in versions:   
    os.chdir('C:/Users/power/Desktop/python/U_VWR_FLOW/log_temp_flow_U_'+version)
    
    namet = []
    namet = glob.glob('./log_temp_flow_U_VWR*.csv')      


    for name in namet[:timelimit]:
        labelname = str(float(name[22:-8]))
        ins = Version(version,labelname)
        ins.time = float(labelname[:])
        with open(name, newline='') as name:
    
            reader = csv.reader(name)
            readerlist = list(reader)
            readerlist = readerlist[1:]
            
            if(len(readerlist[0])==12):ins.centre = [readerlist[0][9],readerlist[0][10]]
            for row in readerlist:
                if not (float(row[2])%0.5) == 0:
                    continue
                ins.Rs.append(float(row[2]))
                ins.HF.append(float(row[8]))
                
                
                ins.delt.append(float(row[1]))
#                ins.vflow.append(float(row[3]))
                ins.temp.append(float(row[7]))   
                ins.Uflow.append(float(row[4]))
                ins.VWRflow.append(float(row[5]))

                
    #        print(ins.name)
            print(ins.label)
    #        print(ins.Rs)
    #        print(ins.delt)
    #        print(ins.cumulative)
            inslist.append(ins)
            
#
#for version in versions:
#    with open('C:/Users/power/Desktop//python/shapeWall/SHAPE_WallRAVE_'+version+'.csv', 'r', newline='') as fg:
#        reader = csv.reader(fg)
#        readerlist = list(reader)
#        for row in readerlist:
#            ins.shapedic[float(row[0])]=float(row[1])
#
#
#    ins.shapedic['ave'] = 12

#        
print('Adjusting data...')
for ins in inslist:
    deltls = ins.delt
    rsls = ins.Rs
    for i in range(0,len(ins.Rs)):
        R = ins.Rs[i]
        ins.Uflow[i] = ins.Uflow[i]/Uunit + R
        ins.VWRflow[i] = ins.VWRflow[i]/VWRunit + R
        ins.temp[i] = (ins.temp[i]-473)/Tunit + R
        rs = sorted(list(set(ins.Rs)))
    for r in rs:
        rls =[]
        for i in range(0,len(ins.Rs)):
            if ins.Rs[i] == r:
                rls.append(i)
        
        VWRpeak = max(ins.VWRflow[rls[0]:rls[-1]])
        ins.VWRpeak.append(VWRpeak)
        ins.VWRpeakdelt.append(ins.delt[ins.VWRflow.index(VWRpeak)])
        Tpeak = max(ins.temp[rls[0]:rls[-1]])
        ins.Tpeak.append(Tpeak)
        ins.Tpeakdelt.append(ins.delt[ins.temp.index(Tpeak)])
        Upeak = max(ins.Uflow[rls[0]:rls[-1]])
        ins.Upeak.append(Upeak)
        ins.Upeakdelt.append(ins.delt[ins.Uflow.index(Upeak)])


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

            
    aveVWR = [0 for x in deltls]
    aveU = [0 for x in deltls]
    aveT = [0 for x in deltls]
    aveHF = [0 for x in deltls]
    for ins in inslist[start:end]:
        for i,velo in enumerate(ins.VWRflow):
            aveVWR[i] += velo
        for i,Uelo in enumerate(ins.Uflow):
            aveU[i] += Uelo
        for i,temp in enumerate(ins.temp):
            aveT[i] += temp
        for i,HF in enumerate(ins.HF):
            aveHF[i] += HF
    for i in range(0,len(aveVWR)):
        aveVWR[i] = aveVWR[i]/len(inslist[start:end])
        aveU[i] = aveU[i]/len(inslist[start:end])
        aveT[i] = aveT[i]/len(inslist[start:end])
        aveHF[i]= aveHF[i]/len(inslist[start:end])
    
    aveVWRpeak = []
    aveTpeak = []
    aveUpeak = []
    aVWRpdelt = []
    aUpdelt = []
    atpdelt = []
    rs = sorted(list(set(rsls)))
    for r in rs:
        rls =[]
        for i in range(0,len(rsls)):
            if rsls[i] == r:
                rls.append(i)
        Tpeak = max(aveT[rls[0]:rls[-1]])
        VWRpeak = max(aveVWR[rls[0]:rls[-1]])
        Upeak = max(aveU[rls[0]:rls[-1]])
        aveVWRpeak.append(VWRpeak)
        aVWRpdelt.append(deltls[aveVWR.index(VWRpeak)])
        aveTpeak.append(Tpeak)
        atpdelt.append(deltls[aveT.index(Tpeak)])    
        aveUpeak.append(Upeak)
        aUpdelt.append(deltls[aveU.index(Upeak)])
        
    ave = Version(version,'ave'+period)
    ave.Rs = rsls
    ave.time = 'ave'
    ave.delt = deltls
    ave.VWRflow = aveVWR
    ave.temp = aveT
    ave.VWRpeak = aveVWRpeak
    ave.Tpeak = aveTpeak
    ave.VWRpeakdelt = aVWRpdelt
    ave.Tpeakdelt = atpdelt
    ave.Upeak = aveUpeak
    ave.Upeakdelt = aUpdelt
    ave.HF = aveHF
    ave.Uflow = aveU
        
    
    inslist.append(ave)
    print('ok')



colorlist = ["r", "g", "b", "m", "y", "k", "w"]
    

'''-------------------plot-------------------------'''  
        
if not os.path.exists("C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_"+shaping+length):
    os.mkdir("C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_"+shaping+length)
    
    
for i,insori in enumerate(inslist[:timelimit]+inslist[-len(versions):-len(versions)+1]):
    sametime = [insori]
    for instance in inslist[inslist.index(insori)+1:]:
        if instance.time == insori.time:
            sametime.append(instance)
    #if (sametime[0].time != 'ave'):continue    
    '''plot temp'''
    plt.figure(figsize=(10,30))
    ax1 = plt.subplot(411)
    ax1.set_xlim(0,20.01)
    ax1.set_ylim(0.0,scale)
    #plt.title("INJECTIONRATE", fontsize=30)
    for num,ins in enumerate(sametime):
        rs = sorted(list(set(ins.Rs)))
        for r in rs[first::sep]:
            rls =[]
            for i in range(0,len(ins.Rs)):
                if ins.Rs[i] == r:
                    rls.append(i)
            ax1.plot(ins.temp[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',color = colorlist[num])
    #        plt.show()
        #plt.plot(ins.Tpeak,ins.Tpeakdelt,color = colorlist[num],linestyle= 'dotted',label = ins.name)   
    ax1.legend(loc='uppper right',
           bbox_to_anchor=(0.8, 0.8, 0.2,0.1), 
           borderaxespad=0.,prop={'size':16,})
#    "近似式の計算"
#    shape = int(2*ins.shapedic[ins.time])
#    if shape <= 5:shape = 6
#    res1=np.polyfit(ins.Tpeak[5:shape], ins.Tpeakdelt[5:shape], 1)
#    print(res1,ins.label,'temp')
#    y1 = np.poly1d(res1)(ins.Tpeak[5:shape])
#    plt.plot(ins.Tpeak[5:shape],y1,color = 'k',linestyle= 'dashed')
    plt.yticks( np.arange(0, scale+0.01, 0.2) )     
#    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R', fontsize=30)
    plt.ylabel('   delt     mm', fontsize=fontz)
    plt.title('Temp_'+ins.label+'ms', fontsize=fontz)
    #plt.title('Temperature [1mm--250K]', fontsize=fontz)
    #plt.setp(ax1.get_xticklabels(), visible=False)
#   plt.legend(fontsize = 'small',frameon = None,prop={'size':20,},bbox_to_anchor=(1.05, 1))
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = fontz,direction = 'in',length = 7)
    #plt.grid()
    #plt.show()
    
    '''plot HF'''
    ax2 = plt.subplot(412, sharex=ax1)
    ax2.set_xlim(0,20)
    #plt.title("Wall_Heat_Flux"+ins.label, fontsize=fontz)
    plt.title('Wall Heat Flux', fontsize=fontz)
    for num,ins in enumerate(sametime):
        ax2.plot(ins.Rs,ins.HF,color =colorlist[num],label = ins.name)
        ax2.legend(loc='uppper right',
           bbox_to_anchor=(0.8, 0.8, 0.2,0.1), 
           borderaxespad=0.,prop={'size':16,})

    
#   ax2.scatter(ins.vflow,ins.delt,label=ins.label,marker ='.')
#    plt.yticks( np.arange(0, scale+0.01, scale/2) )
#    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R      mm', fontsize=20)
    plt.ylabel('   HF    MW/mm2', fontsize=fontz)
    plt.title('HF_'+ins.label+'_'+'ms', fontsize=15)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = fontz,direction = 'in',length = 7)
#   plt.show()
    #plt.grid()
    
    '''plot VWRflow'''
    ax3 = plt.subplot(413, sharex=ax1)
    ax3.set_xlim(0,20)
    ax3.set_ylim(0.0,scale)
    #plt.title("dp/dt_movingAve")
    for num,ins in enumerate(sametime):
            
        rs = sorted(list(set(ins.Rs)))
    
        for r in rs[::sep]:
            rls =[]
            for i in range(0,len(ins.Rs)):
                if ins.Rs[i] == r:
                    rls.append(i)
            ax3.plot(ins.VWRflow[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',color = colorlist[num])
        #plt.plot(ins.VWRpeak,ins.VWRpeakdelt,linestyle= 'dotted',color = colorlist[num],label = ins.name)
    ax3.legend(loc='uppper right',
           bbox_to_anchor=(0.8, 0.8, 0.2,0.1), 
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
    plt.ylabel('   delt     mm', fontsize=fontz)
    plt.title('VWRflow_'+ins.label+'ms', fontsize=fontz)
    #plt.title('Horizontal Flow [1mm--20m/s]', fontsize=fontz)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = fontz,direction = 'in',length = 7)
#   plt.show()
    #plt.grid()
    
    '''plot Uflow'''
    ax4 = plt.subplot(414, sharex=ax1)
    ax4.set_xlim(0,20)
    ax4.set_ylim(0.0,scale)
    #plt.title("dp/dt_movingAve")
    for num,ins in enumerate(sametime):
            
        rs = sorted(list(set(ins.Rs)))
    
        for r in rs[::sep]:
            rls =[]
            for i in range(0,len(ins.Rs)):
                if ins.Rs[i] == r:
                    rls.append(i)
            ax4.plot(ins.Uflow[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',color = colorlist[num])
        #plt.plot(ins.Upeak,ins.Upeakdelt,linestyle= 'dotted',color = colorlist[num],label = ins.name)
    ax4.legend(loc='uppper right',
           bbox_to_anchor=(0.8, 0.8, 0.2,0.1), 
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
    plt.xlabel('R      mm', fontsize=fontz)
    plt.ylabel('   delt     mm', fontsize=fontz)
    plt.title('Uflow_'+ins.label+'ms', fontsize=fontz)
    #plt.title('Vertical Flow [1mm--20m/s]', fontsize=fontz)
    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
    plt.xticks( np.arange(0, 20.01,5 ) )
    plt.tick_params(labelsize = fontz,direction = 'in',length = 7)
    
    filename = "C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_"+shaping+length+"/U_VWR_"+ins.label+"ms.png"
    plt.savefig(filename)
    plt.show()

#with open('../res_temp_vflow_'+version+'.csv', 'w', newline='') as csvfile:
#        spamwriter = csv.writer(csvfile)
#        spamwriter.writerow(['time','a_temp','b_temp','a_vflow','b_vflow'])
#        for ins in inslist:
#            ike = [ins.time]+ins.res
#            spamwriter.writerow(ike)
if period == 'all':
    print('average of '+str(0.05*timelimit)+'ms')
else:
    print('average of '+str(aveDu)+' ms before EOI')
    
    

#'''-------------------plot(safe place)-------------------------'''  
#        
#if not os.path.exists("C:/Users/power/Desktop/python/images/temp_vflow_HF/"+shaping+length):
#    os.mkdir("C:/Users/power/Desktop/python/images/temp_vflow_HF/"+shaping+length)
#    
#    
#for i,insori in enumerate(inslist[:timelimit]+inslist[-len(versions):-len(versions)+1]):
#    sametime = [insori]
#    for instance in inslist[inslist.index(insori)+1:]:
#        if instance.time == insori.time:
#            sametime.append(instance)
#        
#    '''plot temp'''
#    plt.figure(figsize=(15,20))
#    ax1 = plt.subplot(311)
#    ax1.set_xlim(0,20.01)
#    ax1.set_ylim(0.0,0.4)
#    #plt.title("INJECTIONRATE", fontsize=30)
#    for num,ins in enumerate(sametime):
#        rs = sorted(list(set(ins.Rs)))
#        for r in rs[::sep]:
#            rls =[]
#            for i in range(0,len(ins.Rs)):
#                if ins.Rs[i] == r:
#                    rls.append(i)
#            ax1.plot(ins.temp[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',color = colorlist[num])
#    #        plt.show()
#        plt.plot(ins.Tpeak,ins.Tpeakdelt,color = colorlist[num],linestyle= 'dotted',label = ins.name)   
#    ax1.legend(loc='uppper right',
#           bbox_to_anchor=(0.8, 0.8, 0.2,0.1), 
#           borderaxespad=0.,prop={'size':16,})
##    "近似式の計算"
##    shape = int(2*ins.shapedic[ins.time])
##    if shape <= 5:shape = 6
##    res1=np.polyfit(ins.Tpeak[5:shape], ins.Tpeakdelt[5:shape], 1)
##    print(res1,ins.label,'temp')
##    y1 = np.poly1d(res1)(ins.Tpeak[5:shape])
##    plt.plot(ins.Tpeak[5:shape],y1,color = 'k',linestyle= 'dashed')
#    plt.yticks( np.arange(0, 0.4+0.01, 0.2) )     
##    plt.yticks( np.arange(0, 4.01, 1) )
##    plt.xlabel('R', fontsize=30)
#    plt.ylabel('   delt     mm', fontsize=20)
#    plt.title('temp_'+ins.label, fontsize=20)
#    #plt.setp(ax1.get_xticklabels(), visible=False)
##   plt.legend(fontsize = 'small',frameon = None,prop={'size':20,},bbox_to_anchor=(1.05, 1))
#    plt.xticks( np.arange(0, 20.01,5 ) )
#    plt.tick_params(labelsize = 20,direction = 'in',length = 7)
#    #plt.grid()
#    #plt.show()
#    
#    '''plot HF'''
#    ax2 = plt.subplot(312, sharex=ax1)
#    ax2.set_xlim(0,20)
#    #plt.title("dp/dt_movingAve")
#    for num,ins in enumerate(sametime):
#        ax2.plot(ins.Rs,ins.HF,color =colorlist[num],label = ins.name)
#        ax2.legend(loc='uppper right',
#           bbox_to_anchor=(0.8, 0.8, 0.2,0.1), 
#           borderaxespad=0.,prop={'size':16,})
#
#    
##   ax2.scatter(ins.vflow,ins.delt,label=ins.label,marker ='.')
##    plt.yticks( np.arange(0, scale+0.01, scale/2) )
##    plt.yticks( np.arange(0, 4.01, 1) )
##    plt.xlabel('R      mm', fontsize=20)
#    plt.ylabel('   HF    MW/mm2', fontsize=20)
##    plt.title('vflow_'+ins.label+'_'+version, fontsize=15)
#    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
#    plt.xticks( np.arange(0, 20.01,5 ) )
#    plt.tick_params(labelsize = 20,direction = 'in',length = 7)
##   plt.show()
#    #plt.grid()
#    
#    '''plot vflow'''
#    ax3 = plt.subplot(313, sharex=ax1)
#    ax3.set_xlim(0,20)
#    ax3.set_ylim(0.0,scale)
#    #plt.title("dp/dt_movingAve")
#    for num,ins in enumerate(sametime):
#            
#        rs = sorted(list(set(ins.Rs)))
#    
#        for r in rs[::sep]:
#            rls =[]
#            for i in range(0,len(ins.Rs)):
#                if ins.Rs[i] == r:
#                    rls.append(i)
#            ax3.plot(ins.VWRflow[rls[0]:rls[-1]],ins.delt[rls[0]:rls[-1]],marker = '.',color = colorlist[num])
#        plt.plot(ins.VWRpeak,ins.VWRpeakdelt,linestyle= 'dotted',color = colorlist[num],label = ins.name)
#    ax3.legend(loc='uppper right',
#           bbox_to_anchor=(0.8, 0.8, 0.2,0.1), 
#           borderaxespad=0.,prop={'size':16,})
##    "近似式の計算"
##    shape = int(2*ins.shapedic[ins.time])
##    if shape <= 5:shape = 6
##    res2=np.polyfit(ins.Vpeak[5:shape], ins.Vpeakdelt[5:shape], 1)
##    ins.res = list(res1)+list(res2)
##    print(res2,ins.label,'vflow')
#    print('1mm='+str(VWRunit)+'m/s,  1mm = '+str(Tunit)+'K')
##    y2 = np.poly1d(res2)(ins.Vpeak[5:shape])
##    plt.plot(ins.Vpeak[5:shape],y2,color = 'k',linestyle= 'dashed')
#    
##   ax2.scatter(ins.vflow,ins.delt,label=ins.label,marker ='.')
#    plt.yticks( np.arange(0, scale+0.01, scale/2) )
##    plt.yticks( np.arange(0, 4.01, 1) )
#    plt.xlabel('R      mm', fontsize=20)
#    plt.ylabel('   delt     mm', fontsize=20)
#    plt.title('vflow_'+ins.label, fontsize=20)
#    #plt.legend(fontsize = 'small',frameon = True,prop={'size':20,})
#    plt.xticks( np.arange(0, 20.01,5 ) )
#    plt.tick_params(labelsize = 20,direction = 'in',length = 7)
##   plt.show()
#    #plt.grid()
#    
#    filename = "C:/Users/power/Desktop/python/images/temp_vflow_HF/"+shaping+length+"/temp_vflow_"+'_'+ins.label+".png"
#    plt.savefig(filename)
#    plt.show()
#
##with open('../res_temp_vflow_'+version+'.csv', 'w', newline='') as csvfile:
##        spamwriter = csv.writer(csvfile)
##        spamwriter.writerow(['time','a_temp','b_temp','a_vflow','b_vflow'])
##        for ins in inslist:
##            ike = [ins.time]+ins.res
##            spamwriter.writerow(ike)
#if period == 'all':
#    print('average of '+str(0.05*timelimit)+'ms')
#else:
#    print('average of '+str(aveDu)+' ms before EOI')