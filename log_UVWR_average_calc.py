# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:58:11 2019

@author: power

Calc and make average folder for [log_temp_frow_U_version]

"""

import os
import csv
import glob

class Inst:
    def __init__(self, name, time):
        self.name = name
        self.time = time
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []
        self.x5 = []
        self.x0 = []
        self.x6 = []
        self.x7 = []
        self.x8 = []
        self.x9 = []
        self.x10 = []

class Version:
    def __init__(self, name):
        self.name = name
        self.foldername = ''
        self.csvs = []

        
        
        #define your file name form here

'''main'''

#print('case?')
##case=str(input())  
#case = 'p_dpdt'
os.chdir("c:\\Users\\power\\Desktop\\python\\U_VWR_FLOW")

print('version?')
version = str(input())

path = './'
files = os.listdir(path)
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
print(files_dir)  

length = []
inslist = []
namelist = [x for x in files_dir if version in x and 'ave' not in x]
for name in namelist:
    print(name)
    ins = Version(name[16:])
    
    os.chdir("c:\\Users\\power\\Desktop\\python\\U_VWR_FLOW\\"+name)
    ins.csvs = sorted(glob.glob('log*.csv'))
    ins.foldername = name
    length.append(len(ins.csvs))
    inslist.append(ins)



if not os.path.exists("../log_temp_flow_U_"+version+"_ave"):
    os.mkdir("../log_temp_flow_U_"+version+"_ave")   






size = []
this = min(length)
for t in range(this):
    print(t/this*100,'%')
    instls = []
    for ins in inslist:
        inst = Inst(ins.name,ins.csvs[t][20:-8])
        os.chdir("c:\\Users\\power\\Desktop\\python\\U_VWR_FLOW\\"+ins.foldername)

        with open(ins.csvs[t], newline='') as cname:
    
            reader = csv.reader(cname)
            readerlist = list(reader)
            readerlist = readerlist[1:]
            
            for row in readerlist:
                inst.x0.append(float(row[0]))
                inst.x1.append(float(row[1]))
                inst.x2.append(float(row[2]))
                inst.x3.append(float(row[3]))
                inst.x4.append(float(row[4]))
                inst.x5.append(float(row[5]))
                inst.x6.append(float(row[6]))
                inst.x7.append(float(row[7]))
                inst.x8.append(float(row[8]))
                inst.x9.append(float(row[9]))
                inst.x10.append(float(row[10]))
            
            size.append(len(inst.x0))
                
                
    #            ins.x6.append(float(row[6]))
    #            ins.x7.append(float(row[7]))
                
            
    #        print(ins.label)
            #print(ins.x)
    #        print(ins.y)
    #        print(ins.cumulative)
        instls.append(inst)
    

    limit = min(size)
    
    avex1 = [0 for x in range(limit)]
    avex2 = [0 for x in range(limit)]
    avex3 = [0 for x in range(limit)]
    avex4 = [0 for x in range(limit)]
    avex0 = [0 for x in range(limit)]
    avex5 = [0 for x in range(limit)]
    avex6 = [0 for x in range(limit)]
    avex7 = [0 for x in range(limit)]
    avex8 = [0 for x in range(limit)]
    avex9 = [0 for x in range(limit)]
    avex10 = [0 for x in range(limit)]
    for inst in instls:
        for i,x1 in enumerate(inst.x1):
            avex1[i] += x1
        for i,x2 in enumerate(inst.x2):
            avex2[i] += x2
        for i,x3 in enumerate(inst.x3):
            avex3[i] += x3
        for i,x4 in enumerate(inst.x4):
            avex4[i] += x4
        for i,x0 in enumerate(inst.x0):
            avex0[i] += x0
        for i,x5 in enumerate(inst.x5):
            avex5[i] += x5
        for i,x6 in enumerate(inst.x6):
            avex6[i] += x6
        for i,x7 in enumerate(inst.x7):
            avex7[i] += x7
        for i,x8 in enumerate(inst.x8):
            avex8[i] += x8
        for i,x9 in enumerate(inst.x9):
            avex9[i] += x9
        for i,x10 in enumerate(inst.x10):
            avex10[i] += x10
    for i in range(0,limit):
        
        avex1[i] = avex1[i]/len(inslist)
        avex2[i] = avex2[i]/len(inslist)
        avex3[i] = avex3[i]/len(inslist)
        avex4[i] = avex4[i]/len(inslist)
        avex0[i] = avex0[i]/len(inslist)   
        avex5[i] = avex5[i]/len(inslist)
        avex6[i] = avex6[i]/len(inslist)
        avex7[i] = avex7[i]/len(inslist)
        avex8[i] = avex8[i]/len(inslist)
        avex9[i] = avex9[i]/len(inslist)
        avex10[i] = avex10[i]/len(inslist) 
        
    os.chdir("../log_temp_flow_U_"+version+"_ave")
    with open('log_temp_flow_U_VWR_'+inst.time+' ms.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['X(mm)','delta(mm)','R(mm)','F_ave','U_ave(m/s)','VWR_ave(m/s)','UVW_ave(m/s)','T_ave','Heat_Flux[Q/m2]','Zcen_b(mm)','Ycen_b(mm)'])
        for i in range(limit):
            spamwriter.writerow([avex0[i],avex1[i],avex2[i],avex3[i],avex4[i],avex5[i],avex6[i],avex7[i],avex8[i],avex9[i],avex10[i]])

print('Done!')
