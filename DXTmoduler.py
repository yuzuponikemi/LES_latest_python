# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:38:09 2018

@author: power
"""
import math
import matplotlib.pyplot as plt
#nx = 166
#ny = 104
#nz =  82
#
#
#ssx = 50
##elif defined(FINE_GRID_AROUND_WALL)
#magnify = 1.0        # >1.0 --> fine  <1.0 --> coarse
#xmax = 0.3164*10**(-3)      #Max size of X grid             midified by ikemi
#gmin_x = 0.1778*10**(-3)     #Min size of X grid(*nozzle diameter)
#gmin_yz = 0.136355003*10**(-3)    #Min size of Y and Z grid
#ry = 1.0686          #Ratio Setting for Y grid
#rz = 1.0686  
#
#enlerge = 1/magnify
#ry = ry**enlerge
#rz = rz**enlerge
#rmax = 38 * magnify # 1.07^35 = 10
#dx0 = gmin_x * enlerge
#dxmax = xmax * enlerge
#dy0 = gmin_yz * enlerge
#dz0 = gmin_yz * enlerge
#ss = ssx * magnify
#    
#jm=(ny+1)/2
#km=(nz+1)/2
#
#ir = int(6.0*10**(-3)/dxmax+0.5)+3
#dxts=[]
#dxtnx=[]
#dxt1s=[]
#dxt2s=[]
#for i in range (ir,nx-21+1):
#    dxt = dxmax - (dxmax-dx0)*math.exp(-((i-ir)/ss)**2)
#    dxt2 = dxmax - (dxmax-dx0)*math.exp(-((nx-21-i)/ss)**2)
#    dxt1s.append(dxt)
#    dxt2s.append(dxt2)
#    now = min(dxt,dxt2)
#    dxts.append(now)
#    dxtnx.append(i)
#
##plt.figure(figsize=(20,15)) 
#plt.title("ssx="+str(ssx))
#plt.plot(dxtnx,dxts,label="mins")
#plt.plot(dxtnx,dxt1s,label="one")
#plt.plot(dxtnx,dxt2s,label="two")
#plt.legend(fontsize = 'small',frameon = True)
##plt.xlabel('time(ms)')
#plt.ylabel('Qwall')
##plt.setp(ax1.get_xticklabels(), visible=False)
#plt.grid()   
#plt.show()




def checkssx(ssx):
    nx = 166
    ny = 104
    nz =  82
#elif defined(FINE_GRID_AROUND_WALL)
    magnify = 1.0        # >1.0 --> fine  <1.0 --> coarse
    xmax = 0.3164*10**(-3)      #Max size of X grid             midified by ikemi
    gmin_x = 0.1778*10**(-3)     #Min size of X grid(*nozzle diameter)
    gmin_yz = 0.136355003*10**(-3)    #Min size of Y and Z grid
    ry = 1.0686          #Ratio Setting for Y grid
    rz = 1.0686  
    
    enlerge = 1/magnify
    ry = ry**enlerge
    rz = rz**enlerge
    rmax = 38 * magnify # 1.07^35 = 10
    dx0 = gmin_x * enlerge
    dxmax = xmax * enlerge
    dy0 = gmin_yz * enlerge
    dz0 = gmin_yz * enlerge
    ss = ssx * magnify
        
    jm=(ny+1)/2
    km=(nz+1)/2
    
    ir = int(6.0*10**(-3)/dxmax+0.5)+3
    dxts=[]
    dxtnx=[]
    dxt1s=[]
    dxt2s=[]
    for i in range (ir,nx-21+1):
        dxt = dxmax - (dxmax-dx0)*math.exp(-((i-ir)/ss)**2)
        dxt2 = dxmax - (dxmax-dx0)*math.exp(-((nx-21-i)/ss)**2)
        dxt1s.append(dxt)
        dxt2s.append(dxt2)
        now = min(dxt,dxt2)
        dxts.append(now)
        dxtnx.append(i)
    
    #plt.figure(figsize=(20,15)) 
    plt.title("ssx="+str(ssx))
    plt.plot(dxtnx,dxts,label="mins")
#    plt.plot(dxtnx,dxt1s,label="one")
#    plt.plot(dxtnx,dxt2s,label="two")
    plt.legend(fontsize = 'small',frameon = True)
    #plt.xlabel('time(ms)')
    plt.ylabel('DXT')
    #plt.setp(ax1.get_xticklabels(), visible=False)
#    plt.grid()   
    plt.show()


for i in range(24,33,2):
    checkssx(i)


    