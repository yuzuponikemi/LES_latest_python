# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:30:13 2018

@author: power
DYTmoduler
"""
import math
import matplotlib.pyplot as plt
#nx = 166
ny = 104
nz =  82
#
#
#ssx = 50
##elif defined(FINE_GRID_AROUND_WALL)
magnify = 1.0        # >1.0 --> fine  <1.0 --> coarse
xmax = 0.3164*10**(-3)      #Max size of X grid             midified by ikemi
gmin_x = 0.1778*10**(-3)     #Min size of X grid(*nozzle diameter)
gmin_yz = 0.136355003*10**(-3)    #Min size of Y and Z grid
ry = 1.06          #Ratio Setting for Y grid
rz = 1.075  
#
enlerge = 1/magnify
ry = ry**enlerge
rz = rz**enlerge
rmax = 38 * magnify # 1.07^35 = 10
dx0 = gmin_x * enlerge
dxmax = xmax * enlerge
dy0 = gmin_yz * enlerge
dz0 = gmin_yz * enlerge
#ss = ssx * magnify
#    
jm=int((ny+1)/2)
km=int((nz+1)/2)
#
#
#  YV(JM)=0.D0
#  YV(JM+1) = MAX(DY0,GMIN)
#  YV(JM-1) = -YV(JM+1)
#  DYT=DY0
#  DO J=JM+2,NY-3
#IF(J < JM+2+RMAX) DYT = DYT*RY
#IF(DYT < GMIN_YZ) THEN
#					YV(J) = YV(J-1) + GMIN_YZ
#ELSE
#					YV(J) = YV(J-1) + DYT
#END IF
#  END DO
#  YV(NY-2) = 2.D0*YV(NY-3) - YV(NY-4)
#			YV(NY-1) = 2.D0*YV(NY-3) - YV(NY-5)
#			YV(NY)   = 2.D0*YV(NY-3) - YV(NY-6)
#  DYT=DY0
#  DO J=JM-2,3,-1
#IF(JM-2-RMAX < J) DYT = DYT*RY
#IF(DYT < GMIN_YZ) THEN
#					YV(J) = YV(J+1) - GMIN_YZ
#ELSE
#					YV(J) = YV(J+1) - DYT
#END IF
#  END DO
#  YV(1) = 2.D0*YV(3) - YV(5)
#			YV(2) = 2.D0*YV(3) - YV(4)

dyts=[]
dyt=dy0
dy=[]
dyc=0
for i in range(0,jm):
    dy.append(dyc)
    dyt=dyt*ry
    if dyt<gmin_yz:
        dyts.append(gmin_yz)
        print("minus")
        dyc += gmin_yz
    else:
        dyts.append(dyt)
        dyc +=dyt

dzts=[]
dzt=dz0
dz=[]
dzc=0
for i in range(0,km):
    dz.append(dzc)
    dzt=dzt*rz

    if dzt<gmin_yz:
        dzts.append(gmin_yz)
        print("minus")
        dzc += gmin_yz
    else:
        dzts.append(dzt)
        dzc += dzt

plt.figure(figsize=(16,12)) 
plt.title("ry="+str(ry)+"  rz="+str(rz), fontsize=30)
for z in dz:
    nowz=[]
    for i in range(0,len(dy)):
        nowz.append(z)
    
    plt.plot(nowz,dy,'o')
#    plt.plot(dxtnx,dxt1s,label="one")
#    plt.plot(dxtnx,dxt2s,label="two")
plt.legend(fontsize = 'small',frameon = True)
plt.xlabel('DZ', fontsize=30)
plt.ylabel('DY', fontsize=30)
#plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid()   
plt.show()       




