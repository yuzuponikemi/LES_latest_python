# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:22:07 2019

@author: power
"""

import os
import cv2
#import numpy as np
import glob


print('please put [Folder abs PATH].')
#place = str(input())
#C:\Users\power\Desktop\python\images\log_U_VWRflow\U_VWR_T_HF_BdLayer_single_1.0mm
#place = 'C:/Users/power/Desktop/python/images/log_temp_VWR_h_res/last3_ave'
place='C:/Users/power/Desktop/python/images/log_U_VWRflow/U_VWR_T_HF_BdLayer_single_1.0mm'
print('OK! Let\'s process '+place)

#print('put directory name')
#dr = "\"+str(input())
dr = ''
ik = 0
for c in place[::-1]:
    if c == '/':ik += 1
    if ik == 2:break
    dr += c

os.chdir(place)

namelist = glob.glob('./*ms.png')
namelist = [name[2:] for name in namelist]

lim = 5*(len(namelist)-1)

nam = ''
ik = 0
for c in namelist[1][::-1]:
    if c == '_':ik = 1
    if ik == 1:nam = c + nam

st = nam+'0005ms.png'
img = cv2.imread(st[:-9]+'.'+st[-9:])
hight, width, channnels = img.shape[:3]

# 動画作成
fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
video  = cv2.VideoWriter(nam[:-1]+'.mp4', fourcc, 2.0, (width,hight))

for i in range(210,lim,5):
    if i%1000 == 0:
        i = int(i/1000)
        st = nam+str((i))+'0ms.png'
        img = cv2.imread(st[:-7]+'.'+st[-7:])
    elif i%100 == 0  :
        print(str(i/1000)+'ms')
        i = int(i/100)
        st = nam+'{0:02d}ms.png'.format(i)
        img = cv2.imread(st[:-7]+'.'+st[-7:])
    elif i%10 == 0:
        i = int(i/10)
        st = nam+'{0:03d}ms.png'.format(i)
        img = cv2.imread(st[:-8]+'.'+st[-8:])


    else:
        st = nam+'{0:004d}ms.png'.format(i)
        img = cv2.imread(st[:-9]+'.'+st[-9:])


    # img.resize(img,(640,480))
    
    video.write(img)


#os.chdir('C:/Users/power/Desktop/python/movie')

video.release()

print('Done!')



