# -*- coding: utf-8 -*-
"""
Created on Tue Mar 02 17:02:19 2021

@author: gautier
"""

from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from PIL  import Image

def roi_draw(event, x, y, flags, param):
		global refPt
	 	
		if event == cv2.EVENT_LBUTTONDOWN:
			refPt = [(x, y)]
			
	 
		elif event == cv2.EVENT_LBUTTONUP:

			refPt.append((x, y))
			
	 
			# draw a rectangle around the region of interest
			cv2.rectangle(im1, refPt[0], refPt[1], (255, 255, 0), 1)
			#Print position x,y and size w,h
			text= 'x="'+str(refPt[0][0])+'"'+' y="'+str(refPt[0][1])+'"'+' w="'+str(np.abs(refPt[1][0]-refPt[0][0]))+'"'+' h="'+str(np.abs(refPt[1][1]-refPt[0][1]))
			print (text)
			print ('[['+str(refPt[0][0])+','+str(refPt[0][1])+']],')
			cv2.imshow('DIC', im1)

def roi_search(im):
    while (1):
        cv2.imshow('DIC', im)
        cv2.setMouseCallback('DIC', roi_draw) 

        k=cv2.waitKey()
        if k == 27:
	        break	

im1 = cv2.imread('12.png',0)
im2 = cv2.imread('13.png',0)

# GUI for window search in image 1
#roi_search(im1) 



#Markers 
markers=np.array([[750,460,120,100],

                  [270,24,40,40],[323,24,40,40],[376,24,40,40],[429,24,40,40],[482,24,40,40],[535,24,40,40],[588,24,40,40],[641,24,40,40],[694,24,40,40],[747,24,40,40],[800,24,40,40],[853,24,40,40],[906,24,40,40],[959,24,40,40],[1012,24,40,40],[1065,24,40,40],[1118,24,40,40],[1171,24,40,40],
                  [270,77,40,40],[323,77,40,40],[376,77,40,40],[429,77,40,40],[482,77,40,40],[535,77,40,40],[588,77,40,40],[641,77,40,40],[694,77,40,40],[747,77,40,40],[800,77,40,40],[853,77,40,40],[906,77,40,40],[959,77,40,40],[1012,77,40,40],[1065,77,40,40],[1118,77,40,40],[1171,77,40,40],
                  [270,130,40,40],[323,130,40,40],[376,130,40,40],[429,130,40,40],[482,130,40,40],[535,130,40,40],[588,130,40,40],[641,130,40,40],[694,130,40,40],[747,130,40,40],[800,130,40,40],[853,130,40,40],[906,130,40,40],[959,130,40,40],[1012,130,40,40],[1065,130,40,40],[1118,130,40,40],[1171,130,40,40],
                  [323,183,40,40],[376,183,40,40],[429,183,40,40],[482,183,40,40],[535,183,40,40],[588,183,40,40],[641,183,40,40],[694,183,40,40],[747,183,40,40],[800,183,40,40],[853,183,40,40],[906,183,40,40],[959,183,40,40],[1012,183,40,40],[1065,183,40,40],[1118,183,40,40],[1171,183,40,40],
                  #[323,236,40,40],[376,236,40,40],[429,236,40,40],[482,236,40,40],[535,236,40,40],[588,236,40,40],[641,236,40,40],[694,236,40,40],
                  [747,236,40,40],[800,236,40,40],[853,236,40,40],[906,236,40,40],[959,236,40,40],[1012,236,40,40],[1065,236,40,40],[1118,236,40,40],[1171,236,40,40],
#transition       
                                      
                  [320,320,40,40],[373,320,40,40],[426,320,40,40],[479,320,40,40],[532,320,40,40],#[585,340,40,40],[638,340,40,40],#[691,340,40,40],[744,340,40,40],[797,340,40,40],
                  [320,373,40,40],[373,373,40,40],[426,373,40,40],[479,373,40,40],[532,373,40,40],
                  
#gauche                  
                  #[10,176,40,40],[62,176,40,40],[115,176,40,40],
                  #[10,229,40,40],[62,229,40,40],[115,229,40,40],
                  #[10,282,40,40],[62,282,40,40],[115,282,40,40],
                  #[10,335,40,40],[62,335,40,40],[115,335,40,40],
                  #[10,388,40,40],[62,388,40,40],[115,388,40,40],
                  #[10,441,40,40],[62,441,40,40],[115,441,40,40],
                  #[10,494,40,40],[62,494,40,40],[115,494,40,40],
                  #[10,547,40,40],[62,547,40,40],[115,547,40,40],
                  #[10,600,40,40],[62,600,40,40],[115,600,40,40],
                                 
   ])

#markers=np.array([[10,10,43,28],[40,50,26,25],[13,88,26,26],[7,155,23,24],[23,316,19,20]])
#markers=np.array([[483,195,40,20]])

result=np.zeros((np.shape(markers)[0],7),dtype=int)
distance=np.zeros((np.shape(markers)[0],4),dtype=float)

for i in range(0,np.shape(markers)[0]):
    yt=markers[i,1]
    ht=markers[i,3]
    xt=markers[i,0]
    wt=markers[i,2]
    template=im1[yt:yt+ht,xt:xt+wt]
    enlar=10 #roi search 
    template_im2=im2[yt-enlar:yt+ht+enlar,xt-enlar:xt+wt+enlar]
    
    #res=cv2.matchTemplate(im2, template, cv2.TM_SQDIFF_NORMED)
    res=cv2.matchTemplate(template_im2, template, cv2.TM_CCORR_NORMED) 	#Template matching for the fixed point
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 		#Find the position of the template
	
	#cv2.rectangle(im2, (xt,yt), (xt+wt,yt+ht), 100, 1)
    max_loc=(max_loc[0]+(xt-enlar),max_loc[1]+(yt-enlar))
    top_left = max_loc
    bottom_right = (top_left[0] + wt, top_left[1] + ht)
    cv2.rectangle(im2,top_left, bottom_right, 255, 3)
	#cv2.arrowedLine(im2, (xt,yt), top_left, 255, 2)
	
    result[i,:]=np.array([xt,yt,xt+wt,yt+ht, top_left[0]-xt,top_left[1]-yt,i], dtype=int)
    distance[i]=np.array([(result[i,4]-result[0,4]),(result[0,5]-result[i,5]),np.sqrt(((result[i,4]-result[0,4])**2)+(result[0,5]-result[i,5])**2),i])

pt_origin=0
x_shift=result[pt_origin,4]
#x_shift=0
#y_shift=0
y_shift=result[pt_origin,5]	
s=4 #scaling arrows
#text properties
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 2

for  i in range(0,np.shape(markers)[0]):
    cv2.rectangle(im1, (result[i,0], result[i,1]), (result[i,2], result[i,3]), 255, 3)
    cv2.arrowedLine(im2,(result[i,0], result[i,1]), (result[i,0]+s*(result[i,4]-x_shift), result[i,1]+s*(result[i,5]-y_shift)),0,5)
#    cv2.putText(im1,str(i),(result[i,2], result[i,3]),font,fontScale,fontColor,lineType)

 
cv2.imshow('image1',im1)
cv2.imshow('image2',im2)
#cv2.imwrite(os.path.join( "result.png") , im1)
#print (result)
#print (distance)
cv2.waitKey()
cv2.destroyAllWindows()



