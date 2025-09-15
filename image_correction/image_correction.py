import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

img_avant = cv2.imread('avant.BMP')
img_apres = cv2.imread('apres.BMP')
threshold=30

def centroid(img,mode):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	if mode == 'hessian':
		sigma=7
		hxx, hyy, hxy = hessian_matrix(gray, sigma)
		i1, i2 = hessian_matrix_eigvals([hxx, hxy, hyy])
		i=i1/np.max(i1)
		gray=gray/np.max(gray)+i
		gray=np.uint8(gray*255/np.max(gray))
		thresh = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)[1]
		kernel = np.ones((5,5),np.uint8)
		closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
	else:
		gray = cv2.equalizeHist(gray)
		#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,809,110)
		thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)[1]
		# noise removal
		kernel = np.ones((5,5),np.uint8)
		closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 3)

	# sure background area
	sure_bg = cv2.erode(closing,kernel,iterations=3)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,0)
	ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_fg,sure_bg)
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1

	# Now, mark the region of unknown with zero
	markers[unknown==1] = 0
	markers = cv2.watershed(img,markers)
	img[markers >1] = [255,0,0]

	contours, hierarchy = cv2.findContours(markers, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(markers, contours, -1, (0,255,0), 3)
	P=[]
	for i in range(0,len(contours)):
		cnt = contours[i]
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		P=np.append(P,[cx,cy])
		cv2.rectangle(markers,(cx-1,cy-1),(cx+1,cy+1),(0,255,0),4)

	P=np.reshape(P,((int(np.shape(P)[0]/2),2)))
	return markers,P

m_avant,P_avant=centroid(img_avant,'none')
m_apres,P_apres=centroid(img_apres,'none')

plt.imshow(m_avant,cmap='jet')
plt.show()
plt.imshow(m_apres,cmap='jet')
plt.show()

ind_i=[]
ind_j=[]
for i in range(0,P_avant.shape[0]):
    for j in range(0,P_apres.shape[0]):
        if np.linalg.norm(P_avant[i,:]-P_apres[j,:])<threshold:
            ind_i.append(i)
            ind_j.append(j)
P_avant=P_avant[ind_i]
P_apres=P_apres[ind_j]
    
print (P_avant, P_apres)

tform3 = transform.ProjectiveTransform()
tform3.estimate(P_avant, P_apres)
print (tform3)


img_avant = cv2.imread('avant.BMP')
img_apres = cv2.imread('apres.BMP')
warped = transform.warp(img_apres, tform3.inverse)
plt.imshow(warped)
plt.show()


