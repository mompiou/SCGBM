from matplotlib import pyplot as plt
import numpy as np


data=np.genfromtxt('training_resUnet-markers_3.log',delimiter=',')
#epoch,dice_coef,iou,iou_thresholded,loss,val_dice_coef,val_iou,val_iou_thresholded,val_loss

ind=np.argwhere(data[:,0]==0)

for i in range(0,np.shape(ind)[0]-1):
	data[int(ind[i]):int(ind[i+1]),0]=data[int(ind[i]):int(ind[i+1]),0]+int(ind[i])


data[int(ind[-1]):,0]=data[int(ind[-1]):,0]+int(ind[-1])


fig1=plt.figure(1)
#plt.plot(data[2:,0]+1,data[2:,8], 'r--',label='val_loss')
plt.plot(data[2:,0]+1,data[2:,4], 'r-',label='loss')

plt.legend()
fig2=plt.figure(2)
plt.plot(data[:,0]+1,data[:,2], 'b-',label='iou')
#plt.plot(data[:,0]+1,data[:,2], 'b-.',label='iou_thresholded')
#plt.plot(data[:,0]+1,data[:,5], 'b--',label='val_iou')
#plt.plot(data[:,0]+1,data[:,5], 'b-',label='val_iou_thresholded')

plt.legend()

plt.show()
