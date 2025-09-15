import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
from tensorflow.keras import backend as K
from keras_unet.utils import plot_imgs

def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c

def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    #e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e4, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    #u1 = upsample_concat_block(b1, e4)
    #d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(b1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    return model
    

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
                K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)



masks = glob.glob("*.BMP")
orgs = glob.glob("*.BMP")

image_size=384
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    im=Image.open(image).resize((384,384))
    
    if len(np.array(im).shape) == 2:
            	im= im.convert('RGB')
    elif np.array(im).shape[2] == 4:
        	im= im.convert('RGB')
    
    #im = exposure.equalize_adapthist(np.array(im), clip_limit=0.01)
    imgs_list.append(np.array(im))

    imask = Image.open(mask).resize((384,384))
    masks_list.append(np.array(imask))


imgs_np = np.asarray(imgs_list)[:,:,:,0] #remove colors
masks_np = imgs_np
print(imgs_np.shape, masks_np.shape)
print(imgs_np.max(), masks_np.max())

x = np.asarray(imgs_np, dtype=np.float32)/imgs_np.max()
y = np.asarray(masks_np, dtype=np.float32)/masks_np.max()
	
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

print(x.shape,y.shape)

model_filename = 'segm_model_resUnet-markers.h5'
model = ResUNet()
model.load_weights(model_filename)
adam = keras.optimizers.Adam()
model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[iou,dice_coef])

masks_train = glob.glob("train/images_segmented/SegmentationClassPNG/*.png")
orgs_train = glob.glob("train/images_segmented/JPEGImages/*.png")

imgs_list_train = []
masks_list_train = []
for image, mask in zip(orgs_train, masks_train):
    im=Image.open(image).resize((384,384))
    
    if len(np.array(im).shape) == 2:
            	im= im.convert('RGB')
    elif np.array(im).shape[2] == 4:
        	im= im.convert('RGB')
    
    #im = exposure.equalize_adapthist(np.array(im), clip_limit=0.01)
    imgs_list_train.append(np.array(im))

    imask = Image.open(mask).resize((384,384))
    masks_list_train.append(np.array(imask))

imgs_np = np.asarray(imgs_list_train)[:,:,:,0] #remove colors
masks_np = np.asarray(masks_list_train)
print(imgs_np.shape, masks_np.shape)
print(imgs_np.max(), masks_np.max())


#plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)

x_t = np.asarray(imgs_np, dtype=np.float32)/imgs_np.max()
y_t = np.asarray(masks_np, dtype=np.float32)/masks_np.max()

y_t = y_t.reshape(y_t.shape[0], y_t.shape[1], y_t.shape[2], 1)
x_t = x_t.reshape(x_t.shape[0], x_t.shape[1], x_t.shape[2], 1)

print('train', x_t.max(), y_t.max())

x_train, x_val, y_train, y_val = train_test_split(x_t, y_t, test_size=0.3, random_state=0)


#scores = model.evaluate(x_train, y_train, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))



x_val=x[1:3,:,:,:]
y_val=y[1:3,:,:,:]
#x_val=x_train
#y_val=y_train
y_pred = model.predict(x_val)
#for i in range(0,x_val.shape[0]):
#	scores = model.evaluate(x_val[i:i+1,:,:,:], y_val[i:i+1,:,:,:], verbose=0)
#	im=model.predict(x_val[i:i+1,:,:,:])
#	#plt.imshow(im[0,:,:,:])
#	fig=plot_imgs(org_imgs=x_val[i:i+1,:,:,:], mask_imgs=y_val[i:i+1,:,:,:], pred_imgs=im, nm_img_to_plot=9)
#	fig.savefig('result_predict_image/'+str(i))
#	file_predict = open("result_predict_image/result_predict_image.txt","a") 
#	file_predict.write(str(np.around(scores[0]*100,decimals=2))+','+str(np.around(scores[1]*100,decimals=2))+','+str(np.around(scores[2]*100,decimals=2))+'\n')
#	print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
#	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#	print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
#	print ('==============')
#	#plt.show()
	
	

plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=3)
plt.show()



