import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler, TensorBoard
from tensorflow.keras import backend as K
from keras_unet.utils import plot_imgs

masks = glob.glob("train/images_segmented/SegmentationClassPNG/*.png")
orgs = glob.glob("train/images_segmented/JPEGImages/*.png")


image_size=384
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    im=Image.open(image).resize((image_size,image_size))
    
    if len(np.array(im).shape) == 2:
            	im= im.convert('RGB')
    elif np.array(im).shape[2] == 4:
        	im= im.convert('RGB')
    
    imgs_list.append(np.array(im))
    im = Image.open(mask).resize((384,384))
    masks_list.append(np.array(im))

imgs_np = np.asarray(imgs_list)[:,:,:,0] #remove colors
masks_np = np.asarray(masks_list)[:,:,:]
print(imgs_np.shape, masks_np.shape)
print(imgs_np.max(), masks_np.max())

x = np.asarray(imgs_np, dtype=np.float32)/imgs_np.max()
y = np.asarray(masks_np, dtype=np.float32)/masks_np.max()
	
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

print(x.max(), y.max())

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

def get_augmented(
    X_train,
    Y_train,
    X_val=None,
    Y_val=None,
    batch_size=32,
    seed=0,
    data_gen_args=dict(
        rotation_range=10.0,
        width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=5,
        zoom_range=[0.5,1.5],
        brightness_range=[0.8,1.2],
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="constant",
    ),
):
    """[summary]
    
    Args:
        X_train (numpy.ndarray): [description]
        Y_train (numpy.ndarray): [description]
        X_val (numpy.ndarray, optional): [description]. Defaults to None.
        Y_val (numpy.ndarray, optional): [description]. Defaults to None.
        batch_size (int, optional): [description]. Defaults to 32.
        seed (int, optional): [description]. Defaults to 0.
        data_gen_args ([type], optional): [description]. Defaults to dict(rotation_range=10.0,# width_shift_range=0.02,height_shift_range=0.02,shear_range=5,# zoom_range=0.3,horizontal_flip=True,vertical_flip=False,fill_mode="constant",).
    
    Returns:
        [type]: [description]
    """

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(
        X_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    Y_train_augmented = Y_datagen.flow(
        Y_train, batch_size=batch_size, shuffle=True, seed=seed
    )

    train_generator = zip(X_train_augmented, Y_train_augmented)

    if not (X_val is None) and not (Y_val is None):
        # Validation data, no data augmentation, but we create a generator anyway
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=False, seed=seed)
        Y_datagen_val.fit(Y_val, augment=False, seed=seed)
        X_val_augmented = X_datagen_val.flow(
            X_val, batch_size=batch_size, shuffle=False, seed=seed
        )
        Y_val_augmented = Y_datagen_val.flow(
            Y_val, batch_size=batch_size, shuffle=False, seed=seed
        )

        # combine generators into one which yields image and masks
        val_generator = zip(X_val_augmented, Y_val_augmented)

        return train_generator, val_generator
    else:
        return train_generator



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
   # e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e4, f[3], strides=1)
    b1 = conv_block(b0, f[3], strides=1)
    
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

def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y

def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

batch_size=1
train_gen = get_augmented(
    x_train, y_train, batch_size=batch_size,
    data_gen_args = dict(
        rescale=1./255,
        rotation_range=55.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0,
        zoom_range=[0.5,1.5],
        brightness_range=[0.8,1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))

sample_batch = next(train_gen)
xx, yy = sample_batch
print(xx.shape, yy.shape)
plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=2, figsize=6)
plt.show()

#model_filename = 'segm_model_resUnet-markers.h5'
model_filename_save = 'segm_model_resUnet-markers.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename_save, 
    verbose=1, 
    monitor='loss', 
    save_best_only=True,
)

def step_decay_schedule(initial_lr=1e-1, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule,verbose = 1)

lr_sched = step_decay_schedule(initial_lr=5e-4, decay_factor=0.75, step_size=50)



csv_logger = CSVLogger('training_resUnet-markers.log', separator=",", append=True)

#logdir = "logs/fit/resunet-markers"

#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)




model = ResUNet()
adam = keras.optimizers.Adam()
model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[iou,dice_coef,iou_thresholded])
model.summary()
#model.load_weights(model_filename)
history = model.fit(
    train_gen,
    steps_per_epoch=np.ceil(x_train.shape[0]/batch_size),
    epochs=1000,
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint,csv_logger,lr_sched]#,tensorboard_callback]
)



#print("\n      Ground Truth            Predicted Value")

#for i in range(1, 5, 1):
#    ## Dataset for prediction
#    x, y = valid_gen.__getitem__(i)
#    result = model.predict(x)
#    result = result > 0.4
#    
#    for i in range(len(result)):
#        fig = plt.figure()
#        fig.subplots_adjust(hspace=0.4, wspace=0.4)

#        ax = fig.add_subplot(1, 2, 1)
#        ax.imshow(np.reshape(y[i]*255, (image_size, image_size)), cmap="gray")

#        ax = fig.add_subplot(1, 2, 2)
#        ax.imshow(np.reshape(result[i]*255, (image_size, image_size)), cmap="gray")                 
