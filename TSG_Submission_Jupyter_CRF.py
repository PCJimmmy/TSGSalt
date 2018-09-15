
# coding: utf-8

# In[1]:


#VS Code pitches a bitch if these not at the front
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


# VS code does not like % stuff


#get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf
import warnings
import keras

from itertools import chain
from keras import backend as K
from keras import losses
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras_applications import imagenet_utils
from skimage.io import imread, imshow, concatenate_images
from skimage.morphology import label
from skimage.transform import resize
from tqdm import tqdm
from tqdm import tqdm_notebook, tnrange



from skimage.io import imread, imsave

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray



#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# imports from my hold all the crap module
from TSG_definitions import dice_loss
from TSG_definitions import bce_dice_loss
from TSG_definitions import  mean_iou
from TSG_definitions import get_unet_inception_resnet_v2
#from TSG_definitions import rle_encode
from TSG_definitions import rle_decode
from TSG_definitions import crf



from keras import optimizers
from keras import losses

#config = tf.ConfigProto(device_count={"CPU": 12})
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# In[4]:



def rle_encode(im):
        '''
        im: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = im.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)


# In[5]:


# Set some parameters
# will use this in loop to write files with different levels
# model fit settings
batch_size = 16              # 32 should work on single GPU with simple unet - change to 64 for 2 GPU
nb_epoch = 100               # early stopping has always prevented the model fit from reaching 100


# drop out needs to be a not decimal number so the file titles dont end up with . in wrong spot = confuses all
# use two digits   ie   0.10 is 10
drop_out_rate = 0
Model_DOE = 'ResNet_128_73_Savgol_' + str(batch_size) + '_DO_' + str(drop_out_rate)
# now turn the drop out into the correct decimal format
drop_out_rate = drop_out_rate/100
model_name = 'TGS_get_simple_unet_128__resnet_AF_LR001_PT05_DO_' + Model_DOE
model_increment = '_001'

submission_name = 'd:/salt/submission/' + model_name + '.csv'
model_is_named = 'd:/salt/models/' + model_name + model_increment

tb_directory = 'd:/salt/logs/' + Model_DOE

model_save = ('d:/salt/models/' + model_name + '.h5')

# lets try naming as
#  TSG_                    always start with this vs Airbus_
# unet_inceptionv2         the pre-trained model we used
# LR01                     learning rate - LR01 would be rate of 0.01 - should match what parameter is shown below
# AF_                      added flip augmentation   if not use NF_
# DO                      model contains dropout
# PT06_                   prediction threshold - only add this if its not 0.5         PT_06 would be threshold of 0.6
# SMALL_                  use this when the images are from the small testing batch rather than the full set
# 001                     incremental value for runs where the above are the same but something else changed - how to document the change is the quesion
# model_doe               this is the guy we will change most of the time!!!

path_train = 'd:/salt/train/'
path_test = 'd:/salt/test/'

your_history_path = 'd:/salt/history/'
results_log_path = 'd:/salt/history/' + model_name + '.csv'

path_train_images = 'D:/salt/train/savgol_Images/'
path_validation_images = 'd:/salt/train/validation_images/'
path_test_images = 'd:/salt/test/images/'
path_mask_images = 'd:/salt/train/savgol_Masks/'
path_mask_validation = 'd:/salt/train/validation_masks/'


# In[7]:


#parameters that we will want to optimize
prediction_threshold = 0.5
learning_rate = 0.0001
im_width = 256
im_height = 256
im_chan = 3 # Number of channels: first is original and second cumsum(axis=0)
n_features = 0 # Number of extra features, like depth
border = 5
model_to_use = ''
validation_split = 0.3               # percentage to be used for validation


# In[8]:


print(path_train_images)
train_ids = next(os.walk(path_train_images))[2]
print(path_test_images)
test_ids = next(os.walk(path_test_images))[2]

print(model_is_named)
print(submission_name)


# In[9]:


X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)

sizes_test = []
sizes_train = []


# In[10]:


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    img = load_img(path_train_images + id_)
    x = img_to_array(img)#[:,:,1]
    sizes_train.append([x.shape[0], x.shape[1]])
    x = resize(x, (im_width, im_height, 3), mode='constant', preserve_range=True)
    X_train[n] = x
    mask = img_to_array(load_img(path_mask_images + id_))[:,:,1]
    Y_train[n] = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)

print('Done!')


# In[11]:


# add in the train images I have set aside for validation


# In[12]:


print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = load_img(path_test_images + id_)
    x = img_to_array(img)#[:,:,1]
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (im_width, im_height, 3), mode='constant', preserve_range=True)
    X_test[n] = x
    
print('Done!')


# In[13]:


print(model_is_named)


# In[14]:


#model_is_named = 'd:/salt/models/TGS_get_simple_unet_128__resnet_AF_LR001_PT05_DO_ResNet_128_73_Savgol_16_DO_0_001'
from TSG_definitions import  mean_iou


# In[15]:


# another super fast one !!!

print('loading model for predictions ..........!')
model = load_model(model_is_named, custom_objects={'mean_iou': mean_iou,'dice_loss':dice_loss,'bce_dice_loss':bce_dice_loss})
print('ok  was not that bad  ..........!')


# In[ ]:


model.evaluate(X_train, Y_train, verbose=0)
# what does output represent    mean iou  - dice loss - bce dice loss ??


# In[ ]:


print(len(X_test))


# In[ ]:


preds_test = model.predict(X_test, verbose=1)


# In[ ]:


#preds_val = model.predict(X_train[int(X_train.shape[0]*validation_split):], verbose=0)


# In[ ]:


#print(len(preds_val))


# In[ ]:


# predictions run on the GPU so not bad considering

#preds_train = model.predict(X_train[:int(X_train.shape[0]*validation_split)], verbose=0)


# In[ ]:


#print(len(preds_train))


# In[ ]:


preds_full_train = model.predict(X_train,verbose=1)


# In[ ]:


print(len(preds_full_train))


# In[ ]:


# do the code for optimizing the theshold value here - before we make our predictions
# the two def where here - now in definitions

from TSG_definitions import _iou
from TSG_definitions import miou
from TSG_definitions import mean_iou


# In[ ]:


print(len(Y_train))


# In[ ]:


# our training array size is 70 to 90% of the data
# but the thresholds def expect to see 100%
# the code is using 100% of thread 6 and only bit of the other 11
# try using GPU for this one  TODO
print(len(preds_full_train))


# In[ ]:


thresholds = np.linspace(0.1, 0.9, 80)
# orginal
ious = np.array([miou(Y_train, np.int32(preds_full_train > prediction_threshold)) 
                 for prediction_threshold in tqdm_notebook(thresholds)])

#ious = np.array([mean_iou(Y_train, np.int32(preds_full_train > prediction_threshold)) 
#                 for prediction_threshold in tqdm_notebook(thresholds)])


# In[ ]:


print(thresholds)
plt.plot(thresholds, ious);


# In[ ]:


print(ious)


# In[ ]:


threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
print(threshold_best)


# In[ ]:


# when override wanted uncomment this line
threshold_best = 0.5


# In[ ]:


# Threshold predictions
print('making predictions ..........!')
#preds_train_t = (preds_train > threshold_best).astype(np.uint8)
#preds_val_t = (preds_val > threshold_best).astype(np.uint8)
preds_test_t = (preds_test > threshold_best).astype(np.uint8)
preds_full_train_t = (preds_full_train > threshold_best).astype(np.uint8)


# In[ ]:


#del X_test 
# del model  I will remove the comment later - right now I had to have to reload the model


# In[ ]:


from TSG_definitions import iou_metric
from TSG_definitions import iou_metric_batch


# In[ ]:


# test dictionary
print('create dictionary for the submission ..........!')
preds_test_upsampled = []
for i in range(len(preds_test_t)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))


# In[ ]:


test_submission_name = 'd:/salt/submission/' + model_name + model_increment + '_Optimum_Threshold.csv'


# In[ ]:


from TSG_definitions import RLenc


# In[ ]:


pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm(enumerate(test_ids))}


# In[ ]:


sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']


# In[ ]:


sub.to_csv(test_submission_name)

print(' Test submission completed ')


# In[ ]:


#  Generate a rle file for train predictions so we can look at real vs not real in VB program

submission_name = 'd:/salt/submission/' + model_name +  model_increment + '_OT_TRAIN.csv'


# In[ ]:


print(len(preds_full_train))


# In[ ]:


# Train - full set including validation
print('create dictionary for the full train set to create masks ..........!')
preds_train_upsampled = []
for i in range(len(preds_full_train)):
    preds_train_upsampled.append(resize(np.squeeze(preds_full_train_t[i]), 
                                       (sizes_train[i][0], sizes_train[i][1]), 
                                       mode='constant', preserve_range=True))


# In[ ]:


print(len(preds_train_upsampled))


# In[ ]:


pred_dict_train = {fn[:-4]:RLenc(np.round(preds_train_upsampled[i])) for i,fn in tqdm(enumerate(train_ids))}


# In[ ]:


sub = pd.DataFrame.from_dict(pred_dict_train,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']


# In[ ]:


sub.to_csv(submission_name)


# In[ ]:


# CRF Process both train and test submissions


# In[ ]:


print (test_submission_name)


# In[ ]:


df = pd.read_csv(test_submission_name)

i = 0
j = 0
plt.figure(figsize=(30,15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)  #adjust this to change vertical and horiz. spacings..
# Visualizing the predicted outputs
while True:
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])
        plt.subplot(1,12,j+1)
        plt.imshow(decoded_mask)
        plt.title('ID: '+df.loc[i,'id'])
        j = j + 1
        if j>11:
            break
    i = i + 1


# In[ ]:


import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

def crf(original_image, mask_img):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(mask_img.shape)<3):
        mask_img = gray2rgb(mask_img)

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
    
#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 10 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))


# In[ ]:


for i in tqdm(range(df.shape[0])):
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])        
        orig_img = imread(path_test_images+df.loc[i,'id']+'.png')        
        crf_output = crf(orig_img,decoded_mask)
        df.loc[i,'rle_mask'] = rle_encode(crf_output)


# In[ ]:


df.to_csv(test_submission_name,index=False)


# In[ ]:


df = pd.read_csv(test_submission_name)

i = 0
j = 0
plt.figure(figsize=(30,15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)  #adjust this to change vertical and horiz. spacings..
# Visualizing the predicted outputs
while True:
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])
        plt.subplot(1,12,j+1)
        plt.imshow(decoded_mask)
        plt.title('ID: '+df.loc[i,'id'])
        j = j + 1
        if j>11:
            break
    i = i + 1


# In[ ]:


for i in tqdm(range(df.shape[0])):
    if str(df.loc[i,'rle_mask'])!=str(np.nan):        
        decoded_mask = rle_decode(df.loc[i,'rle_mask'])        
        orig_img = imread(path_test_images+df.loc[i,'id']+'.png')        
        crf_output = crf(orig_img,decoded_mask)
        df.loc[i,'rle_mask'] = rle_encode(crf_output)


# In[ ]:


test_submission_name_crf =  test_submission_name + '_CRF.csv'


# In[ ]:


df.to_csv(test_submission_name_crf,index=False)

