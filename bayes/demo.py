
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:

import tkinter
import matplotlib
#matplotlib.use('TkAgg')
import os
print(matplotlib.get_backend())
import sys
print(matplotlib.get_backend())
import random
print(matplotlib.get_backend())
import math
import numpy as np
import skimage.io
print(matplotlib.get_backend())
import matplotlib.pyplot as plt
print(matplotlib.get_backend())
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
print(matplotlib.get_backend())
import mrcnn.model as modellib
print(matplotlib.get_backend())
from mrcnn import visualize
print(matplotlib.get_backend())
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/epu/"))  # To find local version
sys.path.append(os.path.join(ROOT_DIR, "samples/epu"))  # To find local version
from epu import *

#modify..............................................................................
xun = {'insulator','switch','breaker','tank','bushing','fin','pedestal','conservator','pipe','arrester','capacitor','inductor','bus','CT','PT','line','frame','resistor','connecting port','tower','pole'}
equipment = {'transformer','GIS','insulator','switch','breaker','arrester','inductor','bus','line','frame','resistor','whole capacitor','PT+insulator','CT+insulator','filter','connecting port','tower','pole','nest'}
component = {'tank','bushing','fin','pedestal','conservator','pipe','capacitor','CT','PT','bus+bushing','switch+insulator'}

dataset_val = EpuDataset()
dataset_val.load_epu('D:\\Python\\bayes', "val")
dataset_val.prepare()
#print(matplotlib.get_backend())

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_epu_0030.h5")
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(EpuConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
#model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
#class_names = ["BG", "transformer", "GIS", "insulator", "switch", "breaker", "tank", "bushing", "fin", "pedestal", "conservator", "pipe", "arrester", "capacitor", "inductor", "bus", "CT", "PT", "line", "frame", "resistor", "whole capacitor", "bus+bushing", "PT+insualtor", "CT+insulator", "filter", "connecting port", "switch+insulator", "tower", "pole", "nest"]
#class_names = ['cat','dog']

#modify..............................................................................
class_names = list(xun)
class_names.insert(0,'BG')
'''
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
'''

# ## Run Object Detection+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from test_mask import *
# In[5]:
path = "D:\\Python\\bayes"
os.chdir(path)
'''
# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#name = 'cat.jpg'
image = skimage.io.imread('0121.jpg')

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
'''
f = open('typetable.txt','r')
typetable = eval(f.read())
f.close()

f = open('itypetable.txt','r')
itypetable = eval(f.read())
f.close()

f = open('areatable.txt','r')
areatable = eval(f.read())
#print(areatable)
f.close()

f = open('iareatable.txt','r')
iareatable = eval(f.read())
f.close()

f = open('regiontable.txt','r')
regiontable = eval(f.read())
f.close()

f = open('iregiontable.txt','r')
iregiontable = eval(f.read())
f.close()

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)

#modify..............................................................................
APs = {}
for i in range(22):
  APs[i] = list()
APPs = []
for image_id in image_ids:
    print(image_id)
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

#modify..............................................................................

    for typenum in range(1,22):
      ngt_class_id = []
      ngt_bbox = [] 
      ngt_mask = []
      nngt_class_id = []
      nngt_bbox = [] 
      nngt_mask = []
      nngt_score = []
      for i in range(gt_class_id.shape[0]):
        if gt_class_id[i] == typenum:
          ngt_class_id.append(gt_class_id[i])
          ngt_bbox.append(gt_bbox[i])
          ngt_mask.append(gt_mask[:,:,i])
      ngt_class_id = np.array(ngt_class_id)
      ngt_bbox = np.array(ngt_bbox)
      ngt_mask = np.array(ngt_mask)

      if ngt_mask.shape[0] != 0:
        ngt_maskk = np.zeros((ngt_mask.shape[1],ngt_mask.shape[2],ngt_mask.shape[0]))
        for i in range(ngt_mask.shape[0]):
          ngt_maskk[:,:,i] = ngt_mask[i]
    
      for i in range(0,r['class_ids'].shape[0]):
        if (r['class_ids'])[i] == typenum:
          nngt_class_id.append((r['class_ids'])[i])
          nngt_bbox.append((r['rois'])[i])
          nngt_mask.append((r['masks'])[:,:,i])
          nngt_score.append((r['scores'])[i])
      nngt_class_id = np.array(nngt_class_id)
      nngt_bbox = np.array(nngt_bbox)
      nngt_mask = np.array(nngt_mask)
      nngt_score = np.array(nngt_score)
      
      if nngt_mask.shape[0] != 0:
        nngt_maskk = np.zeros((nngt_mask.shape[1],nngt_mask.shape[2],nngt_mask.shape[0]))
        for i in range(nngt_mask.shape[0]):
          nngt_maskk[:,:,i] = nngt_mask[i]
          
    '''
    #bayes
    type_ids = testmask(typetable,areatable,regiontable,itypetable,iareatable,iregiontable,r,class_names,image)
    rr = {}
    rr["class_ids"] = np.array(type_ids)
    print(rr["class_ids"])
   # visualize.display_instances(image, r['rois'], r['masks'], rr['class_ids'], 
    #                        class_names, r['scores'])
    '''
#modify..............................................................................
    # Compute AP
    AP, precisions, recalls, overlaps =\
    utils.compute_ap(ngt_bbox, ngt_class_id, ngt_mask,
                     nngt_bbox, nngt_class_id, nngt_score, nngt_mask)
    APs[typenum].append(AP)
      #print(AP)
    '''
    # Compute AP after bayes
    APP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], rr["class_ids"], r["scores"], r['masks'])
    APPs.append(APP)
    print(APP)
        '''
#modify..............................................................................
for i in range(1,20):
  print(class_names[i], "mAP: ", np.mean(APs[i]))
#print("mAPP: ", np.mean(APPs))