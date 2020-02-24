import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from jsonnn import *
from region_relation import *

def polytomask(nodee,filename):
  mask = []
  type = []
  roi = []
  maskresult = {}
  mkpath = "D:\deskTopppp\毕设\全部原图"
  os.chdir(mkpath)
  path = os.path.join(mkpath,filename)
  img = cv2.imread(filename)
  size = img.shape
  for i in range(len(nodee)):
    img = np.zeros(size,'uint8')
    triangle = []
    a = []
    xs = nodee[i].xs
    ys = nodee[i].ys
    for j in range(len(xs)):
      triangle.append([xs[j],ys[j]]) 
    triangle = np.array(triangle)
    cv2.fillPoly(img, [triangle], (1,1,1))
    mask.append(img[:,:,0])
    type.append(nodee[i].type)
    x1,y1,x2,y2 = bounding_box(nodee[i])
    a.append(x1)
    a.append(y1)
    a.append(x2)
    a.append(y2)
    roi.append(a)
  maskresult['masks'] = mask
  maskresult['type'] = type
  maskresult['rois'] = roi
  #print(roi)
  #print(type(mask))
  #plt.imshow(img)
  #plt.show()
  return maskresult


