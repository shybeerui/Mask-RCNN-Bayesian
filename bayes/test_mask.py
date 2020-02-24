#对于整体的检验
#mask版
from mrcnn import visualize
from jsonnn import *
#from demo import *

def get_mina(areatable):
  tmin = np.ones((5,1))
  #get the minimum of typetable
  for i in range(6):
    if len(typetable[i]) != 0:
      for k in range(len(typetable[i])/3):
        if (typetable[i])[k,3] < min[i]:
          tmin[i] = (typetable[i])[k,3]
  return tmin

def get_minr(regiontable):
  tmin = np.ones((5,1))
  #get the minimum of typetable
  if len(typetable[i]) != 0:
    for k in range(len(typetable[i])/3):
      if (typetable[i])[k,3] < min[i]:
        tmin[i] = (typetable[i])[k,3]
  return tmin

def type_test(my_dict,typetable):
  my_type = {}
  table = {}
  pt = 0
  for i in range(6):
    my_type[i] = {}
    table[i] = {}
  #get the product of probability
  for i in range(6):
    if len(my_dict[i]) != 0:
      for j in range(math.floor(len(my_dict[i])/3)):
        flag = True
        min = 1
        ll = len((my_dict[i])[j,0])
        for k in range(math.floor(len(typetable[ll])/3)):
          if set((my_dict[i])[j,0]) == set((typetable[ll])[k,0]):
            if (typetable[ll])[k,2] < min:
              min = (typetable[ll])[k,2]
            if (my_dict[i])[j,1] == (typetable[ll])[k,1]:
              pt += math.log((typetable[ll])[k,2],10)
              flag = False
              break
        if flag:
          pt += math.log(min/100,10)
  #所有图片type的概率
  return pt

#一个一个进
def area_test(nodee,areatable):
  flag = True
  type = nodee.type
  area = nodee.area
  for i in range(math.floor(len(areatable)/3)):
    if type == areatable[i,0]:
      avg = areatable[i,1]
      sigma2 = areatable[i,2]
      if sigma2 != 0:
        pa = round((np.exp(-0.5 * ((area - avg) / sigma2)) / (np.sqrt(2*np.pi*sigma2))), 4)
        flag = False
        if pa == 0:
          flag = True
        if flag == 0:
          pa = math.log(pa,10)
      break
  if flag:
    pa = math.log(0.001,10)
  return pa

def region_test(nodee,R,regiontable):
  pr = 0
  flag = True
  min = 1
  for i in range(len(nodee)):
    for j in range(len(nodee)):
      for k in range(math.floor(len(regiontable)/4)):
        if regiontable[k,0] == nodee[i] and regiontable[k,1] == nodee[j]:
          if regiontable[k,3] < min:
            min = regiontable[k,3]
          if regiontable[k,2] == R[i,j]:
            pr += math.log(regiontable[k,3],10)
            flag = False 
            break
      if flag:
        pr += math.log(min/100,10)
  return pr

#test part
#def testmask(typetable,areatable,regiontable,itypetable,iareatable,iregiontable,maskresult,class_names,filename):
def testmask(typetable,areatable,regiontable,itypetable,iareatable,iregiontable,maskresult,class_names,image):
  #r = maskresult[0]
  r = maskresult
  rois = r['rois']
  masks = r['masks']
  #type = r['type']
  class_ids = r['class_ids']
  scores = r['scores']
  
  #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
   #                         class_names, r['scores'])

  p = []
  ip = []
  regiontable = {}
  #start = 0
  num = 1
  ptotal = -10000
  type_ids = []
  my_dict = {}
  imy_dict = {}
  my_type = {}
  imy_type = {}
  for i in range(15):
    my_dict[i] = {}
    imy_dict[i] = {}
  pa = 0
  ipa = 0

  area = {}
  centroidx = {}
  centroidy = {}
  nodee = {}
  R = {}
  iR = {}
  maxarea = 0;
  #information from each image
  #type = [class_names[i] for i in class_ids]
  for i in range(0,len(scores)):
    area[i] = np.sum(np.sum(masks[:,:,i]))
    #area[i] = np.count_nonzero(masks[i])
    #print(masks[:,:,i])
    #print(masks[:,:,i].shape)
    #print(type(masks[:,:,i]))
    #print(area[i])
    if area[i] > maxarea:
      maxarea = area[i]
    centroidx[i] = (rois[i][0] + rois[i][2])/2
    centroidy[i] = (rois[i][1] + rois[i][3])/2
      #print(type[i])
    #type在此修改
    nodee[i] = nnode(class_names[class_ids[i]],area[i],centroidx[i],centroidy[i],i)
    nodee[i].mask = masks[:,:,i]
    nodee[i].get_roi(rois[i])
    #print(nodee[3].area)
  for k in range(0,len(scores)):
    nodee[k].aver(maxarea)
    nodee[k].maskcontain(nodee)
  nodeee = get_container(nodee)
  for i in range(0,len(scores)):
    #print('s')
    pa = 0
    ipa = 0
    ptotal = ptotal -1
    tmp  = class_ids[i]
    #print(tmp)
    for ii in range(1,len(class_names)):
      nodee[i].type = class_names[ii]
      for k in range(0,len(nodeee)):
        pa += area_test(nodeee[k],areatable)
      for k in range(0,len(nodee)):
        node = {}
        if nodee[k].children != []:
          #print('ok')
          length = len(nodee[k].children)
            #cnode[len(cnode),0] = nodee[i].type
            #cnode[len(cnode),1] = nodee[i].children
          for j in range(0,len(nodee[k].children)):
            node[len(node)] = nodee[(nodee[k].children)[j]]
            ipa += area_test(nodee[(nodee[k].children)[j]],iareatable)
          iR = mask_get_relation(node)
          imy_type = get_mytype(iR,node,imy_type)
          imy_dict = istatistics(nodee[k].type,node,iR,imy_dict)
      R = mask_get_relation(nodeee)
      my_dict = statistics(nodeee,R,my_dict)
      num += 1
      pt = type_test(my_dict,typetable)
      pr = region_test(nodee,R,regiontable)
      ipt = type_test(imy_dict,itypetable)
      ipr = region_test(node,iR,iregiontable)
      #p = pt + pa + pr
      p = pt + pr
        #p.append(pt + pr)
      #ip = ipt + ipa + ipr
      ip = ipt + ipr
        #ip.append(ipt + ipr)
      #print(p+ip)
      if p+ip > ptotal:
        ptotal = p + ip
        tmp = ii
    nodee[i].type = class_names[tmp]
    type_ids.append(tmp)
        #print(nodee[i].area)
      #print(type)
      #print(xs)
      #print(ys)
      #print(area)
      #print(centroidx) 
      #print(centroidy)
      #print(nodee[3].area)
  print(num)
  return type_ids

if __name__ == '__main__':
  delete_unmarked_merge()
  typetable,regiontable,areatable,itypetable,iregiontable,iareatable = acquire_information()
  print('finished')
  '''
  j1 = json.load(open('neww.json'), strict=False)
  content = list(j1.values())
  for c in content:
    my_dict = {}
    imy_dict = {}
    my_type = {}
    imy_type = {}
    for i in range(7):
      my_dict[i] = {}
      imy_dict[i] = {}
    pa = 0
    ipa = 0
    area = {}
    centroidx = {}
    centroidy = {}
    nodee = {}
    R = {}
    iR = {}
    maxarea = 0;
    #information from each image
    xs = [r['shape_attributes']['all_points_x'] for r in c['regions']]
    ys = [r['shape_attributes']['all_points_y'] for r in c['regions']]
    type = [r['region_attributes']['type'] for r in c['regions']]
    name = c['filename']
    for i in range(0,len(xs)):
      area[i] = get_area(xs[i],ys[i])
      if area[i] > maxarea:
        maxarea = area[i]
      centroidx[i],centroidy[i] = get_centroid(xs[i],ys[i])
      #print(type[i])
      nodee[i] = nnode(type[i],area[i],centroidx[i],centroidy[i],i,xs[i],ys[i])
      #print(nodee[3].area)
    for i in range(0,len(xs)):
      nodee[i].aver(maxarea)
      nodee[i].contain(nodee)
    nodeee = get_container(nodee)
    maskresult = polytomask(nodee,name)
    p,ip = testmask(typetable,areatable,regiontable,itypetable,iareatable,iregiontable,maskresult,name)
    '''
  p,ip = testmask(typetable,areatable,regiontable,itypetable,iareatable,iregiontable,r,class_names)
  print('yes')
  