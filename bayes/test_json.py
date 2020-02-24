#对于整体的检验
#json版

from jsonnn import *
'''
def get_mint(typetable):
  tmin = np.ones((5,1))
  #get the minimum of typetable
  for i in range(6):
    if len(typetable[i]) != 0:
      for k in range(len(typetable[i])/3):
        if (typetable[i])[k,3]) < min[i]:
          tmin[i] = (typetable[i])[k,3])
  return tmin
'''
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
def test(typetable,areatable,regiontable,itypetable,iareatable,iregiontable):
  p = []
  ip = []
  regiontable = {}
  #start = 0
  num = 1
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
    for i in range(0,len(nodeee)):
      pa += area_test(nodeee[i],areatable)
    for i in range(0,len(nodee)):
      node = {}
      if nodee[i].children != []:
        print('ok')
        length = len(nodee[i].children)
        #cnode[len(cnode),0] = nodee[i].type
        #cnode[len(cnode),1] = nodee[i].children
        for j in range(len(nodee[i].children)):
          node[len(node)] = nodee[(nodee[i].children)[j]]
          ipa += area_test(nodee[(nodee[i].children)[j]],iareatable)
        iR = get_relation(node)
        imy_type = get_mytype(iR,node,imy_type)
        imy_dict = istatistics(nodee[i].type,node,iR,imy_dict)
    R = get_relation(nodeee)
    my_dict = statistics(nodeee,R,my_dict)
    num += 1
    #pa = math.log(pa, 10)
    #ipa = math.log(ipa, 10)
    pt = type_test(my_dict,typetable)
    pr = region_test(nodee,R,regiontable)
    #pt = math.log(pt, 10)
    #pr = math.log(pr, 10)
    ipt = type_test(imy_dict,itypetable)
    ipr = region_test(node,iR,iregiontable)
    #ipt = math.log(ipt, 10)
    #ipr = math.log(ipr, 10)
    #print(pt)
    #print(pa)
    #print(pr)
    p.append(pt + pa + pr)
    #p.append(pt + pr)
    ip.append(ipt + ipa + ipr)
    #ip.append(ipt + ipr)
  print(p)
  print(ip)
  '''
  print(my_dict[0])
  print(my_dict[1])
  print(my_dict[2])
  print(my_dict[3])
  print(my_dict[4])
  print(my_dict[5])
  print(my_dict[6])
  '''
        #print(nodee[i].area)
      #print(type)
      #print(xs)
      #print(ys)
      #print(area)
      #print(centroidx) 
      #print(centroidy)
      #print(nodee[3].area)
  print(num)
  return p,ip

if __name__ == '__main__':
  delete_unmarked_merge()
  typetable,regiontable,areatable,itypetable,iregiontable,iareatable = acquire_information()
  p,ip = test(typetable,areatable,regiontable,itypetable,iareatable,iregiontable)