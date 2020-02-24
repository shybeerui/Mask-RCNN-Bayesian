from get_area import *
from region_relation import *
from statistics import *
from polytomask import *
from shuffle import *
import json   
import os
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt  

mkpath = "D:\deskTopppp\变电站标注"
os.chdir(mkpath)

equipment2 = {'transformer','GIS','switch','break','arrester','inductor','bus','frame','resistor','whole capacitor','PT+insulator','CT+insulator','filter','connecting port','tower','pole'}
equipment1 = {'insulator','nest','line'}
#equipment = equipment1|equipment2|{'tank','bushing',''}
class nnode():
  def get_priviledge(self):
    if self.type in equipment2:
      self.priviledge = 2
    elif self.type in equipment1:
      self.priviledge = 1
    else:
      self.priviledge = 0

  def __init__(self,type,area,centroidx,centroidy,number,xs={},ys={}):
    self.type = type
    self.area = area
    self.centroidx = centroidx
    self.centroidy = centroidy
    self.xs = xs
    self.ys = ys
    self.get_priviledge()
    self.number = number
    self.children = []
    self.mask = {}

  def aver(self,maxarea):
    self.area = self.area/maxarea
  
  def contain(self,nodee):
    for i in range(len(nodee)):
      if i != self.number:
        if(inside(self,nodee[i])):
          self.children.append(i)
          
  def maskcontain(self,nodee):
    for i in range(len(nodee)):
      if i != self.number:
        if(maskinside(self,nodee[i])):
          self.children.append(i)
          
  def polytomask(self,filename):
    mkpath = "D:\deskTopppp\毕设\全部原图"
    os.chdir(mkpath)
    #path = os.path.join(mkpath,filename)
    img = cv2.imread(filename)
    size = img.shape
    img = np.zeros(size,'uint8')
    triangle = []
    xs = self.xs
    ys = self.ys
    for j in range(len(xs)):
      triangle.append([xs[j],ys[j]]) 
    triangle = np.array(triangle)
    cv2.fillPoly(img, [triangle], (1,1,1))
    mask = img[:,:,0]
    self.mask = mask
    #plt.imshow(img)
    #plt.show()
    #mkpath = "D:\deskTopppp\毕设\RCNN\mask_rcnn"
    #os.chdir(mkpath)
  
  def get_roi(self,roi):
    self.roi = roi

def probability_distribution(data, bins_interval=0.01, margin=0.1):
    ll = 1 / bins_interval
    start = 0 - bins_interval
    bins = []
    for i in range(math.floor(ll+1)):
      bins.append(start + bins_interval) 
    #print(len(bins))
    #for i in range(0, len(bins)):
    #    print(bins[i])
    plt.xlim(min(data) - margin, max(data) + margin)
    plt.title("probability-distribution")
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    plt.hist(data, 100, histtype='bar', color=['r'])
    plt.show()

def get_centroid(x, y):
  xc = sum(x)/len(x)
  yc = sum(y)/len(y)
  return xc,yc

#delete unmarked and merge
def delete_unmarked_merge():
  nnum = 0
  #clear
  f = open('new.json', 'w')
  f.write('{')
  f.close()
  #delete and merge
  for i in range(0,26):
    if i == 12:
      continue
    num = 0
    j1 = json.load(open('via_region_data ({}).json'.format(i)))
    key = list(j1.keys())
    content = list(j1.values())
    for c in content:
      if c['regions']:
        #print(c['filename'])
        f = open('new.json', 'a')
        if nnum == 0:
          f.write('"'+key[num]+'":')
        else:
          f.write(',"'+key[num]+'":')
        f.close()
        json.dump(c, open('new.json', 'a'))
        nnum += 1
        num += 1
  print(nnum)
  f = open('new.json', 'a')
  f.write('}')
  f.close()

def get_container(nodee):
  nodeee = {}
  num = 0
  containerf = np.ones((len(nodee),1))
  for i in range(len(nodee)):
    if nodee[i].children != []:
      for j in range(len(nodee[i].children)):
        containerf[(nodee[i].children)[j]] = 0
  for i in range(len(containerf)):
    if containerf[i]:
      nodeee[num] = nodee[i]
      num += 1
  return nodeee

def type_table(my_dict):
  my_type = {}
  table = {}
  for i in range(6):
    my_type[i] = {}
    table[i] = {}
  #get all the father types
  for i in range(6):
    if len(my_dict[i]) != 0:
      for j in range(math.floor(len(my_dict[i])/3)):
        flag = True
        for k in range(len(my_type[i])):
          if set((my_dict[i])[j,0]) == set((my_type[i])[k]):
            flag = False
            break
        if flag:
          (my_type[i])[len(my_type[i])] = (my_dict[i])[j,0]
  #get type-table     type type pro
  for i in range(6):
    if len(my_dict[i]) != 0:
      for k in range(len(my_type[i])):
        flag = True
        num = 0
        index = []
        for j in range(math.floor(len(my_dict[i])/3)):
          if set((my_dict[i])[j,0]) == set((my_type[i])[k]):
            md = table[i]
            length = len(md)
            index.append(math.floor(length/3))
            md[math.floor(length/3),0] = (my_type[i])[k]
            md[math.floor(length/3),1] = (my_dict[i])[j,1]
            md[math.floor(length/3),2] = (my_dict[i])[j,2]
            num += (my_dict[i])[j,2]
        for l in range(len(index)):
          md[index[l],2] /= num
  return table

def get_mytype(R,nodee,my_type):
  for i in range(len(R)):
    for j in range(len(R)):
      flag = True
      if R[i,j] != 0:
        if len(my_type) == 0:
          my_type[0,0] = nodee[i].type
          my_type[0,1] = nodee[j].type
        else:
          for k in range(math.floor(len(my_type)/2)):
            if nodee[i].type == my_type[k,0] and nodee[j].type == my_type[k,1]:
              flag = False
              break
          if flag:
            my_type[k+1,0] = nodee[i].type
            my_type[k+1,1] = nodee[j].type
  return my_type

def iregion_table1(R,nodee,my_type,table):
  for i in range(math.floor(len(my_type)/2)):
    for j in range(len(R)):
      for k in range(len(R)):
        if R[j,k] != 0:
          if nodee[j].type == my_type[i,0] and nodee[k].type == my_type[i,1]:
            length = len(table)
            if length == 0:
              table[0,0] = my_type[i,0]
              table[0,1] = my_type[i,1]
              table[0,2] = R[j,k]
              table[0,3] = 1
            flag = True
            for l in range(math.floor(length/4)):
              if nodee[j].type == table[l,0] and nodee[k].type == table[l,1] and R[j,k] == table[l,2]:
                table[l,3] += 1
                flag = False
                break
            if flag:
              table[math.floor(length/4),0] = my_type[i,0]
              table[math.floor(length/4),1] = my_type[i,1]
              table[math.floor(length/4),2] = R[j,k]
              table[math.floor(length/4),3] = 1
  return table

def iregion_table2(my_type,table):
  for i in range(math.floor(len(my_type)/2)):
    length = len(table)
    flag = True
    sum = 0
    index = []
    for l in range(math.floor(length/4)):
      if my_type[i,0] == table[l,0] and my_type[i,1] == table[l,1]:
        sum += table[l,3]
        index.append(l)
    for j in range(len(index)):
      if sum != 0:      
        table[index[j],3] /= sum
  return table

def region_table1(R,nodee,my_type,table):
  for i in range(math.floor(len(my_type)/2)):
    for j in range(len(R)):
      for k in range(len(R)):
        if R[j,k] != 0:
          if nodee[j].type == my_type[i,0] and nodee[k].type == my_type[i,1]:
            length = len(table)
            if length == 0:
              table[0,0] = my_type[i,0]
              table[0,1] = my_type[i,1]
              table[0,2] = R[j,k]
              table[0,3] = 1
            flag = True
            for l in range(math.floor(length/4)):
              if nodee[j].type == table[l,0] and nodee[k].type == table[l,1] and R[j,k] == table[l,2]:
                table[l,3] += 1
                flag = False
                break
            if flag:
              table[math.floor(length/4),0] = my_type[i,0]
              table[math.floor(length/4),1] = my_type[i,1]
              table[math.floor(length/4),2] = R[j,k]
              table[math.floor(length/4),3] = 1
  return table

def region_table2(my_type,table):
  for i in range(math.floor(len(my_type)/2)):
    length = len(table)
    flag = True
    sum = 0
    index = []
    for l in range(math.floor(length/4)):
      if my_type[i,0] == table[l,0] and my_type[i,1] == table[l,1]:
        sum += table[l,3]
        index.append(l)
    for j in range(len(index)):
      if sum != 0:      
        table[index[j],3] /= sum
  return table

def get_typee(nodee,typee):
  for i in range(len(nodee)):
    flag = True
    for j in range(len(typee)):
      if nodee[i].type == typee[j]:
        flag = False
        break
    if flag:
      typee.append(nodee[i].type)
  return typee

def area_table1(nodee,typee,stypee):
  for i in range(len(typee)):
    for j in range(len(nodee)):
      flag = True
      if nodee[j].type == typee[i]:
        for k in range(math.floor(len(stypee)/3)):
          if stypee[k,0] == nodee[j].type and stypee[k,1] == nodee[j].area:
            stypee[k,2] += 1
            flag = False
            break
        if flag:
          stypee[math.floor(len(stypee)/3),0] = nodee[j].type
          stypee[math.floor(len(stypee)/3),1] = nodee[j].area
          stypee[math.floor(len(stypee)/3),2] = 1
  return stypee

def area_table2(typee,stypee):
  table = {}
  for i in range(len(typee)):
    res = []
    for j in range(math.floor(len(stypee)/3)):
      if stypee[j,0] == typee[i]:
        for k in range(stypee[j,2]):
          res.append(stypee[j,1])
    N = len(res)
    nres = np.array(res)
    sum1 = nres.sum()
    nres2 = nres*nres
    sum2 = nres2.sum()
    mean = sum1/N
    var = sum2/N - mean**2
    table[math.floor(len(table)/3),0] = typee[i]
    table[math.floor(len(table)/3),1] = mean
    table[math.floor(len(table)/3),2] = var
    #probability_distribution(res)
    #res.sort()
    #plt.bar(range(len(res)), res)  
    #plt.show()  
    #print(scipy.stats.normaltest(res))
  return table

def deep_acquire_information1(sum,nodee,typee,my_type,my_dict):
  #cnode = {}
  for i in range(len(nodee)):
    con = []
    node = {}
    R = {}
    if nodee[i].children != []:
      sum += 1
      length = len(nodee[i].children)
      #cnode[len(cnode),0] = nodee[i].type
      #cnode[len(cnode),1] = nodee[i].children
      for j in range(len(nodee[i].children)):
        node[len(node)] = nodee[(nodee[i].children)[j]]
      typee = get_typee(node,typee)
      R = get_relation(node)
      my_type = get_mytype(R,node,my_type)
      my_dict = istatistics(nodee[i].type,node,R,my_dict)
  return sum,typee,my_type,my_dict

def deep_acquire_information2(nodee,regiontable,stypee,typee,my_type,my_dict):
  cnode = {}
  for i in range(len(nodee)):
    con = []
    node = {}
    R = {}
    if nodee[i].children != []:
      length = len(nodee[i].children)
      cnode[len(cnode),0] = nodee[i].type
      cnode[len(cnode),1] = nodee[i].children
      for j in range(len(nodee[i].children)):
        node[len(node)] = nodee[(nodee[i].children)[j]]
    R = get_relation(node)
    regiontable = region_table1(R,node,my_type,regiontable)
    stypee = area_table1(node,typee,stypee)
  return regiontable,stypee

#acquire information
def acquire_information():
  my_dict = {}
  my_type = {}
  regiontable = {}
  typee = []
  imy_dict = {}
  imy_type = {}
  iregiontable = {}
  itypee = []
  sum = 0
  #start = 0
  for i in range(7):
    my_dict[i] = {}
    imy_dict[i] = {}
  num = 1;
  j1 = json.load(open('new.json'), strict=False)
  content = list(j1.values())
  for c in content:
    area = {}
    centroidx = {}
    centroidy = {}
    nodee = {}
    R = {}
    maxarea = 0
    #information from each image
    #if num <= 3:
    xs = [r['shape_attributes']['all_points_x'] for r in c['regions']]
    ys = [r['shape_attributes']['all_points_y'] for r in c['regions']]
    type = [r['region_attributes']['type'] for r in c['regions']]
    name = c['filename']
    #print(c['filename'])
    for i in range(0,len(xs)):
      area[i] = get_area(xs[i],ys[i])
      if area[i] > maxarea:
        maxarea = area[i]
      centroidx[i],centroidy[i] = get_centroid(xs[i],ys[i])
      #print(type[i])
      nodee[i] = nnode(type[i],area[i],centroidx[i],centroidy[i],i,xs[i],ys[i])
      #nodee[i].polytomask(name)
      #print(nodee[3].area)
    for i in range(0,len(xs)):
      nodee[i].aver(maxarea)
      nodee[i].contain(nodee)
      #nodee[i].maskcontain(nodee)
    typee = get_typee(nodee,typee)
    sum,itypee,imy_type,imy_dict = deep_acquire_information1(sum,nodee,itypee,imy_type,imy_dict)
    nodeee = get_container(nodee)
    R = get_relation(nodeee)
    my_type = get_mytype(R,nodeee,my_type)
      #print(R)
    my_dict = statistics(nodeee,R,my_dict)
    #if num == 1:
     # break
    num += 1
  
  #print(typee)
  #print("my_type:")
  #print(my_type)

  stypee = {}
  istypee = {}
  for c in content:
    area = {}
    centroidx = {}
    centroidy = {}
    nodee = {}
    R = {}
    maxarea = 0;
    #information from each image
    #if num <= 3:
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
    for i in range(0,len(xs)):
      if len(nodee[i].children) != 0:
        print(nodee[i].type)
        for j in range(len(nodee[i].children)):
          print(nodee[(nodee[i].children)[j]].type)
        print('/s')
    iregiontable,istypee = deep_acquire_information2(nodee,iregiontable,istypee,itypee,imy_type,imy_dict)
    nodeee = get_container(nodee)
    R = get_relation(nodeee)
    regiontable = region_table1(R,nodeee,my_type,regiontable)
    stypee = area_table1(nodee,typee,stypee)
    #mytype = get_mytype(R,nodeee,my_type)
      #print(R)
    #my_dict = statistics(nodeee,R,my_dict)
    num += 1
  regiontable = region_table2(my_type,regiontable)
  areatable = area_table2(typee,stypee)
  iregiontable = region_table2(imy_type,iregiontable)
  iareatable = area_table2(itypee,istypee)
  typetable = type_table(my_dict)
  itypetable = type_table(imy_dict)

        #print(nodee[i].area)
      #print(type)
      #print(xs)
      #print(ys)
      #print(area)
      #print(centroidx) 
      #print(centroidy)
      #print(nodee[3].area)
  #print(num/2)
  #print(stypee)
  #print(regiontable)
  #print(areatable)
  #print(iregiontable)
  #print(iareatable)
  #print("imy_type:")
  #print(imy_type)
  #print(typetable)
  #print(sum)
  #print(itypetable)

  return typetable,regiontable,areatable,itypetable,iregiontable,iareatable



if __name__ == '__main__':
  delete_unmarked_merge()
  shuffle('new.json')
  #typetable,regiontable,areatable,itypetable,iregiontable,iareatable = acquire_information()

  #print(typetable)
  
  
  
'''
f = open('via_project_23Mar2019_17h19m (3)_1.json')    #打开文件 
ff = json.load(f)
aa = list(ff.values())
#aa = [a for a in aa if aa['filename']]
#aa = ff.values()
for a in aa:
  print(a['filename'])
  #print(a)
  #for r in a['filename']:
   # print(r)#把json串变成python的数据类型：字典，传一个文件对象，它会帮你读文件，不需要再单独读文件 
'''