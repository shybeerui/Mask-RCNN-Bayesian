#PGM
#classify according to the number of parent nodes
#traverse child nodes

import math

class nnode():
  def get_priviledge(self):
    if self.type in equipment2:
      self.priviledge = 2
    elif self.type in equipment1:
      self.priviledge = 1
    else:
      self.priviledge = 0

  def __init__(self,type,area,centroidx,centroidy,xs,ys):
    self.type = type
    self.area = area
    self.centroidx = centroidx
    self.centroidy = centroidy
    self.xs = xs
    self.ys = ys
    self.get_priviledge()

  def aver(self,maxarea):
    self.area = self.area/maxarea

  def contain(self,nodee):
    for i in range(len(nodee)):
      if i != self.number:
        if(inside(self,nodee[i])):
          self.children.append(i)


#0：父节点  1：子节点  2：个数
def statistics(nodee,R,my_dict):
  for i in range(len(nodee)):
    #parent = {}
    num = 0
    flag = True
    #if nodee[i].priviledge == 0:
    #Traversing through child nodes
    if nodee[i].children == []:
      T = list()
      for j in range(len(nodee)):
        if R[j,i] != 0:
          #parent[num] = nodee[j].type
          T.append(nodee[j].type)
          num += 1
      #根据父节点的个数分类
      md = my_dict[num]
      length = len(md)
      #print(length)
      if length == 0:
        md[0,0] = T
        md[0,1] = nodee[i].type
        md[0,2] = 1
      else:
        for k in range(math.floor(length/3)):
          if set(T) == set(md[k,0]) and nodee[i].type == md[k,1]:
            md[k,2] += 1
            flag = False
            break
        if flag:
          md[math.floor(length/3),0] = T
          md[math.floor(length/3),1] = nodee[i].type
          md[math.floor(length/3),2] = 1
  return my_dict
        
def istatistics(cftype,nodee,R,my_dict):
  for i in range(len(nodee)):
    #parent = {}
    num = 1
    flag = True
    #if nodee[i].priviledge == 0:
    #Traversing through child nodes
    if nodee[i].children == []:
      T = [cftype]
      for j in range(len(nodee)):
        if R[j,i] != 0:
          #parent[num] = nodee[j].type
          T.append(nodee[j].type)
          num += 1
      #根据父节点的个数分类
      md = my_dict[num]
      length = len(md)
      #print(length)
      if length == 0:
        md[0,0] = T
        md[0,1] = nodee[i].type
        md[0,2] = 1
      else:
        for k in range(math.floor(length/3)):
          if set(T) == set(md[k,0]) and nodee[i].type == md[k,1]:
            md[k,2] += 1
            flag = False
            break
        if flag:
          md[math.floor(length/3),0] = T
          md[math.floor(length/3),1] = nodee[i].type
          md[math.floor(length/3),2] = 1
  return my_dict