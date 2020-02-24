#special region code
#1|2|3
#4|0|6
#7|8|9 

import numpy as np

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

def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
  #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
  if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
    return False
  if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
    return False
  if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
    return False
  if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
    return False
  if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
    return False
  if s_poi[0]<poi[0] and e_poi[0]<poi[0]: #线段在射线左边
    return False

  xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
  if xseg<poi[0]: #交点在射线起点的左侧
    return False
  return True  #排除上述情况之后


def isPoiWithinPoly(poi,poly):
  #输入：点，多边形三维数组
  #poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组

  #可以先判断点是否在外包矩形内 
  #if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
  #但算最小外包矩形本身需要循环边，会造成开销，本处略去
  sinsc=0 #交点个数
  #for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
  s_poi=poly[len(poly)-1]
  e_poi=poly[0]
  if isRayIntersectsSegment(poi,s_poi,e_poi):
    sinsc+=1 #有交点就加1
  for i in range(len(poly)-1): #[0,len-1]
    s_poi=poly[i]
    e_poi=poly[i+1]
    if isRayIntersectsSegment(poi,s_poi,e_poi):
      sinsc+=1 #有交点就加1

  return True if sinsc%2==1 else  False

#to judge whether n2 is inside n1
def inside(n1,n2):
  poly = []
  flag = True
  for i in range(len(n1.xs)):
    poly.append([(n1.xs)[i],(n1.ys)[i]])
  for j in range(len(n2.xs)):
    poi = []
    poi.append((n2.xs)[j])
    poi.append((n2.ys)[j])
    if isPoiWithinPoly(poi,poly)==0:
      flag = False
      break
  return flag

#mask version
#to judge whether n2 is inside n1
def maskinside(n1,n2):
  poly = []
  flag = False
  res = n1.mask + n2.mask
  bing = res
  jiao = n1.mask & n2.mask 
  r1 = sum(sum(jiao))
  r2 = sum(sum(n2.mask))
  r3 = sum(sum(n1.mask))
  if r2 != 0:
    if r1/r2 >= 0.7:
      if r1/r3 < r1/r2:
        flag = True
  return flag


#n1's location relative to n2's
def region_relation(n1, n2):
  if n1.centroidx < n2.centroidx:
    if n1.centroidy < n2.centroidy:
      relation = 1
    elif n1.centroidy > n2.centroidy:
      relation = 7
    else:
      relation = 4
  elif n1.centroidx > n2.centroidx:
    if n1.centroidy < n2.centroidy:
      relation = 3
    elif n1.centroidy > n2.centroidy:
      relation = 9
    else:
      relation = 6
  else:
    if n1.centroidy < n2.centroidy:
      relation = 2
    elif n1.centroidy > n2.centroidy:
      relation = 8
    else:
      relation = 0
  
  return relation


def bounding_box(n):
  xs = n.xs
  ys = n.ys
  leng = len(xs)
  xmin = 10000
  xmax = -1
  ymin = 10000
  ymax = -1
  for i in range(leng):
    if xs[i] < xmin:
      xmin = xs[i]
    if xs[i] > xmax:
      xmax = xs[i]
    if ys[i] < ymin:
      ymin = ys[i]
    if ys[i] > ymax:
      ymax = ys[i]
  return max(0,xmin-5),max(0,ymin-5),xmax+5,ymax+5


#adjacent or not   flag=0 not  flag=1 yes
def whether_adjacent(n1, n2):
  flag = 0
  xmin1,ymin1,xmax1,ymax1 = bounding_box(n1)
  xmin2,ymin2,xmax2,ymax2 = bounding_box(n2)
  if xmin1 <= xmin2:
    if xmin2 <= xmax1:
      if ymin2 > ymax1:
        flag = 0
      elif ymax2 < ymin1:
        flag = 0
      else:
        flag =1
    else:
      flag = 0
  else:
    if xmin1 <= xmax2:
      if ymin1 > ymax2:
        flag = 0
      elif ymax1 < ymin2:
        flag = 0
      else:
        flag =1
    else:
      flag = 0
  
  return flag

#adjacent or not   flag=0 not  flag=1 yes
def mask_whether_adjacent(n1, n2, roi1, roi2):
  flag = 0
  xmin1,ymin1,xmax1,ymax1 = roi1
  xmin2,ymin2,xmax2,ymax2 = roi2
  if xmin1 <= xmin2:
    if xmin2 <= xmax1:
      if ymin2 > ymax1:
        flag = 0
      elif ymax2 < ymin1:
        flag = 0
      else:
        flag =1
    else:
      flag = 0
  else:
    if xmin1 <= xmax2:
      if ymin1 > ymax2:
        flag = 0
      elif ymax1 < ymin2:
        flag = 0
      else:
        flag =1
    else:
      flag = 0
  
  return flag
  
#R[i,j] --> i to j
def get_relation(nodee):
  length = len(nodee) 
  #print(length)
  #print(nodee)
  R = np.zeros((length,length))
  if length > 1:
    for i in range(length-1):
      for j in range(i+1):
        flag = whether_adjacent(nodee[i], nodee[j])
        if flag == 0:
          R[i,j] = 0
        else:
          if nodee[i].priviledge > nodee[j].priviledge:
            R[i,j] = region_relation(nodee[i], nodee[j])
          else:
            R[j,i] = region_relation(nodee[j], nodee[i])
  return R

#R[i,j] --> i to j
def mask_get_relation(nodee):
  length = len(nodee) 
  #print(length)
  #print(nodee)
  R = np.zeros((length,length))
  if length > 1:
    for i in range(length-1):
      for j in range(i+1):
        flag = mask_whether_adjacent(nodee[i], nodee[j], nodee[i].roi, nodee[j].roi)
        if flag == 0:
          R[i,j] = 0
        else:
          if nodee[i].priviledge > nodee[j].priviledge:
            R[i,j] = region_relation(nodee[i], nodee[j])
          else:
            R[j,i] = region_relation(nodee[j], nodee[i])
  return R
