import math

class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y


def GetAreaOfPolyGonbyVector(points):
    # 基于向量叉乘计算多边形面积
    area = 0
    if(len(points)<3):

         raise Exception("error")

    for i in range(0,len(points)-1):
        p1 = points[i]
        p2 = points[i + 1]

        triArea = (p1.x*p2.y - p2.x*p1.y)/2
        area += triArea
    return abs(area)

def get_area(x, y):

    points = []

    #x = [1,2,3,4,5,6,5,4,3,2]
    #y = [1,2,2,3,3,3,2,1,1,1] 
    for index in range(len(x)):
        points.append(Point(x[index],y[index]))

    area = GetAreaOfPolyGonbyVector(points)
    #print(area)
    #print(math.ceil(area))
    #assert math.ceil(area)==1
    return area
'''
if __name__ == '__main__':
    get_area()
    print("OK") 
'''