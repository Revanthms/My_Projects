import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull #for plotting convex polygon
from shapely.geometry import Polygon #for finding area and perimeter
import math

#This function is to sort the random corners of the polygon in anticlock-wise direction by calculating the centroid and measuring the angles between the centroid and the centres

def algo(point):
    return (math.atan2(point[0] - mean_x, point[1] - mean_y) + 2 * math.pi) % (2*math.pi)

img=cv2.imread('/home/revanth/bytemInternship/images_for_programs/hexi.png')#give the path to the image you want to find the perimeter and area
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray scale image
gray=np.float32(gray)

#corner detection
corners=cv2.goodFeaturesToTrack(gray,6,0.1,30) #get the number of edges that the polygon has from the user to give it in the second parameter
corners=np.int0(corners)
xvalues=[]
yvalues=[]

for corner in corners:
  x,y=corner.ravel()
  xvalues.append(x)
  yvalues.append(y)
  cv2.circle(img, (x,y),3,255,-1)
  
cv2.imshow('',img)
cv2.waitKey()

#print xvalues
#print yvalues
mean_x=float(sum(xvalues))/len(xvalues)
mean_y=float(sum((yvalues)))/len(xvalues)


point=[]

for i in range(len(xvalues)):
    point.append((xvalues[i],yvalues[i]))

#plt.show()  
point.sort(key=algo) #sort corners in anticlockwise direction
sides=len(point)

perimeter= Polygon(point).length  # this finds the perimeter by measuring the length between succesive points in the list point
Area= Polygon(point).area # this finds the area through shoelace algorithm

print ('perimeter of the polygon is',perimeter)
print ('Area of the polygon is',Area)

points=np.array(point)
hull = ConvexHull(points)
 
#this is to plot a convex polygon for a given set of points
#plt.plot(points[:,0], points[:,1], 'o')


plt.plot(xvalues,yvalues,'ro')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')


plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()


#upperx=max(xvalues)
#uppery=max(yvalues)
#lowerx=min(xvalues)
#lowery=min(yvalues)
#
#length=upperx-lowerx
#breadth=uppery-lowery
#
#perimeter=2*(length+breadth)
#print perimeter
#area=length*breadth
#print area
#
#ratio=float(length)/breadth
#
#print ratio

#cv2.imshow('img',img)
#cv2.waitKey()



