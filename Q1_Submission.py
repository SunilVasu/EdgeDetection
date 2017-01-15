import cv2
import numpy as np
import cmath
from PIL import Image
from matplotlib import pyplot as plt
from scipy import signal

image = cv2.imread('UBCampus.jpg',0)
cv2.imshow("OriginalImage.jpg",image)
(r,c)=image.shape
mask = np.array([[0,0,-1,-1,-1,0,0],[0,-2,-3,-3,-3,-2,0],[-1,-3,5,5,5,-3,-1],[-1,-3,5,16,5,-3,-1],[-1,-3,5,5,5,-3,-1],[0,-2,-3,-3,-3,-2,0],[0,0,-1,-1,-1,0,0]])

lap_mask = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])

#a)Compute and display the DoG image by applying the following DoG mask to the test image
dst = cv2.filter2D(image,-1,mask)

#using convolve2d func
#grad = signal.convolve2d(image, mask, boundary='symm', mode='valid')
#print "Here",grad
#dst = grad

cv2.imshow("DoG.jpg",dst)
cv2.imwrite("DoG.jpg",dst)
#b)Compute and display the zero-crossing of the DoG image obtained in (a)
(r,c)=dst.shape
array=dst
zc=np.zeros_like(array)

for x in range(r):
    for y in range (c):
	zc[x,y]=255
	
#marking the zero crossing edges from top to bottom & left to Right
for x in range(r):
    temp=np.zeros(c)
    for y in range (c):
	temp[y]=array[x,y]
    zero_crossings = np.where(np.diff(np.sign(temp)))[0]
    for q in range (len(zero_crossings)):
	zc[x,zero_crossings[q]]=array[x,zero_crossings[q]]
       
for x in range(c):
    temp=np.zeros(c)
    for y in range (r):
	temp[y]=array[y,x]
    zero_crossings = np.where(np.diff(np.sign(temp)))[0]
    for q in range (len(zero_crossings)):
	zc[zero_crossings[q],x]=array[zero_crossings[q],x]

#If using the convolve2d Function
#for x in range(0,r-2):
 #  for y in range (0,c-2):
#	if(array[x,y]*array[x,y+1]<0):
#		zc[x,y]=0
       
#for x in range(0,c-3):
 # for y in range (0,r-3):
#	if(array[y,x]*array[y+1,x]<0):
#		zc[y,x]=0
#		print zc.shape


print (r,c)	
cv2.imshow("DOG_ZeroCrossing.jpg",zc)
cv2.imwrite("DOG_ZeroCrossing.jpg",zc)
pic=zc

#c)Compute and display the zero-crossing strong edges by removing weak edges that do not
#have first derivative support in (b)

#First derivative image obtained using sobel filter as the vertical and horizontal lines can be clearly seen using this filter
sobel_mask_h = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#sobel_mask_h = np.array([[3,4,5,6,5,4,3],[2,3,4,5,4,3,2],[1,2,3,4,3,2,1],[0,0,0,0,0,0,0],[-1,-2,-3,-4,-3,-2,-1],[-2,-3,-4,-5,-4,-3,-2],[-3,-4,-5,-6,-5,-4,-3]])

sobel_mask_v = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
#sobel_mask_v = np.array([[3 ,  2,   1,   0,   -1,  -2,  -3],[4,   3,   2,   0,   -2,  -3,  -4],[5,   4,   3,   0,   -3,  -4,  -5],[6,   5,   4,   0,   -4,  -5,  -6],[5,   4,   3,   0,   -3,  -4,  -5],[4,   3,   2,   0,   -2,  -3,  -4],[3,   2,   1,   0,   -1,  -2,  -3]])

#other direction sobel filters
sobel_mask_1 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
sobel_mask_2 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
sobel_mask_3 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
sobel_mask_4 = np.array([[0,1,2],[-1,0,-1],[-2,-1,0]])

dst_h = cv2.filter2D(image,-1,sobel_mask_h)
dst_v = cv2.filter2D(image,-1,sobel_mask_v)

dst_1 = cv2.filter2D(image,-1,sobel_mask_1)
dst_2 = cv2.filter2D(image,-1,sobel_mask_2)
dst_3 = cv2.filter2D(image,-1,sobel_mask_3)
dst_4 = cv2.filter2D(image,-1,sobel_mask_4)
cv2.imwrite("sobel_h_DoG.jpg",dst_h)
cv2.imwrite("sobel_v_DoG.jpg",dst_v)
cv2.imwrite("sobel_1_DoG.jpg",dst_1)
cv2.imwrite("sobel_2_DoG.jpg",dst_2)
cv2.imwrite("sobel_3_DoG.jpg",dst_3)
cv2.imwrite("sobel_4_DoG.jpg",dst_4)

#removing weak edges that do not have first derivative(in sobel zero crossing images) from zero crossing image from (b)

for x in range(r):
    for y in range (c):
	if( (dst_v[x,y]<30)and(dst_h[x,y]<30)and(dst_1[x,y]<30)and(dst_2[x,y]<30)and(dst_3[x,y]<30)and(dst_4[x,y]<30) and zc[x,y]<100):
		pic[x,y]=255;
	if( (dst_v[x,y]>200)or(dst_h[x,y]>200)or(dst_1[x,y]>200)or(dst_2[x,y]>200)or(dst_3[x,y]>200)or(dst_4[x,y]>200) and zc[x,y]<60):
		pic[x,y]=zc[x,y];
	
	

cv2.imshow("Final DoG.jpg",pic)
cv2.imwrite("Final DoG.jpg",pic)


#d)Compute and display the LoG zero-crossing edges by applying the following LoG mask
#to the test image
#remove noise by applying a Guassian blur to the test Image

gaussian_mask=np.array([[(0.109),(0.111),(0.109)],[(0.111),(0.135),(0.111)],[(0.109),(0.111),(0.109)]],dtype='f')

logImg = cv2.filter2D(image,-1,gaussian_mask)

logDst = cv2.filter2D(image,-1,lap_mask)

#using convolve2d func
#grad = signal.convolve2d(logImg, lap_mask, boundary='symm', mode='valid')
#print "Here",grad.shape
#logDst = grad

cv2.imshow("logimgBlur.jpg",logImg)
cv2.imshow("logimg.jpg",logDst)

cv2.imwrite("logimgBlur.jpg",pic)
cv2.imwrite("logimg.jpg",pic)

#marking the zero crossing edges from top to bottom & left to Right
(r,c)=logDst.shape
array2=logDst
zc2=np.zeros_like(array2)
for x in range(r):
    for y in range (c):
  	zc2[x,y]=255;

for x in range(r):
    temp=np.zeros(c)
    for y in range (c):
	temp[y]=array2[x,y]
    zero_crossings2 = np.where(np.diff(np.sign(temp)))[0]
    for q in range (len(zero_crossings2)):
	zc2[x,zero_crossings2[q]]=array2[x,zero_crossings2[q]]

for x in range(c):
    temp=np.zeros(c)
    for y in range (r):
	temp[y]=array2[y,x]
    zero_crossings2 = np.where(np.diff(np.sign(temp)))[0]
    for q in range (len(zero_crossings2)):
	zc2[zero_crossings2[q],x]=array2[zero_crossings2[q],x]


#If using the convolve2d Function
#for x in range(0,r-2):
#   for y in range (0,c-2):
#	if(array2[x,y]*array2[x+1,y]<0):
#		zc2[x,y]=0
       
#for x in range(0,c-2):
 #  for y in range (0,r-2):
#	if(array2[y,x]*array2[y,x+1]<0):
#		zc2[y,x]=0



pic2=zc2	
cv2.imshow("LOG ZeroCrossing.jpg",zc2)
cv2.imwrite("LOG ZeroCrossing.jpg.jpg",zc2)

#Removing weak edges using sobel filter. We use sobel in all 6 directions
dst_h = cv2.filter2D(logImg,-1,sobel_mask_h)
dst_v = cv2.filter2D(logImg,-1,sobel_mask_v)
dst_1 = cv2.filter2D(logImg,-1,sobel_mask_1)
dst_2 = cv2.filter2D(logImg,-1,sobel_mask_2)
dst_3 = cv2.filter2D(logImg,-1,sobel_mask_3)
dst_4 = cv2.filter2D(logImg,-1,sobel_mask_4)
cv2.imwrite("sobel_h_LoG.jpg",dst_h)
cv2.imwrite("sobel_v_LoG.jpg",dst_v)
cv2.imwrite("sobel_1_LoG.jpg",dst_1)
cv2.imwrite("sobel_2_LoG.jpg",dst_2)
cv2.imwrite("sobel_3_LoG.jpg",dst_3)
cv2.imwrite("sobel_4_LoG.jpg",dst_4)
#removing weak edges that do not have first derivative(in sobel magnitude image) from zero crossing image zc2

for x in range(r):
    for y in range (c):
    	if(dst_h[x,y]<30 and dst_v[x,y]<30 and (dst_1[x,y]<30)and(dst_2[x,y]<30)and(dst_3[x,y]<30)and(dst_4[x,y]<30) and zc2[x,y]<100):
		pic2[x,y]=255
	if(dst_h[x,y]>200 or dst_v[x,y]>200 or(dst_1[x,y]>200)or(dst_2[x,y]>200)or(dst_3[x,y]>200)or(dst_4[x,y]>200) and zc2[x,y]<60):
		pic2[x,y]=zc2[x,y]


cv2.imshow("Final LOG.jpg",pic2)
cv2.imwrite("LoG.jpg",pic2)

cv2.waitKey(0)
cv2.destroyAllWindows()















