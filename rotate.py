import PIL
from matplotlib import pyplot as plt
img1 = plt.imread('./w_4849.jpg')
import cv2
import numpy as np
from scipy.ndimage.interpolation import affine_transform
 
def affine(img = None):
    # 画像読み込み

    # 画像サイズの取得(横, 縦)
    size = tuple(np.array([img.shape[1], img.shape[0]]))
 
    # 回転させたい角度
    rad = np.pi / 4
    # x軸方向に平行移動させたい距離
    move_x = 0
    # y軸方向に平行移動させたい距離
    #move_y = img.shape[0] * -0.5
    move_y = 0
    matrix = [
                [np.cos(rad),  -1 * np.sin(rad), move_x],
                [np.sin(rad),   np.cos(rad), move_y]
            ]
    affine_matrix = np.float32(matrix)
    img_afn = cv2.warpAffine(img, affine_matrix, size, flags=cv2.INTER_LINEAR)
    return img_afn
"""
def affine2(img = None):
    # 画像サイズの取得(横, 縦)
    size = tuple(np.array([img.shape[1], img.shape[0]]))
    
    # 回転させたい角度
    rad = np.pi / 4
    # x軸方向に平行移動させたい距離
    move_x = 0
    # y軸方向に平行移動させたい距離
    #move_y = img.shape[0] * -0.5
    move_y = 0
 
    matrix = [
                [np.cos(rad),  -1 * np.sin(rad), move_x],
                [np.sin(rad),   np.cos(rad), move_y]
            ]
 
    affine_matrix = np.float32(matrix)
    img_afn = affine_transform(img, affine_matrix, size)
    return img_afn
"""

from matplotlib import cm

#plt.imshow(img1[:,:,0]/255.,cmap = plt.cm.binary)
#img1 = 1.-img1
#img2 = affine(img1[:,:,0])
#img1 /= 255.
rows,cols = img1[:,:,0].shape
theta = 45
#img1 = 1-img1

#
theta = np.random() * 360
M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
img2 = cv2.warpAffine(img1,M,(cols,rows))
"""
for x in xrange(rows):
    for y in xrange(cols):
        if img2[x,y,0] == 255 and img2[x,y,1] == 255 and img2[x,y,2] == 255:
            img2[x,y,0] = 0
            img2[x,y,1] = 0
            img2[x,y,2] = 0
"""         
#print img2[:,:,0]
#ax1 = plt.subplot()
#plt.imshow(img2[:,:,0],cmap=plt.cm.binary)
#ax2 = plt.subplot()
plt.imshow((0.299*img2[:,:,0]+0.587*img2[:,:,1]+0.114*img2[:,:,2]),cmap = plt.cm.binary)
