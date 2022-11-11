import numpy as np
import cv2
import matplotlib.pyplot as plt

#1 TODO
#find rgb
real_img = cv2.imread('ball.png')
albedo_image = cv2.imread("ball_albedo.png")
shading_img = cv2.imread('ball_shading.png')


# print(set(image.flatten()))
#RGB 184, 141, 108

#2 
green = [0,255,0]
albedo_image[np.where((albedo_image==[108,141,184]).all(axis=2))]=green

res = np.multiply(albedo_image/255, shading_img/255)

# Plot
f, axarr = plt.subplots(1,2)
axarr[0].imshow(real_img[:,:,::-1])
axarr[0].set_title('Original Image')
axarr[1].imshow(res[:,:,::-1])
axarr[1].set_title('Green Image')

plt.show()

#3
