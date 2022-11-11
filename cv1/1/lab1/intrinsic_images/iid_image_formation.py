import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load images 
real_img = cv2.imread('ball.png')
shading_img = cv2.imread('ball_shading.png')
albedo_img = cv2.imread('ball_albedo.png')

# I=R*S
res = np.multiply(albedo_img/255, shading_img/255)

# Plot
f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(real_img[:,:,::-1])
axarr[0, 0].set_title('Original Image')
axarr[0,1].imshow(shading_img[:,:,::-1])
axarr[0, 1].set_title('Shading Image')
axarr[1,0].imshow(albedo_img[:,:,::-1])
axarr[1,0].set_title('Albedo Image')
axarr[1,1].imshow(res[:,:,::-1])
axarr[1,1].set_title('Result')
plt.show()