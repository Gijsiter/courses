import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import getColourChannels

def visualize(input_image, colourspace):
    
    # Get the channels from the image
    if colourspace != 'gray':
        C1, C2, C3 = getColourChannels.getColourChannels(input_image)

        # Create sub plots to plot the whole image 
        # and the three different channels.
        _, im = plt.subplots(2,2)
        im[0,0].imshow(input_image)
        im[0,0].set_title('Full Image')
        im[0,1].imshow(C1)
        im[1,0].imshow(C2)
        im[1,1].imshow(C3)

    # Choose the correct label 
    if colourspace.lower() == 'opponent':
        L1 = 'O1'
        L2 = 'O2'
        L3 = 'O3'

    elif colourspace.lower() == 'rgb':
        L1 = 'R'
        L2 = 'G'
        L3 = 'B'

    elif colourspace.lower() == 'hsv':
        L1 = 'H'
        L2 = 'S'
        L3 = 'V'

    elif colourspace.lower() == 'ycbcr':
        L1 = 'Y'
        L2 = 'Cb'
        L3 = 'Cr'

    elif colourspace.lower() == 'gray':
        plt.imshow(input_image, cmap='gray')

    else:
        print('Error: Unknown colorspace type [%s]...' % colourspace)

    # Set the correct title
    if colourspace != 'gray':
        im[0,0].set_title('Full Image')
        im[0,1].set_title(L1)
        im[1,0].set_title(L2)
        im[1,1].set_title(L3)

    plt.show()