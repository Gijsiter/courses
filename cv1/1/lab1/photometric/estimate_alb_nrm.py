import numpy as np
import cv2


def estimate_alb_nrm(image_stack, scriptV, shadow_trick=True):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w = image_stack.shape[:2]
    # create arrays for 
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])
    # Handle RGB images (NOT COMPATIBLE WITH THE SHADOW TRICK)
    if len(image_stack.shape) == 4:
        image_stack[image_stack!=image_stack] = 0
        albedo = []
        # Estimate normals with grayscale.
        gray_stack = np.dstack([
            cv2.cvtColor(image_stack[:,:,i,:].astype(np.float32), cv2.COLOR_RGB2GRAY)
            for i in range(image_stack.shape[2])
        ])
        _, normal = estimate_alb_nrm(gray_stack, scriptV,
                                          shadow_trick=False)
        # Approximate albedos per channel.
        VN = scriptV@(normal.reshape(-1,3).T)
        sumVN2 = np.sum(VN, axis=0).reshape(h,w)**2
        for i in range(3):
            pixels = image_stack[:,:,:,i]
            alb_channel = np.zeros([h,w])
            for y in range(h):
                for x in range(w):
                    alb_channel[y,x] = pixels[y,x]@VN[:,y*h+x].T / sumVN2[y,x]
            alb_channel[alb_channel!=alb_channel] = 0
            albedo.append(alb_channel)
        albedo = sum(albedo) / 3
    else:
        """
        ================
        Your code here
        ================
        for each point in the image array
            stack image values into a vector i
            construct the diagonal matrix scriptI
            solve scriptI * scriptV * g = scriptI * i to obtain g for this point
            albedo at this point is |g|
            normal at this point is g / |g|
        """
        for y in range(h):
            for x in range(w):
                i = image_stack[y, x, :].T
                if shadow_trick:
                    scriptI = np.diag(i)
                    A = scriptI @ scriptV
                    B = scriptI @ i
                else:
                    A = scriptV
                    B = i
                g = np.linalg.lstsq(A, B, rcond=-1)[0]
                norm = np.linalg.norm(g)
                albedo[y, x] = norm
                normal[y, x, :] = g / norm

    return albedo, normal

if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10,10,n])
    scriptV = np.zeros([n,3])
    estimate_alb_nrm( image_stack, scriptV, shadow_trick=True)