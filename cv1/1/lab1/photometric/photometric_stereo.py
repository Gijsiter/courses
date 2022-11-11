import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='./SphereGray5/', shadow_trick=False, color=False):

    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    if color:
        # Get images per color channel and stack in 4th dimension.
        image_stack = None
        for i in range(3):
            [images, scriptV] = load_syn_images(image_dir, channel=i)
            if image_stack is None:
                h, w, n = images.shape
                image_stack = np.zeros((h, w, n, 3))
            image_stack[:,:,:,i] = images
        # BGR to RGB
        image_stack = image_stack[:,:,:,::-1]
    else:
        [image_stack, scriptV] = load_syn_images(image_dir)
        [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV,
                                         shadow_trick=shadow_trick)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    threshold = np.mean(SE)
    print(f"Largest SE: {np.amax(SE)}")
    print(f"MSE: {np.mean(SE)}")
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface(p, q, path_type='average')

    # show results
    show_results(albedo, normals, height_map, SE)

## Face
def photometric_stereo_face(image_dir='./photometrics_images/yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV, shadow_trick=False)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)
    print(f"Largest SE: {np.amax(SE)}")
    print(f"MSE: {np.mean(SE)}")
    threshold = 0.001
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q, path_type='average')

    # show results
    show_results(albedo, normals, height_map, SE)

if __name__ == '__main__':
    # photometric_stereo('./photometrics_images/MonkeyColor/', color=True)
    photometric_stereo_face()
