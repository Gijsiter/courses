import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from supplemental_code.supplemental_code import *
from util import *
import pickle as pk


#****************#
# EXERCISE 4.2.1 #
#****************#
m_id = 30
m_exp = 20

a = np.random.uniform(-1, 1, 30)
d = np.random.uniform(-1, 1, 20)

G = torch.tensor(get_face(a, d, mean_id, pcaBasis_id, pcaStd_id, mean_exp,
                          pcaBasis_exp, pcaStd_exp), dtype=torch.float32)

# UNCOMMENT TO SAVE FACE AS OBJECT

# save_obj("objects/test.obj", G, color, cells.T)


#****************#
# EXERCISE 4.2.2 #
#****************#

theta = np.pi / 18
R10 = torch.tensor([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.float32)
R_10 = torch.tensor([[np.cos(-theta), 0, np.sin(-theta)],
                     [0, 1, 0],
                     [-np.sin(-theta), 0, np.cos(-theta)]], dtype=torch.float32)

G10 = G @ R10.T
G_10 = G @ R_10.T

# UNCOMMENT TO SAVE ROTATED FACES

# save_obj("objects/test_r10.obj", G10, color, cells.T)
# save_obj("objects/test_r-10.obj", G_10, color, cells.T)

o = torch.tensor([0, theta, 0]).reshape(3,1)
t = torch.tensor([0, 0, -500]).reshape(3, 1)
T = trans_mat(o, t)

file = "supplemental_code/Landmarks68_model2017-1_face12_nomouth.anl"
annotations = np.fromfile(file, sep='\n', dtype=int)

aspect_ratio = 4/3
FOV = 0.5
n = .1
f = 1

P = proj_mat(aspect_ratio, FOV, n, f)
V = vwpt_mat(0, 480, 0, -640)

VP = V @ P
GT = c2h(G) @ T.T
GT = h2c(GT @ VP.T)

GTuv = h2c(GT)[annotations]

# UNCOMMENT FOR PLOT OF LANDMARKS

# plt.scatter(GTuv[:,0], GTuv[:,1])
# plt.savefig("landmarks.png", bbox_inches='tight')
# plt.show()


#****************#
# EXERCISE 4.2.3 #
#****************#

im = plt.imread("images/gijs.jpg")
detection = torch.tensor(detect_landmark(im))

aspect_ratio = im.shape[1] / im.shape[0]

P = proj_mat(aspect_ratio, FOV, n, f)
V = vwpt_mat(0, im.shape[1], 0, im.shape[0])

VP = V @ P

# plt.scatter(detection[:,0], detection[:,1])

# plt.imshow(im)
# plt.show()

model = {
    'mean_id' : torch.tensor(mean_id),
    'pcaBasis_id' : torch.tensor(pcaBasis_id),
    'pcaStd_id' : torch.tensor(pcaStd_id),
    'mean_exp' : torch.tensor(mean_exp),
    'pcaBasis_exp' : torch.tensor(pcaBasis_exp),
    'pcaStd_exp' : torch.tensor(pcaStd_exp)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# UNCOMMENT TO ESTIMATE LATENT PARAMETERS

# a, d, o, t = estimate_latents(detection, annotations, VP, model, device,
#                               save='gijsparams.pth', epochs=4000)

a, d, o, t = torch.load('gijsparams.pth')
a = a.detach()
d = d.detach()
o = o.detach()
t = t.detach()

G = get_face(a, d, model['mean_id'], model['pcaBasis_id'],
                   model['pcaStd_id'], model['mean_exp'],
                   model['pcaBasis_exp'], model['pcaStd_exp'])



T = trans_mat(o, t)
GT = c2h(G) @ T.T
GT = h2c(GT @ VP.T)
GT = c2h(h2c(GT)).numpy()

# UNCOMMENT FOR PLOT OF FACE AND LANDMARK LOCATIONS AFTER ESTIMATION

# plt.imshow(im)
# landmarks = h2c(GT)[annotations]
# plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='x', label="Estimated")
# plt.scatter(detection[:, 0], detection[:, 1], c='g', marker='o', label="GT")
# plt.legend()
# plt.show()


#****************#
# EXERCISE 4.2.5 #
#****************#

images = [plt.imread(f'multiframe/{i}.jpg') for i in range(5)]

# UNCOMMENT TO ESTIMATE PARAMETERS FOR FRAMES IN ARRAY ABOVE

# res = multi_frame_estimation(images, annotations, model, device)
# pk.dump(res, open('mf.pk', 'wb'))

# res = pk.load(open('mf.pk', 'rb'))

FOV = 0.5
n = .1
f = 1
im = images[0]
aspect_ratio = im.shape[1] / im.shape[0]

P = proj_mat(aspect_ratio, FOV, n, f)
V = vwpt_mat(0, im.shape[1], 0, im.shape[0])

VP = V @ P

# CODE BELOW PLOTS THE RESULTS

# color = bilinear_interpolation(GT, images[i])
# a = res['a'].detach().numpy()
# for i in range(5):
#     d, o, t = res[i]['d'].detach().numpy(), res[i]['o'].detach(), res[i]['t'].detach()
#     G = torch.tensor(get_face(a, d, mean_id, pcaBasis_id, pcaStd_id, mean_exp,
#                           pcaBasis_exp, pcaStd_exp), dtype=torch.float32)
#     T = trans_mat(o, t)
#     G = c2h(G) @ T.T
#     GT = h2c(G @ VP.T)
#     GT = c2h(h2c(GT)).numpy()
#     landmarks = GT[annotations]
#     plt.subplot(1, 5, i+1)
#     plt.imshow(images[i])
#     plt.scatter(landmarks[:,0], landmarks[:,1])
#     save_obj(f"objects/frame{i}.obj", h2c(G), color, cells.T)
# plt.show()


#****************#
# EXERCISE 4.2.5 #
#****************#

im = plt.imread("images/gijs.jpg")
detection = torch.tensor(detect_landmark(im))

aspect_ratio = im.shape[1] / im.shape[0]

P = proj_mat(aspect_ratio, FOV, n, f)
V = vwpt_mat(0, im.shape[1], 0, im.shape[0])
VP = V @ P

# a, d, o, t = estimate_latents(detection, annotations, VP, model, device,
#                               save='images/gijsparams.pth', epochs=3001)

a, d, o, t = torch.load('images/gijsparams.pth')
a = a.detach()
d = d.detach()
o = o.detach()
t = t.detach()

im2 = plt.imread("images/random.jpg")
detection2 = torch.tensor(detect_landmark(im2))

aspect_ratio = im2.shape[1] / im2.shape[0]
P = proj_mat(aspect_ratio, FOV, n, f)
V = vwpt_mat(0, im2.shape[1], 0, im2.shape[0])
VP2 = V @ P

# a2, d2, o2, t2 = estimate_latents(detection2, annotations, VP2, model, device,
#                               save='images/randomparams.pth', epochs=3001)

a2, d2, o2, t2 = torch.load('images/randomparams.pth')
a2 = a2.detach()
d2 = d2.detach()
o2 = o2.detach()
t2 = t2.detach()

Ps = {'a': a,
      'd': d,
      'o': o,
      't': t}

Pt = {'a': a2,
      'd': d2,
      'o': o2,
      't': t2}

# UNCOMMENT TO SWAP FACES

# # SOURCE --> TARGET
# swapped = swap_faces(im, im2, Ps, Pt, model, VP, VP2, savedir='images/st')
# plt.imshow(swapped); plt.show()
# # TARGET --> SOURCE
# swapped = swap_faces(im2, im, Pt, Ps, model, VP2, VP, savedir='images/ts')
# plt.imshow(swapped); plt.show()
