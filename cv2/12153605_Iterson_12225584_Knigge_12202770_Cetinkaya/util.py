import numpy as np
from numpy import linalg as LA
import torch
from torch.autograd import Variable
import h5py
from collections import defaultdict
import pickle as pk
from PIL import Image

from supplemental_code.supplemental_code import *



################# LOAD FACE MODEL AS GLOBAL VARIABLES ##########################
bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')

# Load idendity data
mean_id = np.asarray(bfm['shape/model/mean'], dtype=np.float32)
mean_id = mean_id.reshape(-1, 3)

pcaBasis_id = np.asarray(bfm['shape/model/pcaBasis'],
                            dtype=np.float32)
pcaStd_id = np.sqrt(np.asarray(bfm['shape/model/pcaVariance'],
                    dtype=np.float32))

# Load expression data
mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32)
mean_exp = mean_exp.reshape(-1, 3)

pcaBasis_exp = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
pcaStd_exp = np.sqrt(np.asarray(bfm['expression/model/pcaVariance'],
                        dtype=np.float32))

# Load color and geometry
cells = np.asarray(bfm['shape/representer/cells'], dtype=np.int32)
color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
color = color.reshape(-1, 3)

###############################################################################


def trans_mat(angles, t, device=torch.device('cpu')):
    """Most part taken from:
    https://stackoverflow.com/questions/59387182/construct-a-rotation-matrix-in-pytorch
    to be able to take gradients w.r.t. the angles.
    """

    yaw, pitch, roll = angles

    tensor_0 = torch.zeros(1, device=device)
    tensor_1 = torch.ones(1, device=device)
    # Rotation around X-axis
    RX = torch.stack([
                torch.stack([tensor_1, tensor_0, tensor_0]),
                torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])
            ]).reshape(3,3)
    # Rotation around Y-axis
    RY = torch.stack([
                torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                torch.stack([tensor_0, tensor_1, tensor_0]),
                torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])
            ]).reshape(3,3)
    # Rotation around Z-axis
    RZ = torch.stack([
                torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                torch.stack([tensor_0, tensor_0, tensor_1])
            ]).reshape(3,3)

    R = RZ @ (RY @ RX)

    T = torch.vstack([
        torch.hstack([R, t]),
        torch.tensor([[0, 0, 0, 1]])
    ])
    return T


def proj_mat(aspect_ratio, FOV, n, f):
    t = np.tan(FOV / 2) * n
    b = -t
    r = t * aspect_ratio
    l = b = -t*aspect_ratio
    return torch.tensor([[(2*n)/(r - l), 0, (r + l)/(r - l), 0],
                         [0, (2*n)/(t - b), (t + b)/(t - b), 0],
                         [0, 0, -(f + n)/(f - n), -(2*f*n)/(f - n)],
                         [0, 0, -1, 0]], dtype=torch.float32)


def vwpt_mat(vl, vr, vt, vb):
    return torch.tensor([[(vr - vl)/2, 0, 0, (vr + vl)/2],
                         [0, (vt - vb)/2, 0, (vt + vb)/2],
                         [0, 0, 0.5, 0.5],
                         [0, 0, 0, 1]], dtype=torch.float32)


def c2h(X):
    ones = torch.ones((X.shape[0], 1))
    return torch.hstack([X, ones])


def h2c(X):
    return X[:, :-1] / X[:, [-1]]


def energy_loss(P, L, a, d, la=2, ld=2):
    L_lan = torch.mean(torch.sum((P - L)**2, axis=1))
    L_reg = la*sum(a**2) + ld*sum(d**2)
    return L_lan + L_reg


def get_face(a, d, mean_id, pcaBasis_id, pcaStd_id, mean_exp, pcaBasis_exp,
             pcaStd_exp):
    m_id = a.shape[0]
    m_exp = d.shape[0]

    PCA_id = (pcaBasis_id[:, :m_id] @ (a * pcaStd_id[:m_id])).reshape(-1, 3)
    PCA_exp = (pcaBasis_exp[:, :m_exp] @ (d * pcaStd_exp[:m_exp])).reshape(-1, 3)
    G = mean_id + PCA_id + mean_exp + PCA_exp

    return G


def estimate_latents(detection, annotations, VP, model, device, epochs=10000, 
                     logfreq=1000, save=None, a=None):
    # Initialize parameters.
    d = Variable(torch.empty(20, device=device).uniform_(-1, 1), requires_grad=True)
    o = Variable(torch.tensor([[0.0, 0.0, 0.0]], device=device).T, requires_grad=True)
    t = Variable(torch.tensor([[0.0, 0.0, -500.0]], device=device).T, requires_grad=True)
    if a is None:
        a = Variable(torch.empty(30, device=device).uniform_(-1, 1), requires_grad=True)
        optimizer = torch.optim.Adam([a, d, o, t], lr=0.1)
    else:
        optimizer = torch.optim.Adam([d, o, t])
    for key in model:
        model[key] = model[key].to(device)
    VP = VP.to(device)
    print("ESTIMATING PARAMETERS")
    i = 0
    while True:
        # a_clamped = a.clamp(-1, 1)
        # d_clamped = d.clamp(-1, 1)
        G = get_face(a, d, model['mean_id'], model['pcaBasis_id'],
                           model['pcaStd_id'], model['mean_exp'],
                           model['pcaBasis_exp'], model['pcaStd_exp'])

        T = trans_mat(o, t, device=device)
        GT = c2h(G) @ T.T
        GT = h2c(GT @ VP.T)
        landmarks = h2c(GT)[annotations]
        loss = energy_loss(landmarks, detection, a, d, la=0.03, ld=0.02)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not i % logfreq:
            print(i, loss.item())
            print("RANGE A: %f, %f" %(torch.min(a), torch.max(a)))
            print("RANGE D: %f, %f" %(torch.min(d), torch.max(d)))
            print()
        if 100 < loss.item() < 200:
            print(i, "DONE")
            print(loss.item())
            break
        elif i + 1 == epochs:
            print(i, f"TERMINATED WITH {loss.item()}")
            break
        i += 1
    if save is not None:
        torch.save([a.detach(), d.detach(), o.detach(), t.detach()], save)
    return a, d, o, t


def bilinear_interpolation(points, source):
    """
    Perform Bilinear interpolation for points.
    """
    X = points[:,1]
    Y = points[:,0]
    # Get closest pixel coordinates
    x1 = np.floor(X).astype(np.int32)
    x2 = np.ceil(X).astype(np.int32)
    y1 = np.floor(Y).astype(np.int32)
    y2 = np.ceil(Y).astype(np.int32)
    # Get closest pixel values
    F11 = source[x1, y1]
    F21 = source[x2, y1]
    F12 = source[x1, y2]
    F22 = source[x2, y2]

    # Interpolate color for each point in points.
    color = []
    for i, (x, y) in enumerate(zip(X, Y)):
        FR = np.array([[F11[i, 0], F12[i, 0]],
                       [F21[i, 0], F22[i, 0]]])
        FG = np.array([[F11[i, 1], F12[i, 1]],
                       [F21[i, 1], F22[i, 1]]])
        FB = np.array([[F11[i, 2], F12[i, 2]],
                       [F21[i, 2], F22[i, 2]]])

        x_ = np.array([x2[i] - x, x - x1[i]])
        y_ = np.array([y2[i] - y, y - y1[i]]).T
        vals = np.array([x_ @ FR @ y_,
                         x_ @ FG @ y_,
                         x_ @ FB @ y_])
        color.append(vals)
    return np.clip(np.vstack(color), 0, 255).astype(np.uint8)


def multi_frame_estimation(frames, annotations, model, device):
    im = frames[0]
    FOV = 0.5
    n = .1
    f = 1
    aspect_ratio = im.shape[1] / im.shape[0]

    P = proj_mat(aspect_ratio, FOV, n, f)
    V = vwpt_mat(0, im.shape[1], 0, im.shape[0])

    VP = V @ P
    # Compute values for first frame to get a
    detection = torch.tensor(detect_landmark(im))
    res = defaultdict()
    a, d1, o1, t1 = estimate_latents(detection, annotations, VP, model, device)
    res['a'] = a
    res[0] = {'d': d1, 'o': o1, 't': t1}
    # Estimate parameters for the rest of the frames, fixing a
    for i, frame in enumerate(frames[1:], 1):
        detection = torch.tensor(detect_landmark(frame))
        _, d, o, t = estimate_latents(detection, annotations, VP, model, device,
                                      a=a.detach())
        res[i] = {'d': d, 'o': o, 't': t}

    return res


def generate_and_transform(parameters, VP, model, returnface=False):
    a, d, o, t = parameters['a'], parameters['d'], parameters['o'], parameters['t']
    G = get_face(a, d, model['mean_id'], model['pcaBasis_id'],
                       model['pcaStd_id'], model['mean_exp'],
                       model['pcaBasis_exp'], model['pcaStd_exp'])
    T = trans_mat(o, t)
    GT = h2c(h2c(c2h(G) @ T.T @ VP.T))
    if returnface:
        return GT, G
    else:
        return GT


def swap_faces(source, target, Ps, Pt, model, VP_s, VP_t, savedir=None):
    # Generate face models and transform to image plane.
    Gs, G1 = generate_and_transform(Ps, VP_s, model, returnface=True)
    Gt, G2 = generate_and_transform(Pt, VP_t, model, returnface=True)
    # Interpolate pixel values.
    Cs = bilinear_interpolation(Gs.numpy(), source)
    Ct = bilinear_interpolation(Gt.numpy(), target)
    # Save face objects.
    if savedir is not None:
        save_obj(f'{savedir}/source.obj', G1, Cs, cells.T)
        save_obj(f'{savedir}/target.obj', G2, Ct, cells.T)

    swap = render(c2h(Gt).numpy(), Cs, cells.T,
                  H=target.shape[0], W=target.shape[1]).astype(np.uint8)
    pk.dump(swap, open(f'{savedir}/swap_raw.pk', 'wb'))

    swapped = np.where(swap == 0, target, swap)
    im = Image.fromarray(swapped)
    im.save(f'{savedir}/final.jpg')

    return swapped




