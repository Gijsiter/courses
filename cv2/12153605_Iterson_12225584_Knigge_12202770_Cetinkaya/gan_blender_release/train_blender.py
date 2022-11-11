import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import imageio

import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils

torch.cuda.empty_cache()

# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'Blender'
data_root = 'data_set/data_set/data/'
train_list = 'data_set/data_set/train.str'
test_list = 'data_set/data_set/test.str'
batch_size = 8
nthreads = 4
max_epochs = 1
displayIter = 20
saveIter = 1
img_resolution = 256

lr_gen = 1e-4
lr_dis = 1e-4

momentum = 0.9
weightDecay = 1e-4
step_size = 30
gamma = 0.1

pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001
######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size
check_point_loc = 'Exp_%s/checkpoints/' % experiment_name.replace(' ', '_')
visuals_loc = 'Exp_%s/visuals/' % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
checkpoint_pattern = check_point_loc + 'checkpoint_%s_%d.pth'
logTrain = check_point_loc + 'LogTrain.txt'

torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    # if torch.cuda.is_available():
        # device = torch.device('cuda')
        # print('[*] GPU Device selected as default execution device.')
    # else:
    device = torch.device('cpu')
    print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
          'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

done = u'\u2713'
print('[I] STATUS: Initiate Network and transfer to device...', end='')
# Define your generators and Discriminators here
gen = res_unet.MultiScaleResUNet(in_nc=7).to(device)
dis = discriminators_pix2pix.MultiscaleDiscriminator().to(device)
print(done)

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
    # use the .to(device) function, where device is automatically detected
    # above.
d_path = os.path.join('data_set/tuned_model',"checkpoint_D.pth")
g_path = os.path.join('data_set/tuned_model',"checkpoint_G.pth")

if os.path.isfile(g_path):
    G, _, _ = utils.loadModels(gen, g_path, device=device)

if os.path.isfile(d_path):
    D, _, _ = utils.loadModels(dis, d_path, device=device)
print(done)


print('[I] STATUS: Initiate optimizer...', end='')
print(torch.__version__)
# Define your optimizers and the schedulers and connect the networks from
# before
optimizer_G = torch.optim.SGD(G.parameters(), lr=lr_gen, weight_decay=weightDecay,
                              momentum=momentum)
optimizer_D = torch.optim.SGD(D.parameters(), lr=lr_dis, weight_decay=weightDecay,
                              momentum=momentum)

G_sched = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size, gamma=gamma)
D_sched = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size, gamma=gamma)
print(done)


print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
L_id = vgg_loss.VGGLoss().to(device)
L_pixel = nn.L1Loss()
L_gan = gan_loss.GANLoss().to(device)
print(done)

print('[I] STATUS: Initiate Dataloaders...')
# Initialize your datasets here
trainSet = SwappedDatasetLoader(train_list, data_root, resize=img_resolution)
testSet = SwappedDatasetLoader(test_list, data_root, resize=img_resolution)

trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True,
                         num_workers=nthreads, pin_memory=True, drop_last=True)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True,
                        num_workers=nthreads, pin_memory=True, drop_last=True)

batches_train = len(trainLoader)
batches_test = len(testLoader)
print(done)


print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
print(done)


def transfer_mask(img1, img2, mask):
    return img1 * mask + img2 * (1 - mask)


def blend_imgs_bgr(source_img, target_img, mask):
    # Implement poisson blending here. You can us the built-in seamlessclone
    # function in opencv which is an implementation of Poisson Blending.
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return target_img
    H, W = target_img.shape[:2]
    center = (W // 2, H // 2)
    output = cv2.seamlessClone(source_img, target_img, mask*255, center,
                               cv2.NORMAL_CLONE)
    return output


def blend_imgs(source_tensor, target_tensor, mask_tensor):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        mask = np.round(mask * 255).astype('uint8')
        out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)

# def laplacian_blend(source,target,mask):



def Train(G, D, epoch_count, iter_count):
    G.train(True)
    D.train(True)
    epoch_count += 1
    pbar = tqdm(enumerate(trainLoader), total=batches_train, leave=False)

    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1
        images, _ = data

        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        # 2) Feed the data to the networks.
        # 4) Calculate the losses.
        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.

        src = images['source'].to(device)
        tgt = images['target'].to(device)
        swp = images['swap'].to(device)
        msk = images['mask'].to(device)

        m = torch.where(swp == -1, 0, 1)[:,[0]]
        t_hat = transfer_mask(swp, tgt, m)
        X = torch.cat([t_hat, tgt, m], dim=1)

        I_fake = G(X)
        gt = blend_imgs(src, tgt, msk)

        # alpha =0.5
        # beta =0.5
        # gt = cv2.addWeighted(src, alpha, tgt, beta, 0.01)

        Dx = D(tgt)
        DGx = D(I_fake)

        LD = gan_weight*L_gan(Dx, True) + gan_weight*L_gan(DGx, False)
        LG = gan_weight*L_gan(DGx, True) \
             + rec_weight*L_id(I_fake, gt) + pix_weight*L_pixel(I_fake, gt)

        # Optimize generator
        optimizer_G.zero_grad()
        LG.backward()
        optimizer_G.step()

        # Optimize discriminator
        optimizer_D.zero_grad()
        LD.backward()
        optimizer_G.step()

        if iter_count % displayIter == 0:
            # Write to the log file.
            trainLogger.write(
                "G loss: %f :: D loss %f" %(LG, LD)
            )
        # Print out the losses here. Tqdm uses that to automatically print it
        # in front of the progress bar.
        pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator

    t_source, t_swap, t_target, t_pred, t_blend = Test(G)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    # return np.nanmean(total_loss_pix),\
    #     np.nanmean(total_loss_id), np.nanmean(total_loss_attr),\
    #     np.nanmean(total_loss_rec), np.nanmean(total_loss_G_Gan),\
    #     np.nanmean(total_loss_D_Gan), iter_count
    return iter_count

def Test(G):
    with torch.no_grad():
        G.eval()
        t = enumerate(testLoader)
        i, (data) = next(t)

        # print(data)
        images,_ = data
        # print(images)
        src = images['source'].to(device)
        tgt = images['target'].to(device)
        swp = images['swap'].to(device)
        msk = images['mask'].to(device)


        m = torch.where(swp == -1, 0, 1)[:,[0]]
        t_hat = transfer_mask(swp, tgt, m)
        X = torch.cat([t_hat, tgt, m], dim=1)

        pred = G(X)
        # gt = blend_imgs(src, tgt, msk)

        alpha =0.5
        beta =0.5

        gt = np.zeros(src.shape)

        for i in range(8):
            a = img_utils.tensor2bgr(src[i])
            b = img_utils.tensor2bgr(tgt[i])
            c = cv2.addWeighted(a, alpha, b, beta, 0.01)

            print(c.shape)
            print(np.transpose(c,1,2,0).shape)
            gt[i] = np.transpose(c,1,2,0)

        gt = img_utils.bgr2tensor(gt)
        img_transfer_input = None
        # Feed the network with images from test set

        # Blend images
        # pred = G(img_transfer_input)
        # You want to return 4 components:
            # 1) The source face.
            # 2) The 3D reconsturction.
            # 3) The target face.
            # 4) The prediction from the generator.
            # 5) The GT Blend that the network is targettting.
        return src, swp, tgt, pred, gt


if __name__ == '__main__':

    iter_count = 0
    # Print out the experiment configurations. You can also save these to a file if
    # you want them to be persistent.
    print('[*] Beginning Training:')
    print('\tMax Epoch: ', max_epochs)
    print('\tLogging iter: ', displayIter)
    print('\tSaving frequency (per epoch): ', saveIter)
    print('\tModels Dumped at: ', check_point_loc)
    print('\tVisuals Dumped at: ', visuals_loc)
    print('\tExperiment Name: ', experiment_name)

    # for i in range(max_epochs):
    #     # Call the Train function here
    #     # Step through the schedulers if using them.
    #     # You can also print out the losses of the network here to keep track of
    #     # epoch wise loss.
    #     Train(G, D, 0, iter_count)
    #     G_sched.step()
    #     D_sched.step()
    # trainLogger.close()
    src, swp, tgt, pred, gt = Test(G)

    for i in range(8):


        a = img_utils.tensor2bgr(src[i])
        b = img_utils.tensor2bgr(swp[i])
        c = img_utils.tensor2bgr(tgt[i])
        d = img_utils.tensor2bgr(pred[i])
        e = img_utils.tensor2bgr(gt[i])

        pp = np.hstack((a,b,c,d,e))
        # print(src)
        if i==0:
            total = pp
        else:
            total = np.vstack((total,pp))

    cv2.imshow('plot',total)
    cv2.imwrite(str(i)+'test2.png',total)
    cv2.waitKey(500)

    # src = img_utils.tensor2rgb(src[0])
    # # print(src)
    # cv2.imshow('plot',src)
    # cv2.imwrite('test.png',src)
    # cv2.waitKey(500)
