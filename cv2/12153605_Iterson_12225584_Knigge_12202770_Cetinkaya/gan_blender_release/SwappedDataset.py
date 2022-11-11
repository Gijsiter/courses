from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import img_utils
from img_utils import *
import cv2
import matplotlib.pyplot as plt
# Helper function to quickly see the values of a list or dictionary of data
def printTensorList(data, detailed=False):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
            if detailed:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
        print(']')


class SwappedDatasetLoader(Dataset):

    def __init__(self, data_file, prefix, resize=256):
        self.prefix = prefix
        self.resize = resize
        # Define your initializations and the transforms here. You can also
        # define your tensor transforms to normalize and resize your tensors.
        # As a rule of thumb, put anything that remains constant here.
        with open(data_file, 'r') as f:
            data_paths = f.read().splitlines()

        self.data_paths = data_paths[:100]
        self.len = len(self.data_paths)
        self.resizer = transforms.Resize(size=(resize, resize))

    def __len__(self):
        # Return the length of the datastructure that is your dataset
        return self.len

    def __getitem__(self, index):
        # Write your data loading logic here. It is much more efficient to
        # return your data modalities as a dictionary rather than list. So you
        # can return something like the follows:
        #     image_dict = {'source': source,
        #                   'target': target,
        #                   'swap': swap,
        #                   'mask': mask}

        #     return image_dict, self.data_paths[index]
        path = self.data_paths[index].split("_")

        src = path[0]+"_fg_"+path[3]
        tgt = path[0]+"_bg_"+path[2]+".png"
        msk = path[0]+"_mask_"+path[2]+"_"+path[3]
        swp = self.data_paths[index]

        # print('test',rgb2tensor(self.resizer(Image.open(self.prefix + src))).shape)
        source = rgb2tensor(self.resizer(Image.open(self.prefix + src))).squeeze()
        # print('test2',source.shape)
        target = rgb2tensor(self.resizer(Image.open(self.prefix + tgt))).squeeze()
        swap = rgb2tensor(self.resizer(Image.open(self.prefix + swp))).squeeze()
        mask = rgb2tensor(self.resizer(Image.open(self.prefix + msk)),
                          normalize=False).squeeze(0)

        image_dict = {'source': source,
                      'target': target,
                      'swap': swap,
                      'mask': mask}

        return image_dict, self.data_paths[index]


# def blend_imgs_bgr(source_img, target_img, mask):
#     # Implement poisson blending here. You can us the built-in seamlessclone
#     # function in opencv which is an implementation of Poisson Blending.
#     a = np.where(mask != 0)
#     if len(a[0]) == 0 or len(a[1]) == 0:
#         return target_img
#     if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
#         return target_img
#     H, W = target_img.shape[:2]

#     # ret,thresh = cv2.threshold(mask,127,255,0)
#     # contours,_ = cv2.findContours(thresh[:,:,0], 1, 2)
#     # cnt = contours[0]
#     # M = cv2.moments(cnt)
#     # cx = int(M['m10']/M['m00'])
#     # cy = int(M['m01']/M['m00'])
#     # print(cx, cy)

#     center = (W // 2, H // 2)
#     output = cv2.seamlessClone(source_img, target_img, mask, center,
#                                cv2.NORMAL_CLONE)
#     return output


# def blend_imgs(source_tensor, target_tensor, mask_tensor):
#     out_tensors = []
#     for b in range(source_tensor.shape[0]):
#         source_img = img_utils.tensor2bgr(source_tensor[b])
#         target_img = img_utils.tensor2bgr(target_tensor[b])
#         mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
#         mask = np.round(mask * 255).astype('uint8')
#         out_bgr = blend_imgs_bgr(source_img, target_img, mask)
#         out_tensors.append(img_utils.bgr2tensor(out_bgr))

#     return torch.cat(out_tensors, dim=0)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    # It is always a good practice to have separate debug section for your
    # functions. Test if your dataloader is working here. This template creates
    # an instance of your dataloader and loads 20 instances from the dataset.
    # Fill in the missing part. This section is only run when the current file
    # is run and ignored when this file is imported.

    # This points to the root of the dataset
    data_root = 'data_set/data_set/data/'
    # This points to a file that contains the list of the filenames to be
    # loaded.
    test_list = 'data_set/data_set/test.str'
    print('[+] Init dataloader')
    # Fill in your dataset initializations
    testSet = SwappedDatasetLoader(test_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    ims, _ = next(iter(loader))

    src = ims['source']
    tgt = ims['target']
    swp = ims['swap']
    msk = ims['mask']
    m = torch.where(swp == -1, 0, 1).squeeze(0)

    plt.subplot(131); plt.imshow(tensor2rgb(ims['source']))
    plt.subplot(132); plt.imshow(tensor2rgb(ims['target']))
    bruh = blend_imgs(swp, ims['target'], m)
    plt.subplot(133); plt.imshow(tensor2rgb(bruh[0])); plt.show()
    # enu = enumerate(loader)
    # for i in range(20):
    #     a = time.time()
    #     i, (images) = next(enu)
    #     b = time.time()
    #     # Uncomment to use a prettily printed version of a dict returned by the
    #     # dataloader.
    #     # printTensorList(images[0], True)
    #     print('[*] Time taken: ', b - a)
