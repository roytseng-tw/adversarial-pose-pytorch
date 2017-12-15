import os
import random
import numpy as np
import h5py
import skimage as skim
import skimage.io as skio
import skimage.transform as sktf
import torch
import torch.utils.data
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
from datasets import utils

def randn():
    return random.gauss(0, 1)
def rand():
    return random.random()


class LSPMPIIData(torch.utils.data.Dataset):
    def __init__(self, data_root, split,
                 inp_res=256, out_res=64, sigma=1, label_type='Gaussian',
                 scale_factor=0.25, rot_factor=30):
        self.data_root = data_root
        self.split = split
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.label_type = label_type
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor

        self.nJoints = 16
        self.nClasses = self.nJoints
        self.accIdxs = [1, 2, 3, 4, 5, 6, 11, 12, 15, 16]  # joint idxs for accuracy calculation
        self.flipRef = [[0, 5],   [1, 4],   [2, 3],   # noqa
                        [10, 15], [11, 14], [12, 13]]
        # Pairs of joints for drawing skeleton
        self.skeletonRef = [[1, 2, 1],   [2, 3, 1],   [4, 5, 2],  # noqa
                            [5, 6, 2],   [9, 10, 0],  [13, 9, 3], # noqa
                            [11, 12, 3], [12, 13, 3], [14, 9, 4],
                            [14, 15, 4], [15, 16, 4]]
        f = h5py.File(os.path.join(data_root, 'lsp_mpii.h5'), 'r')
        tags = ['imgname', 'part', 'center', 'scale', 'visible', 'istrain']
        self.annot = {}
        for tag in tags:
            if tag == 'imgname':
                self.annot[tag] = list(map(
                    lambda x: bytes(x).decode('utf-8').split('\x00', 1)[0],
                    f[tag].value.astype('uint8')))
            else:
                self.annot[tag] = f[tag].value
        f.close()

        allIdxs = np.arange(len(self.annot['istrain']))
        self.idxRef = {
            'train': allIdxs[self.annot['istrain'] == 1],
            'val': allIdxs[self.annot['istrain'] == 2],
            'test': allIdxs[self.annot['istrain'] == 0]
        }

    def _getPartInfo(self, index):
        # get a COPY
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index].copy()

        # Small adjustment so cropping is less likely to take feet out
        dataset = self.annot['imgname'][index].split('/', 1)[0]
        if dataset == 'mpii':
            c[1] = c[1] + 15 * s
        s = s * 1.25
        return pts, c, s

    def _loadImage(self, index):
        impath = os.path.join(self.data_root, self.annot['imgname'][index])
        im = skim.img_as_float(skio.imread(impath))
        return im

    def __getitem__(self, index):
        index = self.idxRef[self.split][index]
        im = self._loadImage(index)
        pts, c, s = self._getPartInfo(index)
        r = 0
        if self.split == 'train':
            # scale and rotation
            s = s * np.clip(randn() * self.scale_factor + 1, 1 - self.scale_factor, 1 + self.scale_factor)
            r = np.clip(randn() * self.rot_factor, -self.rot_factor, self.rot_factor) if rand() <= 0.6 else 0
            # Flip LR
            if rand() < 0.5:
                im = im[:, ::-1, :]
                pts = utils.fliplr_coords(pts, width=im.shape[1], matchedParts=self.flipRef)
                c[0] = im.shape[1] - c[0]  # flip center point also
            # Color jitter
            im = np.clip(im * np.random.uniform(0.6, 1.4, size=3), 0, 1)
        # Prepare image
        im = utils.crop(im, c, s, r, self.inp_res)

        if im.ndim == 2:
            im = np.tile(im, [1, 1, 3])

        # small size image
        im_s = sktf.resize(im, [self.out_res, self.out_res], preserve_range=True)

        # (h, w, c) to (c, h, w)
        im = np.transpose(im, [2, 0, 1])
        im_s = np.transpose(im_s, [2, 0, 1])

        # Prepare label
        labels = np.zeros((self.nJoints, self.out_res, self.out_res))
        new_pts = utils.transform(pts.T, c, s, r, self.out_res).T
        for i in range(self.nJoints):
            if pts[i, 0] > 0:
                labels[i] = utils.create_label(
                    labels.shape[1:],
                    new_pts[i],
                    self.sigma)

        return im.astype(np.float32), labels.astype(np.float32), im_s.astype(np.float32)

    def __len__(self):
        return len(self.idxRef[self.split])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = LSPMPIIData('/home/roytseng/study/pose/adversarial-pose-pytorch/data', 'train')
    im, label, im_s = dataset[39605]
    plt.imshow(np.transpose(im, [1, 2, 0]))
    plt.show()

    # im_vis = utils.show_sample(im, label)
    # print(im_vis.shape)
    # plt.imshow(im_vis)
    # plt.show()
