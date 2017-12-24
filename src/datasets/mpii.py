import os
import numpy as np
import h5py
import skimage as skim
import skimage.io as skio
import skimage.transform as sktf
import torch
import torch.utils.data

from .utils import rand, rnd, crop, fliplr_coords, transform, create_label


class MPII_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split,
                 inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, return_meta=False, small_image=True):
        self.data_root = data_root
        self.split = split
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.return_meta = return_meta
        self.small_image = small_image

        self.nJoints = 16
        self.accIdxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]  # joint idxs for accuracy calculation
        self.flipRef = [[0, 5],   [1, 4],   [2, 3],   # noqa
                        [10, 15], [11, 14], [12, 13]]

        self.annot = {}
        tags = ['imgname', 'part', 'center', 'scale']
        f = h5py.File('{}/mpii/{}.h5'.format(data_root, split), 'r')
        for tag in tags:
            self.annot[tag] = np.asarray(f[tag]).copy()
        f.close()

    def _getPartInfo(self, index):
        # get a COPY
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index].copy()

        # Small adjustment so cropping is less likely to take feet out
        c[1] = c[1] + 15 * s
        s = s * 1.25
        return pts, c, s

    def _loadImage(self, index):
        impath = os.path.join(self.data_root, 'mpii/images', self.annot['imgname'][index].decode('utf-8'))
        im = skim.img_as_float(skio.imread(impath))
        return im

    def __getitem__(self, index):
        im = self._loadImage(index)
        pts, c, s = self._getPartInfo(index)
        r = 0
        if self.split == 'train':
            # scale and rotation
            s = s * (2 ** rnd(self.scale_factor))
            r = 0 if rand() < 0.6 else rnd(self.rot_factor)
            # flip LR
            if rand() < 0.5:
                im = im[:, ::-1, :]
                pts = fliplr_coords(pts, width=im.shape[1], matchedParts=self.flipRef)
                c[0] = im.shape[1] - c[0]  # flip center point also
            # Color jitter
            im = np.clip(im * np.random.uniform(0.6, 1.4, size=3), 0, 1)
        # Prepare image
        im = crop(im, c, s, r, self.inp_res)
        if im.ndim == 2:
            im = np.tile(im, [1, 1, 3])
        if self.small_image:
            # small size image
            im_s = sktf.resize(im, [self.out_res, self.out_res], preserve_range=True)

        # (h, w, c) to (c, h, w)
        im = np.transpose(im, [2, 0, 1])
        if self.small_image:
            im_s = np.transpose(im_s, [2, 0, 1])

        # Prepare label
        labels = np.zeros((self.nJoints, self.out_res, self.out_res))
        new_pts = transform(pts.T, c, s, r, self.out_res).T
        for i in range(self.nJoints):
            if pts[i, 0] > 0:
                labels[i] = create_label(
                    labels.shape[1:],
                    new_pts[i],
                    self.sigma)

        ret_list = [im.astype(np.float32), labels.astype(np.float32)]
        if self.small_image:
            ret_list.append(im_s)
        if self.return_meta:
            meta = [pts, c, s, r]
            ret_list.append(meta)
        return tuple(ret_list)

    def __len__(self):
        return len(self.annot['imgname'])
