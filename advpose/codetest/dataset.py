import sys
import numpy as np
import skimage.transform as sktf
sys.path.insert(0, '..')
from datasets.lsp import LSPMPIIData, rand, randn
from datasets import utils


class Dataset(LSPMPIIData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        index = self.idxRef[self.split][index]
        im = self._loadImage(index)
        pts, c, s = self._getPartInfo(index)
        r = 0
        # print('Original c: {}, s: {}, r: {}'.format(c, s, r))
        return im, c, s, r, self.annot['imgname'][index]

        if self.split == 'train':
            # scale and rotation
            s = s * np.clip(randn() * self.scale_factor + 1, 1 - self.scale_factor, 1 + self.scale_factor)
            r = np.clip(randn() * self.rot_factor, -self.rot_factor, self.rot_factor) if rand() <= 0.6 else 0
            # Flip LR
            if rand() < 0.5:
                im = im[:, ::-1, :]
                pts = utils.fliplr_coords(pts, width=im.shape[1], matchedParts=self.flipRef)
                c[0] = im.shape[1] - c[0]
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


def test_all():
    sf = 0.25
    rf = 30
    dataset = Dataset('/home/roytseng/study/pose/adversarial-pose-pytorch/data', 'train')
    def read_data(idx):
        im, c, s, r, imgname = dataset[idx]
        try:
            s_list = s * np.arange(1 - sf, 1 + sf, 1)
            r_list = np.arange(-rf, rf, 1)
            for s in s_list:
                for r in r_list:
                    utils.crop(im, c, s, r, dataset.inp_res)
        except Exception:
            return imgname, s, r
    import multiprocessing as mp
    import tqdm

    idxs = np.arange(21500, len(dataset))
    try:
        pool = mp.Pool(processes=8)
        ret = []
        for i, x in enumerate(tqdm.tqdm(
                pool.imap_unordered(read_data, idxs, chunksize=1),  # FIXME: getting slower over time
                ncols=80, total=len(idxs), desc='%d proc' % pool._processes, leave=True)):
            ret.append(x)
    except KeyboardInterrupt:
        pass
    finally:
        ret = np.array(ret)
        print('result:', ret[ret != None])


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # test_all()
    dataset = LSPMPIIData('/home/roytseng/study/pose/adversarial-pose-pytorch/data', 'train')
    for im, label, im_s in dataset:
        print('*********', im.shape)
