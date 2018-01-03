import os
import numpy as np
import skimage as skim
import skimage.io as skio
import skimage.color as skcolor
import skimage.transform as sktf
import torch
import torch.utils.data
from pycocotools.coco import COCO
from pycocotools import mask

from .utils import rand, get_transform


num_parts = 17
part_ref = {'ankle': [15, 16], 'knee': [13, 14], 'hip': [11, 12],
            'wrist': [9, 10], 'elbow': [7, 8], 'shoulder': [5, 6],
            'face': [0, 1, 2], 'ears': [3, 4]}
part_labels = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
               'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
               'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r']
pair_ref = [
    [1, 2], [2, 3], [1, 3],
    [6, 8], [8, 10], [12, 14], [14, 16],
    [7, 9], [9, 11], [13, 15], [15, 17],
    [6, 7], [12, 13], [6, 12], [7, 13]
]
pair_ref = np.array(pair_ref) - 1
flip_ref = [i-1 for i in [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]]
part_label2idx = {label: idx for idx, label in enumerate(part_labels)}

coco, imgIds_train, imgIds_valid = None, None, None

def init_cocoapi(ann_path):
    global coco
    coco = COCO(ann_path)

def setup_train_val_split(validIds_path):
    validIds = set(np.loadtxt(validIds_path, dtype=int))
    imgIds = coco.getImgIds()
    tmpIds = []
    for img_id in imgIds:
        if num_people(img_id) > 0:  # or > 1 for mult-person ?
            tmpIds.append(img_id)
    valid_list = []
    train_list = []
    for img_id in tmpIds:
        if img_id in validIds:
            valid_list.append(img_id)
        else:
            train_list.append(img_id)
    return train_list, valid_list

def get_mask(img_id):
    ''' Get mask for crowd of people '''
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs(img_id)[0]
    m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
    for ann in anns:
        if ann['iscrowd']:
            rle = mask.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
            m += mask.decode(rle)  # mask: {0, 1} unit8
    return m > 0.5

def get_anns(img_id):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    return [ann for ann in anns if ann['num_keypoints'] > 0]

def num_people(img_id):
    anns = get_anns(img_id)
    return len(anns)

def get_keypoints(img_id=None, anns=None):
    assert (img_id is not None) or (anns is not None)
    if anns is None:
        anns = get_anns(img_id)
    n_people = len(anns)
    kps = np.zeros((n_people, num_parts, 3))
    for i in range(n_people):
        kps[i] = np.array(anns[i]['keypoints']).reshape(-1, 3)
    return kps

def loadImage(imgId, data_root, split):
        img_info = coco.loadImgs(imgId)[0]
        imgname = img_info['file_name']
        if '_' in imgname:
            imgname = imgname.split('_')[-1]
        return skim.img_as_float(skio.imread(os.path.join(data_root,
                                 'images/{}2017/{}'.format(split, imgname))))

def kpts_affine(kpts, mat):
    '''Transfrom keypoints
    mat: a 2 by three 3 matrix. top-two rows of the tranformation matrix
    '''
    kpts = np.array(kpts)
    shape = kpts.shape
    kpts_ = np.ones((kpts.shape[0] * kpts.shape[1], 3))
    kpts_[:, :2] = kpts.reshape(-1, 2)
    return (kpts_ @ mat.T).reshape(shape)


class GenerateHeatmaps():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        self.sigma = self.output_res / 64
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros((self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        for points in keypoints:
            for idx, pt in enumerate(points):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue
                    ul = int(x - 3 * self.sigma - 1), int(y - 3 * self.sigma - 1)
                    br = int(x + 3 * self.sigma + 2), int(y + 3 * self.sigma + 2)

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.gaussian[a:b, c:d])
        return hms


class FliterKeypoints():
    def __init__(self, max_num_people, num_parts):
        self.max_num_people = max_num_people
        self.num_parts = num_parts

    def __call__(self, keypoints, output_res):
        visible_nodes = np.zeros((self.max_num_people, self.num_parts, 2))
        for i in range(len(keypoints)):
            tot = 0
            for idx, pt in enumerate(keypoints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 and x < output_res and y < output_res:
                    visible_nodes[i][tot] = (idx * output_res * output_res + y * output_res + x, 1)
                    tot += 1
        return visible_nodes


class COCOPose_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, inp_res, out_res,
                 sigma, scale_factor=0.25, rot_factor=30, max_num_people=30):
        self.data_root = data_root
        self.inp_res = inp_res
        self.out_res = out_res
        self.split = split
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor

        self.generateHeatmaps = GenerateHeatmaps(out_res, num_parts)
        self.fliterKeypoints = FliterKeypoints(max_num_people, num_parts)

        global coco, imgIds_train, imgIds_valid
        if coco is None:
            init_cocoapi(os.path.join(data_root, 'annotations/person_keypoints_train2014.json'))
            imgIds_train, imgIds_valid = setup_train_val_split(os.path.join(data_root, 'valid_id'))

        if split == 'train':
            self.img_ids = imgIds_train
        elif split == 'valid':
            self.img_ids = imgIds_valid
        else:
            raise ValueError

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        im = loadImage(img_id, self.data_root, self.split)
        loss_mask = get_mask(img_id)

        anns = get_anns(img_id)
        keypoints = get_keypoints(anns=anns)

        height, width = im.shape[0:2]
        center = np.array((width/2, height/2))
        scale = max(height, width) / 200

        rot = (rand() * 2 - 1) * self.rot_factor
        scale *= (rand() * 2 - 1) * self.scale_factor + 1

        dx = np.random.randint(-40 * scale, 40 * scale) / center[0]
        dy = np.random.randint(-40 * scale, 40 * scale) / center[1]
        center[0] += dx * center[0]
        center[1] += dy * center[1]

        tform = get_transform(center, scale, rot, (self.out_res, self.out_res), invert=True)
        tform_inv = np.linalg.inv(tform)
        loss_mask = sktf.warp(loss_mask, tform_inv, output_shape=(self.out_res, self.out_res))
        loss_mask = (loss_mask > 0.5).astype(np.float32)
        keypoints[:, :, :2] = kpts_affine(keypoints[:, :, :2], tform[:2])

        tform_inv = get_transform(center, scale, rot, (self.inp_res, self.inp_res), invert=True)
        im = sktf.warp(im, tform_inv, output_shape=(self.inp_res, self.inp_res))

        # Flip LR
        if rand() < 0.5:
            im = im[:, ::-1]
            loss_mask = loss_mask[:, ::-1]
            keypoints = keypoints[:, flip_ref]
            keypoints[:, :, 0] = self.out_res - keypoints[:, :, 0]

        # TODO
        heatmaps = self.generateHeatmaps(keypoints)
        keypoints = self.fliterKeypoints(keypoints, self.out_res)
        im = self.preprocess(im).transpose(2, 0, 1)

        return (im.astype(np.float32),
                loss_mask.astype(np.float32),
                keypoints.astype(np.int32), heatmaps.astype(np.float32))

    def preprocess(self, data):
        # random hue and saturation
        data = skcolor.rgb2hsv(data)
        delta = (rand() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:, :, 0] + (delta * 360 + 360.), 360.)

        delta_sature = rand() + 0.5
        data[:, :, 1] *= delta_sature
        data[:, :, 1] = np.maximum(np.minimum(data[:, :, 1], 1), 0)
        data = skcolor.hsv2rgb(data)

        # adjust brightness
        delta = (rand() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (rand() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)

        return data
