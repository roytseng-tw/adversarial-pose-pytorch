import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable

from group import HeatmapParser
sys.path.insert(1, '..')
import src.datasets.cocopose_umichvl as ref
from src.datasets.utils import get_transform
from src.models.hg_ae import HourglassAENet


parser = argparse.ArgumentParser()
parser.add_argument('ckpt', help='path of checkpoint to load')
parser.add_argument('--cpu', action='store_false', dest='cuda')
parser.add_argument('--coco_root', default='../data/coco')
parser.add_argument('--mode', default='single')
# Model options
parser.add_argument('--nStacks', type=int, default=4)
parser.add_argument('--inpDim', type=int, default=256)
parser.add_argument('--outDim', type=int, default=68)
F = parser.parse_args()


def resize(im, res):
    return np.array([cv2.resize(im[i], res) for i in range(im.shape[0])])

hm_parser = HeatmapParser(detection_val=0.1)

def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    """
    if len(tag.shape) == 3:
        tag = tag[:, :, :, None]

    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0:
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y])

    prev_tag = np.mean(tags, axis=0)
    ans = []

    for i in range(keypoints.shape[0]):
        tmp = det[i, :, :]
        tt = (((tag[i, :, :] - prev_tag[None, None, :])**2).sum(axis=2) ** 0.5)
        tmp2 = tmp - np.round(tt)

        x, y = np.unravel_index(np.argmax(tmp2), tmp.shape)
        xx = x
        yy = y
        val = tmp[x, y]
        x += 0.5
        y += 0.5

        if tmp[xx, min(yy+1, det.shape[1]-1)] > tmp[xx, max(yy-1, 0)]:
            y += 0.25
        else:
            y -= 0.25

        if tmp[min(xx+1, det.shape[0]-1), yy] > tmp[max(0, xx-1), yy]:
            x += 0.25
        else:
            x -= 0.25

        x, y = np.array([y, x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(17):
            if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1

    return keypoints

def multiperson(img, func, mode):
    """
    1. Resize the image to different scales and pass each scale through the network
    2. Merge the outputs across scales and find people by HeatmapParser
    3. Find the missing joints of the people with a second pass of the heatmaps
    """
    if mode == 'multi':
        scales = [2, 1., 0.5]
    else:
        scales = [1]

    height, width = img.shape[0:2]
    center = (width/2, height/2)
    dets, tags = None, []
    for idx, i in enumerate(scales):
        scale = max(height, width)/200
        # input_res = max(height, width)
        inp_res = int((i * 512 + 63)//64 * 64)
        res = (inp_res, inp_res)

        mat_ = get_transform(center, scale, 0, res)[:2]
        inp = cv2.warpAffine(img, mat_, res)/255

        def array2dict(tmp):
            return {
                'det': tmp[0][:, :, :17],
                'tag': tmp[0][:, -1, 17:34]
            }

        tmp1 = array2dict(func(inp))
        tmp2 = array2dict(func(inp[:, ::-1]))
        # print(tmp1['det'].shape, tmp1['tag'].shape)

        tmp = {}
        for ii in tmp1:
            tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]), axis=0)
        # print(tmp['det'].shape, tmp['tag'].shape)

        det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][ref.flip_ref]
        if det.max() > 10:
            continue
        if dets is None:
            dets = det
            mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2]
        else:
            dets = dets + resize(det, dets.shape[1:3])

        if abs(i-1)<0.5:
            res = dets.shape[1:3]
            tags += [resize(tmp['tag'][0], res), resize(tmp['tag'][1, :, :, ::-1][ref.flip_ref], res)]

    if dets is None or len(tags) == 0:
        print('QQ')
        return [], []

    tags = np.concatenate([i[:,:,:,None] for i in tags], axis=3)
    dets = dets/len(scales)/2

    dets = np.minimum(dets, 1)
    grouped = hm_parser.parse(np.float32([dets]), np.float32([tags]))[0]

    scores = [i[:, 2].mean() for i in grouped]

    for i in range(len(grouped)):
        grouped[i] = refine(dets, tags, grouped[i])

    if len(grouped) > 0:
        grouped[:,:,:2] = ref.kpts_affine(grouped[:,:,:2] * 4, mat)
    return grouped, scores

def coco_eval(prefix, dt, gt, valid_dict):
    """
    Evaluate the result with COCO API
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import copy
    import json

    for _, i in enumerate(sum(dt, [])):
        i['id'] = _+1

    image_ids = []
    gt = copy.deepcopy(gt)

    paths, anns, idxes, info = [valid_dict[i] for i in ['path', 'anns', 'idxes', 'info']]

    widths = {}
    heights = {}
    for idx, (a, b) in enumerate(zip(gt, dt)):
        if len(a) > 0:
            for i in b:
                i['image_id'] = a[0]['image_id']
            image_ids.append(a[0]['image_id'])
        if info[idx] is not None:
            widths[a[0]['image_id']] = info[idx]['width']
            heights[a[0]['image_id']] = info[idx]['height']
        else:
            widths[a[0]['image_id']] = 0
            heights[a[0]['image_id']] = 0
    image_ids = set(image_ids)

    cat = [{'supercategory': 'person', 'id': 1, 'name': 'person', 'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]], 'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']}]
    with open(prefix + '/gt.json', 'w') as f:
        json.dump({'annotations':sum(gt, []), 'images':[{'id':i, 'width': widths[i], 'height': heights[i]} for i in image_ids], 'categories':cat}, f)

    with open(prefix + '/dt.json', 'w') as f:
        json.dump(sum(dt, []), f)

    coco = COCO(prefix + '/gt.json')
    coco_dets = coco.loadRes(prefix + '/dt.json')
    coco_eval = COCOeval(coco, coco_dets, "keypoints")
    coco_eval.params.imgIds = list(image_ids)
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

def genDtByPred(pred, image_id=0):
    """
    Generate the json-style data for the output
    """
    ans = []
    for i in pred:
        val = pred[i] if type(pred) == dict else i
        if val[:, 2].max() > 0:
            tmp = {'image_id': int(image_id), "category_id": 1, "keypoints": [], "score": float(val[:, 2].mean())}
            p = val[val[:, 2] > 0][:, :2].mean(axis=0)
            for j in val:
                if j[2] > 0.:
                    tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                else:
                    tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
            ans.append(tmp)
    return ans

def build_valid():
    path, gts, info = [], [], []
    _, valid_ids = ref.setup_train_val_split(os.path.join(F.coco_root, 'valid_id'))
    for img_id in tqdm(valid_ids):
        ann_ids = ref.coco.getAnnIds(imgIds=img_id)
        ann = ref.coco.loadAnns(ann_ids)
        gts.append(ann)

        img_info = ref.coco.loadImgs(img_id)[0]
        _path = img_info['file_name'].split('_')[1] + '/' + img_info['file_name']
        path.append(os.path.join(F.coco_root, 'images', _path))
        assert os.path.exists(path[-1])
        info.append(img_info)

    valid_dict = {
        'path': path,
        'anns': gts,
        'idxes': valid_ids,
        'info': info
    }
    pickle.dump(valid_dict, open(os.path.join(F.coco_root, 'validation.pkl'), 'wb'))
    return valid_dict

def get_valid_img(valid_dict):
    paths, anns, idxes, info = [valid_dict[i] for i in ['path', 'anns', 'idxes', 'info']]

    for i, p in enumerate(tqdm(paths)):
        img = cv2.imread(p)[:, :, ::-1]
        yield anns[i], img

def main():
    ref.init_cocoapi(os.path.join(F.coco_root, 'annotations/person_keypoints_train2014.json'))
    valid_fpath = os.path.join(F.coco_root, 'validation.pkl')
    if not os.path.exists(valid_fpath):
        valid_dict = build_valid()
    else:
        valid_dict = pickle.load(open(valid_fpath, 'rb'))

    netHg = HourglassAENet(nStacks=F.nStacks, inp_nf=F.inpDim, out_nf=F.outDim)
    netHg = nn.DataParallel(netHg)
    ckpt = torch.load(F.ckpt, map_location=lambda storage, loc: storage)
    netHg.load_state_dict(ckpt['netHg'])
    epoch = ckpt['epoch']
    if F.cuda:
        torch.backends.cudnn.benchmark = True
        netHg.cuda()
    netHg.eval()

    def runner(imgs):
        imgs = np.transpose(imgs[np.newaxis, ...], (0, 3, 1, 2))
        imgs = Variable(torch.Tensor(imgs.astype(np.float32)), volatile=True)
        if F.cuda:
            imgs = imgs.cuda()
        out = netHg(imgs)
        preds = [out.data.cpu().numpy()]
        return preds

    gts = []
    preds = []
    for anns, img in get_valid_img(valid_dict):
        gts.append(anns)
        ans, scores = multiperson(img, runner, F.mode)
        if len(ans) > 0:
            ans = ans[:, :, :3]
        pred = genDtByPred(ans)
        for i, score in zip(pred, scores):
            i['score'] = float(score)
        preds.append(pred)

    prefix = F.ckpt.split('ckpts')[0]
    coco_eval(prefix, preds, gts, valid_dict)


if __name__ == '__main__':
    main()
