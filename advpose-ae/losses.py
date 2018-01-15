import torch
import sys
sys.path.insert(1, '..')
from src.extensions.AE.AE_loss import AEloss


def HeatmapLoss(pred, gt, masks):
    """
    loss for detection heatmap
    mask is used to mask off the crowds in coco dataset
    """
    assert pred.size() == gt.size()
    l = ((pred - gt)**2) * masks[:, None, :, :].expand_as(pred)
    l = l.mean(dim=3).mean(dim=2).mean(dim=1)
    return l

def calc_loss(preds, keypoints=None, heatmaps=None, masks=None):
    dets = preds[:, :, :17]
    tags = preds[:, :, 17:34]
    nStack = preds.size(1)

    keypoints = keypoints.cpu().long()
    batchsize = tags.size()[0]

    tag_loss = []
    for i in range(nStack):
        tag = tags[:, i].contiguous().view(batchsize, -1, 1)
        tag_loss.append(AEloss(tag, keypoints))
    tag_loss = torch.stack(tag_loss, dim=1).cuda(tags.get_device())
    detection_loss = []
    for i in range(nStack):
        detection_loss.append(HeatmapLoss(dets[:, i], heatmaps, masks))
    detection_loss = torch.stack(detection_loss, dim=1)
    return tag_loss[:, :, 0], tag_loss[:, :, 1], detection_loss
