import torch
from torch.autograd import Function
from src.extensions.AE._ext import my_lib


class AElossFunction(Function):
    def forward(self, tags, keypoints):
        output = torch.zeros(torch.Size((tags.size()[0], 2)))
        mean_tags = torch.zeros(torch.Size((tags.size()[0], keypoints.size()[1], tags.size()[2]+1)))
        self.mean_tags = mean_tags

        my_lib.my_lib_loss_forward(tags, keypoints, output, mean_tags)
        self.save_for_backward(tags, keypoints)
        return output

    def backward(self, grad_output):
        tags, keypoints = self.saved_tensors
        grad_input = torch.zeros(tags.size()).cuda(tags.get_device())
        # grad_input = tags.new(tags.size()).zero_()
        my_lib.my_lib_loss_backward(tags, keypoints, self.mean_tags, grad_output, grad_input)
        self.mean_tags = None
        return grad_input, torch.zeros(keypoints.size())


def AEloss(input, input1):
    if not input.is_cuda:
        input = input.cuda()
    output = AElossFunction()(input, input1)
    return output

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
