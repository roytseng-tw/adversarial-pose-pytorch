import torch
from torch.autograd import Function
from torch import nn
from ._ext import my_lib


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


def AEloss(tags, keypoints):
    if not tags.is_cuda:
        tags = tags.cuda()
    output = AElossFunction()(tags, keypoints)
    return output
