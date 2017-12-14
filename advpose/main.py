import os
import torch
from tensorboardX import SummaryWriter
# import opts
from datasets.lsp import LSPMPIIData

# FLAGS = opts.get_args()

dataset = LSPMPIIData('/home/roytseng/study/pose/adversarial-pose-pytorch/data', 'train')
dataset[0]
