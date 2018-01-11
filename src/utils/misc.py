import os
import socket
from datetime import datetime
from glob import glob
import torch

def getValue(x):
    '''Convert Torch tensor/variable to numpy array/python scalar
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    elif isinstance(x, torch.autograd.Variable):
        x = x.data.cpu().numpy()
    if x.size == 1:
        x = x.item()
    return x

def getLogDir(root_dir=None, comment=''):
    '''Get logging directory for summary writer
    '''
    if not root_dir:
        root_dir = 'runs'
    log_dir = os.path.join(
        root_dir,
        datetime.now().strftime('%b%d-%H-%M-%S') +
        '_' + socket.gethostname())
    if comment:
        log_dir += '_' + comment
    return log_dir

def makeCkptDir(log_dir):
    ckpt_dir = os.path.join(log_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    return ckpt_dir

def getLatestCkpt(log_dir, suffix='.pth'):
    ckpts = glob(os.path.join(log_dir, 'ckpts/*'+suffix))
    return sorted(ckpts, key=os.path.getmtime, reverse=True)[0]

