import argparse
import sys
import torch

parser = argparse.ArgumentParser()

# General options
parser.add_argument('--cpu', action='store_false', dest='cuda')
parser.add_argument('--log_root', default='runs')
parser.add_argument('--comment', default='', help='text append to default log directory name')

# Dataset
parser.add_argument('--debug', action='store_true', help='output debug info')
parser.add_argument('--dataDir', default='../data/coco')
parser.add_argument('--inputRes', type=int, default=512)
parser.add_argument('--outputRes', type=int, default=128)
parser.add_argument('--scale', type=float, default=0.25)
parser.add_argument('--rotate', type=float, default=30)
parser.add_argument('--maxNumPeople', type=int, default=30)

# Model options
parser.add_argument('--nStacks', type=int, default=4)
parser.add_argument('--inpDim', type=int, default=256)
parser.add_argument('--outDim', type=int, default=68)

# Dataloader
parser.add_argument('--pinMem', action='store_true', help='dataloader pin memory')
parser.add_argument('--nThreads', type=int, default=6)

# Training
parser.add_argument('--trainBatch', type=int, default=6)
parser.add_argument('--validBatch', type=int, default=6)
parser.add_argument('--maxEpochs', type=int, default=150)

parser.add_argument('--continue_exp', '--c')

parser.add_argument('--epoch_init', type=int, default=0)
parser.add_argument('--iter_init', type=int, default=0)
parser.add_argument('--step_init', type=int, default=0)

parser.add_argument('--lr', type=float, default=2e-4)

parser.add_argument('--lrD', type=float, default=8e-5)
parser.add_argument('--lambda_G', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.5, help='Balance weight of real and fake')
parser.add_argument('--kt_lr', type=float, default=0.001)
parser.add_argument('--kt_init', type=float, default=0.0)


def get_args():
    opts = parser.parse_args()
    if opts.cuda and not torch.cuda.is_available():
        sys.exit('Cuda is not available.')
    return opts
