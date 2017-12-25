import argparse
import sys
import torch

parser = argparse.ArgumentParser()

# General options
parser.add_argument('--cpu', action='store_false', dest='cuda')
parser.add_argument('--nThreads', type=int, default=4)
parser.add_argument('--log_root', default='runs')

# Model options
parser.add_argument('--nStacks', type=int, default=8)
parser.add_argument('--nFeats', type=int, default=256)
parser.add_argument('--nModules', type=int, default=1)

# Hyperparameters
parser.add_argument('--lr', type=float, default=2.5e-4)
parser.add_argument('--alpha', type=float, default=0.99)  # for rmsprop
parser.add_argument('--eps', type=float, default=1e-8)  # for rmsprop

# Training
parser.add_argument('--maxEpochs', type=int, default=100)
parser.add_argument('--trainBatch', type=int, default=6)
parser.add_argument('--validBatch', type=int, default=6)

parser.add_argument('--epoch_init', type=int, default=0)
parser.add_argument('--iter_init', type=int, default=0)
parser.add_argument('--step_init', type=int, default=0)

# Data options
parser.add_argument('--dataDir', default='../data')
parser.add_argument('--inputRes', type=int, default=256)
parser.add_argument('--outputRes', type=int, default=64)
parser.add_argument('--scale', type=float, default=0.25)
parser.add_argument('--rotate', type=float, default=30)
parser.add_argument('--hmSigma', type=float, default=1, help='Heatmap gaussian size')


def get_args():
    opts = parser.parse_args()
    if opts.cuda and not torch.cuda.is_available():
        sys.exit('Cuda is not available.')
    return opts
