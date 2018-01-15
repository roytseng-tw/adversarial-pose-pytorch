import argparse
import sys
import torch

parser = argparse.ArgumentParser()

# General options
parser.add_argument('--cpu', action='store_false', dest='cuda')
parser.add_argument('--nThreads', type=int, default=4)
parser.add_argument('--log_root', default='runs')

# Model options
parser.add_argument('--nStacks', type=int, default=4)
parser.add_argument('--nFeats', type=int, default=256)
parser.add_argument('--nModules', type=int, default=2)
parser.add_argument('--modelCkpt', help='Path to the checkpoint file to load')

# Hyperparameters
parser.add_argument('--lr', type=float, default=2.5e-4)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--lrD', type=float, default=8e-5)

parser.add_argument('--lambda_G', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.5, help='Balance weight of real and fake')
parser.add_argument('--lambda_kt', type=float, default=0.001, help='Weight to clip k_t')
parser.add_argument('--kt_init', type=float, default=0.0)

# Training
parser.add_argument('--maxEpoch', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=4)
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


if __name__ == '__main__':
    get_args()
