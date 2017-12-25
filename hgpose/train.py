import os
import sys
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import opts
sys.path.insert(0, '..')
from src.models.hg import HourglassNet
from src.datasets.mpii import MPII_Dataset
from src.utils.misc import getLogDir, makeCkptDir, getValue
from src.utils.evals import accuracy

# Parse arguments
FLAGS = opts.get_args()

epoch_init = FLAGS.epoch_init
iter_init = FLAGS.iter_init
global_step = FLAGS.step_init  # for summary writer (will start on 1)

# Prepare dataset
train_set = MPII_Dataset(
    FLAGS.dataDir, split='train',
    inp_res=FLAGS.inputRes, out_res=FLAGS.outputRes,
    scale_factor=FLAGS.scale, rot_factor=FLAGS.rotate, sigma=FLAGS.hmSigma)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=FLAGS.trainBatch, shuffle=True,
    num_workers=FLAGS.nThreads, pin_memory=True)
valid_set = MPII_Dataset(
    FLAGS.dataDir, split='valid',
    inp_res=FLAGS.inputRes, out_res=FLAGS.outputRes,
    scale_factor=FLAGS.scale, rot_factor=FLAGS.rotate, sigma=FLAGS.hmSigma, small_image=False)
valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=FLAGS.validBatch, shuffle=False,
    num_workers=FLAGS.nThreads, pin_memory=True)

netHg = HourglassNet(
    nStacks=FLAGS.nStacks, nModules=FLAGS.nModules, nFeat=FLAGS.nFeats,
    nClasses=train_set.nJoints)  # ref `nClasses` from dataset
criterion = nn.MSELoss()

optimHg = torch.optim.RMSprop(
    netHg.parameters(),
    lr=FLAGS.lr,
    alpha=FLAGS.alpha, eps=FLAGS.eps)

# TODO: restore model, optim, hyperparameter ...
# if FLAGS.netHg:
#     pass
# if FLAGS.netD:
#     pass

if FLAGS.cuda:
    torch.backends.cudnn.benchmark = True
    # make parallel
    netHg = nn.DataParallel(netHg)
    netHg.cuda()
    criterion.cuda()

# network arch summary
print('Total params of netHg: %.2fM' % (sum(p.numel() for p in netHg.parameters())/1000000.0))

log_dir = getLogDir(FLAGS.log_root)
sumWriter = SummaryWriter(log_dir)
ckpt_dir = makeCkptDir(log_dir)

def run(epoch, iter_start=0):
    netHg.train()

    global global_step
    pbar = tqdm.tqdm(train_loader, desc='Epoch %02d' % epoch, dynamic_ncols=True)
    pbar_info = tqdm.tqdm(bar_format='{bar}{postfix}')
    avg_acc = 0
    for it, sample in enumerate(pbar, start=iter_start):
        global_step += 1
        image, label, image_s = sample
        image = Variable(image)
        label = Variable(label)
        image_s = Variable(image_s)
        if FLAGS.cuda:
            image = image.cuda(async=True)  # TODO: check the affect of async
            label = label.cuda(async=True)
            image_s = image_s.cuda(async=True)

        # generator
        outputs = netHg(image)
        loss_hg_content = 0
        for out in outputs:  # TODO: speed up with multiprocessing map?
            loss_hg_content += criterion(out, label)

        loss_hg = loss_hg_content

        optimHg.zero_grad()
        loss_hg.backward()
        optimHg.step()

        accs = accuracy(outputs[-1].data.cpu(), label.data.cpu(), train_set.accIdxs)

        sumWriter.add_scalar('loss_hg', loss_hg, global_step)
        sumWriter.add_scalar('acc', accs[0], global_step)
        # TODO: learning rate scheduling
        # sumWriter.add_scalar('lr', lr, global_step)

        pbar_info.set_postfix({
            'loss_hg': getValue(loss_hg),
            'acc': accs[0]
        })
        pbar_info.update()
        avg_acc += accs[0] / len(train_loader)

    pbar_info.set_postfix_str('avg_acc: {}'.format(avg_acc))
    pbar.close()
    pbar_info.close()


def valid():
    netHg.eval()

    pbar = tqdm.tqdm(valid_loader, desc='Valid', dynamic_ncols=True)
    avg_acc = 0
    for it, sample in enumerate(pbar):
        image, label = sample
        image = Variable(image)
        label = Variable(label)
        if FLAGS.cuda:
            image = image.cuda(async=True)
            label = label.cuda(async=True)
        output = netHg(image)[-1]
        accs = accuracy(output.data.cpu(), label.data.cpu(), valid_set.accIdxs)
        avg_acc += accs[0] / len(valid_loader)
        if it == (len(valid_loader) - 1):
            pbar.set_postfix_str('avg_acc: {}'.format(avg_acc))


def save(epoch):
    torch.save({
        'netHg': netHg.state_dict()
    }, os.path.join(ckpt_dir, 'model_ep{}.pth'.format(epoch)))


if __name__ == '__main__':
    # Suppress skimage warning ('constant' change to 'reflect' in 0.15)
    import warnings
    warnings.filterwarnings("ignore")

    valid()
    for ep in range(epoch_init, FLAGS.maxEpochs):
        run(ep, iter_init)
        valid()
        save(ep)
        iter_init = 0  # reset after first epoch
