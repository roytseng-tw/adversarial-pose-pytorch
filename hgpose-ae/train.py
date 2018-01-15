import os
import sys
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

sys.path.insert(1, '..')
import opts
from src.models.hg_ae import HourglassAENet
from src.datasets.cocopose_umichvl import COCOPose_Dataset
from src.utils.misc import getLogDir, makeCkptDir, getValue, getLatestCkpt
from loss import calc_loss

# # original author's code
# from coco_pose import dp
# from models.posenet import PoseNet

# Parse arguments
FLAGS = opts.get_args()

epoch_init = FLAGS.epoch_init
iter_init = FLAGS.iter_init
global_step = FLAGS.step_init  # for summary writer (will start on 1)

# Prepare dataset
train_set = COCOPose_Dataset(
    FLAGS.dataDir, split='train',
    inp_res=FLAGS.inputRes, out_res=FLAGS.outputRes,
    scale_factor=FLAGS.scale, rot_factor=FLAGS.rotate, max_num_people=FLAGS.maxNumPeople, debug=FLAGS.debug)
# train_set = dp.init()
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=FLAGS.trainBatch, shuffle=True,
    num_workers=FLAGS.nThreads, pin_memory=FLAGS.pinMem)

# valid_set = COCOPose_Dataset(
#     FLAGS.dataDir, split='valid',
#     inp_res=FLAGS.inputRes, out_res=FLAGS.outputRes,
#     scale_factor=FLAGS.scale, rot_factor=FLAGS.rotate, max_num_people=FLAGS.maxNumPeople)
# valid_loader = torch.utils.data.DataLoader(
#     valid_set, batch_size=FLAGS.validBatch, shuffle=False,
#     num_workers=FLAGS.nThreads, pin_memory=True)

netHg = HourglassAENet(nStacks=FLAGS.nStacks, inp_nf=FLAGS.inpDim, out_nf=FLAGS.outDim)
# netHg = PoseNet(FLAGS.nStacks, FLAGS.inpDim, FLAGS.outDim)
netHg = nn.DataParallel(netHg)

# network arch summary
print('Total params of network: %.2fM' % (sum(p.numel() for p in netHg.parameters()) / 1e6))

if FLAGS.cuda:
    torch.backends.cudnn.benchmark = True
    netHg.cuda()

optimHg = torch.optim.Adam(netHg.parameters(), lr=FLAGS.lr)

if FLAGS.continue_exp:
    log_dir = FLAGS.continue_exp
    ckpt = torch.load(getLatestCkpt(FLAGS.continue_exp))
    netHg.load_state_dict(ckpt['netHg'])
    optimHg.load_state_dict(ckpt['optimHg'])
    epoch_init = ckpt['epoch'] + 1
    global_step = ckpt['global_step']
else:
    log_dir = getLogDir(FLAGS.log_root, comment=FLAGS.comment)
sumWriter = SummaryWriter(log_dir)
ckpt_dir = makeCkptDir(log_dir)


def train(epoch, iter_start=0):
    netHg.train()

    global global_step
    pbar = tqdm.tqdm(train_loader, desc='Epoch %02d' % epoch, dynamic_ncols=True)
    pbar_info = tqdm.tqdm(bar_format='{bar}{postfix}')
    for it, sample in enumerate(pbar, start=iter_start):
        global_step += 1
        if FLAGS.debug:
            image, masks, keypoints, heatmaps, img_ids = sample
        else:
            image, masks, keypoints, heatmaps = sample
        image = Variable(image)
        masks = Variable(masks)
        keypoints = Variable(keypoints)
        heatmaps = Variable(heatmaps)
        if FLAGS.cuda:
            image = image.cuda(async=FLAGS.pinMem)
            masks = masks.cuda(async=FLAGS.pinMem)
            keypoints = keypoints.cuda(async=FLAGS.pinMem)
            heatmaps = heatmaps.cuda(async=FLAGS.pinMem)

        outputs = netHg(image)
        push_loss, pull_loss, detection_loss = calc_loss(outputs, keypoints, heatmaps, masks)

        loss_hg = 0
        toprint = ''
        sum_dict = {}
        for loss, weight, name in zip([push_loss, pull_loss, detection_loss], [1e-3, 1e-3, 1],
                                      ['push_loss', 'pull_loss', 'detection_loss']):
            loss_temp = torch.mean(loss)
            sum_dict[name] = getValue(loss_temp)
            loss_temp *= weight
            loss_hg += loss_temp
            toprint += '{:.8f} '.format(getValue(loss_temp))

        optimHg.zero_grad()
        loss_hg.backward()
        optimHg.step()

        # Summary
        sumWriter.add_scalar('loss_hg', loss_hg, global_step)
        for key, value in sum_dict.items():
            sumWriter.add_scalar(key, loss_temp, global_step)

        pbar_info.set_postfix_str(toprint)
        pbar_info.update()

    pbar.close()
    pbar_info.close()

def save(epoch):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'netHg': netHg.state_dict(),
        'optimHg': optimHg.state_dict(),
    }, os.path.join(ckpt_dir, 'model_ep{}.pth'.format(epoch)))


if __name__ == '__main__':
    # Suppress skimage warning ('constant' change to 'reflect' in 0.15)
    # import warnings
    # warnings.filterwarnings("ignore")

    for ep in range(epoch_init, FLAGS.maxEpochs):
        train(ep, iter_init)
        # valid()
        save(ep)
        iter_init = 0  # reset after first epoch
