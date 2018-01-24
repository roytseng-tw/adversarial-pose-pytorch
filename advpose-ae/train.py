import os
import sys
import tqdm
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

sys.path.insert(1, '..')
import opts
from src.models.hg_ae import HourglassAENet
from src.models.dis import HourglassDisNet
from src.datasets.cocopose_umichvl import COCOPose_Dataset, num_parts
from src.utils.misc import getLogDir, makeCkptDir, getValue, getLatestCkpt
from losses import calc_loss


# Parse arguments
FLAGS = opts.get_args()
epoch_init = FLAGS.epoch_init
iter_init = FLAGS.iter_init
global_step = FLAGS.step_init  # for summary writer (will start on 1)

# Prepare dataset
train_set = COCOPose_Dataset(
    FLAGS.dataDir, split='train',
    inp_res=FLAGS.inputRes, out_res=FLAGS.outputRes,
    scale_factor=FLAGS.scale, rot_factor=FLAGS.rotate,
    max_num_people=FLAGS.maxNumPeople, debug=FLAGS.debug)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=FLAGS.trainBatch, shuffle=True,
    num_workers=FLAGS.nThreads, pin_memory=FLAGS.pinMem)

netHg = HourglassAENet(nStacks=FLAGS.nStacks, inp_nf=FLAGS.inpDim, out_nf=FLAGS.outDim)
netD = HourglassDisNet(
    nStacks=FLAGS.nStacks, nModules=1, nFeat=128,
    nClasses=num_parts, inplanes=3+num_parts)
netHg = nn.DataParallel(netHg)
netD = nn.DataParallel(netD)

def criterion_D(pred, gt):
    l = (pred - gt)**2
    l = l.mean()
    return l

kt = FLAGS.kt_init

# network arch summary
print('Total params of network: %.2fM' % (sum(p.numel() for p in netHg.parameters()) / 1e6))

if FLAGS.cuda:
    torch.backends.cudnn.benchmark = True
    netHg.cuda()
    netD.cuda()

optimHg = torch.optim.Adam(netHg.parameters(), lr=FLAGS.lr)
optimD = torch.optim.Adam(netD.parameters(), lr=FLAGS.lrD)

if FLAGS.continue_exp:
    log_dir = FLAGS.continue_exp
    ckpt = torch.load(getLatestCkpt(FLAGS.continue_exp))
    netHg.load_state_dict(ckpt['netHg'])
    netD.load_state_dict(ckpt['netD'])
    optimHg.load_state_dict(ckpt['optimHg'])
    optimD.load_state_dict(ckpt['optimD'])
    epoch_init = ckpt['epoch'] + 1
    global_step = ckpt['global_step']
else:
    comment = 'lambda_G{}-gamma{}-kt_lr{}'.format(
        FLAGS.lambda_G, FLAGS.gamma, FLAGS.kt_lr)
    if FLAGS.comment:
        comment += '_' + FLAGS.comment
    log_dir = getLogDir(FLAGS.log_root, comment=comment)
sumWriter = SummaryWriter(log_dir)
ckpt_dir = makeCkptDir(log_dir)


def train(epoch, iter_start=0):
    global global_step, kt

    netHg.train()
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

        image_s = nn.functional.avg_pool2d(image, 4)
        inp_real = torch.cat([image_s, heatmaps], dim=1)
        d_real = netD(inp_real)
        loss_d_real = criterion_D(d_real, heatmaps)

        pred_heatmaps = outputs[:, -1, :17].squeeze(dim=1)  # Notice: manually assign the dimension to avoid unexcepted freeze
        inp_fake = torch.cat([image_s, pred_heatmaps], dim=1)
        d_fake = netD(inp_fake)
        loss_d_fake = criterion_D(d_fake, pred_heatmaps)

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

        loss_d = loss_d_real - kt * loss_d_fake
        loss_hg = loss_hg + FLAGS.lambda_G * loss_d_fake

        optimD.zero_grad()
        loss_d.backward(retain_graph=True)
        optimD.step()
        optimHg.zero_grad()
        loss_hg.backward()
        optimHg.step()

        # update kt
        loss_d_real_ = getValue(loss_d_real)
        loss_d_fake_ = getValue(loss_d_fake)
        balance = FLAGS.gamma * loss_d_real_ - loss_d_fake_
        kt = kt + FLAGS.kt_lr * balance
        kt = min(1, max(0, kt))
        measure = loss_d_real_ + abs(balance)

        # Summary
        sumWriter.add_scalar('loss_hg', loss_hg, global_step)
        for key, value in sum_dict.items():
            sumWriter.add_scalar(key, loss_temp, global_step)
        sumWriter.add_scalar('loss_d', loss_d, global_step)
        toprint += ', loss_d: {:.8f}'.format(getValue(loss_d))
        sumWriter.add_scalar('loss_d_real', loss_d_real, global_step)
        sumWriter.add_scalar('loss_d_fake', loss_d_fake, global_step)
        sumWriter.add_scalar('measure', measure, global_step)
        sumWriter.add_scalar('kt', kt, global_step)

        pbar_info.set_postfix_str(toprint)
        pbar_info.update()

        del outputs, push_loss, pull_loss, detection_loss, loss_hg, \
            d_real, d_fake, loss_d_real, loss_d_fake, loss_d

    pbar.close()
    pbar_info.close()


def save(epoch):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'netHg': netHg.state_dict(),
        'optimHg': optimHg.state_dict(),
        'netD': netD.state_dict(),
        'optimD': optimD.state_dict(),
        'kt': kt,
    }, os.path.join(ckpt_dir, 'model_ep{}.pth'.format(epoch)))


if __name__ == '__main__':
    cv2.setNumThreads(0)  # disable multithreading in OpenCV for main thread to avoid problems after fork
    for ep in range(epoch_init, FLAGS.maxEpochs):
        train(ep, iter_init)
        # valid()
        save(ep)
        iter_init = 0  # reset after first epoch
