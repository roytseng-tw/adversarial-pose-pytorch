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
from src.models.dis import HourglassDisNet
from src.datasets.lsp_mpii import LSPMPII_Dataset
from src.utils.misc import getValue, getLogDir, makeCkptDir
from src.utils.evals import accuracy

# Parse arguments
FLAGS = opts.get_args()

epoch_init = FLAGS.epoch_init
iter_init = FLAGS.iter_init
global_step = FLAGS.step_init  # for summary writer (will start on 1)

# Prepare dataset
dataset = LSPMPII_Dataset(
    FLAGS.dataDir, split='train',
    inp_res=FLAGS.inputRes, out_res=FLAGS.outputRes,
    scale_factor=FLAGS.scale, rot_factor=FLAGS.rotate, sigma=FLAGS.hmSigma)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=FLAGS.batchSize, shuffle=True,
    num_workers=FLAGS.nThreads, pin_memory=True)

print('Number of training samples: %d' % len(dataset))
print('Number of training batches per epoch: %d' % len(dataloader))

netHg = HourglassNet(
    nStack=FLAGS.nStack, nModules=FLAGS.nModules, nFeat=FLAGS.nFeats,
    nClasses=dataset.nClasses)  # ref `nClasses` from dataset
netD = HourglassDisNet(
    nStack=FLAGS.nStack, nModules=FLAGS.nModules, nFeat=FLAGS.nFeats,
    nClasses=dataset.nClasses, inplanes=3+dataset.nClasses)
# make parallel (identity op if cpu)
netHg = nn.DataParallel(netHg)
netD = nn.DataParallel(netD)

criterion = nn.MSELoss()
criterion_D = nn.MSELoss()

kt = FLAGS.kt_init

optimHg = torch.optim.RMSprop(netHg.parameters(), lr=FLAGS.lr, alpha=FLAGS.alpha)
optimD = torch.optim.Adam(netD.parameters(), lr=FLAGS.lrD, betas=(0.9, 0.999))

# TODO: restoring: model, optim, hyperparameter ...
if FLAGS.modelCkpt:
    ckpt = torch.load(FLAGS.modelCkpt, map_location=lambda storage, loc: storage)
    netHg.load_state_dict(ckpt['netHg'])
    netD.load_state_dict(ckpt['netD'])
# if FLAGS.netHg:
#     pass
# if FLAGS.netD:
#     pass

if FLAGS.cuda:
    torch.backends.cudnn.benchmark = True
    netHg.cuda()
    netD.cuda()
    criterion.cuda()
    criterion_D.cuda()

# TODO: network arch summary
# print('    Total params of netHg: %.2fM' % (sum(p.numel() for p in netHg.parameters())/1000000.0))

log_dir = getLogDir('runs')
sumWriter = SummaryWriter(log_dir)
ckpt_dir = makeCkptDir(log_dir)

def run(epoch):
    global kt, global_step
    pbar = tqdm.tqdm(dataloader, desc='Epoch %02d' % epoch, dynamic_ncols=True)
    pbar_info = tqdm.tqdm(None, bar_format='{bar}{postfix}')  # showing info on the second line
    avg_acc = 0
    for it, sample in enumerate(pbar, start=iter_init):
        global_step += 1
        image, label, image_s = sample
        image = Variable(image)
        label = Variable(label)
        image_s = Variable(image_s)
        if FLAGS.cuda:
            image = image.cuda()
            label = label.cuda(async=True)
            image_s = image_s.cuda()

        # generator
        outputs = netHg(image)
        loss_hg_content = 0
        for out in outputs:  # TODO: speed up with multiprocessing map?
            loss_hg_content += criterion(out, label)

        # Discriminator
        input_real = torch.cat([image_s, label], dim=1)
        d_real = netD(input_real)
        loss_d_real = criterion_D(d_real, label)

        input_fake = torch.cat([image_s, outputs[-1]], dim=1)
        d_fake = netD(input_fake)
        loss_d_fake = criterion_D(d_fake, label)

        loss_d = loss_d_real - kt * loss_d_fake
        loss_hg = loss_hg_content + FLAGS.lambda_G * loss_d_fake.detach()

        ''' Backward seperatedly '''
        optimD.zero_grad()
        loss_d.backward(retain_graph=True)
        optimD.step()
        optimHg.zero_grad()
        loss_hg.backward()
        # optimD.step()
        optimHg.step()

        ''' Backward all at once (slightly different ?) '''
        # loss_total = loss_d + loss_hg
        # optimD.zero_grad()
        # optimHg.zero_grad()
        # loss_total.backward()
        # optimD.step()
        # optimHg.step()

        # update kt
        loss_d_real_ = getValue(loss_d_real)
        loss_d_fake_ = getValue(loss_d_fake)
        balance = FLAGS.gamma * loss_d_real_ - loss_d_fake_ / FLAGS.lambda_G  # dividing is Good?
        kt = kt + FLAGS.lambda_kt * balance
        kt = min(1, max(0, kt))
        measure = loss_d_real_ + abs(balance)

        accs = accuracy(outputs[-1].data.cpu(), label.data.cpu(), dataset.accIdxs)

        # summary
        sumWriter.add_scalar('loss_d_real', loss_d_real_, global_step)
        sumWriter.add_scalar('loss_d_fake', loss_d_fake_, global_step)
        sumWriter.add_scalar('loss_d', loss_d, global_step)
        sumWriter.add_scalar('loss_hg_content', loss_hg_content, global_step)
        sumWriter.add_scalar('loss_hg', loss_hg, global_step)
        sumWriter.add_scalar('kt', kt, global_step)
        sumWriter.add_scalar('balance', balance, global_step)
        sumWriter.add_scalar('meauser', measure, global_step)
        sumWriter.add_scalar('acc', accs[0], global_step)
        # TODO: learning rate scheduling
        # sumWriter.add_scalar('lr', lr, global_step)

        pbar_info.set_postfix({
            'balance': balance,
            'loss_hg': getValue(loss_hg),
            'loss_d': getValue(loss_d),
            'acc': accs[0],
        })
        pbar_info.update()

        avg_acc += accs[0] / len(dataloader)
    pbar_info.set_postfix_str('avg_acc: {}'.format(avg_acc))
    pbar.close()
    pbar_info.close()


def save(epoch):
    torch.save(
        {'netD': netD.state_dict(),
         'netHg': netHg.state_dict(),
         'kt': kt},
        os.path.join(ckpt_dir, 'model_ep{}.pth'.format(epoch)))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")  # suppress skimage warning ('constant' change to 'reflect' in 0.15)
    netD.train()
    netHg.train()
    save('_init')
    for ep in range(epoch_init, FLAGS.maxEpoch):
        run(ep)
        save(ep)
