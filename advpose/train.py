import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter
import opts
from models.hg import HourglassNet
from models.dis import HourglassDisNet
from datasets.lsp import LSPMPIIData
from utils import getValue, getLogDir
from evals import accuracy

epoch_init = 0
iter_init = 0
global_step = 0
FLAGS = opts.get_args()

dataset = LSPMPIIData('../data', split='train')
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=FLAGS.batchSize, shuffle=True,
    num_workers=FLAGS.nThreads, pin_memory=True)

netHg = HourglassNet(
    nStack=FLAGS.nStack, nModules=FLAGS.nModules, nFeat=FLAGS.nFeats,
    nClasses=dataset.nClasses)  # ref `nClasses` from dataset
netD = HourglassDisNet(
    nStack=FLAGS.nStack, nModules=FLAGS.nModules, nFeat=FLAGS.nFeats,
    nClasses=dataset.nClasses, inplanes=3+dataset.nClasses)
criterion = nn.MSELoss()
criterion_D = nn.MSELoss()

kt = FLAGS.kt_init

optimHg = torch.optim.RMSprop(netHg.parameters(), lr=FLAGS.lr, alpha=FLAGS.alpha)
optimD = torch.optim.Adam(netD.parameters(), lr=FLAGS.lrD, betas=(0.9, 0.999))

# TODO: restore model, optim, hyperparameter ...
# if FLAGS.netHg:
#     pass
# if FLAGS.netD:
#     pass

if FLAGS.cuda:
    torch.backends.cudnn.benchmark = True
    # make parallel
    netHg = nn.DataParallel(netHg)
    netD = nn.DataParallel(netD)
    netHg.cuda()
    netD.cuda()
    criterion.cuda()
    criterion_D.cuda()

# TODO: network arch summary
# print('    Total params of netHg: %.2fM' % (sum(p.numel() for p in netHg.parameters())/1000000.0))
# exit()

sumWriter = SummaryWriter(getLogDir('../runs'))

def run(epoch):
    global kt, global_step
    for it, sample in enumerate(dataloader):
        global_step += 1
        print('iter:', it)
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
        optimHg.step()

        ''' Backward all at once (slightly different) '''
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

        # TODO: summary, printing (maybe tqdm)
        sumWriter.add_scalar('loss_d_real', loss_d_real_, global_step)
        sumWriter.add_scalar('loss_d_fake', loss_d_fake_, global_step)
        sumWriter.add_scalar('loss_d', loss_d, global_step)
        sumWriter.add_scalar('loss_hg_content', loss_hg_content, global_step)
        sumWriter.add_scalar('loss_hg', loss_hg, global_step)
        sumWriter.add_scalar('kt', kt, global_step)
        sumWriter.add_scalar('meauser', measure, global_step)
        sumWriter.add_scalar('acc', accs[0], global_step)
        # sumWriter.add_scalar('lr', lr, global_step)

def save(epoch, ):
    torch.save({
        'netD': netD.state_dict(),
        'netHg': netHg.state_dict()
    }, 'model_ep{}.pth'.format(epoch))

if __name__ == '__main__':
    netD.train()
    netHg.train()
    for ep in range(epoch_init, FLAGS.nEpochs):
        run(ep)
        # save(ep)
