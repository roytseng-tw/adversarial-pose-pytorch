import torch
import torch.nn as nn


def conv_module(fin, fout):
    return nn.Sequential(nn.Conv2d(fin, fout, 3, 1, 1),
                         nn.ReLU(inplace=True))

class Hourglass(nn.Module):
    def __init__(self, n, nFeats, nModules=1, f_inc=128,
                 module=conv_module):
        super().__init__()
        self.n = n
        self.nFeats = nFeats
        self.nModules = nModules
        self.module = module

        nFeats_2 = nFeats + f_inc

        self.up1 = self._make_modules(nModules, nFeats, nFeats)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = self._make_modules(nModules, nFeats, nFeats_2)
        if self.n > 1:  # Recursive
            self.low2 = Hourglass(n - 1, nFeats_2, nModules, f_inc, module)
        else:
            self.low2 = self._make_modules(nModules, nFeats_2, nFeats_2)
        self.low3 = self._make_modules(nModules, nFeats_2, nFeats)

        self.up2 = nn.Upsample(scale_factor=2)

    def _make_modules(self, n, nf1, nf2, change_nf_first=True):
        if change_nf_first:
            mlist = [self.module(nf1, nf2)] + [self.module(nf2, nf2) for _ in range(n-1)]
        else:
            mlist = [self.module(nf1, nf1) for _ in range(n-1)] + [self.module(nf1, nf2)]
        return nn.Sequential(*mlist)

    def forward(self, x):
        up1_ = x
        up1_ = self.up1(up1_)
        low_ = self.pool1(x)
        low_ = self.low1(low_)
        low_ = self.low2(low_)
        low_ = self.low3(low_)
        up2_ = self.up2(low_)
        return up1_ + up2_


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.conv = nn.Conv2d(x_dim, y_dim, 1)

    def forward(self, x):
        return self.conv(x)


class HourglassAENet(nn.Module):
    def __init__(self, nStacks, inp_nf, out_nf, inplanes=3):
        super().__init__()
        self.nStacks = nStacks
        self.inp_nf = inp_nf
        self.out_nf = out_nf

        self.head = nn.Sequential(
            nn.Conv2d(inplanes, 64, 7, 2, 3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),  # pose-ae-demo has this
            nn.Conv2d(128, inp_nf, 3, 1, 1), nn.ReLU(inplace=True),
        )

        self.features = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_nf),
                nn.Conv2d(inp_nf, inp_nf, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(inp_nf, inp_nf, 3, padding=1), nn.ReLU(inplace=True),  # pose-ae-demo kernel_size=1
            ) for _ in range(nStacks)])
        self.outs = nn.ModuleList([nn.Conv2d(inp_nf, out_nf, 1) for _ in range(nStacks)])
        self.merge_features = nn.ModuleList([Merge(inp_nf, inp_nf) for _ in range(nStacks - 1)])
        self.merge_preds = nn.ModuleList([Merge(out_nf, inp_nf) for _ in range(nStacks - 1)])

    def forward(self, x):
        x = self.head(x)
        preds = []
        for i in range(self.nStacks):
            feature = self.features[i](x)
            preds.append(self.outs[i](feature))
            if i < self.nStacks - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)
        return torch.stack(preds, 1)
