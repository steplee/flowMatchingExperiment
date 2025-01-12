import torch, torch.nn as nn, torch.nn.functional as F

# Did not spend a lot of time making sure this is good.
# Just a first impl to get started.


if 1:
    # I think this is more sensible?
    INITIAL_RESIDUAL_WEIGHT = dict(up = .1, across = 1)
else:
    INITIAL_RESIDUAL_WEIGHT = dict(up = 1, across = .1)

# Well it's not a residual, but let's lower initial weight of `into_t` relative to `into_x`.
# (Weighing time less than input pixels...)
INITIAL_RESIDUAL_WEIGHT['into_t'] = .1

bias = False
def nl(C):
    return nn.GroupNorm(32, C)
def act():
    return nn.ReLU(True)

def make_block(C0, C2):
    C1 = C2 * 2
    return nn.Sequential(
            nn.Conv2d(C0, C1, 3, padding=0, bias=bias),
            nl(C1),
            act(),
            nn.Conv2d(C1, C1, 1, padding=1, bias=bias),
            nl(C1),
            act(),
            nn.Conv2d(C1, C2, 3, padding=1, bias=bias),
            nl(C2),
            act())


# Maps `x` (original image) and `t` (time scalar in [0,1]) to the first feature
class Into(nn.Module):
    def __init__(self, C1):
        super().__init__()

        self.into_x = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False),
                act())

        self.into_t = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                act())

        with torch.no_grad():
            self.into_t[-2].weight.data.mul_(INITIAL_RESIDUAL_WEIGHT['into_t'])


    def forward(self, x, t):
        assert x.size(0) == t.size(0) and t.size(1) == 1 and t.size(2) == 1 and t.size(3) == 1
        x = self.into_x(x)
        t = self.into_t(t)
        return x+t



class Down(nn.Module):
    def __init__(self, C0, C2):
        super().__init__()

        self.net = make_block(C0, C2)
        self.halfscale = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.net(x)
        x = self.halfscale(x)
        return x

class Up(nn.Module):
    def __init__(self, C0, C2):
        super().__init__()

        if 0:
            self.net = make_block(C0, C2)
            with torch.no_grad():
                self.net[-2].weight.data.mul_(INITIAL_RESIDUAL_WEIGHT['up'])
        else:
            # Larger model...
            Cmid = C0 * 2
            self.net = nn.Sequential(make_block(C0, Cmid), make_block(Cmid, C2))
            with torch.no_grad():
                self.net[-1][-2].weight.data.mul_(INITIAL_RESIDUAL_WEIGHT['up'])

        self.doublescale = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        x = self.net(x)
        x = self.doublescale(x)
        return x

class Across(nn.Module):
    def __init__(self, C0):
        super().__init__()

        self.net = nn.Sequential(
                nn.Conv2d(C0, C0, 1, bias=bias),
                nl(C0),
                act())

        with torch.no_grad():
            self.net[-2].weight.data.mul_(INITIAL_RESIDUAL_WEIGHT['across'])

    def forward(self, x):
        return self.net(x)


class ModelA(nn.Module):
    def __init__(self, modelConf):
        super().__init__()

        self.into = Into(32)

        # channels = [32, 64, 128, 256]
        channels = [32, 64, 128, 256, 256]
        nblocks = len(channels)-1

        self.downs = nn.ModuleList([
            Down(channels[i], channels[i+1]) for i in range(nblocks)
        ])
        self.ups = nn.ModuleList([
            # Up(channels[i+1], channels[i]) for i in range(nblocks)
            Up(channels[i+1], channels[i]) for i in range(nblocks-1,-1,-1)
        ])
        self.acrosses = nn.ModuleList([
            # Across(channels[i]) for i in range(nblocks)
            Across(channels[i]) for i in range(nblocks-1,-1,-1)
        ])

        self.outof = nn.Sequential(
                nn.Conv2d(channels[0], channels[0]//2, 3, padding=1, bias=False),
                act(),
                nn.Conv2d(channels[0]//2, 3, 3, padding=1))

    def forward(self, x, t):
        x = self.into(x, t)

        xd = [x]

        for down in self.downs:
            xd.append(down(xd[-1]))
        # for xd_ in xd: print(xd_.shape)

        y = xd[-1]
        for i, up in enumerate(self.ups):
            if i > 0:
                a = self.acrosses[i-1](xd[-i-1])
                # print(f'upi={i}, y0={y.shape}, xdi={xd[-i-1].shape}, a={a.shape}')
                y = y + a
            y = up(y)

        y = self.outof(y)
        return y


if __name__ == '__main__':
    m = ModelA({})
    x = torch.randn(1,3,128,128)
    t = torch.rand(1,1,1,1)
    y = m(x,t )
