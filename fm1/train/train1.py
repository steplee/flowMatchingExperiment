import torch, cv2, os, sys, numpy as np, importlib
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ..data.dset import HalfScaleDataset
from .summary_writer import SummaryWriter

#
# https://arxiv.org/pdf/2210.02747
#

#
# A note about `pixelScale`:
#
#       Imagine sampling a random image with values ~ N(0, 1).
#       Then mapping those values back to RGB pixels would be like: (x / pixelScale + .4) * 255
#
#       Now consider generating values N(0,1) and mapping to an RGB image tensor.
#       Take pixelScale to be  3: then the random values map to a relatively high contrast image.
#       Take pixelScale to be .3: then the random values map to a relatively low  contrast image.
#
#       To follow the paper as close as possible I *do* use p(x) ~ N(0,1) (independent along each pixel of the noise image).
#       In effect, pixelScale controls in this code controls what sigma_max would in a diffusion model.
#       (the FM sigma_min below is fixed to a certainly sensible value)
#
#       To 'explore' all areas of the state space, pixelScale should be sufficiently high.
#       I don't fully understand all of the connections yet, but I think:
#           If it were too low, then for small values of time, our velocity field would not have seen enough
#           training data far from the true image distribution and our sampling process would go off the rails
#           at low times before it even gets a chance.
#
#           If it is too high, we waste lots of walltime training on samples too far from data manifold,
#           and also our model wastes wall-time/capacity learning to push all velocities inward.
#
#       Note that cnf loss is somehow dependent on pixelScale and so is not a reliable way of comparing models.
#       Note that imagenet stddev of pixels is about .22 (IIRC), so a pixelScale of 1/.22 = 4.5 would be a sensible default.
#
#       FIXME: Actually I think doing p(x) ~ N(0, 3) would be very easy -- does not play into loss at all AFAICT
#              Indeed this seems to be the case. pixelScale should be left 1.0 now...
#

def resolve(p):
    module, clazz = p.rsplit('.',1)
    return getattr(importlib.import_module(module), clazz)

def load_model_maybe_new_conf(conf):

    sd = None
    if 'load' in conf.model:
        d = torch.load(conf.model.load)
        sd = d['sd']
        if 'iter' in d: conf.iter = d['iter']
        del conf.model.load
        conf = OmegaConf.merge(d['conf'], conf)

    model = resolve(conf.model.kind)(conf.model)
    model = model.train().cuda()

    if sd is not None:
        model.load_state_dict(sd)

    return model, conf

def get_optim(conf, model, iter0=None):
    c = conf.train
    o = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weightDecay)

    from .sched import WarmupThenExpCosineScheduler
    o.sched = WarmupThenExpCosineScheduler(o,
                                             warmupIters=c.get('warmupIters', 100),
                                             cosFreq=c.get('cosFreq', 6.2/1000),
                                             gamma=c.get('gamma',.99997),
                                             cosOffset=c.get('cosOffset',.8),
                                             restart_epoch=iter0
                                           )

    return o


class Trainer:
    def __init__(self, conf):

        self.model, conf = load_model_maybe_new_conf(conf)
        self.conf = conf

        self.modelCallable = self.model
        if len(conf.dataParallelIds) > 0:
            self.modelCallable = torch.nn.DataParallel(self.model, device_ids=conf.dataParallelIds)

        self.imgSize = conf.data.imgSize
        self.epoch = 0
        self.iter0 = conf.get('iter', 0)
        self.iter = self.iter0

        self.pixelScale = conf.pixelScale
        self.pSigma = conf.pSigma
        self.sigma_min = self.pixelScale * 2 / 255

        if not conf.get('inferenceOnly', False):
            self.dataset = HalfScaleDataset(conf.data.root, conf.data.imgSize)

            self.opt = get_optim(conf, self.modelCallable, self.iter0)
            print('title', conf.title)
            self.SW = SummaryWriter(conf.title, conf.outDir)


    def run(self):
        for iter in range(self.iter0, self.conf.train.iters):
            self.iter = iter
            self.train_one_batch(iter)
            if iter % self.conf.saveEvery == 0 and iter > self.iter0 + 5:
                self.save()
            if iter % self.conf.evalEvery == 0:
                self.run_eval(iter)

    def train_one_batch(self, iter):
        y,xhalf = self.get_batch(iter)
        with torch.no_grad():
            y = y.cuda().float().permute(0,3,1,2).div_(255).sub_(.4).mul_(self.pixelScale)
            xhalf = xhalf.cuda().float().permute(0,3,1,2).div_(255).sub_(.4).mul_(self.pixelScale)

        # WARNING: This is all unconditional for now?.?.?

        # print(y.shape,x.shape)
        x1 = F.upsample(xhalf, scale_factor=2)
        y1 = y

        if iter == self.iter0:
            self.show_some_training_examples(x1,y1)

        B = x1.size(0)
        x0 = torch.randn_like(x1) * self.pSigma # (to make conditional on lower res img, this would change...)
        t = torch.rand(B,1,1,1,device=x1.device)
        # upsilon_t = (1 - (1 - self.sigma_min) * t) * x0 + t * x1 # eqns 22 & 23
        upsilon_t = (1 - (1 - self.sigma_min) * t) * x0 + t * y1 # eqns 22 & 23. NOTE: should use x1 not y1 when conditional
        v_t = self.modelCallable(upsilon_t, t)

        # res = v_t - (x1 - (1 - self.sigma_min) * x0)
        res = v_t - (y1 - (1 - self.sigma_min) * x0)
        loss = res.norm(dim=1).mean()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.opt.sched.step(epoch=iter)

        if iter % 25 == 0:
            print(iter,loss.item())
        if iter % 5 == 0:
            self.SW.add_scalar('train/lr', self.opt.sched.get_last_lr()[0], iter=iter)
        self.SW.add_scalar('loss/cnf', loss.item(), iter)

    def get_batch(self, iteration):
        try:
            batch = next(self.dloader_iter)
            return batch
        except (AttributeError, StopIteration):
            print('recreate DataLoader')
            dloader = DataLoader(
                    self.dataset, batch_size=conf.data.batchSize, num_workers=conf.data.numWorkers,
                    worker_init_fn=self.dataset.worker_init if hasattr(self.dataset, 'worker_init') else None)
            self.dloader_iter = iter(dloader)
            return self.get_batch(iteration)

    # Mainly to debug pSigma / pixelScale.
    # This is very helpful for tuning pSigma, in fact.
    def show_some_training_examples(self, x1,y1, T=16):
        with torch.no_grad():
            x0 = torch.randn_like(x1[:1]) * self.pSigma
            t = torch.linspace(0,1,T).view(T,1,1,1).to(x1.device)
            print(x0.shape, t.shape, y1.shape)
            xt = (1 - (1 - self.sigma_min) * t) * x0 + t * y1[:1]
            xt = xt.permute(2,0,3,1).reshape(self.imgSize, self.imgSize*T, 3)
            xt = xt.div(self.pixelScale).add(.4).mul_(255)
            xt = xt.clamp(0,255).byte().cpu().numpy()
            self.SW.add_image('train/x_t', xt, self.iter)



    def sample_images_unconditional(self, B=None, x0=None, solver='euler'):
        with torch.no_grad():
            self.modelCallable.eval()
            if x0 is None:
                x0 = torch.randn(B, 3, self.imgSize, self.imgSize).cuda() * self.pSigma
            else:
                B = x0.size(0)
            xt = x0

            N = 100
            dt = 1 / N
            for i,t in enumerate(torch.linspace(0, 1, N, device=xt.device)):

                if solver == 'euler':
                    v_t = self.modelCallable(xt, t.view(1,1,1,1).repeat(B,1,1,1))
                    xt = xt + v_t * dt
                elif solver == 'midpoint':
                    v_t = self.modelCallable(xt, t.view(1,1,1,1).repeat(B,1,1,1))
                    xt5 = xt + v_t * dt * .5
                    v_t5 = self.modelCallable(xt5, t.view(1,1,1,1).repeat(B,1,1,1) + dt * .5)
                    xt = xt + v_t5 * dt
                else:
                    ValueError('invalid ODE solver')

                if hasattr(self, 'SW'):
                    if i == N//2:
                        xx = xt.div(self.pixelScale).add(.4).mul_(255)
                        self.SW.add_scalar('eval/xtPixMeanMean@t=0.5', xx.flatten(2).mean(dim=-1).mean(), self.iter)
                        self.SW.add_scalar('eval/xtPixMeanStd @t=0.5', xx.flatten(2).std(dim=-1).mean(), self.iter)
                        self.SW.add_scalar('eval/vtNormMean@t=0.5', v_t.norm(dim=1).mean() * 255/self.pixelScale, self.iter)
                    if i == N-1:
                        xx = xt.div(self.pixelScale).add(.4).mul_(255)
                        self.SW.add_scalar('eval/xtPixMeanMean@t=1', xx.flatten(2).mean(dim=-1).mean(), self.iter)
                        self.SW.add_scalar('eval/xtPixMeanStd @t=1', xx.flatten(2).std(dim=-1).mean(), self.iter)
                        self.SW.add_scalar('eval/vtNormMean@t=1', v_t.norm(dim=1).mean() * 255/self.pixelScale, self.iter)

            self.modelCallable.train()
            return xt

    def run_eval(self, iter):
        with torch.no_grad():
            y = self.sample_images_unconditional(B=32)
            B,C,H,W = y.size()
            y = y.view(4,8,3,H,W)
            y = torch.cat((y, torch.full((4,8,3,H,1), -9e4).to(y.device)), -1)
            y = torch.cat((y, torch.full((4,8,3,1,W+1), -9e4).to(y.device)), -2)
            y = y.permute(0,3, 1,4, 2).reshape(4*(H+1), 8*(H+1), 3)
            y = y.div(self.pixelScale).add_(.4).mul_(255).clamp(0, 255).byte().cpu().numpy()
            self.SW.add_image('eval/samples',y,iter)
            # cv2.imshow('samples', y)
            # cv2.waitKey(1)

    def save(self):
        path = os.path.join(self.conf.outDir, f'{self.conf.title}.{self.iter:>06d}.pt')
        print(f' - Saving \'{path}\'...')
        torch.save(dict(conf=conf,iter=self.iter,sd=self.model.state_dict()), path)


if __name__ == '__main__':

    conf0 = {
        'title': 'firstJan11',
        'dataParallelIds': [],
        'saveEvery': 20_000,
        'evalEvery': 500,
        'outDir': '/data/flowMatching/',

        'pixelScale': 1,
        'pSigma': 3,

        'model': {
            # 'kind': 'A'
            'kind': 'B'
        },
        'train': {
            'iters': 75_000,
            'weightDecay': 0,
            'lr': 2e-4,
        },
        'data': {
            'root': '/data/multiDataset1/',
            # 'imgSize': 128,
            'imgSize': 64,
            # 'batchSize': 64,
            'batchSize': 24,
            'numWorkers': 4,
        }
    }

    conf = OmegaConf.merge(conf0, OmegaConf.from_cli())

    if conf.model.kind.lower() == 'a': conf.model.kind = 'fm1.model.modelA.ModelA'
    if conf.model.kind.lower() == 'b': conf.model.kind = 'fm1.model.modelB.ModelB'
    if conf.model.kind.lower() == 'c': conf.model.kind = 'fm1.model.modelC.ModelC'
    if conf.model.kind.lower() == 'd': conf.model.kind = 'fm1.model.modelD.ModelD'
    if conf.model.kind.lower() == 'unet': conf.model.kind = 'fm1.model.unet_wrapper.ModelUnet'

    tr = Trainer(conf)

    with open('/tmp/model.txt','w') as fp:
        print(tr.model, file=fp)
        print(' - Wrote model info to \'/tmp/model.txt\'')

    try:
        tr.run()
    except KeyboardInterrupt:
        pass
    if tr.iter > tr.iter0 + 150:
        tr.save()

