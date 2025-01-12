
import torch, numpy as np

class WarmupThenExpCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    '''
    I plan to use this scheduler all around since it handles warmup, decay, and a cyclic cosine profile.
    It is better to call step(iter), rather than the usual step().

    The cosine part is pointless. But I'm still using this class because the warmp-up can be very important.

    @cosOffset: a default value of 0.9 means that the curve will hardly be affected.

    '''
    def __init__(self, opt, warmupIters=-1, cosFreq=3.141/750, gamma=1, cosOffset=.9, last_epoch=-1, verbose=False, restart_epoch=None):
        super().__init__(opt, last_epoch, verbose)
        self.warmupIters = warmupIters
        self.restartEpoch = restart_epoch
        self.gamma = gamma
        self.cosFreq = cosFreq
        if cosOffset <= 0: cosOffset = 1
        self.cosOffset = cosOffset

    def get_lr(self):
        if self.last_epoch < 1:
            return [0 for group in self.optimizer.param_groups]
        assert False, 'must use sched.step(epoch=iter)'

    def _get_closed_form_lr(self):
        i = self.last_epoch
        m = 1
        if i >= 0 and i < self.warmupIters:
            m = (i+1) / (self.warmupIters-2)
        elif (i-self.restartEpoch) >= 0 and (i-self.restartEpoch) < self.warmupIters:
            m = ((i-self.restartEpoch)+1) / (self.warmupIters-2)
        elif i >= self.warmupIters:
            j = i - self.warmupIters
            m = (self.gamma ** j) * ((np.cos(j * self.cosFreq) * .5 + .5) * (1-self.cosOffset) + self.cosOffset)

        # return [group['lr'] * m for group in self.optimizer.param_groups]
        return [group['initial_lr'] * m for group in self.optimizer.param_groups]

    def __repr__(self):
        return f'WarmupThenExpCosineScheduler(warmup={self.warmupIters}, freq={self.cosFreq}, gamma={self.gamma} [halflife {-1/np.log2(self.gamma):.1f}], cosOffset={self.cosOffset})'
