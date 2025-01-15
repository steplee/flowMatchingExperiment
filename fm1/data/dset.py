import torch, numpy as np, os, sys, cv2
from .gdalRaster import GdalRaster

def get_paired_paths(root):
    files = []
    for r,ds,fs in os.walk(root):
        for f in fs:
            if '.tif' in f:
                files.append(os.path.join(r,f))
    pairs = []
    for f1 in files:
        if '1a' in f1:
            f2 = f1.replace('1a', '1b')
            if f2 in files:
                pairs.append((f1,f2))

    return pairs


# FIXME: Probably not as good as using a non-Iterable (normal) Dataset,
#        and in the constructor enumerating all candidate patches at multiple fixed scales.
#        Then also order in a way that is friendly to gdal/disk caching.

class HalfScaleDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, size=256):
        self.size = size
        self.sampleExtentRange = (self.size, self.size*10) # in pixels
        self.allowFlip = True

        self.pathPairs = get_paired_paths(root)
        assert len(self.pathPairs) > 0

        self.pairs = None

    def __iter__(self):
        self.pairs = [(GdalRaster(a), GdalRaster(b)) for a,b in self.pathPairs]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            id = worker_info.id
        else:
            id = 0
        seed = id * 1234 + 1000
        print(f'worker {id} using seed {seed}')

        return self

    def __next__(self):
        assert self.pairs is not None, 'must call __iter__ first'

        for _ in range(100):
            out = self.try_once()
            if out is not None:
                return out

        assert False, 'failed too many times'


    def try_once(self):
        # Pick random dataset.
        # FIXME: Should use sample weight as proportional to dataset w*h.

        a,b = self.pairs[np.random.choice(len(self.pairs))]

        # ab = (a,b), (a,a), (b,b)
        # a,b = ab[np.random.choice(len(ab))] # Only use two different datasets 33% of time.

        a,b = b,b # FIXME: Only using ONE dataset for now...

        W, H = a.w, a.h

        ss = int(np.random.uniform(*self.sampleExtentRange))
        x = int(np.random.uniform(W - ss - 4) + 2)
        y = int(np.random.uniform(H - ss - 4) + 2)


        # WARNING:
        # FIXME: full res disabled for now
        if 0:
            b = b.readPixelRange((x,y,ss,ss), self.size//2, self.size//2, 3)
            if b is None or (b == b[0,0]).all(): return None
            a = a.readPixelRange((x,y,ss,ss), self.size, self.size, 3)
        else:
            a = a.readPixelRange((x,y,ss,ss), self.size, self.size, 3)
            b = a

        if self.allowFlip:
            r = np.random.randint(0,4)
            if r & 0b01:
                a,b = a[::-1], b[::-1]
            if r & 0b10:
                a,b = a[:, ::-1], b[:, ::-1]
            if r > 0:
                a,b = np.copy(a,'C'), np.copy(b,'C')

        return a,b




if __name__ == '__main__':
    r = '/data/multiDataset1/'
    d = HalfScaleDataset(r, size=64)
    d = iter(d)
    for i in range(1000):
        a,b = next(d)
        cv2.imshow('a',a)
        cv2.imshow('b',b)
        cv2.waitKey(1)
