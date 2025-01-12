'''
Thin wrapper around pytorch's tensorboard SummaryWriter impl.
This modification just helps format the text and also keeps track of the last iter from any add_*() call
'''

from torch.utils.tensorboard import SummaryWriter as SummaryWriter_
import os,sys

globalSw = None

def getGlobalSw():
    global globalSw
    return globalSw

def destroyGlobalSw():
    global globalSw
    del globalSw
    globalSw = None

class SummaryWriter(SummaryWriter_):
    def __init__(self, title, rootPath, iter=0):
        self.path = os.path.join(rootPath, title)
        print(' - Creating SummaryWriter at', self.path)
        super().__init__(self.path)

        global globalSw
        assert globalSw == None
        globalSw = self

        self.iter = iter

    def __del__(self):
        global globalSw
        globalSw = None

    def add_text(self, key, txt, iter=None):
        if iter is not None:
            self.iter = iter
        super().add_text(key, txt.replace('\n','  \n').replace('\t','&nbsp;'))

    def add_image(self, key, image, iter=None, dataformats='HWC'):
        if iter is not None:
            self.iter = iter
        super().add_image(key, image, iter, dataformats=dataformats)

    def add_scalar(self, key, image, iter=None):
        if iter is not None:
            self.iter = iter
        super().add_scalar(key, image, iter)

    def add_yaml_text(self, key, txt, iter=None):
        if iter is not None:
            self.iter = iter
        from pprint import pformat
        super().add_text(key, pformat(txt).replace('\n','  \n'))

    def add_structured_text(self, key, txt, iter=None):
        if iter is not None:
            self.iter = iter
        # from pprint import pformat
        super().add_text(key, txt.replace('  ', '..').replace('\t', '...').replace('\n', '  \n'))

    def getIter(self):
        return self.iter

    def __del__(self):
        print(f' - closing SummaryWriter \'{self.path}\'')

