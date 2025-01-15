from .train1 import load_model_maybe_new_conf, Trainer, cv2, torch
from omegaconf import OmegaConf
import os

def main(conf):
    tr = Trainer(conf)
    # print(tr.model)

    B = conf.B
    assert B % 8 == 0

    exportVideoShowingAllSamplesInRow = False

    if conf.initialState == 'random':
        x0 = torch.randn(B, 3, tr.imgSize, tr.imgSize).cuda() * tr.pSigma
    elif conf.initialState.startswith('randomInterpolated'):
        n = int(conf.initialState.split('_')[-1])
        xs = torch.randn(n, 3, tr.imgSize, tr.imgSize).cuda() * tr.pSigma

        # Come up with interpolation weights to create B mixed samples from n base samples.
        # Rather than linearly interpolating, let's use a softmax scheme.
        # This means coming up with a weight matrix that gives weights of each `xs` sample with each batch index.
        l = torch.linspace(0,n, B)[None].to(xs.device)
        nodes = torch.arange(n)[:,None].to(xs.device)
        w = torch.nn.functional.softmax(-(nodes-l)**2/conf.interpolationTemp, dim=0) # [n,B]

        x0 = (xs[:,None] * w[:,:, None,None,None]).sum(0)

        # NOTE: We get very bad results unless we re-standardize the interpolated values!
        for i in range(len(x0)):
            x0[i] = (x0[i] - x0[i].mean()) / x0[i].std() * tr.pSigma

        exportVideoShowingAllSamplesInRow = True

    if d := conf.outputVideoDir: os.makedirs(d, exist_ok=True)

    if conf.yield_results:

        for method in conf.methods:
            vcap = None
            for y0 in tr.sample_images_unconditional(x0=x0, solver=method, T=conf.T, yield_results=True):
                B,C,H,W = y0.size()
                y = y0.view(B//8,8,3,H,W)
                y = torch.cat((y, torch.full((B//8,8,3,H,1), -9e4).to(y.device)), -1)
                y = torch.cat((y, torch.full((B//8,8,3,1,W+1), -9e4).to(y.device)), -2)
                y = y.permute(0,3, 1,4, 2).reshape(B//8*(H+1), 8*(H+1), 3)
                y = y.div(tr.pixelScale).add_(.4).mul_(255).clamp(0, 255).byte().cpu().numpy()
                y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
                if conf.outputVideoDir:
                    if vcap is None:
                        vcap = cv2.VideoWriter(os.path.join(conf.outputVideoDir, f'{method}.mp4'), frameSize=y.shape[:2][::-1], fps=10, fourcc=cv2.VideoWriter_fourcc(*'mp4v'))
                    vcap.write(y)
                cv2.imshow(method,y)
                cv2.waitKey(1)

        if exportVideoShowingAllSamplesInRow and conf.outputVideoDir:
            vcap = cv2.VideoWriter(os.path.join(conf.outputVideoDir, f'{method}.inRow.mp4'), frameSize=(tr.imgSize,tr.imgSize), fps=15, fourcc=cv2.VideoWriter_fourcc(*'mp4v'))
            for y in y0:
                y = y.permute(1,2,0)
                y = y.div(tr.pixelScale).add_(.4).mul_(255).clamp(0, 255).byte().cpu().numpy()
                y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
                print(y.shape)
                vcap.write(y)

        cv2.waitKey(0)

    else:

        assert not conf.outputVideo, 'only supported if yield_results'

        for method in conf.methods:
            y = tr.sample_images_unconditional(x0=x0, solver=method, T=conf.T, yield_results=False)
            B,C,H,W = y.size()
            y = y.view(B//8,8,3,H,W)
            y = torch.cat((y, torch.full((B//8,8,3,H,1), -9e4).to(y.device)), -1)
            y = torch.cat((y, torch.full((B//8,8,3,1,W+1), -9e4).to(y.device)), -2)
            y = y.permute(0,3, 1,4, 2).reshape(B//8*(H+1), 8*(H+1), 3)
            y = y.div(tr.pixelScale).add_(.4).mul_(255).clamp(0, 255).byte().cpu().numpy()
            y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
            cv2.imshow(method,y)

if __name__ == '__main__':

    conf0 = {
        'inferenceOnly': True,
        'T': 100,
        'yield_results': True,
        'outputVideoDir': None,
        # 'methods': 'euler midpoint'.split(' '),
        'methods': 'midpoint'.split(' '),
        'seed': 0,
        'B': 32,

        # 'initialState': 'random',
        'initialState': 'randomInterpolated_4', # Interpolate 4 base samples!
        'interpolationTemp': 1,
    }


    conf = OmegaConf.merge(conf0, OmegaConf.from_cli())

    torch.manual_seed(int(conf.seed))

    with torch.no_grad():
        main(conf)


