from .train1 import load_model_maybe_new_conf, Trainer, cv2, torch
from omegaconf import OmegaConf

if __name__ == '__main__':

    conf0 = {
        'inferenceOnly': True,
    }
    conf = OmegaConf.merge(conf0, OmegaConf.from_cli())

    tr = Trainer(conf)
    print(tr.model)

    B = 32
    x0 = torch.randn(B, 3, tr.imgSize, tr.imgSize).cuda() * tr.pSigma

    for method in 'euler midpoint'.split(' '):
        y = tr.sample_images_unconditional(x0=x0, solver=method)
        B,C,H,W = y.size()
        y = y.view(4,8,3,H,W)
        y = torch.cat((y, torch.full((4,8,3,H,1), -9e4).to(y.device)), -1)
        y = torch.cat((y, torch.full((4,8,3,1,W+1), -9e4).to(y.device)), -2)
        y = y.permute(0,3, 1,4, 2).reshape(4*(H+1), 8*(H+1), 3)
        y = y.div(tr.pixelScale).add_(.4).mul_(255).clamp(0, 255).byte().cpu().numpy()
        cv2.imshow(method,y)

    cv2.waitKey(0)

