import torch

def entropy(x, eps=1e-8):
    x = (x.permute(1, 0) - x.min(dim=1)[0])
    x = (x/(torch.sum(x, dim=0)+eps)).permute(1, 0)
    baselog = torch.log(torch.tensor([x.shape[-1]]).float())
    return -torch.sum(x* (torch.log(x)/baselog), dim=1)
