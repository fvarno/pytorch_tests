import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
np.set_printoptions(precision=6)

input = torch.randn(20, 5, 10, 10)
eps = 1e-5
def batchnorm1(input, mean, var):
    out = F.batch_norm(input, mean, var,\
        weight, bias, True, momentum=1, eps=eps)
    return out, mean, var
def batchnorm2(input, mean, var):
    out = F.batch_norm(input, mean, var,\
        weight, bias, False, momentum=0, eps=eps)
    return out, mean, var
def batchnorm3(input):
    # C, N, H, W
    features = input.permute([1, 0, 2, 3])
    reshaped = features.reshape([input.shape[1], -1])
    mean = reshaped.mean(-1)
    var = reshaped.var(-1)
    # H, W, N, C
    features = features.permute([2, 3, 1, 0])
    normalized = (features - mean) / (var.sqrt()+eps)
    out = torch.mul(normalized, weight) + bias
    # N, C, H, W
    out = out.permute(2, 3, 0, 1)
    return out, mean, var
def batchnorm4(input):
    # C, N, H, W
    features = input.permute([1, 0, 2, 3])
    reshaped = features.reshape([input.shape[1], -1])
    mean = reshaped.mean(-1)
    var = reshaped.var(-1)
    # H, W, N, C
    features = features.permute([2, 3, 1, 0])
    normalized = (features - mean) / (var+eps).sqrt()
    out = torch.mul(normalized, weight) + bias
    # N, C, H, W
    out = out.permute(2, 3, 0, 1)
    return out, mean, var


bn = nn.BatchNorm2d(5, track_running_stats=False, momentum=1)
weight=bn.weight
bias =bn.bias

mean=torch.zeros(input.shape[1])
var=torch.zeros(input.shape[1])

bch1 = batchnorm1(input, mean, var)
bch2 = batchnorm2(input, bch1[1], bch1[2])
bch3 = batchnorm3(input)
bch4 = batchnorm4(input)


print(np.array(bch1[0].data[0][0][0]))
print(np.array(bch2[0].data[0][0][0]))
print(np.array(bch3[0].data[0][0][0]))
print(np.array(bch4[0].data[0][0][0]))
