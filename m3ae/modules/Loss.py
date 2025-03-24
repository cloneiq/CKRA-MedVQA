import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys

def distanceL2(h, t):
    s = h - t
    sum = torch.square(s).sum(-1)
    return sum

def dot_sim(im, s):
    im_normalized = F.normalize(im, p=2, dim=1)
    s_normalized = F.normalize(s, p=2, dim=1)
    cosine_sim = torch.matmul(im_normalized, s_normalized.t())
    epsilon = 1e-8
    return cosine_sim.clamp(min=-1.0 + epsilon, max=1.0 - epsilon)

def l2_sim(im, s):
    im = F.normalize(im, dim=1)
    s = F.normalize(s, dim=1)
    b_im = im.shape[0]
    b_s = s.shape[0]
    return distanceL2(im.unsqueeze(0).repeat(b_s,1,1),s.unsqueeze(1).repeat(1,b_im,1)).transpose(0,1)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.45, measure='l2', max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.measure = measure
        if measure == 'l2':
            self.sim = l2_sim
        if measure == 'dot':
            self.sim = dot_sim
        self.max_violation = max_violation

    def forward(self, im, s, matrix):
        matrix = torch.tensor(matrix).cuda()
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        if self.measure == 'l2':
            cost_s = (self.margin + d1  - scores).clamp(min=0).to('cuda:0')
            cost_im = (self.margin + d2  - scores).clamp(min=0).to('cuda:0')
        else:
            cost_s = (self.margin + scores - d1).clamp(min=0).to('cuda:0')
            cost_im = (self.margin + scores - d2).clamp(min=0).to('cuda:0')
        cost_s = torch.nan_to_num(cost_s, nan=0.0)
        cost_im = torch.nan_to_num(cost_im, nan=0.0)
        mask1 = scores.eq(d1).cuda()
        mask2 = mask1.t()
        mask3 = matrix.eq(1).cuda()
        cost_s = cost_s.masked_fill_(mask1, 0)
        cost_im = cost_im.masked_fill_(mask2, 0)
        cost_s = cost_s.masked_fill_(mask3, 0)
        cost_im = cost_im.masked_fill_(mask3, 0)
        cost_s = torch.nan_to_num(cost_s, nan=0.0, posinf=1.0, neginf=0.0)
        cost_im = torch.nan_to_num(cost_im, nan=0.0, posinf=1.0, neginf=0.0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        epsilon = 1e-8
        numerator = cost_s.sum() + cost_im.sum()
        denominator = cost_s.shape[0] * cost_s.shape[1] - mask3.sum() - cost_s.shape[0]
        if denominator == 0:
            denominator = epsilon
        contra_loss = numerator / denominator
        return contra_loss
