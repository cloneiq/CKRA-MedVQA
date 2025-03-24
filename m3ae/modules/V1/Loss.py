import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys


# 在计算文本向量和图像向量之间的距离时，通常使用余弦相似度而不是L2距离。
# 这是因为余弦相似度考虑了向量的方向，而文本向量和图像向量在方向上可能存在很大的差异，而L2距离则更关注向量的长度。

def distanceL2(h, t):
    s = h - t
    sum = torch.square(s).sum(-1)
    return sum

def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    # 对输入张量进行 L2 归一化
    im_normalized = F.normalize(im, p=2, dim=1)
    s_normalized = F.normalize(s, p=2, dim=1)

    # 直接计算归一化后的点积
    cosine_sim = torch.matmul(im_normalized, s_normalized.t())

    # 加入 epsilon 防止数值精度问题（根据需要）
    epsilon = 1e-8
    return cosine_sim.clamp(min=-1.0 + epsilon, max=1.0 - epsilon)

    # return im.mm(s.t())

def l2_sim(im, s):
    # 归一化，在第二个维度上，
    im = F.normalize(im, dim=1)
    s = F.normalize(s, dim=1)

    b_im = im.shape[0]
    b_s = s.shape[0]
    return distanceL2(im.unsqueeze(0).repeat(b_s,1,1),s.unsqueeze(1).repeat(1,b_im,1)).transpose(0,1)



# 计算图像和句子之间的相似度
# 使用margin-based损失函数来计算对比损失
# 可以选择使用L2距离或余弦相似度作为度量
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.45, measure='l2', max_violation=False):
        # max_violation 是否用最难样本
        super(ContrastiveLoss, self).__init__()

        # 常数 margin 表示正负样本之间的最小期望距离
        self.margin = margin
        self.measure = measure
        if measure == 'l2':
            self.sim = l2_sim
            # self.margin = -self.margin
        if measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    # matrix 用来过滤的，源代码中防止包含源自同一图像的标题被当作负样本对
    # 我的代码中，知识来自问题，应该不用管是否作用于同一图像
    def forward(self, im, s, matrix):    # matrix

        matrix = torch.tensor(matrix).cuda()
        # if torch.isnan(matrix).any():
        #     print("NaN detected in matrix")
        # compute image-sentence score matrix
        #im,s维度相同，默认将除了配对的都视为负样本

        # 计算图像和句子特征之间的相似度矩阵
        # s.detach(),使知识的提取不受对比学习的影响
        scores = self.sim(im, s)
        # [[0.0359, 0.0239],
        #  [0.0380, 0.0302]]
        # 获取对角线上的相似度值 diagonal   对角线上是正例，其他是反例
        # 以对角线上的元素为基准，计算图像不变对应不同知识的损失 和 知识不变对不同知识的损失
        diagonal = scores.diag().view(im.size(0), 1)
        # [[0.0359],
        #  [0.0302]]
        # 将diagonal扩展为与 scores 相同形状的矩阵 d1 和 d2
        d1 = diagonal.expand_as(scores)
        #[[0.0359, 0.0359],
        # [0.0302, 0.0302]]    这一行知识不变，图像不一样
        d2 = diagonal.t().expand_as(scores)
        #  第一列图像相同，知识不一样
        # [[0.0359, 0.0302],
        #  [0.0359, 0.0302]]

        # compare every diagonal score to scores in its column

        if self.measure == 'l2':
            # h+r, t-
            cost_s = (self.margin + d1  - scores).clamp(min=0).to('cuda:0')   # min=0 表示将张量中的所有负值都设为0，正值保持不变
            # compare every diagonal score to scores in its row
            # (h+r)-, t
            cost_im = (self.margin + d2  - scores).clamp(min=0).to('cuda:0')
        else:
            # h+r, t-; 0.3+0.2 -0.8<0; 0.7+0.2-0.6>0
            cost_s = (self.margin + scores - d1).clamp(min=0).to('cuda:0')
            # [0.4500, 0.4380],
            # [0.4578, 0.4500]]
            # compare every diagonal score to scores in its row
            # (h+r)-, t
            cost_im = (self.margin + scores - d2).clamp(min=0).to('cuda:0')
            # [[0.4500, 0.4436],
            #  [0.4521, 0.4500]]
        cost_s = torch.nan_to_num(cost_s, nan=0.0)
        cost_im = torch.nan_to_num(cost_im, nan=0.0)

        # cost_s = (self.margin + scores - d1).clamp(min=0)
        # # compare every diagonal score to scores in its row
        # # (h+r)-, t
        # cost_im = (self.margin + scores - d2).clamp(min=0)

        # # clear diagonals
        # mask = torch.eye(scores.size(0)) > .5
        # I = mask
        # if torch.cuda.is_available():
        #     I = I.cuda()
        # cost_s = cost_s.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # another mask method
        # 创建三个掩码矩阵
        mask1 = scores.eq(d1).cuda() # 是一个与 scores 大小相同的布尔矩阵。scores.eq(d1) 检查 scores 中的每个元素是否等于 d1
        #[[ True, False],
        # [False,  True]]
        mask2 = mask1.t()  # mask1的转置矩阵
        mask3 = matrix.eq(1).cuda()  # 是一个与 matrix 大小相同的布尔矩阵。matrix.eq(1) 检查 matrix 中的每个元素是否等于1
        #[[False, False],
        # [False, False]],

        # 使用 mask1 来填充 cost_s。masked_fill_ 方法会将 cost_s 中对应 mask1 为 True 的位置填充为0
        cost_s = cost_s.masked_fill_(mask1, 0)
        # [[0.0000, 0.4380],
        #  [0.4578, 0.0000]]
        cost_im = cost_im.masked_fill_(mask2, 0)
        # [[0.0000, 0.4436],
        #  [0.4521, 0.0000]]

        cost_s = cost_s.masked_fill_(mask3, 0)
        # [[0.0000, 0.4380],
        #  [0.4578, 0.0000]]
        cost_im = cost_im.masked_fill_(mask3, 0)
        cost_s = torch.nan_to_num(cost_s, nan=0.0, posinf=1.0, neginf=0.0)
        cost_im = torch.nan_to_num(cost_im, nan=0.0, posinf=1.0, neginf=0.0)


        # keep the maximum violating negative for each query
        # 最难负样本：与正样本最不相似，相似度最小的负样本
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        epsilon = 1e-8  # 小常数
        # 计算分子
        numerator = cost_s.sum() + cost_im.sum()
        # 计算分母
        denominator = cost_s.shape[0] * cost_s.shape[1] - mask3.sum() - cost_s.shape[0]
        if denominator == 0:
            denominator = epsilon
        contra_loss = numerator / denominator

        return contra_loss

        # return (cost_s.sum() + cost_im.sum())/(cost_s.shape[0]*cost_s.shape[1]-cost_s.shape[0])
        #                 损失和               /（batch_size * batch_size）=对比个数  -   相同图像    -    正例    =  反例的平均损失
        # return (cost_s.sum() + cost_im.sum()) / (cost_s.shape[0] * cost_s.shape[1] - mask3.sum() - cost_s.shape[0])
        # return cost_s.sum() + cost_im.sum()