"""Attention, auxiliary-head and angular-margin components."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.last_ca = None
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        out = self.mlp(avg) + self.mlp(mx)
        out = self.sigmoid(out).view(b, c, 1, 1)
        self.last_ca = out.detach()
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_sa = None
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        self.last_sa = s.detach()
        return x * s

class CBAM(nn.Module):
    def __init__(self, in_ch, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_ch, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.last_weights = None
    def forward(self, x):
        y = self.avg_pool(x)                                    # (B,C,1,1)
        y1d = y.squeeze(-1).transpose(-1, -2)                   # (B,1,C)
        y1d = self.conv(y1d)
        y1d = self.sigmoid(y1d)                                 # (B,1,C)
        w = y1d.transpose(-1, -2).unsqueeze(-1)                 # (B,C,1,1)
        self.last_weights = w.detach()
        return x * w.expand_as(x)

class EDHHead(nn.Module):
    """
    Predicts a (downsampled) distance map from fused Stage-3 features.
    Training is self-supervised against a pseudo Signed/Unsigned Distance Transform (SDT) from edges.
    """
    def __init__(self, in_ch=1024, out_stride=4):
        super().__init__()
        ch = max(64, in_ch // 16)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1)
        )
        self.out_stride = out_stride

    def forward(self, x3):
        # x3: (B,1024,h,w), predict at h/out_stride, w/out_stride -> we'll downsample x3 internally
        if self.out_stride > 1:
            x3 = F.avg_pool2d(x3, kernel_size=self.out_stride, stride=self.out_stride)
        d = self.proj(x3)  # (B,1,h',w')
        return d

def eikonal_loss(d):
    # |∇d| -> target 1
    gx = F.conv2d(d, torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=d.dtype, device=d.device), padding=1)
    gy = F.conv2d(d, torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=d.dtype, device=d.device), padding=1)
    grad_mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
    return F.l1_loss(grad_mag, torch.ones_like(grad_mag))

class OccluderEncoder(nn.Module):
    """
    Encodes occluder-focused pooled features into a direction u in embedding space.
    """
    def __init__(self, in_ch=1024, emb_dim=512):
        super().__init__()
        hid = max(128, in_ch // 4)
        self.net = nn.Sequential(
            nn.Linear(in_ch, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, emb_dim)
        )
    def forward(self, v):
        u = self.net(v)                           # (B,emb)
        u = F.normalize(u, dim=1)
        return u

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

    def forward(self, embeddings, labels):
        # recompute trig terms from *current* margin each call
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        th    = math.cos(math.pi - self.m)
        mm    = math.sin(math.pi - self.m) * self.m

        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine   = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi    = cosine * cos_m - sine * sin_m
        if not self.easy_margin:
            phi = torch.where(cosine > th, phi, cosine - mm)
        else:
            phi = torch.where(cosine > 0,  phi,  cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s
