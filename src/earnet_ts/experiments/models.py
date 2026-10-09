"""Model variants with explicit effective schedules and auxiliary gradients."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.vision_transformer import VisionTransformer
from earnet_ts.experiments.uerc2019 import CBAM, ECA, EDHHead, OccluderEncoder, ArcMarginProduct, eikonal_loss


def resnet(root, layers=None, pretrained=True):
    model = resnet50(weights=None) if layers is None else ResNet(Bottleneck, layers, zero_init_residual=True)
    if pretrained:
        weights = Path(root) / "weights/imagenet/resnet50-11ad3fa6.pth"
        if not weights.is_file():
            raise FileNotFoundError(f"ImageNet initialization required: {weights}")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        compatible = {k: v for k, v in state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
    return model


class EarNetBackbone(nn.Module):
    def __init__(self, root, config, pretrained=True):
        super().__init__()
        self.config = config
        self.kind = config.get("architecture", "earnet")
        self.two = self.kind in ("earnet", "rgb_rgb")
        self.vit = self.kind == "vit_tiny"
        if self.vit:
            # An explicitly scratch-trained transformer reference, not Arun et al.'s implementation.
            self.transformer = VisionTransformer(image_size=config.get("input_size", 256), patch_size=16,
                                                 num_layers=12, num_heads=3, hidden_dim=192,
                                                 mlp_dim=768, num_classes=512)
            self.bn = nn.BatchNorm1d(512)
            return
        depth = config.get("single_stage3_blocks", 6)
        base = resnet(root, [3, 4, depth, 3] if self.kind == "single_fusion" else None, pretrained)
        if self.kind == "single_fusion":
            original = base.conv1
            base.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                base.conv1.weight.copy_(original.weight.repeat(1, 2, 1, 1) / 2)
        self.A_stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.A_l1, self.A_l2, self.A_l3, self.A_l4 = base.layer1, base.layer2, base.layer3, base.layer4
        if self.two:
            branch = resnet(root, pretrained=pretrained)
            self.B_stem = nn.Sequential(branch.conv1, branch.bn1, branch.relu, branch.maxpool)
            self.B_l1, self.B_l2, self.B_l3 = branch.layer1, branch.layer2, branch.layer3
            self.eca = ECA(config.get("eca_kernel", 3)) if config.get("eca", True) else nn.Identity()
            self.fuse_reduce = nn.Conv2d(2048, 1024, 1, bias=False)
            self.fuse_bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.use_attention = config.get("cbam", True) if self.kind != "rgb" else False
        self.attn = CBAM(2048, reduction=16, kernel_size=7) if self.use_attention else None
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        nn.init.xavier_normal_(self.embed.weight); nn.init.zeros_(self.embed.bias)
        kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        self.register_buffer("_kx", kernel.view(1, 1, 3, 3))
        self.register_buffer("_ky", kernel.T.contiguous().view(1, 1, 3, 3))
        self.register_buffer("_lap", torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.trunk = nn.ModuleList([self.A_stem, self.A_l1, self.A_l2, self.A_l3] +
                                  ([self.B_stem, self.B_l1, self.B_l2, self.B_l3, self.eca,
                                    self.fuse_reduce, self.fuse_bn] if self.two else []) + [self.A_l4])

    @torch.no_grad()
    def _rgb_to_contour(self, x):
        rgb = x * self._std + self._mean
        r, g, b = rgb[:, :1], rgb[:, 1:2], rgb[:, 2:3]
        green = (g > .6) & (g > r + .25) & (g > b + .25)
        rgb = torch.where(green.expand_as(rgb), F.avg_pool2d(rgb, 3, 1, 1), rgb)
        gray = .2989 * rgb[:, :1] + .5870 * rgb[:, 1:2] + .1140 * rgb[:, 2:3]
        variance = (F.avg_pool2d(gray.square(), 9, 1, 4) - F.avg_pool2d(gray, 9, 1, 4).square()).clamp_min(0)
        mask = (variance < self.config.get("variance_threshold", 1e-4)).float()
        mask = F.max_pool2d(mask, 5, 1, 2)
        mask = 1 - F.max_pool2d(1 - mask, 3, 1, 1)
        suppress = (1 - mask).clamp(0, 1)
        responses = []
        for kernel in [self._kx, self._ky, self._lap]:
            value = F.conv2d(gray, kernel, padding=1) * suppress
            lo, hi = value.amin((2, 3), keepdim=True), value.amax((2, 3), keepdim=True)
            responses.append((value - lo) / (hi - lo + 1e-8))
        return torch.cat(responses, 1), suppress

    def forward(self, x, return_aux=False):
        if self.vit:
            z = F.normalize(self.bn(self.transformer(x)).float(), dim=1)
            return (z, {}) if return_aux else z
        contour, suppress = self._rgb_to_contour(x)
        inp = torch.cat([x, contour], 1) if self.kind == "single_fusion" else x
        a = self.A_l3(self.A_l2(self.A_l1(self.A_stem(inp))))
        weights = None
        if self.two:
            inp = x if self.kind == "rgb_rgb" else contour
            b = self.B_l3(self.B_l2(self.B_l1(self.B_stem(inp))))
            a = self.eca(torch.cat([a, b], 1))
            weights = getattr(self.eca, "last_weights", None)
            a = self.relu(self.fuse_bn(self.fuse_reduce(a)))
        x4 = self.A_l4(a)
        if self.attn is not None:
            x4 = self.attn(x4)
        z = F.normalize(self.bn(self.embed(self.gap(x4).flatten(1))).float(), dim=1)
        if return_aux:
            return z, {"contour": contour, "suppress": suppress, "fused3": a, "eca_w": weights}
        return z


class ArcFaceHead(ArcMarginProduct):
    """Evaluate current angular coefficients in FP32 with finite derivatives."""

    def forward(self, embeddings, labels):
        with torch.autocast(embeddings.device.type, enabled=False):
            cosine = F.linear(F.normalize(embeddings.float(), dim=1),
                              F.normalize(self.weight.float(), dim=1))
            cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
            sine = (1 - cosine.square()).clamp_min(1e-7).sqrt()
            phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                threshold = math.cos(math.pi - self.m)
                phi = torch.where(cosine > threshold, phi,
                                  cosine - math.sin(math.pi - self.m) * self.m)
            one_hot = torch.zeros_like(cosine).scatter_(1, labels[:, None], 1)
            return (one_hot * phi + (1 - one_hot) * cosine) * self.s


class EarNetModel(nn.Module):
    def __init__(self, root, classes, config, pretrained=True):
        super().__init__()
        self.config = config
        self.backbone = EarNetBackbone(root, config, pretrained)
        self.head = ArcFaceHead(512, classes, s=config.get("scale", 64.), m=config.get("margin", .35))
        auxiliary = self.backbone.two
        self.edh_head = EDHHead(1024, 4) if config.get("edh", True) and auxiliary else None
        self.occ_encoder = OccluderEncoder(1024, 512) if config.get("onsa", True) and auxiliary else None

    def set_epoch(self, epoch):
        ratio = min(1., max(0., (epoch - 1) / max(1, self.config.get("schedule_epochs", 10) - 1)))
        self.head.m = self.config.get("margin", .35) * (.5 * (1 - math.cos(math.pi * ratio))
                                                       if self.config.get("margin_schedule", True) else 1.)
        end = self.config.get("scale", 64.)
        self.head.s = 16 + (end - 16) * ratio if self.config.get("scale_schedule", True) else end

    def forward(self, x, labels=None, losses=False):
        z, aux = self.backbone(x, return_aux=True)
        logits = None if labels is None else self.head(z, labels)
        terms = {}
        if losses and self.edh_head is not None:
            import cv2
            with torch.no_grad():
                magnitude = aux["contour"].abs().mean(1, keepdim=True).float()
                magnitude = (magnitude - magnitude.amin((2, 3), keepdim=True)) / (magnitude.amax((2, 3), keepdim=True) - magnitude.amin((2, 3), keepdim=True) + 1e-8)
                inverse = (magnitude <= self.config.get("edge_threshold", .25)).to(torch.uint8).cpu().numpy()
                targets = []
                for image in inverse[:, 0]:
                    distance = cv2.distanceTransform(image * 255, cv2.DIST_L2, 3)
                    targets.append(distance / (distance.max() + 1e-8))
                target = torch.from_numpy(np.stack(targets)[:, None]).to(z.device)
                target = F.avg_pool2d(target, 4, 4)
            features = aux["fused3"] if self.config.get("edh_connected", False) else aux["fused3"].detach()
            with torch.autocast(z.device.type, enabled=False):
                predicted = self.edh_head(features.float())
                target = F.interpolate(target, size=predicted.shape[-2:], mode="area")
                terms["edh_l1"] = F.l1_loss(predicted.sigmoid(), target) * self.config.get("edh_l1", .2)
                terms["edh_eik"] = eikonal_loss(predicted) * self.config.get("edh_eik", .1)
        if losses and self.occ_encoder is not None:
            mask = 1 - F.interpolate(aux["suppress"], size=aux["fused3"].shape[-2:], mode="nearest")
            pooled = (aux["fused3"].detach() * mask).sum((2, 3)) / (mask.sum((2, 3)) + 1e-8)
            with torch.autocast(z.device.type, enabled=False):
                u = self.occ_encoder(pooled.detach().float())
                terms["onsa_identity"] = (z * u.detach()).sum(1).square().mean() * self.config.get("onsa_identity", .05)
                terms["onsa_adversary"] = -(z.detach() * u).sum(1).square().mean() * self.config.get("onsa_adversary", .05)
        return z, logits, aux, terms


def batch_hard_triplet(z, labels, margin=.2):
    distance = 1 - z @ z.T
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), device=z.device, dtype=torch.bool)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        raise ValueError("Triplet batches require at least two identities and two samples per identity.")
    hardest_positive = distance.masked_fill(~positive, -torch.inf).max(1).values
    hardest_negative = distance.masked_fill(~negative, torch.inf).min(1).values
    return (hardest_positive[valid] - hardest_negative[valid] + margin).clamp_min(0).mean()
