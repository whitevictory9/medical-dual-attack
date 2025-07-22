#!/usr/bin/env python3
# attack_cli_unified.py

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.models import (
    vgg16_bn, resnet50, densenet121, efficientnet_v2_s
)
from captum.attr import (
    IntegratedGradients, DeepLift, LayerGradCam, Saliency
)
from scipy.ndimage import gaussian_filter
from utils import ScoreCAM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Location & Top-k dual adversarial attack"
    )
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to test set (ImageFolder)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Model checkpoint (.pth)")
    parser.add_argument("--model-name", type=str, default="efficientnet_v2_s",
                        choices=["vgg16_bn", "resnet50", "densenet121", "efficientnet_v2_s"])
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224],
                        help="Resize size: WIDTH HEIGHT")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of random samples to attack")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="L\u221E perturbation bound")
    parser.add_argument("--alpha", type=float, default=0.005,
                        help="PGD step size")
    parser.add_argument("--iterations", type=int, default=20,
                        help="PGD iterations")
    parser.add_argument("--attack-type", type=str, choices=["location", "topk"],
                        default="location", help="Attack type: location or topk")
    parser.add_argument("--k-percent", type=int, default=90,
                        help="Top-k percentage for mask")
    parser.add_argument("--epsilon-top-k", type=float, default=0.25,
                        help="Threshold for Top-k EMD success")
    parser.add_argument("--method", type=str, default="gradcam",
                        choices=["gradcam", "scorecam", "ig", "deeplift", "saliency"],
                        help="Attribution method")
    return parser.parse_args()


def infer_dataset_stats(path):
    lower = path.lower()
    if "dermoscopy" in lower:
        return (4, {0: "BCC", 1: "BKL", 2: "MEL", 3: "NV"},
                [0.6715, 0.5314, 0.5249], [0.2201, 0.2015, 0.2134])
    if "chest" in lower:
        return (2, {0: "Normal", 1: "Pneumonia"},
                [0.4676, 0.4676, 0.4676], [0.2484, 0.2484, 0.2484])
    if "fundoscopy" in lower:
        return (2, {0: "DR", 1: "Normal"},
                [0.4043, 0.2154, 0.0708], [0.2794, 0.1518, 0.0792])
    raise ValueError(f"Cannot infer dataset from {path}")


def get_dataloader(path, img_size, batch_size, sample_size):
    tf = transforms.Compose([
        transforms.Resize(tuple(img_size)),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(path, transform=tf)
    if sample_size and sample_size < len(ds):
        idx = torch.randperm(len(ds))[:sample_size]
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


class NormalizedModel(nn.Module):
    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def load_model(name, num_classes, ckpt, mean, std, device):
    name = name.lower()
    if name == "vgg16_bn":
        net = vgg16_bn(pretrained=False)
        net.classifier[6] = nn.Linear(4096, num_classes)
    elif name == "resnet50":
        net = resnet50(pretrained=False)
        net.fc = nn.Linear(2048, num_classes)
    elif name == "densenet121":
        net = densenet121(pretrained=False)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    elif name == "efficientnet_v2_s":
        net = efficientnet_v2_s(pretrained=False)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    state = torch.load(ckpt, map_location=device)
    net.load_state_dict(state, strict=True)
    return NormalizedModel(net, mean, std).to(device).eval()


def get_target_layer(model, name):
    backbone = model.net
    m = name.lower()
    if m == "densenet121":
        return list(backbone.features.denseblock4.children())[-1].conv2
    if m == "vgg16_bn":
        return backbone.features[40]
    if m == "resnet50":
        return backbone.layer4[2].conv3
    if m == "efficientnet_v2_s":
        return backbone.features[-1]
    raise ValueError(f"No target layer for {name}")


# Attribution computations

def compute_ig(model, x, t):
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, target=t)
    return np.abs(attr[0].detach().cpu().numpy()).sum(0)


def compute_deeplift(model, x, t):
    dl = DeepLift(model)
    attr = dl.attribute(x, target=t)
    return np.abs(attr[0].detach().cpu().numpy()).sum(0)


def compute_saliency(model, x, t):
    s = Saliency(model)
    attr = s.attribute(x, target=t)
    return np.abs(attr[0].detach().cpu().numpy()).sum(0)


def compute_gradcam(model, x, t, layer):
    gc = LayerGradCam(model, layer)
    cam = gc.attribute(x, target=t).squeeze().cpu().numpy()
    cam = gaussian_filter(cam, sigma=2)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def compute_scorecam(model, x, t, layer):
    sc = ScoreCAM(model, layer, n_batch=32)
    cam, _ = sc(x, idx=t)
    return cam.squeeze().squeeze().cpu().numpy()


def compute_explanation(model, x, t, method, layer=None):
    if method == "ig":          return compute_ig(model, x, t)
    if method == "deeplift":    return compute_deeplift(model, x, t)
    if method == "saliency":    return compute_saliency(model, x, t)
    if method == "gradcam":     return compute_gradcam(model, x, t, layer)
    if method == "scorecam":    return compute_scorecam(model, x, t, layer)
    raise ValueError(f"Unknown method: {method}")


# Mask & loss / success criteria

def compute_mask(attr_map, k_percent):
    thresh = np.percentile(attr_map.flatten(), k_percent)
    return (attr_map >= thresh).astype(float)


def location_loss(orig, pert):
    mask = compute_mask(orig, args.k_percent)
    return np.linalg.norm(pert - mask) / pert.size


def location_success(orig, pert):
    thresh = np.percentile(orig.flatten(), args.k_percent)
    mask = (orig >= thresh).astype(float)
    rel_imp = (pert * mask).sum() / (mask.sum() + 1e-8)
    return 1 if rel_imp < thresh else 0


def topk_loss(orig, pert):
    mask = compute_mask(orig, args.k_percent)
    return -np.sum(pert * mask)


def topk_success(orig, pert):
    m1 = compute_mask(orig, args.k_percent)
    m2 = compute_mask(pert, args.k_percent)
    overlap = (m1 * m2).sum()
    union = ((m1 + m2) > 0).sum()
    score = overlap / union if union > 0 else 0
    return 1 if score < args.epsilon_top_k else 0


def dual_attack(x, y, model, layer, cfg):
    x, y = x.to(cfg.device), y.to(cfg.device)
    with torch.no_grad():
        if model(x).argmax(1) != y: return x, 0
    orig_attr = compute_explanation(model, x, y.item(), cfg.method, layer)
    delta = torch.empty_like(x).uniform_(-cfg.epsilon, cfg.epsilon)
    pert = (x + delta).clamp(0,1).detach().requires_grad_(True)
    for _ in range(cfg.iterations):
        out = model(pert)
        loss_c = F.cross_entropy(out, y)
        pert_attr = compute_explanation(model, pert, y.item(), cfg.method, layer)
        if cfg.attack_type == 'location':
            loss_e = location_loss(orig_attr, pert_attr)
            succ_f = location_success(orig_attr, pert_attr)
        else:
            loss_e = topk_loss(orig_attr, pert_attr)
            succ_f = topk_success(orig_attr, pert_attr)
        lam = 0.01 + (0.1-0.01) * max(0, min(1, (0.01-loss_c.item())/0.01))
        total = loss_c + lam * loss_e
        model.zero_grad()
        total.backward()
        with torch.no_grad():
            delta.data = (delta + cfg.alpha * delta.grad.sign()).clamp(-cfg.epsilon, cfg.epsilon)
            pert = (x + delta).clamp(0,1).detach().requires_grad_(True)
    final = model(pert).argmax(1)
    success = int((final.item() != y.item()) and succ_f)
    return pert.detach(), success


def main():
    global args
    args = parse_args()
    # reproducibility
    random.seed(33); np.random.seed(33)
    torch.manual_seed(33); torch.cuda.manual_seed_all(33)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, _, mean, std = infer_dataset_stats(args.dataset_path)
    loader = get_dataloader(
        args.dataset_path, args.img_size,
        args.batch_size, args.sample_size
    )
    model = load_model(
        args.model_name, num_classes, args.model_path,
        mean, std, args.device
    )
    layer = get_target_layer(model, args.model_name)

    total, succ = 0, 0
    for imgs, labels in loader:
        total += 1
        _, ok = dual_attack(imgs, labels, model, layer, args)
        succ += ok
        print(f"[{total}] Attack {'SUCCESS' if ok else 'FAIL'}")
    print(f"\nOverall success rate: {succ}/{total} = {succ/total:.2%}")

if __name__ == "__main__":
    main()
