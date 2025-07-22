# ================= utils.py =================
import torch
import torch.nn.functional as F
from typing import Tuple

# ---------------------------------------------------------------------
# 1. Forward-hook：记录 target_layer 激活
# ---------------------------------------------------------------------
class _SaveValues:
    def __init__(self, layer: torch.nn.Module):
        self.activations: torch.Tensor | None = None
        layer.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        self.activations = out.detach()


# ---------------------------------------------------------------------
# 2. Score-CAM  (gradient-free CAM, CVPRW 2020)
# ---------------------------------------------------------------------
class ScoreCAM:
    """
    scorecam = ScoreCAM(model, target_layer, n_batch=32)
    heatmap, cls_idx = scorecam(img, idx=label)
    heatmap 形状 (1,1,H,W)，已归一化到 [0,1]。
    """
    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 n_batch: int = 32):
        self.model = model.eval()
        self.n_batch = n_batch
        self._saved = _SaveValues(target_layer)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def __call__(self,
                 x: torch.Tensor,
                 idx: int | None = None
                 ) -> Tuple[torch.Tensor, int]:
        """
        x   : Tensor (1,3,H,W) —— 已放到与 model 相同的 device
        idx : 要解释的类别；None 则取 model(x).argmax()
        """
        assert x.ndim == 4 and x.size(0) == 1, \
            "ScoreCAM 仅支持一次解释单张图片。"

        device = x.device
        _, _, H, W = x.shape

        # 1) 前向得到目标类别
        logits = self.model(x)
        if idx is None:
            idx = logits.argmax(dim=1).item()

        # 2) 获取目标层激活并归一化
        acts: torch.Tensor = self._saved.activations
        if acts is None:
            raise RuntimeError("未捕获到激活，请确认 target_layer 正常前向。")

        acts = F.relu(acts)                                          # (1,C,h,w)
        acts = F.interpolate(
            acts, size=(H, W), mode='bilinear', align_corners=False
        )                                                            # (1,C,H,W)
        C = acts.size(1)

        amin = acts.view(C, -1).min(dim=1)[0].view(1, C, 1, 1)
        amax = acts.view(C, -1).max(dim=1)[0].view(1, C, 1, 1)
        acts = (acts - amin) / (amax - amin + 1e-8)                  # 0-1

        # 3) 生成掩膜批次 → 前向 → 收集 w_k
        weights = []
        for s in range(0, C, self.n_batch):
            mask = acts[:, s:s + self.n_batch]                       # (1,C',H,W)
            mask = mask.squeeze(0)                                   # (C',H,W)
            C_prime = mask.size(0)

            # 复制到 3 通道，与输入逐元素相乘
            mask3 = mask.unsqueeze(1).repeat(1, x.size(1), 1, 1)     # (C',3,H,W)
            imgs  = x.repeat(C_prime, 1, 1, 1) * mask3               # (C',3,H,W)

            # softmax 概率作为权重
            probs = F.softmax(self.model(imgs), dim=1)[:, idx]       # (C',)
            weights.append(probs.view(-1))                           # 保证 >=1 维

        weights = torch.cat(weights).to(device)                      # (C,)

        # 4) 得到 CAM
        cam = (weights.view(C, 1, 1) * acts.squeeze(0)).sum(0, keepdim=True)  # (1,H,W)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)                             # 归一化 [0,1]

        return cam.unsqueeze(0), idx                                 # (1,1,H,W), int


