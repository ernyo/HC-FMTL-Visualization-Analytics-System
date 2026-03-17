"""
HCA2 on SHAPES — Optimized PyTorch PoC
======================================

Goal
----
A minimal, **fast** federated multi‑task training script to prove the effectiveness of
HCA2 (conflict‑averse encoder + cross‑attention decoder) on a tiny synthetic SHAPES dataset.

Design choices for speed & accuracy
-----------------------------------
- **Synthetic data** generated once in memory (no disk I/O).
- **Two tasks only**: semantic segmentation (5 classes incl. background) and edge.
- **Tiny backbone + FPN‑lite decoder + light heads**.
- **AMP** on CUDA, channels‑last tensors, AdamW + warmup+cosine.
- **Single‑process federated simulation** with N clients and R rounds.
- Aggregators: `none`, `fedavg`, `hca2`.
- HCA2 encoder uses a **projected GD** simplex solver (no SciPy).
- HCA2 decoder uses **layer‑wise cross‑attention**.

Usage
-----
python hca2_shapes_poc.py \
  --aggregator hca2 \
  --clients 3 \
  --rounds 8 \
  --local_epochs 1 \
  --train_per_client 800 \
  --val_size 200 \
  --batch 64 \
  --alpha 0.1 --beta 0.1

Expected: >95% mIoU in a couple of rounds on GPU; Edge BCE will drop quickly.
"""
from __future__ import annotations
import math, random, argparse, time, copy
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

# =========================
# Config & Utilities
# =========================
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels_last = torch.cuda.is_available()

def set_deterministic():
    torch.backends.cudnn.benchmark = True

# =========================
# Synthetic SHAPES dataset
# =========================
SHAPES = ["square","circle","triangle","star"]
PALETTE = np.array([[0,0,0], [255,0,0],[0,200,0],[0,128,255],[255,200,0]], np.uint8)  # bg + 4

@dataclass
class DataSpec:
    size: int = 128
    n_objects: Tuple[int,int] = (2,3)  # min,max inclusive


def _draw_shape(img: np.ndarray, mask: np.ndarray, shape: str, color: Tuple[int,int,int], cls: int, cx: int, cy: int, s: int, rot: float):
    H, W = img.shape[:2]
    if shape == "square":
        x1,y1 = max(0,cx-s//2), max(0,cy-s//2)
        x2,y2 = min(W-1,cx+s//2), min(H-1,cy+s//2)
        cv2.rectangle(img, (x1,y1),(x2,y2), color.tolist(), -1)
        cv2.rectangle(mask,(x1,y1),(x2,y2), int(cls), -1)
    elif shape == "circle":
        cv2.circle(img,(cx,cy), s//2, color.tolist(), -1)
        cv2.circle(mask,(cx,cy), s//2, int(cls), -1)
    elif shape == "triangle":
        pts = np.array([[0,-s//2],[-s//2,s//2],[s//2,s//2]], np.int32)
        c, si = math.cos(rot), math.sin(rot)
        R = np.array([[c,-si],[si,c]])
        pts = (pts @ R.T + np.array([cx,cy])).astype(np.int32)
        cv2.fillConvexPoly(img, pts, color.tolist())
        cv2.fillConvexPoly(mask, pts, int(cls))
    elif shape == "star":
        pts = []
        for i in range(10):
            r = 0.5*s if i%2==0 else 0.25*s
            t = -math.pi/2 + i*(2*math.pi/10) + rot
            pts.append([cx + int(r*math.cos(t)), cy + int(r*math.sin(t))])
        pts = np.array(pts, np.int32)
        cv2.fillConvexPoly(img, pts, color.tolist())
        cv2.fillConvexPoly(mask, pts, int(cls))
    else:
        raise ValueError(shape)


def synth_sample(spec: DataSpec, rng: np.random.Generator) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    H = W = spec.size
    img = np.zeros((H,W,3), np.uint8)
    seg = np.zeros((H,W), np.uint8)  # 0..4
    n = rng.integers(spec.n_objects[0], spec.n_objects[1]+1)
    for _ in range(n):
        shape = SHAPES[rng.integers(0,len(SHAPES))]
        cls = 1 + SHAPES.indexOf(shape) if hasattr(SHAPES,"indexOf") else 1 + SHAPES.index(shape)
        color = PALETTE[cls]
        cx, cy = int(rng.integers(20, W-20)), int(rng.integers(20, H-20))
        s = int(rng.integers(24, 72))
        rot = float(rng.random()*math.pi)
        _draw_shape(img, seg, shape, color, cls, cx, cy, s, rot)
    # edge: 4-neighbour change
    edge = np.zeros_like(seg, np.uint8)
    edge[1:,:]  |= (seg[1:,:]!=seg[:-1,:])
    edge[:-1,:] |= (seg[1:,:]!=seg[:-1,:])
    edge[:,1:]  |= (seg[:,1:]!=seg[:,:-1])
    edge[:,:-1] |= (seg[:,1:]!=seg[:,:-1])
    return img, seg, edge


class ShapesMem(Dataset):
    def __init__(self, n_samples: int, spec: DataSpec, seed: int = 0):
        super().__init__()
        self.H = spec.size
        rng = np.random.default_rng(seed)
        imgs = np.empty((n_samples,self.H,self.H,3), np.uint8)
        segs = np.empty((n_samples,self.H,self.H), np.uint8)
        edges= np.empty((n_samples,self.H,self.H), np.uint8)
        for i in range(n_samples):
            imgs[i], segs[i], edges[i] = synth_sample(spec, rng)
        # to tensors (channels-last kept for storage; convert on load)
        self.imgs = imgs
        self.segs = segs
        self.edges= edges

    def __len__(self): return self.imgs.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.imgs[idx]).to(dtype=torch.float32) # HWC
        x = x.permute(2,0,1).contiguous() / 255.0  # CHW
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        seg = torch.from_numpy(self.segs[idx].copy()).long()
        edge= torch.from_numpy(self.edges[idx].copy()).float().unsqueeze(0) # 1HW
        return x, seg, edge


# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# def show_batch(ds: ShapesMem, palette: np.ndarray, batch_size=8):
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
#     x, seg, edge = next(iter(dl))  # take one batch
#     x = x[:8]; seg = seg[:8]; edge = edge[:8]

#     cols = 8
#     plt.figure(figsize=(2*cols, 6))

#     # Row 1: images
#     for i in range(x.size(0)):
#         plt.subplot(3, cols, i+1)
#         plt.imshow(x[i].permute(1,2,0).numpy())
#         plt.axis("off"); plt.title(f"img {i}")

#     # Row 2: seg
#     for i in range(seg.size(0)):
#         plt.subplot(3, cols, cols+i+1)
#         seg_rgb = palette[seg[i].numpy()].astype(np.float32)/255.0
#         plt.imshow(seg_rgb); plt.axis("off"); plt.title("seg")

#     # Row 3: overlay edges
#     for i in range(edge.size(0)):
#         plt.subplot(3, cols, 2*cols+i+1)
#         img = x[i].permute(1,2,0).numpy().copy()
#         mask = (edge[i,0].numpy() > 0.5)
#         img[mask] = 1.0
#         plt.imshow(img); plt.axis("off"); plt.title("edges")

#     plt.tight_layout(); plt.show()

# ds = ShapesMem(32, DataSpec(size=128), seed=SEED)
# show_batch(ds, PALETTE)

# =========================
# Model: Tiny backbone + FPN-lite decoder + heads
# =========================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None: p = k//2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.bn(self.conv(x)))

class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k//2
        self.dw = ConvBNReLU(in_ch, in_ch, k, s, p, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, 1, 1, 0)
    def forward(self,x): return self.pw(self.dw(x))

class TinyBackbone(nn.Module):
    def __init__(self, C=24):
        super().__init__()
        self.stem1 = ConvBNReLU(3, C, 3, 2)
        self.stem2 = DWSeparableConv(C, C, 3, 2) # /4
        self.s1 = nn.Sequential(DWSeparableConv(C,C), DWSeparableConv(C,C))
        self.s2 = nn.Sequential(DWSeparableConv(C,2*C,3,2), DWSeparableConv(2*C,2*C))
        self.s3 = nn.Sequential(DWSeparableConv(2*C,4*C,3,2), DWSeparableConv(4*C,4*C))
        self.s4 = nn.Sequential(DWSeparableConv(4*C,8*C,3,2), DWSeparableConv(8*C,8*C))
        self.outC = C
    def forward(self,x):
        x = self.stem1(x); x = self.stem2(x)
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return [f1,f2,f3,f4]

class Decoder(nn.Module):
    def __init__(self, in_dims: List[int], E: int):
        super().__init__()
        C1,C2,C3,C4 = in_dims
        self.lat4 = nn.Conv2d(C4, E, 1, bias=False)
        self.lat3 = nn.Conv2d(C3, E, 1, bias=False)
        self.lat2 = nn.Conv2d(C2, E, 1, bias=False)
        self.lat1 = nn.Conv2d(C1, E, 1, bias=False)
        self.ref3 = DWSeparableConv(E,E)
        self.ref2 = DWSeparableConv(E,E)
        self.ref1 = DWSeparableConv(E,E)
        self.final= DWSeparableConv(E,E)
    def forward(self, feats):
        f1,f2,f3,f4 = feats
        p4 = self.lat4(f4)
        p3 = self.ref3(self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.ref2(self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.ref1(self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode='bilinear', align_corners=False))
        return self.final(p1)  # /4

class Head(nn.Module):
    def __init__(self, dim, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DWSeparableConv(dim, dim), 
            nn.Conv2d(dim, out_ch, 1)
        )
    def forward(self,x): return self.block(x)

class MultiDecoderModel(nn.Module):
    def __init__(self, tasks: List[str], baseC=24):
        super().__init__()
        self.tasks = tasks
        self.backbone = TinyBackbone(baseC)
        in_dims = [baseC,2*baseC,4*baseC,8*baseC]
        self.decoders = nn.ModuleDict({ 'shared': Decoder(in_dims, baseC) })
        self.heads = nn.ModuleDict()
        if 'semseg' in tasks: self.heads['semseg'] = Head(baseC, 5)
        if 'edge' in tasks:   self.heads['edge']   = Head(baseC, 1)
    def forward(self,x):
        feats = self.backbone(x)
        z = self.decoders['shared'](feats)
        outs = {}
        if 'semseg' in self.tasks:
            outs['semseg'] = F.interpolate(self.heads['semseg'](z), size=x.shape[-2:], mode='bilinear', align_corners=False)
        if 'edge' in self.tasks:
            outs['edge']   = F.interpolate(self.heads['edge'](z),   size=x.shape[-2:], mode='bilinear', align_corners=False)
        return outs

# =========================
# Losses & Metrics
# =========================
class MultiTaskLoss(nn.Module):
    def __init__(self, weights: Dict[str,float]):
        super().__init__()
        self.w = weights
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.95))
    def forward(self, pred: Dict[str,torch.Tensor], seg: torch.Tensor, edge: torch.Tensor):
        total = 0.0
        out = {}
        if 'semseg' in pred:
            ls = self.ce(pred['semseg'], seg)
            out['semseg'] = ls; total = total + self.w.get('semseg',1.0)*ls
        if 'edge' in pred:
            # edge shape [B,1,H,W] vs label [B,1,H,W]
            le = self.bce(pred['edge'], edge)
            out['edge'] = le; total = total + self.w.get('edge',50.0)*le
        out['total'] = total
        return out

def miou(pred_logits: torch.Tensor, target: torch.Tensor, n_classes=5) -> float:
    with torch.no_grad():
        pred = pred_logits.argmax(dim=1)
        ious = []
        for k in range(n_classes):
            p = (pred==k); g=(target==k)
            inter = (p & g).sum().item()
            union = (p | g).sum().item()
            ious.append( inter/union if union>0 else 0.0 )
        return float(np.mean(ious))

# =========================
# Optimizer / Scheduler
# =========================
class WarmupCosineLR:
    def __init__(self, optimizer, max_epochs, base_lr, min_lr=1.25e-6, warmup_epochs=0, warmup_init_lr=1.25e-7):
        self.opt=optimizer; self.max_epochs=max(1,int(max_epochs)); self.warmup=max(0,int(warmup_epochs))
        self.base_lr=float(base_lr); self.min_lr=float(min_lr); self.warmup_init_lr=float(warmup_init_lr)
    def _set_lr(self, lr):
        for pg in self.opt.param_groups: pg['lr']=lr
    def step(self, epoch):
        if epoch < self.warmup and self.warmup>0:
            t=(epoch+1)/self.warmup; lr=self.warmup_init_lr + t*(self.base_lr-self.warmup_init_lr)
        else:
            if self.max_epochs==self.warmup: lr=self.min_lr
            else:
                t=(epoch-self.warmup)/max(1,(self.max_epochs-self.warmup)); lr=self.min_lr + 0.5*(self.base_lr-self.min_lr)*(1+math.cos(math.pi*t))
        self._set_lr(lr); return lr

# =========================
# Federated helpers
# =========================
@torch.no_grad()
def clone_weights(model: nn.Module) -> Dict[str,torch.Tensor]:
    return {k: v.detach().clone() for k,v in model.state_dict().items()}

@torch.no_grad()
def assign_weights(model: nn.Module, weights: Dict[str,torch.Tensor]):
    model.load_state_dict(weights, strict=True)

@torch.no_grad()
def add_inplace(dst: Dict[str,torch.Tensor], src: Dict[str,torch.Tensor], alpha: float=1.0):
    for k in dst.keys(): dst[k].add_(src[k], alpha=alpha)

@torch.no_grad()
def sub_weights(a: Dict[str,torch.Tensor], b: Dict[str,torch.Tensor]):
    return {k: a[k]-b[k] for k in a.keys()}

# parameter grouping by name

def split_keys(state: Dict[str,torch.Tensor]):
    enc = [k for k in state if k.startswith('backbone')]
    dec = [k for k in state if k.startswith('decoders')]
    head= [k for k in state if k.startswith('heads')]
    return enc, dec, head

# ---- HCA2: encoder (conflict‑averse) ----
# @torch.no_grad()
# def hca2_encoder(last_list: List[Dict[str,torch.Tensor]], delta_list: List[Dict[str,torch.Tensor]], alpha: float=0.1):
#     enc_keys,_,_ = split_keys(last_list[0])
#     # enc_keys = [k for k in enc_keys_raw if torch.is_floating_point(last_list[0][k])]
#     # flattens
#     flats_last=[]; flats_delta=[]
#     for i in range(len(last_list)):
#         flats_last.append(torch.cat([ last_list[i][k].flatten() for k in enc_keys ]))
#         flats_delta.append(torch.cat([ delta_list[i][k].flatten() for k in enc_keys ]))
#     F_last = torch.stack(flats_last) # [N,D]
#     F_del  = torch.stack(flats_delta) # [N,D]
#     N,D = F_del.shape
#     # Gram matrix & PGD on simplex for weights x
#     G = F_del @ F_del.t()  # [N,N]
#     x = torch.full((N,), 1.0/N, device=G.device)
#     for _ in range(25):
#         grad = 2*(G @ x)           # grad of || sum x_i g_i ||^2
#         x = x - 0.05*grad
#         # project to simplex
#         # (Michelot projection)
#         u,_ = torch.sort(x, descending=True); css = torch.cumsum(u,0)
#         rho = torch.nonzero(u*torch.arange(1,N+1,device=x.device) > (css-1)).max()
#         theta = (css[rho]-1)/(rho+1)
#         x = torch.clamp(x-theta, min=0)
#         s = x.sum(); x = x/s if s>0 else torch.full_like(x, 1.0/N)
#     g_mean = F_del.mean(0)
#     gw = (x.view(N,1)*F_del).sum(0)
#     delta_update = (g_mean + alpha*gw)/(1+alpha)
#     new_flat = F_last + F_del + alpha*delta_update
#     # unflatten back into dicts
#     out_list=[]
#     for i in range(N):
#         off=0; new_state={}
#         for k in enc_keys:
#             numel = last_list[i][k].numel()
#             new_state[k] = new_flat[i,off:off+numel].view_as(last_list[i][k])
#             off+=numel
#         # out_list.append(float(new_state))
#         out_list.append(new_state)
#     return out_list

@torch.no_grad()
def hca2_encoder(last_list, delta_list, alpha: float = 0.1):
    enc_keys_all, _, _ = split_keys(last_list[0])
    # work only on float tensors for the conflict-averse math
    enc_float = [k for k in enc_keys_all if torch.is_floating_point(last_list[0][k])]

    # flatten floats
    flats_last, flats_delta = [], []
    for i in range(len(last_list)):
        flats_last.append(torch.cat([last_list[i][k].flatten() for k in enc_float]).to(torch.float32))
        flats_delta.append(torch.cat([delta_list[i][k].flatten() for k in enc_float]).to(torch.float32))

    F_last = torch.stack(flats_last)   # [N, D]
    F_del  = torch.stack(flats_delta)  # [N, D]
    N, D = F_del.shape

    # PGD on simplex to get conflict-averse weights
    G = F_del @ F_del.t()
    x = torch.full((N,), 1.0 / max(1, N), device=G.device)
    for _ in range(25):
        grad = 2 * (G @ x)
        x = x - 0.05 * grad
        # project to simplex (Michelot)
        u, _ = torch.sort(x, descending=True); css = torch.cumsum(u, 0)
        rho = (u * torch.arange(1, N + 1, device=x.device) > (css - 1)).nonzero().max()
        theta = (css[rho] - 1) / (rho + 1)
        x = torch.clamp(x - theta, min=0)
        s = x.sum(); x = x / s if s > 0 else torch.full_like(x, 1.0 / max(1, N))

    g_mean = F_del.mean(0)
    gw = (x.view(N, 1) * F_del).sum(0)
    delta_update = (g_mean + alpha * gw) / (1 + alpha)

    new_flat = F_last + F_del + alpha * delta_update  # [N, D]

    # unflatten floats + add back non-floats as pass-through (last + delta)
    out_list = []
    for i in range(N):
        off = 0
        new_state = {}
        # floats: HCA²
        for k in enc_float:
            n = last_list[i][k].numel()
            new_state[k] = new_flat[i, off:off + n].view_as(last_list[i][k]).to(last_list[i][k].dtype)
            off += n
        # non-floats: pass-through update
        for k in enc_keys_all:
            if k not in new_state:
                new_state[k] = last_list[i][k] + delta_list[i][k]
        out_list.append(new_state)
    return out_list

# ---- HCA2: decoder (cross‑attention) ----
# @torch.no_grad()
# def hca2_decoder(last_list: List[Dict[str,torch.Tensor]], delta_list: List[Dict[str,torch.Tensor]], beta: float=0.1):
#     # _,dec_keys,_ = split_keys(last_list[0])
#     _,dec_keys_raw,_ = split_keys(last_list[0])
#     dec_keys = [k for k in dec_keys_raw if torch.is_floating_point(last_list[0][k])]
#     K = len(last_list)
#     out_list = []
#     for i in range(K):
#         upd = {}
#         for k in dec_keys:
#             q = delta_list[i][k].flatten()
#             keys = [ delta_list[j][k].flatten() for j in range(K) ]
#             Kmat = torch.stack(keys, dim=1)  # [D,K]
#             attn = (q.view(1,-1) @ Kmat / math.sqrt(q.numel())).softmax(dim=1).view(-1) # [K]

#             print(f"Kmat: {Kmat.dtype}")
#             print(f"attn: {attn.dtype}")

#             ctx  = (Kmat @ attn).view_as(delta_list[i][k])
#             upd[k] = last_list[i][k] + delta_list[i][k] + beta*ctx
#         out_list.append(upd)
#     return out_list

@torch.no_grad()
def hca2_decoder(last_list, delta_list, beta: float = 0.1):
    _, dec_keys_all, _ = split_keys(last_list[0])
    K = len(last_list)
    out_list = []
    for i in range(K):
        upd = {}
        for k in dec_keys_all:
            # non-float buffers (e.g., num_batches_tracked): pass-through
            if not torch.is_floating_point(last_list[i][k]):
                upd[k] = last_list[i][k] + delta_list[i][k]
                continue

            # attention math in float32
            q = delta_list[i][k].flatten().to(torch.float32)                  # [D]
            keys = [delta_list[j][k].flatten().to(torch.float32) for j in range(K)]
            Kmat = torch.stack(keys, dim=1)                                   # [D, K]
            d = max(1, q.numel())
            attn = (q.view(1, -1) @ Kmat / math.sqrt(d)).softmax(dim=1).view(-1)  # [K]
            ctx = (Kmat @ attn).view_as(delta_list[i][k]).to(delta_list[i][k].dtype)

            upd[k] = last_list[i][k] + delta_list[i][k] + beta * ctx
        out_list.append(upd)
    return out_list

# =========================
# One local epoch
# =========================

def train_one_local_epoch(model: nn.Module, dl: DataLoader, opt: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, crit: MultiTaskLoss):
    model.train()
    tot_time = 0.0
    meters = {'n':0, 'total':0.0, 'semseg':0.0, 'edge':0.0}

    for x, seg, edge in dl:
        x = x.to(dev).to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        seg = seg.to(dev); edge = edge.to(dev)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            pred = model(x)
            loss_dict = crit(pred, seg, edge)     # {'total', 'semseg', 'edge'}
            loss = loss_dict['total']

        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()

        bs = x.size(0)
        meters['n']      += bs
        meters['total']  += loss_dict['total'].item()  * bs
        if 'semseg' in loss_dict: meters['semseg'] += loss_dict['semseg'].item() * bs
        if 'edge'   in loss_dict: meters['edge']   += loss_dict['edge'].item()   * bs

        tot_time += (time.time() - t0)

    n = max(1, meters['n'])
    logs = {k: meters[k]/n for k in ('total','semseg','edge')}
    return tot_time, logs

@torch.no_grad()
def eval_model(model: nn.Module, dl: DataLoader):
    model.eval()
    mi, eb, n= 0.0, 0.0, 0
    for x, seg, edge in dl:
        x=x.to(dev).to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        seg=seg.to(dev); edge=edge.to(dev)
        pred = model(x)
        if 'semseg' in pred:
            mi += miou(pred['semseg'], seg)*x.size(0)
        if 'edge' in pred:
            eb += F.binary_cross_entropy_with_logits(pred['edge'], edge).item()*x.size(0)
        n += x.size(0)
    return {'mIoU': mi/max(n,1), 'edgeBCE': eb/max(n,1)}

# =========================
# Main
# =========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aggregator', default='hca2', choices=['none','fedavg','hca2'])
    p.add_argument('--clients', type=int, default=3)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--local_epochs', type=int, default=1)
    p.add_argument('--train_per_client', type=int, default=800)
    p.add_argument('--val_size', type=int, default=200)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    args = p.parse_args()

    set_deterministic()

    # Build fixed validation set
    spec = DataSpec(size=128)
    val_ds = ShapesMem(args.val_size, spec, seed=123)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # Client setup: heterogenous tasks to induce conflicts
    client_tasks = []
    for i in range(args.clients):
        if i==0: client_tasks.append(['semseg'])
        elif i==1: client_tasks.append(['edge'])
        else: client_tasks.append(['semseg','edge'])

    clients = []
    for i in range(args.clients):
        train_ds = ShapesMem(args.train_per_client, spec, seed=SEED+10+i)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, persistent_workers=False)
        model = MultiDecoderModel(['semseg','edge']).to(dev)
        if channels_last: model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        sched = WarmupCosineLR(opt, max_epochs=args.rounds*args.local_epochs, base_lr=args.lr, warmup_epochs=1)

        # have to tune different proportions for losses due to different tasks
        w = {
            'semseg': 1.0 if 'semseg' in client_tasks[i] else 0.0,
            'edge'  : 50.0 if 'edge'   in client_tasks[i] else 0.0,
        }
        crit = MultiTaskLoss(weights=w).to(dev)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        clients.append({'model':model,'opt':opt,'sched':sched,'crit':crit,'scaler':scaler,'dl':train_dl,'tasks':client_tasks[i]})

    # Training rounds
    for r in range(args.rounds):
        print(f"\n== Round {r+1}/{args.rounds} ==")
        # snapshot pre
        pre = [clone_weights(c['model']) for c in clients]
        # local steps
        epoch_time=0.0
        train_logs = []
        for i,c in enumerate(clients):
            t_round = 0.0
            for _ in range(args.local_epochs):
                t, logs = train_one_local_epoch(c['model'], c['dl'], c['opt'], c['scaler'], c['crit'])
                t_round += t
                c['sched'].step(r)  # epoch-based schedule; ok to call each local epoch too
            epoch_time += t_round
            train_logs.append(logs)
        # print per-client train losses
        for i, logs in enumerate(train_logs):
            print(f"Client {i} TrainLoss total={logs['total']:.4f}  "
                f"semseg={logs['semseg']:.4f}  edge={logs['edge']:.4f}")

        post = [clone_weights(c['model']) for c in clients]
        print(f"Round training time: {epoch_time:.1f}s")

        # aggregate
        if args.aggregator!='none':
            # compute deltas
            deltas = [ sub_weights(post[i], pre[i]) for i in range(len(clients)) ]
            # keys
            enc_keys, dec_keys, head_keys = split_keys(pre[0])
            if args.aggregator=='fedavg':
                avg = copy.deepcopy(pre[0])
                for k in avg.keys():
                    avg[k].zero_()
                    for i in range(len(clients)): avg[k].add_(post[i][k])
                    avg[k].div_(len(clients))
                for i in range(len(clients)):
                    assign_weights(clients[i]['model'], avg)
            else:
                # HCA2
                # Build lists of last & delta dicts for enc/dec only
                last_enc = [{k: pre[i][k].clone() for k in enc_keys} for i in range(len(clients))]
                delta_enc= [{k: deltas[i][k].clone() for k in enc_keys} for i in range(len(clients))]
                new_enc_list = hca2_encoder(last_enc, delta_enc, alpha=args.alpha)

                last_dec = [{k: pre[i][k].clone() for k in dec_keys} for i in range(len(clients))]
                delta_dec= [{k: deltas[i][k].clone() for k in dec_keys} for i in range(len(clients))]
                new_dec_list = hca2_decoder(last_dec, delta_dec, beta=args.beta)

                # merge back; keep heads personalized (post)
                for i in range(len(clients)):
                    merged = copy.deepcopy(pre[i])
                    for k in enc_keys: merged[k] = new_enc_list[i][k]
                    for k in dec_keys: merged[k] = new_dec_list[i][k]
                    for k in head_keys: merged[k] = post[i][k]
                    assign_weights(clients[i]['model'], merged)

        # eval (client 0 as proxy)
        res = eval_model(clients[0]['model'], val_dl)
        fps = (args.train_per_client/args.batch)/ (epoch_time/len(clients)) if epoch_time>0 else 0
        print(f"mIoU {res['mIoU']*100:.1f}% | Edge BCE {res['edgeBCE']:.3f} | Train FPS ~{fps:.1f}")

    # after aggregation
    val_logs = []
    for i,c in enumerate(clients):
        res = eval_model(c['model'], val_dl)
        val_logs.append(res)
        print(f"Client {i} Val mIoU={res['mIoU']*100:.1f}%  EdgeBCE={res['edgeBCE']:.3f}")

if __name__ == '__main__':
    main()
