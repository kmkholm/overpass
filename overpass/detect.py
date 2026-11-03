"""Automatic flyover detection: a small U-Net ridge segmenter on whitened spectrograms.

Trained on tiles from overpass.synth (curves rendered from real ADS-B tracks), applied to a
whole recording in time chunks, then turned into dopplertrack EventFit objects by connected
components + the existing ideal-curve fit. The classical scan (dopplertrack.tracks.detect_events)
stays available as the ablation baseline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

from dopplertrack.audio import Spectrogram
from dopplertrack.doppler import speed_of_sound
from dopplertrack.tracks import EventFit, fit_event


# ----------------------------------------------------------------------------- model
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_model(base: int = 16):
    torch, nn = _torch()

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class RidgeNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = base
            self.e1 = block(1, b); self.e2 = block(b, 2 * b); self.e3 = block(2 * b, 4 * b); self.e4 = block(4 * b, 8 * b)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d3 = block(8 * b + 4 * b, 4 * b); self.d2 = block(4 * b + 2 * b, 2 * b); self.d1 = block(2 * b + b, b)
            self.out = nn.Conv2d(b, 1, 1)

        def forward(self, x):
            # input in whitened dB; squash so bursts of +40 dB do not dominate
            x = torch.tanh(x / 10.0)
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([self.up(e4), e3], 1))
            d2 = self.d2(torch.cat([self.up(d3), e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
            return self.out(d1)

    return RidgeNet()


def _pad8(a: np.ndarray):
    """Pad (F,T) to multiples of 8 for the three pooling levels."""
    F, T = a.shape
    pf = (-F) % 8; pt = (-T) % 8
    return np.pad(a, ((0, pf), (0, pt)), mode="edge"), (F, T)


def train(X: np.ndarray, Y: np.ndarray, epochs: int = 20, batch: int = 16, lr: float = 1e-3,
          device: str | None = None, seed: int = 0, val_frac: float = 0.1, log=print, out_path=None):
    """Train the ridge segmenter. X, Y: (N,1,F,T). Returns (model, history dict)."""
    torch, nn = _torch()
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(X); nv = max(1, int(n * val_frac)); idx = np.random.permutation(n)
    vi, ti = idx[:nv], idx[nv:]
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=device))
    hist = {"train_loss": [], "val_loss": [], "val_dice": []}

    def loss_fn(logit, y):
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        return bce(logit, y) + dice

    for ep in range(epochs):
        model.train(); tl = 0.0; nb = 0
        for k in range(0, len(ti), batch):
            b = ti[k:k + batch]
            x = torch.from_numpy(X[b]).to(device); y = torch.from_numpy(Y[b]).to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            tl += float(loss); nb += 1
        sched.step()
        model.eval(); vl = 0.0; vd = 0.0; nv_b = 0
        with torch.no_grad():
            for k in range(0, len(vi), batch):
                b = vi[k:k + batch]
                x = torch.from_numpy(X[b]).to(device); y = torch.from_numpy(Y[b]).to(device)
                logit = model(x); vl += float(loss_fn(logit, y))
                p = (torch.sigmoid(logit) > 0.5).float()
                vd += float((2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)); nv_b += 1
        hist["train_loss"].append(tl / max(nb, 1)); hist["val_loss"].append(vl / max(nv_b, 1)); hist["val_dice"].append(vd / max(nv_b, 1))
        log(f"epoch {ep + 1}/{epochs} train {hist['train_loss'][-1]:.4f} val {hist['val_loss'][-1]:.4f} dice {hist['val_dice'][-1]:.3f}")
    if out_path:
        torch.save(model.state_dict(), out_path)
        json.dump(hist, open(str(out_path) + ".history.json", "w"), indent=1)
    return model, hist


def load_model(path, device: str | None = None):
    torch, _ = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def predict_map(model, sp: Spectrogram, chunk_frames: int = 2048, overlap: int = 64, device: str | None = None) -> np.ndarray:
    """Ridge probability (F, T) for a whole spectrogram, in overlapping time chunks."""
    torch, _ = _torch()
    device = device or next(model.parameters()).device
    F, T = sp.S.shape
    P = np.zeros((F, T), np.float32); Wt = np.zeros(T, np.float32)
    model.eval()
    with torch.no_grad():
        for a in range(0, T, chunk_frames - overlap):
            b = min(a + chunk_frames, T)
            x, (f0, t0) = _pad8(sp.S[:, a:b])
            logit = model(torch.from_numpy(x[None, None]).to(device))
            p = torch.sigmoid(logit)[0, 0, :f0, :t0].cpu().numpy()
            w = np.ones(t0, np.float32)
            if a > 0: w[:overlap] = np.linspace(0, 1, overlap)
            if b < T: w[-overlap:] = np.minimum(w[-overlap:], np.linspace(1, 0, overlap))
            P[:, a:b] += p * w[None, :]; Wt[a:b] += w
            if b == T:
                break
    return P / np.maximum(Wt, 1e-6)[None, :]


# ----------------------------------------------------------------------------- events
@dataclass
class Detection:
    t_start: float
    t_end: float
    f_lo: float
    f_hi: float
    harmonics: int
    prob: float          # mean ridge probability inside the component
    area: int            # pixels
    fit: EventFit | None = None

    def to_dict(self):
        d = asdict(self); d["fit"] = self.fit.to_dict() if self.fit else None; return d


def components(P: np.ndarray, sp: Spectrogram, thr: float = 0.5, min_len_s: float = 40.0, min_area: int = 60):
    """Connected components of the thresholded ridge map -> (t0, t1, f0, f1, prob, area) list."""
    lab, n = ndimage.label(P > thr, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) < min_area:
            continue
        t0, t1 = sp.t[xs.min()], sp.t[xs.max()]
        if t1 - t0 < min_len_s:
            continue
        out.append((float(t0), float(t1), float(sp.f[ys.min()]), float(sp.f[ys.max()]), float(P[ys, xs].mean()), int(len(ys))))
    out.sort(key=lambda r: r[0])
    return out


def group_harmonics(comps, tol: float = 0.12):
    """Merge components that overlap in time and sit at ~2x / 3x the frequency of a lower one."""
    used = [False] * len(comps); groups = []
    for i, a in enumerate(comps):
        if used[i]:
            continue
        used[i] = True; harm = 1; members = [a]
        fa = 0.5 * (a[2] + a[3])
        for j, b in enumerate(comps):
            if used[j]:
                continue
            fb = 0.5 * (b[2] + b[3])
            ov = min(a[1], b[1]) - max(a[0], b[0])
            if ov > 0.5 * min(a[1] - a[0], b[1] - b[0]):
                for h in (2, 3):
                    if abs(fb / fa - h) < tol * h:
                        used[j] = True; harm = max(harm, h); members.append(b); break
        groups.append((a, harm, members))
    return groups


def detect(model, sp: Spectrogram, thr: float = 0.5, c: float | None = None, refine: bool = True,
           top: int | None = None, min_len_s: float = 40.0, min_area: int = 60, log=None) -> list[Detection]:
    """Full detector: ridge map -> components -> harmonic groups -> EventFit per group."""
    c = speed_of_sound() if c is None else c
    P = predict_map(model, sp)
    comps = components(P, sp, thr, min_len_s, min_area)
    dets = []
    for (a, harm, members) in group_harmonics(comps):
        t0, t1, flo, fhi, prob, area = a
        area = sum(m[5] for m in members)
        d = Detection(t0, t1, flo, fhi, harm, prob, area)
        if refine:
            try:
                d.fit = fit_event(sp, max(t0 - 20, sp.t[0]), min(t1 + 20, sp.t[-1]),
                                  max(flo * 0.9, sp.f[0] + 1), min(fhi * 1.1, sp.f[-1] - 1), c=c, harmonics=harm,
                                  v_bounds=(25.0, 220.0), d_bounds=(150.0, 20000.0))
            except Exception as e:      # a degenerate component; keep the detection without a fit
                if log: log(f"fit failed for {t0:.0f}-{t1:.0f}s: {e}")
        dets.append(d)
    dets.sort(key=lambda d: -(d.prob * np.sqrt(d.area)))
    return dets[:top] if top else dets


def save_detections(dets: list[Detection], path):
    json.dump([d.to_dict() for d in dets], open(path, "w"), indent=1)


def load_detections(path) -> list[Detection]:
    out = []
    for d in json.load(open(path)):
        fit = EventFit(**d.pop("fit")) if d.get("fit") else None
        d.pop("fit", None)
        out.append(Detection(**d, fit=fit))
    return out


# ----------------------------------------------------------------------------- evaluation
def pixel_roc(P_list: list[np.ndarray], M_list: list[np.ndarray], n_thr: int = 101):
    """Pixel-level ROC (fpr, tpr, auc) over held-out tiles."""
    p = np.concatenate([x.ravel() for x in P_list]); m = np.concatenate([x.ravel() > 0.5 for x in M_list])
    thr = np.linspace(1, 0, n_thr)
    tpr = np.array([(p[m] >= t).mean() for t in thr]); fpr = np.array([(p[~m] >= t).mean() for t in thr])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def event_match(dets: list[Detection], truth: list[tuple[float, float]], tol_s: float = 30.0):
    """Event-level matching: truth = list of (t_cpa_heard, f0). Returns (tp, fp, fn)."""
    used = [False] * len(truth); tp = 0
    for d in dets:
        tc = d.fit.t_cpa if d.fit else 0.5 * (d.t_start + d.t_end)
        hit = None
        for i, (tt, f0) in enumerate(truth):
            if not used[i] and abs(tc - tt) <= tol_s and d.f_lo * 0.8 <= f0 <= d.f_hi * 1.25:
                hit = i; break
        if hit is not None:
            used[hit] = True; tp += 1
    return tp, len(dets) - tp, len(truth) - tp
