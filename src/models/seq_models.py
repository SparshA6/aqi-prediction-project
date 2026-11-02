"""GRU / LSTM regressors on the environment-only windows. Fixed, untuned hyper-parameters (same for every cell).

Early stopping holds out the latest dates of the TRAINING rows and purges window overlap at the boundary.
The outer evaluated splits are never used for stopping. Final prediction = mean over SEEDS independently trained networks.
"""

import numpy as np
import torch
import torch.nn as nn

from validation import temporal_validation_split

HIDDEN = 64
LR = 2e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
MAX_EPOCHS = 60
PATIENCE = 8
VAL_DATE_FRACTION = 0.15


class SeqNet(nn.Module):
    def __init__(self, n_in, arch):
        super().__init__()
        rnn = {"GRU": nn.GRU, "LSTM": nn.LSTM}[arch]
        self.rnn = rnn(n_in, HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class AttnNet(nn.Module):
    """GRU encoder with optional attention.
    temporal = "attn"    learned additive attention over the per-day hidden states -> weights alpha (B,T)
               "uniform" fixed 1/T weights (the attention-ablated control: identical network, weights not learned)
    feature  = True      learned softmax gate over the input features of each day -> weights beta (B,T,F);
               inputs are rescaled by F*beta so uniform beta == no gating (False = no gate at all)."""

    def __init__(self, n_in, temporal, feature):
        super().__init__()
        self.temporal, self.feature = temporal, feature
        if feature:
            self.gate = nn.Linear(n_in, n_in)
            nn.init.normal_(self.gate.weight, std=0.01)
            nn.init.zeros_(self.gate.bias)
        self.rnn = nn.GRU(n_in, HIDDEN, batch_first=True)
        if temporal == "attn":
            self.score = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, return_attn=False):
        beta = None
        if self.feature:
            beta = torch.softmax(self.gate(x), dim=-1)
            x = x * beta * x.shape[-1]
        h, _ = self.rnn(x)
        if self.temporal == "attn":
            alpha = torch.softmax(self.score(h).squeeze(-1), dim=1)
        else:
            alpha = h.new_full(h.shape[:2], 1.0 / h.shape[1])
        y = self.head((alpha.unsqueeze(-1) * h).sum(1)).squeeze(-1)
        return (y, alpha, beta) if return_attn else y


class SpatialNet(nn.Module):
    """Spatiotemporal attention over a patch of neighbouring cells. Input x: (B, C, T, D+1) -- last channel = cell-present mask.
    1. one GRU shared by all cells encodes each cell's T-day window (cell offsets are input channels, so the shared
       encoder knows where its cell is);
    2. temporal attention alpha (B,C,T) pools each cell's states into a cell vector      ("attn"; "uniform" = mean-pool);
    3. spatial attention gamma (B,C) pools the cell vectors over the patch (absent cells masked) ("attn"; "uniform" = mean over present cells);
    4. MLP head -> AQI."""

    def __init__(self, n_in_with_mask, temporal, spatial):
        super().__init__()
        self.temporal, self.spatial = temporal, spatial
        self.rnn = nn.GRU(n_in_with_mask - 1, HIDDEN, batch_first=True)
        if temporal == "attn":
            self.tscore = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        if spatial == "attn":
            self.sscore = nn.Sequential(nn.Linear(HIDDEN, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, return_attn=False):
        B, C, T, D = x.shape
        mask = x[:, :, 0, -1]                                               # (B,C) 1 = cell exists
        h, _ = self.rnn(x[..., :-1].reshape(B * C, T, D - 1))               # (B*C,T,H)
        if self.temporal == "attn":
            alpha = torch.softmax(self.tscore(h).squeeze(-1), dim=1)
        else:
            alpha = h.new_full(h.shape[:2], 1.0 / T)
        ctx = (alpha.unsqueeze(-1) * h).sum(1).reshape(B, C, -1)            # (B,C,H)
        if self.spatial == "attn":
            gamma = torch.softmax(self.sscore(ctx).squeeze(-1).masked_fill(mask == 0, -1e9), dim=1)
        else:
            gamma = mask / mask.sum(1, keepdim=True).clamp(min=1)
        y = self.head((gamma.unsqueeze(-1) * ctx).sum(1)).squeeze(-1)
        return (y, alpha.reshape(B, C, T), gamma) if return_attn else y


SPATIAL_VARIANTS = {"S-Uniform": ("uniform", "uniform"), "S-TA": ("attn", "uniform"), "S-SA": ("uniform", "attn"), "S-TA+SA": ("attn", "attn")}
ATTN_VARIANTS = {"Uniform": ("uniform", False), "TA": ("attn", False), "FA": ("uniform", True), "TA+FA": ("attn", True)}


def make_net(name, n_in):
    if name in ATTN_VARIANTS:
        return AttnNet(n_in, *ATTN_VARIANTS[name])
    if name in SPATIAL_VARIANTS:
        return SpatialNet(n_in, *SPATIAL_VARIANTS[name])
    return SeqNet(n_in, name)


def _predict(net, X, bs=256):
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(net(torch.from_numpy(X[i:i + bs]).to(next(net.parameters()).device)).cpu().numpy())
    return np.concatenate(outs)


def fit_predict_seq(arch, Xtr, ytr, dates_tr, X_evals, seed, masks=None, device="cpu"):
    """Xtr float32 preprocessed, ytr raw AQI, dates_tr = calendar date of each row.
    -> ({eval_name: raw-scale predictions}, n_epochs_run, trained net, (target mean, target std))"""
    rng = np.random.default_rng(seed)
    fit_mask, val_mask = masks if masks is not None else temporal_validation_split(dates_tr, Xtr.shape[-2], VAL_DATE_FRACTION)
    mu, sd = ytr[fit_mask].mean(), ytr[fit_mask].std()
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("Fitting targets must be finite and have nonzero variance")
    yz = ((ytr - mu) / sd).astype(np.float32)
    Xf, yf, Xv, yv = Xtr[fit_mask], yz[fit_mask], Xtr[val_mask], yz[val_mask]
    if not np.isfinite(Xtr).all() or not np.isfinite(ytr).all():
        raise ValueError("Training inputs and targets must be finite")

    torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    net = make_net(arch, Xtr.shape[-1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Xf_t, yf_t = torch.from_numpy(Xf).to(device), torch.from_numpy(yf).to(device)
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Xf))).to(device)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xf_t[b]), yf_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        v = float(np.mean((_predict(net, Xv) - yv) ** 2))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return {k: _predict(net, X) * sd + mu for k, X in X_evals.items()}, ep, net.cpu(), (mu, sd)
