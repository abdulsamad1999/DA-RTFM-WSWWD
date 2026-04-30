import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss



def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    """Legacy score smoothness (per-video).

    The original code flattened scores across the batch, which unintentionally
    smooths across video boundaries. Here we compute smoothness along the
    temporal dimension for each video independently.

    Expected shapes:
      - (B, T) or (B, T, 1)
    """
    if arr.dim() == 3:
        arr = arr.squeeze(-1)
    if arr.dim() != 2:
        raise ValueError(f"smooth() expects (B,T) or (B,T,1), got {arr.shape}")
    diff = arr[:, 1:] - arr[:, :-1]
    loss = torch.mean(diff ** 2)
    return lamda1 * loss


def temporal_feature_smoothness(feats: torch.Tensor, normalize_dim: bool = False) -> torch.Tensor:
    """Feature-level temporal consistency (supervisor requirement).

    Penalizes large variations between consecutive segment feature representations.

    feats: (B, T, D)
    - Always normalized by (T-1) via averaging over the temporal-difference axis.
    - If normalize_dim=True, also normalizes by feature dimension D for more stable scaling.
    """
    if feats.dim() != 3:
        raise ValueError(f"temporal_feature_smoothness expects (B,T,D), got {feats.shape}")
    B, T, D = feats.shape
    if T <= 1:
        return torch.zeros((), device=feats.device)
    diff = feats[:, 1:, :] - feats[:, :-1, :]          # (B, T-1, D)
    loss = torch.sum(diff ** 2, dim=2)                 # (B, T-1)  sum over D
    loss = loss.mean()                                 # mean over B*(T-1)  => normalized by (T-1)
    if normalize_dim and D > 0:
        loss = loss / float(D)
    return loss


def direction_opposition_penalty(D: torch.Tensor, mag: torch.Tensor, margin: float) -> torch.Tensor:
    """Penalize flow directions that oppose the dominant traffic direction.

    D: direction-consistency score per segment, ideally D_t = cos(theta_t - theta_ref)
       shape (B, T)
    mag: mean flow magnitude per segment, shape (B, T)
    margin: penalize when D_t < margin (margin=0 penalizes negative alignment)
    """
    if D.dim() != 2:
        raise ValueError(f"direction_opposition_penalty expects (B,T), got {D.shape}")
    if mag.dim() != 2:
        raise ValueError(f"direction_opposition_penalty expects mag (B,T), got {mag.shape}")
    # Hinge penalty; weight by magnitude so strong opposing motion is penalized more.
    hinge = torch.relu(margin - D)
    w = torch.clamp(mag, min=0.0)
    return torch.mean(hinge * w)


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin, label_smooth_eps=0.0):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.label_smooth_eps = float(label_smooth_eps)
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        # Ensure labels reside on the same device as the scores.  Using
        # ``cuda()`` unconditionally breaks CPU‑only environments.
        # Instead we cast to the same device as the concatenated scores.
        label = label.to(score.device)

        # Label smoothing: prevents sigmoid saturation from magnitude ranking
        # pressure. Without this, 2/3 seeds converge to score distributions
        # compressed near 1.0, making threshold calibration degenerate.
        if self.label_smooth_eps > 0:
            label_smooth = label * (1.0 - self.label_smooth_eps) + \
                           (1.0 - label) * self.label_smooth_eps
        else:
            label_smooth = label
        loss_cls = self.criterion(score, label_smooth)  # BCE loss in the score space

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total


def train(train_nloader, train_aloader, model, batch_size, optimizer, device, args):
    """Perform one training epoch.

    Visualization has been removed from the training loop. Loss values
    and components are printed to stdout for tracking progress.
    """
    model.train()
    eps = float(getattr(args, 'label_smooth_eps', 0.0))
    loss_criterion = RTFM_loss(0.0001, 100, label_smooth_eps=eps)

    normal_iter = iter(train_nloader)
    anomaly_iter = iter(train_aloader)

    num_batches = min(len(train_nloader), len(train_aloader))

    for batch_idx in range(num_batches):
        if batch_idx == 0:
            pos_label = 1.0 * (1.0 - eps) + 0.0 * eps
            neg_label = 0.0 * (1.0 - eps) + 1.0 * eps
            print(f"Label smoothing: eps={eps} "
                  f"(positive label -> {pos_label:.2f}, negative label -> {neg_label:.2f})",
                  flush=True)
        try:
            ninput, nlabel = next(normal_iter)
        except StopIteration:
            normal_iter = iter(train_nloader)
            ninput, nlabel = next(normal_iter)

        try:
            ainput, alabel = next(anomaly_iter)
        except StopIteration:
            anomaly_iter = iter(train_aloader)
            ainput, alabel = next(anomaly_iter)

        ninput = ninput.to(device)
        ainput = ainput.to(device)

        inputs = torch.cat((ninput, ainput), dim=0)

        # Model returns an extra tensor `features` (B, T, 512) for feature-level temporal smoothness.
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, features = model(inputs)

        # scores: (2B, T, 1)
        abnormal_scores = scores[batch_size:]
        normal_scores_full = scores[:batch_size]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        # --- Supervisor requirement 1: Temporal smoothness (feature-level) ---
        # Apply to all videos if args.temp_on_all is True; otherwise apply to normal videos only.
        lambda_temp = float(getattr(args, 'lambda_temp', 0.0))
        if lambda_temp > 0:
            if getattr(args, 'temp_on_all', False):
                loss_temp = temporal_feature_smoothness(features, normalize_dim=getattr(args, 'temp_norm_dim', False))
            else:
                loss_temp = temporal_feature_smoothness(features[:batch_size], normalize_dim=getattr(args, 'temp_norm_dim', False))
        else:
            loss_temp = torch.zeros((), device=device)

        # --- Supervisor requirement 2: Direction-opposition penalty (normal videos) ---
        # Penalize flow vectors that oppose the dominant traffic direction on NORMAL videos only,
        # using the direction-consistency score D_t stored in the last flow channel.
        lambda_dir = float(getattr(args, 'lambda_dir', 0.0))
        dir_margin = float(getattr(args, 'dir_margin', 0.0))
        if lambda_dir > 0 and getattr(args, 'use_flow', False):
            # Flow layout appended at the end of the RGB feature vector: [u, v, mag, D]
            # D is last channel, magnitude is second last channel.
            D_t = inputs[:batch_size, :, -1]
            mag_t = inputs[:batch_size, :, -2]
            loss_dir = direction_opposition_penalty(D_t, mag_t, margin=dir_margin)
        else:
            loss_dir = torch.zeros((), device=device)

        # Optional: score-level smoothness (legacy). If enabled, compute per-video smoothness.
        lambda_score_smooth = float(getattr(args, 'lambda_score_smooth', 0.0))
        if lambda_score_smooth > 0:
            loss_score_smooth = smooth(normal_scores_full, lambda_score_smooth) + smooth(abnormal_scores, lambda_score_smooth)
        else:
            loss_score_smooth = torch.zeros((), device=device)

        # Base RTFM objective (BCE + feature magnitude ranking)
        loss_base = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

        # Total cost (sparsity removed; supervisor wants temporal smoothness added)
        cost = loss_base + (lambda_temp * loss_temp) + (lambda_dir * loss_dir) + loss_score_smooth

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(
            f"Batch {batch_idx + 1}/{num_batches} | Loss: {cost.item():.6f} | "
            f"Base: {loss_base.item():.6f} | Temp: {loss_temp.item():.6f} | Dir: {loss_dir.item():.6f}"
        )
