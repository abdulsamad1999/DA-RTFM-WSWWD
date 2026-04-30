#!/usr/bin/env python3
"""Generate all paper figures for DA-RTFM v3.

Figures generated:
  1. comparison_v3   — horizontal ROC-AUC bar chart (10 methods)
  2. version_progression — V1/V2/V3 bars with gain annotations
  3. ablation_v3     — grouped bar chart, conditions A-D
  4. roc_curve_v3    — mean ± std ROC curves from v3 checkpoints

Usage:
    python helpers/generate_paper_figures.py \
        --runs-root runs_da_rtfm_v3 \
        --test-list list/splits/test_base_v3.list \
        --output-dir paper_assets/figures \
        --use-flow --num-segments 32
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DA-RTFM paper figures")
    p.add_argument("--runs-root",    default="runs_da_rtfm_v3")
    p.add_argument("--test-list",    default="list/splits/test_base_v3.list")
    p.add_argument("--output-dir",   default="paper_assets/figures")
    p.add_argument("--use-flow",     action="store_true")
    p.add_argument("--num-segments", type=int, default=32)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def savefig(fig, out_dir: Path, stem: str) -> List[str]:
    saved = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(str(p), dpi=300, bbox_inches="tight")
        saved.append(str(p))
        print(f"  Saved: {p}")
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Figure 1: ROC-AUC comparison bar chart
# ---------------------------------------------------------------------------

_COMPARISON = [
    ("YOLO+Tracking",   0.605, "heuristic"),
    ("Flow+Rules",      0.612, "heuristic"),
    ("BN-WVAD",         0.626, "weakly_sup"),
    ("VadCLIP",         0.630, "weakly_sup"),
    ("ProDISC-VAD",     0.717, "weakly_sup"),
    ("MIL",             0.827, "weakly_sup"),
    ("LogReg",          0.843, "weakly_sup"),
    ("UMIL",            0.858, "weakly_sup"),
    ("Supervised",      0.940, "supervised"),
    ("DA-RTFM (Ours)",  0.966, "ours"),
]

_COLORS = {
    "heuristic":  "#adb5bd",
    "weakly_sup": "#a8c8e8",
    "supervised": "#f4a261",
    "ours":       "#1f4e79",
}


def fig1_comparison(out_dir: Path) -> List[str]:
    pdf_path = out_dir / "comparison_v3.pdf"
    if pdf_path.exists():
        print(f"  [SKIP] {pdf_path} already exists.")
        return [str(pdf_path), str(out_dir / "comparison_v3.png")]

    names  = [m[0] for m in _COMPARISON]
    aucs   = [m[1] for m in _COMPARISON]
    colors = [_COLORS[m[2]] for m in _COMPARISON]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, aucs, color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, aucs):
        text_color = "white" if val > 0.85 else "black"
        ax.text(bar.get_width() - 0.006,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha="right", va="center",
                fontsize=8.5, color=text_color, fontweight="bold")

    ax.axvline(0.90, color="#e63946", linestyle="--", linewidth=1.2,
               label="0.90 reference")
    ax.set_xlim(0.55, 1.02)
    ax.set_xlabel("ROC-AUC", fontsize=11)
    ax.set_title("Wrong-Way Driving Detection: ROC-AUC Comparison",
                 fontsize=12, fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor=_COLORS["ours"],        label="DA-RTFM (Ours)"),
        mpatches.Patch(facecolor=_COLORS["supervised"],  label="Supervised"),
        mpatches.Patch(facecolor=_COLORS["weakly_sup"],  label="Weakly Supervised"),
        mpatches.Patch(facecolor=_COLORS["heuristic"],   label="Heuristics"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return savefig(fig, out_dir, "comparison_v3")


# ---------------------------------------------------------------------------
# Figure 2: Version progression
# ---------------------------------------------------------------------------

def fig2_version_progression(out_dir: Path) -> List[str]:
    versions = ["V1\n(zero-pad)", "V2\n(adaptive\nresample)", "V3\n(+data +\nlabel smooth)"]
    aucs     = [0.851, 0.937, 0.966]
    colors   = ["#adb5bd", "#6baed6", "#1f4e79"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(versions, aucs, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    for i in range(len(aucs) - 1):
        x1   = bars[i].get_x() + bars[i].get_width()
        x2   = bars[i + 1].get_x()
        xmid = (x1 + x2) / 2
        gain = aucs[i + 1] - aucs[i]
        ax.annotate(
            f"+{gain * 100:.1f}pp",
            xy=(bars[i + 1].get_x() + bars[i + 1].get_width() / 2, aucs[i + 1]),
            xytext=(xmid, (aucs[i] + aucs[i + 1]) / 2 + 0.018),
            ha="center", fontsize=9, color="#e63946",
            arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.0),
        )

    ax.set_ylim(0.80, 1.01)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("DA-RTFM Version Progression", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return savefig(fig, out_dir, "version_progression")


# ---------------------------------------------------------------------------
# Figure 3: Ablation grouped bar chart
# ---------------------------------------------------------------------------

_ABLATION = {
    "A": {"ROC": 0.975, "F1": 0.835, "ROC_std": 0.003, "F1_std": 0.062,
          "label": "A\nRGB only"},
    "B": {"ROC": 0.964, "F1": 0.854, "ROC_std": 0.003, "F1_std": 0.047,
          "label": "B\n+Flow"},
    "C": {"ROC": 0.967, "F1": 0.846, "ROC_std": 0.003, "F1_std": 0.015,
          "label": "C\n+Flow\n+Temp"},
    "D": {"ROC": 0.967, "F1": 0.846, "ROC_std": 0.003, "F1_std": 0.015,
          "label": "D\n+Dir\n(=Full)"},
}


def fig3_ablation(out_dir: Path) -> List[str]:
    conds    = list(_ABLATION.keys())
    x        = np.arange(len(conds))
    w        = 0.35
    roc_vals = [_ABLATION[c]["ROC"]     for c in conds]
    f1_vals  = [_ABLATION[c]["F1"]      for c in conds]
    roc_stds = [_ABLATION[c]["ROC_std"] for c in conds]
    f1_stds  = [_ABLATION[c]["F1_std"]  for c in conds]
    xlabels  = [_ABLATION[c]["label"]   for c in conds]

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w / 2, roc_vals, w, label="ROC-AUC",
                color="#1f4e79", yerr=roc_stds, capsize=4,
                error_kw={"elinewidth": 1.2})
    b2 = ax.bar(x + w / 2, f1_vals,  w, label="F1",
                color="#a8c8e8", yerr=f1_stds,  capsize=4,
                error_kw={"elinewidth": 1.2})

    for bar, val in zip(b1, roc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    for bar, val in zip(b2, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylim(0.75, 1.04)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ablation Study: Conditions A–D (mean ± std, 3 seeds)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return savefig(fig, out_dir, "ablation_v3")


# ---------------------------------------------------------------------------
# Figure 4: ROC curves from v3 checkpoints
# ---------------------------------------------------------------------------

def _score_test_set(ckpt_path: Path, test_list_path: Path,
                    use_flow: bool, num_segments: int,
                    device) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        import torch
        import argparse as _ap
        from model import Model
        from dataset import Dataset
        from torch.utils.data import DataLoader

        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        feat_dim = 1028 if use_flow else 1024
        for k, v in state.items():
            if "Aggregate" in k and "conv" in k and "weight" in k and v.dim() == 3:
                feat_dim = int(v.shape[1])
                break

        model = Model(n_features=feat_dim, batch_size=16,
                      num_segments=num_segments, topk_ratio=0.25)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

        ds_args = _ap.Namespace(
            rgb_list=str(test_list_path),
            test_rgb_list=str(test_list_path),
            dataset="wrongway-dataset",
            modality="RGB",
            use_flow=use_flow,
            flow_suffix="_flow",
            flow_dim=4,
            split_by_label=False,
        )
        loader = DataLoader(
            Dataset(ds_args, is_normal=True, test_mode=True),
            batch_size=16, shuffle=False, num_workers=0,
        )
        all_scores, all_labels = [], []
        import torch as _t
        with _t.no_grad():
            for feats, lbls in loader:
                vs, _, _ = model.infer(feats.to(device))
                all_scores.append(vs.cpu().numpy())
                all_labels.append(lbls.numpy())
        return (np.concatenate(all_labels).astype(np.float32),
                np.concatenate(all_scores).astype(np.float32))
    except Exception as exc:
        print(f"  [WARN] Scoring failed for {ckpt_path.name}: {exc}")
        return None


def _interpolate_roc(roc_list: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr_grid = np.linspace(0, 1, 200)
    tprs = [np.interp(fpr_grid, fpr, tpr) for fpr, tpr, _ in roc_list]
    tprs_arr = np.array(tprs)
    return fpr_grid, tprs_arr.mean(0), tprs_arr.std(0)


def fig4_roc_curves(runs_root: Path, test_list_path: Path,
                    use_flow: bool, num_segments: int,
                    out_dir: Path) -> List[str]:
    try:
        import torch
        from sklearn.metrics import roc_curve, auc as sk_auc
    except ImportError as exc:
        print(f"  [SKIP] ROC curves: missing dependency: {exc}")
        return []

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roc_list  = []
    auc_vals  = []

    for run in sorted(runs_root.glob("*_rtfm_wd*")):
        ckpts = sorted((run / "ckpt").glob("*_best.pkl"))
        if not ckpts:
            continue
        result = _score_test_set(ckpts[-1], test_list_path,
                                 use_flow, num_segments, device)
        if result is None:
            continue
        labels, scores = result
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_val = float(sk_auc(fpr, tpr))
        roc_list.append((fpr, tpr, auc_val))
        auc_vals.append(auc_val)
        print(f"  {run.name}: AUC={auc_val:.4f}")

    if not roc_list:
        print("  [WARN] No ROC curves — checkpoints or feature files not found.")
        return []

    fpr_grid, mean_tpr, std_tpr = _interpolate_roc(roc_list)
    mean_auc = float(np.mean(auc_vals))
    std_auc  = float(np.std(auc_vals))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr_grid, mean_tpr, color="#1f4e79", lw=2,
            label=(f"DA-RTFM v3  AUC={mean_auc:.3f}"
                   f"±{std_auc:.3f}  (n={len(roc_list)})"))
    ax.fill_between(fpr_grid,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.2, color="#1f4e79")
    ax.axhline(0.843, color="#adb5bd", linestyle="--", lw=1.2,
               label="LogReg reference (AUC=0.843)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — DA-RTFM v3 (3 seeds)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return savefig(fig, out_dir, "roc_curve_v3")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args          = parse_args()
    out_dir       = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root     = REPO_ROOT / args.runs_root
    test_list_path = REPO_ROOT / args.test_list

    saved_all: list = []

    print("\n=== Figure 1: Comparison bar chart ===")
    saved_all += fig1_comparison(out_dir)

    print("\n=== Figure 2: Version progression ===")
    saved_all += fig2_version_progression(out_dir)

    print("\n=== Figure 3: Ablation study ===")
    saved_all += fig3_ablation(out_dir)

    print("\n=== Figure 4: ROC curves (re-scoring from checkpoints) ===")
    saved_all += fig4_roc_curves(runs_root, test_list_path,
                                 args.use_flow, args.num_segments, out_dir)

    print("\n=== All figures ===")
    for f in saved_all:
        print(f"  {f}")
    print(f"\nTotal: {len(saved_all)} files in {out_dir}")


if __name__ == "__main__":
    main()
