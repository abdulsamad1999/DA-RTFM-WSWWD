import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import option

import random
import numpy as np
from datetime import datetime


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from config import Config
from dataset import Dataset
from model import Model, weight_init
from train import train


def evaluate(test_loader, model, device):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for feats, lbl in test_loader:
            feats = feats.to(device)  # (B,T,F)
            lbl = lbl.to(device)
            video_score, _, _ = model.infer(feats)
            all_scores.append(video_score.detach().cpu())
            all_labels.append(lbl.detach().cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    # AUC-ROC and AUC-PR (Average Precision)
    auc_roc = roc_auc_score(labels, scores) if len(set(labels.tolist())) > 1 else float('nan')
    auc_pr = average_precision_score(labels, scores) if len(set(labels.tolist())) > 1 else float('nan')
    return auc_roc, auc_pr


def main():
    args = option.parser.parse_args()

    # Backwards compatibility: if --output-dir was provided by older scripts, map it to output_root.
    if hasattr(args, 'output_dir') and not hasattr(args, 'output_root'):
        args.output_root = args.output_dir

    # Reproducibility
    set_seed(int(getattr(args, 'seed', 123)), bool(getattr(args, 'deterministic', False)))

    # For GTA5 wrongway lists, splitting by label is required.
    if args.dataset.startswith('wrongway') and not args.split_by_label:
        args.split_by_label = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_feature_size = args.feature_size + (args.flow_dim if getattr(args, 'use_flow', False) else 0)

    # ---------------------------------------------------------------------
    # Validation handling. If a validation list is provided via --val-rgb-list
    # we will evaluate on it. Otherwise, if no validation list is provided
    # (and val_ratio is specified), we simply warn the user. Automatic
    # splitting of the training list has been removed; dataset splitting
    # should be performed ahead of time using helpers/split_dataset.py.
    train_list_file = args.rgb_list
    val_list_file = args.val_rgb_list

    if not val_list_file:
        if float(getattr(args, 'val_ratio', 0)) > 0:
            print(
                "[WARNING] --val-rgb-list not provided. Automatic splitting using --val-ratio "
                "has been removed. Please create a validation list with helpers/split_dataset.py "
                "or set --val-ratio to 0 to disable validation."
            )
        # Disable validation
        val_list_file = None

    # Prepare dataset instances
    # Training sets: normal and anomalous for MIL. Use split_by_label if requested.
    train_nset = Dataset(args, is_normal=True, test_mode=False)
    train_aset = Dataset(args, is_normal=False, test_mode=False)

    # Validation set: treat as test_mode=True so that no label filtering occurs (we want both classes).
    val_loader = None
    if val_list_file:
        import copy
        val_args = copy.copy(args)
        val_args.test_rgb_list = val_list_file
        val_args.rgb_list = val_list_file
        val_set = Dataset(val_args, is_normal=True, test_mode=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Test set: only loaded when evaluating after training.
    test_args = args
    test_set = Dataset(test_args, is_normal=True, test_mode=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # DataLoaders for training
    train_nloader = DataLoader(train_nset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=True)
    train_aloader = DataLoader(train_aset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=True)

    # Model
    model = Model(
        n_features=input_feature_size,
        batch_size=args.batch_size,
        num_segments=args.num_segments,
        topk_ratio=args.topk_ratio,
    ).to(device)
    model.apply(weight_init)
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)

    # Optimizer / LR schedule
    config = Config(args)
    optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)

    # Create a unique run directory based on timestamp and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(getattr(args, 'output_root', 'runs'), f"{timestamp}_{args.model_name}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    metrics_dir = os.path.join(run_dir, 'metrics')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Track best validation AUC (if applicable)
    best_auc = -1.0
    best_epoch = -1
    print("\nStarting training...\n")

    # Open a text log for per‑epoch metrics
    auc_log_path = os.path.join(metrics_dir, 'train_val_auc.tsv')
    with open(auc_log_path, 'w', encoding='utf-8') as auc_log:
        auc_log.write('epoch\tAUC_ROC_val\tAUC_PR_val\n')
        for epoch in tqdm(range(1, args.max_epoch + 1), desc="Epoch"):
            # Update LR according to schedule
            lr_now = config.lr[epoch - 1] if (epoch - 1) < len(config.lr) else config.lr[-1]
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now
            # Training step
            train(train_nloader, train_aloader, model, args.batch_size, optimizer, device, args)
            # Validation
            auc_roc, auc_pr = float('nan'), float('nan')
            if val_loader is not None:
                auc_roc, auc_pr = evaluate(val_loader, model, device)
                print(f"Epoch {epoch:03d} | Val AUC-ROC: {auc_roc:.4f} | Val AUC-PR: {auc_pr:.4f}")
                # Save best checkpoint based on validation AUC
                if auc_roc > best_auc:
                    best_auc = auc_roc
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{args.model_name}_best.pkl'))
                    with open(os.path.join(metrics_dir, 'best_auc.txt'), 'w', encoding='utf-8') as f:
                        f.write(f"epoch={epoch}\nauc_roc={auc_roc}\nauc_pr={auc_pr}\n")
            # Log metrics
            auc_log.write(f"{epoch}\t{auc_roc}\t{auc_pr}\n")
            auc_log.flush()

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{args.model_name}_final.pkl'))
    print(f"\nTraining complete. Best validation AUC-ROC: {best_auc} (epoch {best_epoch})")
    # Evaluate on test set using best or final checkpoint
    best_ckpt_path = os.path.join(ckpt_dir, f'{args.model_name}_best.pkl')
    final_ckpt_path = os.path.join(ckpt_dir, f'{args.model_name}_final.pkl')
    ckpt_to_load = best_ckpt_path if os.path.exists(best_ckpt_path) else final_ckpt_path
    model.load_state_dict(torch.load(ckpt_to_load, map_location=device), strict=False)
    test_auc_roc, test_auc_pr = evaluate(test_loader, model, device)
    print(f"Test AUC-ROC: {test_auc_roc:.4f} | Test AUC-PR: {test_auc_pr:.4f}")


if __name__ == '__main__':
    main()
