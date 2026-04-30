import os
import numpy as np
import torch
import torch.utils.data as data

torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    """Dataset for pre-extracted video features.

    Expected list format (one per line):
        <path_to_rgb_feature.npy> <label>
    where label is 0 (normal) or 1 (wrong-way / abnormal).

    If --use-flow is enabled, this loader will also load:
        <path_without_ext><flow_suffix>.npy
    which must have shape (T, flow_dim), aligned to RGB segments.
    """

    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.args = args
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.transform = transform
        self.test_mode = test_mode

        self.rgb_list_file = args.test_rgb_list if test_mode else args.rgb_list
        self.list = []
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file, 'r', encoding='utf-8'))

        # Custom datasets (e.g., GTA5 wrong-way): split by label column.
        if (not self.test_mode) and getattr(self.args, 'split_by_label', False):
            filtered = []
            for line in self.list:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                lbl = parts[1]
                try:
                    lbl_f = float(lbl)
                except ValueError:
                    continue
                if self.is_normal and lbl_f == 0.0:
                    filtered.append(line)
                elif (not self.is_normal) and lbl_f == 1.0:
                    filtered.append(line)
            self.list = filtered
            return

        # Legacy splits for original RTFM datasets (kept for compatibility)
        if not self.test_mode:
            if self.dataset == 'shanghai':
                self.list = self.list[63:] if self.is_normal else self.list[:63]
            elif self.dataset == 'ucf':
                self.list = self.list[810:] if self.is_normal else self.list[:810]

    def __getitem__(self, index):
        line = self.list[index].strip()
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid list line (expected 'path label'): {line}")

        path, label = parts[0], parts[1]
        features = np.load(path, allow_pickle=True).astype(np.float32)

        # Optional: concatenate optical-flow direction features
        if getattr(self.args, 'use_flow', False):
            base, ext = os.path.splitext(path)
            flow_path = f"{base}{self.args.flow_suffix}{ext}"
            flow_feat = np.load(flow_path, allow_pickle=True).astype(np.float32)

            if flow_feat.shape[0] != features.shape[0]:
                raise ValueError(
                    f"Flow/RGB segment mismatch: {flow_path} has {flow_feat.shape[0]} vs RGB {features.shape[0]}"
                )
            features = np.concatenate([features, flow_feat], axis=1)

        if self.transform is not None:
            features = self.transform(features)

        features = torch.from_numpy(features)  # (T, F)
        label = torch.tensor(float(label), dtype=torch.float32)
        return features, label

    def __len__(self):
        return len(self.list)
