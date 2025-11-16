# %% [markdown]
# Replace ResNet models with EfficientNet-B1, MobileNetV4-Small, ConvNeXt-Femto

# %%
%pip install pandas numpy mne scipy scikit-learn torch imbalanced-learn gdown py7zr matplotlib seaborn tqdm -q

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
import sys
import gc
import time
import random
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import mne
mne.set_log_level('ERROR')

from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             cohen_kappa_score, confusion_matrix, classification_report)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import gdown
import py7zr

# -------------------------
# OPTIMIZED CONFIGURATION FOR A40 (48GB VRAM)
# -------------------------
GDRIVE_FILE_URL = "https://drive.google.com/file/d/1SpxCWWAPWkyhfagPbvuSCCgNfEsCPKlg/view?usp=sharing"
DATA_DIR = Path("sleep_edf_data")
OUT_DIR = Path("output")
PLOT_DIR = Path("plots")
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_SUBJECTS = 20
EPOCH_LEN = 30.0
RANDOM_STATE = 42

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True

USE_CLASS_WEIGHTS = False
USE_WEIGHTED_SAMPLER = True
EARLY_STOP_MODE = 'hybrid'

USE_AMP = True
GRADIENT_CLIP = 1.0

CLASS_NAMES = ['Wake', 'N1+N2', 'N3', 'REM']
STAGE_MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 1,
    'Sleep stage 3': 2,
    'Sleep stage 4': 2,
    'Sleep stage R': 3,
}

DESIRED_EEG = ['EEG Fpz-Cz', 'EEG Pz-Oz']
PLOT_DPI = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# -------------------------
# Memory Management
# -------------------------
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# -------------------------
# Checkpoint Management
# -------------------------
def save_pipeline_checkpoint(state, filename="pipeline_checkpoint.pkl"):
    checkpoint_path = CHECKPOINT_DIR / filename
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

def load_pipeline_checkpoint(filename="pipeline_checkpoint.pkl"):
    checkpoint_path = CHECKPOINT_DIR / filename
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None

# -------------------------
# Utility functions
# -------------------------
def download_and_extract_gdrive(gdrive_url, out_dir: Path, archive_name='sleep_edf.7z'):
    archive_path = out_dir / archive_name
    if not archive_path.exists():
        print(f"Downloading dataset...")
        gdown.download(gdrive_url, str(archive_path), fuzzy=True, quiet=False)
    extracted_flag = out_dir / '.extracted'
    if not extracted_flag.exists():
        print("Extracting archive...")
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=out_dir)
        extracted_flag.touch()
    return out_dir

def discover_valid_file_pairs(data_dir: Path, n_subjects_limit: int = None):
    dirs = sorted([d for d in data_dir.glob("subject_*") if d.is_dir()])
    if len(dirs) == 0:
        parents = sorted(set([p.parent for p in data_dir.glob("**/*-PSG.edf")]))
        dirs = parents
    if n_subjects_limit:
        dirs = dirs[:n_subjects_limit]
    file_pairs = []
    for sd in dirs:
        psg_files = [p for p in sd.iterdir() if p.is_file() and ('psg' in p.name.lower() or '-psg' in p.name.lower())]
        hyp_files = [p for p in sd.iterdir() if p.is_file() and ('hyp' in p.name.lower() or 'hypnogram' in p.name.lower())]
        if len(psg_files) > 0 and len(hyp_files) > 0:
            psg = psg_files[0]
            hyp = hyp_files[0]
            try:
                ann = mne.read_annotations(hyp)
                if len(ann) > 0:
                    file_pairs.append((psg, hyp))
            except:
                pass
    return file_pairs

def pick_eeg_channels(raw, desired_channels=None):
    if desired_channels is None:
        desired_channels = DESIRED_EEG
    available = [ch for ch in desired_channels if ch in raw.ch_names]
    if available:
        return available
    picks = mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        raise RuntimeError("No EEG channels")
    chosen = [raw.ch_names[picks[i]] for i in range(min(2, len(picks)))]
    return chosen

def load_and_extract_epochs(psg_file: Path, hyp_file: Path):
    try:
        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        ann = mne.read_annotations(hyp_file)
        raw.set_annotations(ann, emit_warning=False)
        eeg_chs = pick_eeg_channels(raw)
        raw.pick(eeg_chs)
        raw.filter(l_freq=0.5, h_freq=30.0, verbose=False)
        event_id = {}
        for k, v in STAGE_MAPPING.items():
            event_id[k] = v
            event_id[k.lower()] = v
        events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=EPOCH_LEN, verbose=False)
        if events.shape[0] == 0:
            return None, None, None
        tmax = EPOCH_LEN - 1.0 / raw.info['sfreq']
        epochs = mne.Epochs(raw=raw, events=events, event_id=None, tmin=0.0, tmax=tmax, baseline=None, preload=True, verbose=False)
        X = epochs.get_data()
        y = epochs.events[:, -1]
        return X, y, raw.info['sfreq']
    except Exception as e:
        return None, None, None

def process_subjects_leak_free(file_pairs_subset):
    X_list = []
    y_list = []
    subj_ids = []
    for idx, (psg, hyp) in enumerate(file_pairs_subset):
        X, y, sfreq = load_and_extract_epochs(psg, hyp)
        if X is None:
            continue
        X_list.append(X)
        y_list.append(y)
        subj_ids.extend([idx] * len(y))
    if len(X_list) == 0:
        return None, None, None
    X_concat = np.vstack(X_list)
    y_concat = np.hstack(y_list)
    return X_concat, y_concat, np.array(subj_ids)

def normalize_channelwise_stats(X_train, X_val, X_test, eps=1e-8):
    n_channels = X_train.shape[1]
    X_train_n = np.zeros_like(X_train)
    X_val_n = np.zeros_like(X_val)
    X_test_n = np.zeros_like(X_test)
    for ch in range(n_channels):
        chdata = X_train[:, ch, :]
        mean = chdata.mean()
        std = chdata.std()
        std = std if std > 0 else 1.0
        X_train_n[:, ch, :] = (X_train[:, ch, :] - mean) / (std + eps)
        X_val_n[:, ch, :] = (X_val[:, ch, :] - mean) / (std + eps)
        X_test_n[:, ch, :] = (X_test[:, ch, :] - mean) / (std + eps)
    return X_train_n, X_val_n, X_test_n

def compute_class_weights_safe(y_train, n_classes=len(CLASS_NAMES), eps=1e-6):
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.maximum(counts, eps)
    weights = len(y_train) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_weighted_sampler_safe(y_train, n_classes=len(CLASS_NAMES), eps=1e-6):
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.maximum(counts, eps)
    class_weights = 1.0 / counts
    sample_weights = class_weights[y_train]
    sample_weights = sample_weights / (sample_weights.sum() + 1e-12)
    sample_weights = torch.from_numpy(sample_weights.astype(np.float32))
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# -------------------------
# Comprehensive Metrics
# -------------------------
def compute_comprehensive_metrics(y_true, y_pred):
    m = {}
    m['accuracy'] = float(accuracy_score(y_true, y_pred))
    m['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    m['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    m['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    m['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    m['precision_micro'] = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
    m['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    m['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    m['recall_micro'] = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
    m['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    m['kappa'] = float(cohen_kappa_score(y_true, y_pred))
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(CLASS_NAMES):
        m[f'precision_{class_name}'] = float(precision_per_class[i])
        m[f'recall_{class_name}'] = float(recall_per_class[i])
        m[f'f1_{class_name}'] = float(f1_per_class[i])
    
    m['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    m['classification_report'] = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    
    return m

# -------------------------
# Plotting Functions
# -------------------------
def plot_training_curves(epoch_logs, fold_idx, model_type):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = [log['epoch'] for log in epoch_logs]
    train_loss = [log['train_loss'] for log in epoch_logs]
    val_loss = [log['val_loss'] for log in epoch_logs]
    train_acc = [log['train_acc'] for log in epoch_logs]
    val_f1 = [log['val_f1_macro'] for log in epoch_logs]
    
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, val_f1, 'g-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Macro')
    axes[1, 0].set_title('Validation F1 Macro')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    ax2 = axes[1, 1].twinx()
    axes[1, 1].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[1, 1].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax2.plot(epochs, val_f1, 'g-', label='Val F1', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    ax2.set_ylabel('F1 Macro')
    axes[1, 1].set_title('Combined Metrics')
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOT_DIR / f"training_curves_{model_type}_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def plot_confusion_matrix(cm, fold_idx, model_type):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_type} Fold {fold_idx+1}')
    plt.tight_layout()
    plot_path = PLOT_DIR / f"confusion_matrix_{model_type}_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def plot_per_class_metrics(metrics, fold_idx, model_type):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    classes = CLASS_NAMES
    precision = [metrics[f'precision_{c}'] for c in classes]
    recall = [metrics[f'recall_{c}'] for c in classes]
    f1 = [metrics[f'f1_{c}'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    axes[0].bar(x, precision, width, label='Precision', color='skyblue')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision per Class')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, recall, width, label='Recall', color='lightcoral')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Recall per Class')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, f1, width, label='F1', color='lightgreen')
    axes[2].set_ylabel('Score')
    axes[2].set_title('F1 per Class')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(classes)
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = PLOT_DIR / f"per_class_metrics_{model_type}_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

# -------------------------
# Model Architectures
# -------------------------

# EfficientNet-B1
class SqueezeExcitation1D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv1d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(reduced_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        scale = F.adaptive_avg_pool1d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class MBConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 expand_ratio=6, se_ratio=0.25, drop_rate=0.2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        expanded_channels = in_channels * expand_ratio
        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        
        padding = kernel_size // 2
        self.depthwise = nn.Sequential(
            nn.Conv1d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=expanded_channels, bias=False),
            nn.BatchNorm1d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        self.se = SqueezeExcitation1D(expanded_channels, reduction=int(1/se_ratio))
        
        self.project = nn.Sequential(
            nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.drop_rate = drop_rate
    
    def forward(self, x):
        identity = x
        
        if self.expand is not None:
            x = self.expand(x)
        
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_residual:
            if self.training and self.drop_rate > 0:
                if torch.rand(1).item() > self.drop_rate:
                    x = x + identity
            else:
                x = x + identity
        
        return x

class EfficientNet1D(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, width_mult=1.0, depth_mult=1.1):
        super().__init__()
        
        stem_channels = self._scale_width(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(inplace=True)
        )
        
        blocks_config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        self.blocks = nn.ModuleList()
        in_ch = stem_channels
        
        for expand_ratio, channels, num_layers, stride, kernel_size in blocks_config:
            out_ch = self._scale_width(channels, width_mult)
            num_layers = self._scale_depth(num_layers, depth_mult)
            
            for i in range(num_layers):
                s = stride if i == 0 else 1
                self.blocks.append(
                    MBConv1D(in_ch, out_ch, kernel_size=kernel_size, 
                            stride=s, expand_ratio=expand_ratio)
                )
                in_ch = out_ch
        
        head_channels = self._scale_width(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(head_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(head_channels, num_classes)
        )
        
        self._initialize_weights()
    
    def _scale_width(self, channels, multiplier):
        channels = int(channels * multiplier)
        new_channels = max(8, int(channels + 4) // 8 * 8)
        if new_channels < 0.9 * channels:
            new_channels += 8
        return new_channels
    
    def _scale_depth(self, num_layers, multiplier):
        return int(num_layers * multiplier)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# MobileNetV4-Small
class UIB1D(nn.Module):
    """Universal Inverted Bottleneck for 1D"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        padding = kernel_size // 2
        
        layers = []
        
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)

class MobileNetV4_1D(nn.Module):
    def __init__(self, in_channels=2, num_classes=4):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        config = [
            [3, 1, 16, 1],
            [3, 4, 24, 2],
            [3, 3, 24, 1],
            [5, 3, 32, 2],
            [5, 3, 32, 1],
            [5, 3, 64, 2],
            [5, 4, 64, 1],
            [3, 6, 96, 1],
            [3, 6, 96, 1],
        ]
        
        self.blocks = nn.ModuleList()
        in_ch = 16
        for kernel, exp_ratio, out_ch, stride in config:
            self.blocks.append(UIB1D(in_ch, out_ch, kernel, stride, exp_ratio))
            in_ch = out_ch
        
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ConvNeXt-Femto
class LayerNorm1D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm1D(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = drop_path
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)
        
        if self.training and self.drop_path > 0:
            if torch.rand(1).item() > self.drop_path:
                x = input + x
        else:
            x = input + x
        return x

class ConvNeXt1D(nn.Module):
    def __init__(self, in_channels=2, num_classes=4, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384]):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm1D(dims[0])
        )
        
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock1D(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            
            if i < 3:
                self.stages.append(nn.Sequential(
                    LayerNorm1D(dims[i]),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2)
                ))
        
        self.norm = LayerNorm1D(dims[-1])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.head(x)
        return x

def get_model(model_type='efficientnet_b1', in_ch=2, num_classes=4):
    if model_type == 'efficientnet_b1':
        return EfficientNet1D(in_ch, num_classes, width_mult=1.0, depth_mult=1.1)
    elif model_type == 'mobilenetv4_small':
        return MobileNetV4_1D(in_ch, num_classes)
    elif model_type == 'convnext_femto':
        return ConvNeXt1D(in_ch, num_classes, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384])
    else:
        raise ValueError(f"Unknown model: {model_type}")

# -------------------------
# Training
# -------------------------
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(Xb)
            loss = criterion(logits, yb)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += float(loss.item()) * Xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return (running_loss / total) if total>0 else 0.0, (correct / total) if total>0 else 0.0

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(Xb)
                loss = criterion(logits, yb)
            
            running_loss += float(loss.item()) * Xb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    total = len(all_labels)
    return (running_loss / total) if total>0 else 0.0, np.array(all_preds), np.array(all_labels)

def train_and_evaluate_fold(X_train, y_train, X_val, y_val, X_test, y_test, fold_idx, model_type='efficientnet_b1'):
    result = {}
    X_tr, X_v, X_te = normalize_channelwise_stats(X_train, X_val, X_test)
    X_tr_t = torch.FloatTensor(X_tr); y_tr_t = torch.LongTensor(y_train)
    X_v_t = torch.FloatTensor(X_v); y_v_t = torch.LongTensor(y_val)
    X_te_t = torch.FloatTensor(X_te); y_te_t = torch.LongTensor(y_test)
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_v_t, y_v_t)
    test_ds = TensorDataset(X_te_t, y_te_t)
    
    if USE_WEIGHTED_SAMPLER:
        sampler = get_weighted_sampler_safe(y_train)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, 
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                 prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                 prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS)
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                            prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS)
    
    model = get_model(model_type, in_ch=X_tr.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    class_weights = compute_class_weights_safe(y_train) if USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    best_val_loss = float('inf')
    best_val_f1 = -float('inf')
    patience = 0
    best_state = None
    ckpt_path = None
    epoch_logs = []
    
    pbar = tqdm(range(NUM_EPOCHS), desc=f"Fold {fold_idx+1} {model_type}")
    for epoch in pbar:
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_preds, val_true = evaluate(model, val_loader, criterion)
        val_metrics = compute_comprehensive_metrics(val_true, val_preds)
        val_f1 = val_metrics['f1_macro']
        epoch_logs.append({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc, 
                          'val_loss': val_loss, 'val_f1_macro': val_f1})
        scheduler.step(val_loss)
        
        pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_f1': f'{val_f1:.4f}'})
        
        improved = False
        if EARLY_STOP_MODE == 'val_loss':
            if val_loss < best_val_loss - 1e-8:
                improved = True
                best_val_loss = val_loss
        elif EARLY_STOP_MODE == 'f1_macro':
            if val_f1 > best_val_f1 + 1e-8:
                improved = True
                best_val_f1 = val_f1
        else:
            if val_f1 > best_val_f1 + 1e-8:
                improved = True
                best_val_f1 = val_f1
                best_val_loss = val_loss
            elif abs(val_f1 - best_val_f1) < 1e-6 and val_loss < best_val_loss - 1e-8:
                improved = True
                best_val_loss = val_loss
        
        if improved:
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt_path = CHECKPOINT_DIR / f"best_{model_type}_fold{fold_idx+1}.pt"
            torch.save(best_state, ckpt_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                pbar.set_postfix({'status': 'early_stop'})
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, y_pred, y_true = evaluate(model, test_loader, criterion)
    metrics = compute_comprehensive_metrics(y_true, y_pred)
    
    preds_file = OUT_DIR / f"predictions_{model_type}_fold{fold_idx+1}.npz"
    np.savez_compressed(preds_file, y_true=y_true, y_pred=y_pred)
    
    plot_paths = {}
    plot_paths['training_curves'] = plot_training_curves(epoch_logs, fold_idx, model_type)
    plot_paths['confusion_matrix'] = plot_confusion_matrix(np.array(metrics['confusion_matrix']), fold_idx, model_type)
    plot_paths['per_class_metrics'] = plot_per_class_metrics(metrics, fold_idx, model_type)
    
    result.update({
        'model': model_type,
        'fold': fold_idx+1,
        'checkpoint': str(ckpt_path) if ckpt_path is not None else None,
        'epoch_logs': epoch_logs,
        'metrics': metrics,
        'test_loss': float(test_loss),
        'best_val_loss': float(best_val_loss),
        'best_val_f1': float(best_val_f1),
        'predictions_npz': str(preds_file),
        'plots': plot_paths
    })
    
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    cleanup_memory()
    return result

# -------------------------
# Main
# -------------------------
def main():
    t0 = time.time()
    print("="*80)
    print("SLEEP STAGE CLASSIFICATION - MODERN ARCHITECTURES")
    print("="*80)
    print_gpu_memory()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint = load_pipeline_checkpoint()
    if checkpoint:
        print(f"\n✓ Resuming from checkpoint...")
        all_results = checkpoint['all_results']
        completed_folds = checkpoint['completed_folds']
        file_pairs = checkpoint['file_pairs']
        n_subjects = len(file_pairs)
        models = checkpoint['models']
        splits = checkpoint['splits']
        start_fold = checkpoint['next_fold']
        print(f"✓ Resuming from fold {start_fold+1}/{n_subjects}")
    else:
        download_and_extract_gdrive(GDRIVE_FILE_URL, DATA_DIR)
        file_pairs = discover_valid_file_pairs(DATA_DIR, n_subjects_limit=N_SUBJECTS)
        n_subjects = len(file_pairs)
        
        if n_subjects < 3:
            raise RuntimeError("Need at least 3 valid subjects")
        
        print(f"\n✓ Using {n_subjects} subjects")
        models = ['efficientnet_b1', 'mobilenetv4_small', 'convnext_femto']
        all_results = {m: [] for m in models}
        completed_folds = set()
        
        splits = []
        for test_idx in range(n_subjects):
            val_idx = (test_idx + 1) % n_subjects
            train_idxs = [i for i in range(n_subjects) if i != test_idx and i != val_idx]
            splits.append((train_idxs, val_idx, test_idx))
        
        start_fold = 0
    
    print(f"\n{'='*80}")
    print(f"LOSO CROSS-VALIDATION: {n_subjects} folds")
    print(f"Models: {models}")
    print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: {USE_AMP}")
    print(f"Early Stop Mode: {EARLY_STOP_MODE}")
    print(f"{'='*80}\n")
    
    try:
        for fold_idx in range(start_fold, n_subjects):
            if fold_idx in completed_folds:
                continue
                
            train_idxs, val_idx, test_idx = splits[fold_idx]
            
            print(f"\n{'─'*80}")
            print(f"FOLD {fold_idx+1}/{n_subjects} | Test: Subject {test_idx+1} | Val: Subject {val_idx+1}")
            print(f"{'─'*80}")
            
            train_files = [file_pairs[i] for i in train_idxs]
            val_files = [file_pairs[val_idx]]
            test_files = [file_pairs[test_idx]]
            
            print("Loading data...")
            X_train, y_train, _ = process_subjects_leak_free(train_files)
            X_val, y_val, _ = process_subjects_leak_free(val_files)
            X_test, y_test, _ = process_subjects_leak_free(test_files)
            
            if X_train is None or X_val is None or X_test is None:
                print(f"✗ Skipping fold {fold_idx+1}")
                completed_folds.add(fold_idx)
                continue
            
            print(f"✓ Train: {X_train.shape[0]} epochs | Val: {X_val.shape[0]} epochs | Test: {X_test.shape[0]} epochs")
            print_gpu_memory()
            print()
            
            for model_name in models:
                res = train_and_evaluate_fold(X_train, y_train, X_val, y_val, X_test, y_test, fold_idx, model_type=model_name)
                all_results[model_name].append(res)
                print(f"  {model_name}: Acc={res['metrics']['accuracy']:.4f}, F1={res['metrics']['f1_macro']:.4f}, "
                      f"Kappa={res['metrics']['kappa']:.4f} (best_val_f1={res['best_val_f1']:.4f})")
            
            completed_folds.add(fold_idx)
            
            checkpoint_state = {
                'all_results': all_results,
                'completed_folds': completed_folds,
                'file_pairs': file_pairs,
                'models': models,
                'splits': splits,
                'next_fold': fold_idx + 1,
                'timestamp': timestamp
            }
            save_pipeline_checkpoint(checkpoint_state)
            
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print("✓ Progress saved in checkpoint. Run again to resume.")
        raise
    
    total_time = time.time() - t0
    
    report_path = OUT_DIR / f"comprehensive_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("SLEEP STAGE CLASSIFICATION - COMPREHENSIVE REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Hardware: A40 48GB | Batch: {BATCH_SIZE} | Workers: {NUM_WORKERS}\n")
        f.write(f"Early Stop Mode: {EARLY_STOP_MODE}\n")
        f.write(f"Subjects: {n_subjects}\n")
        f.write(f"Models: {models}\n")
        f.write(f"Total runtime: {total_time:.1f}s\n\n")
        for m in models:
            f.write("="*80 + "\n")
            f.write(f"MODEL: {m}\n")
            f.write("="*80 + "\n\n")
            for res in all_results.get(m, []):
                f.write(f"FOLD {res['fold']}\n")
                f.write("-"*40 + "\n")
                f.write(f"Best Val Loss: {res['best_val_loss']:.6f}\n")
                f.write(f"Best Val F1: {res['best_val_f1']:.4f}\n")
                f.write(f"Test Loss: {res['test_loss']:.6f}\n\n")
                for k, v in res['metrics'].items():
                    if k != 'classification_report':
                        f.write(f"{k}: {v}\n")
                f.write(f"\n{res['metrics']['classification_report']}\n")
                f.write(f"\nPredictions: {res['predictions_npz']}\n")
                f.write(f"Checkpoint: {res['checkpoint']}\n")
                f.write(f"Plots: {res['plots']}\n\n")
    
    rows = []
    for m in models:
        lst = all_results.get(m, [])
        if not lst:
            continue
        metrics_list = [r['metrics'] for r in lst]
        keys = ['accuracy','f1_macro','f1_micro','f1_weighted','precision_macro','precision_micro',
                'precision_weighted','recall_macro','recall_micro','recall_weighted','kappa']
        row = {'model': m}
        for k in keys:
            vals = [md.get(k, 0.0) for md in metrics_list]
            row[k+'_mean'] = float(np.mean(vals))
            row[k+'_std'] = float(np.std(vals))
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_csv = OUT_DIR / f"model_comparison_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    per_class_rows = []
    for m in models:
        lst = all_results.get(m, [])
        if not lst:
            continue
        for class_name in CLASS_NAMES:
            row = {'model': m, 'class': class_name}
            for metric in ['precision', 'recall', 'f1']:
                vals = [r['metrics'].get(f'{metric}_{class_name}', 0.0) for r in lst]
                row[f'{metric}_mean'] = float(np.mean(vals))
                row[f'{metric}_std'] = float(np.std(vals))
            per_class_rows.append(row)
    
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = OUT_DIR / f"per_class_metrics_{timestamp}.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    
    print(f"\n{'='*80}")
    print("COMPRESSING OUTPUTS")
    print(f"{'='*80}\n")
    
    archive_path = Path(f"sleep_classification_results_{timestamp}.7z")
    with py7zr.SevenZipFile(archive_path, 'w') as archive:
        for item in OUT_DIR.rglob('*'):
            if item.is_file():
                archive.write(item, arcname=f"output/{item.relative_to(OUT_DIR)}")
        for item in PLOT_DIR.rglob('*'):
            if item.is_file():
                archive.write(item, arcname=f"plots/{item.relative_to(PLOT_DIR)}")
        for item in CHECKPOINT_DIR.rglob('*'):
            if item.is_file():
                archive.write(item, arcname=f"checkpoints/{item.relative_to(CHECKPOINT_DIR)}")
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))
    print(f"\n✓ Report: {report_path}")
    print(f"✓ Summary: {summary_csv}")
    print(f"✓ Per-class metrics: {per_class_csv}")
    print(f"✓ Compressed archive: {archive_path}")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    print_gpu_memory()
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()