%pip install pandas numpy mne scipy scikit-learn torch imbalanced-learn gdown py7zr matplotlib seaborn tqdm -q
%pip install -q mamba-ssm==1.2.0.post1 --no-build-isolation

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

from mamba_ssm import Mamba

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
# hybrid
EARLY_STOP_MODE = 'f1_macro'

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
# Mamba Model Architecture (Using Official Implementation)
# -------------------------
class MambaBlock(nn.Module):
    """
    Wrapper around official Mamba layer with residual connection and normalization
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    
    def forward(self, x):
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        return x + self.mamba(self.norm(x))

class Mamba1D(nn.Module):
    """
    Complete Mamba model for 1D time series classification
    Uses official Mamba implementation from mamba-ssm package
    """
    def __init__(self, in_ch=2, num_classes=4, d_model=64, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        # Input embedding: project from multi-channel input to d_model
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand) 
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        x: (B, C, L) - batch, channels, length
        Returns: (B, num_classes)
        """
        # Input projection
        x = self.input_proj(x)  # (B, d_model, L')
        
        # Transpose for Mamba: (B, L', d_model)
        x = x.transpose(1, 2)
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Transpose back: (B, d_model, L')
        x = x.transpose(1, 2)
        
        # Global pooling
        x = self.avgpool(x)  # (B, d_model, 1)
        x = torch.flatten(x, 1)  # (B, d_model)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)
        
        return x

def get_model(model_type='mamba_small', in_ch=2, num_classes=4):
    """
    Factory function to create Mamba models with different configurations
    """
    configs = {
        'mamba_small': {
            'd_model': 64,
            'n_layers': 4,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2
        },
        'mamba_base': {
            'd_model': 128,
            'n_layers': 6,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2
        },
    }
    config = configs[model_type]
    return Mamba1D(in_ch=in_ch, num_classes=num_classes, **config)

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

def train_and_evaluate_fold(X_train, y_train, X_val, y_val, X_test, y_test, fold_idx, model_type='mamba_small'):
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
# Detailed Report Generation
# -------------------------
def write_detailed_report(all_results, models, n_subjects, total_time, timestamp):
    report_path = OUT_DIR / f"detailed_results_{timestamp}.txt"
    
    with open(report_path, "w") as f:
        f.write("="*100 + "\n")
        f.write("SLEEP STAGE CLASSIFICATION - COMPREHENSIVE DETAILED REPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Architecture: Official Mamba (Selective State Space Models)\n")
        f.write(f"Package: mamba-ssm==1.2.0.post1\n")
        f.write(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n" if torch.cuda.is_available() else "")
        f.write(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: {USE_AMP}\n")
        f.write(f"Early Stop Mode: {EARLY_STOP_MODE}\n")
        f.write(f"Number of Subjects (Folds): {n_subjects}\n")
        f.write(f"Models: {', '.join(models)}\n")
        f.write(f"Total Runtime: {total_time/60:.2f} minutes ({total_time:.1f} seconds)\n")
        f.write("="*100 + "\n\n")
        
        for model_name in models:
            f.write("\n" + "="*100 + "\n")
            f.write(f"MODEL: {model_name.upper()}\n")
            f.write("="*100 + "\n\n")
            
            results_list = all_results.get(model_name, [])
            
            if not results_list:
                f.write("No results available for this model.\n\n")
                continue
            
            for res in results_list:
                f.write("-"*100 + "\n")
                f.write(f"FOLD {res['fold']}\n")
                f.write("-"*100 + "\n\n")
                
                f.write("TRAINING INFO:\n")
                f.write(f"  Best Validation Loss: {res['best_val_loss']:.6f}\n")
                f.write(f"  Best Validation F1 Macro: {res['best_val_f1']:.4f}\n")
                f.write(f"  Final Test Loss: {res['test_loss']:.6f}\n")
                f.write(f"  Checkpoint: {res['checkpoint']}\n")
                f.write(f"  Predictions: {res['predictions_npz']}\n\n")
                
                f.write("TEST SET METRICS:\n")
                metrics = res['metrics']
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Cohen's Kappa: {metrics['kappa']:.4f}\n\n")
                
                f.write("  F1 Scores:\n")
                f.write(f"    Macro: {metrics['f1_macro']:.4f}\n")
                f.write(f"    Micro: {metrics['f1_micro']:.4f}\n")
                f.write(f"    Weighted: {metrics['f1_weighted']:.4f}\n\n")
                
                f.write("  Precision:\n")
                f.write(f"    Macro: {metrics['precision_macro']:.4f}\n")
                f.write(f"    Micro: {metrics['precision_micro']:.4f}\n")
                f.write(f"    Weighted: {metrics['precision_weighted']:.4f}\n\n")
                
                f.write("  Recall:\n")
                f.write(f"    Macro: {metrics['recall_macro']:.4f}\n")
                f.write(f"    Micro: {metrics['recall_micro']:.4f}\n")
                f.write(f"    Weighted: {metrics['recall_weighted']:.4f}\n\n")
                
                f.write("  Per-Class Metrics:\n")
                for class_name in CLASS_NAMES:
                    f.write(f"    {class_name}:\n")
                    f.write(f"      Precision: {metrics[f'precision_{class_name}']:.4f}\n")
                    f.write(f"      Recall: {metrics[f'recall_{class_name}']:.4f}\n")
                    f.write(f"      F1: {metrics[f'f1_{class_name}']:.4f}\n")
                f.write("\n")
                
                f.write("  Confusion Matrix:\n")
                cm = np.array(metrics['confusion_matrix'])
                f.write("    " + " ".join([f"{c:>10s}" for c in CLASS_NAMES]) + "\n")
                for i, row in enumerate(cm):
                    f.write(f"    {CLASS_NAMES[i]:>10s} " + " ".join([f"{val:10d}" for val in row]) + "\n")
                f.write("\n")
                
                f.write("  Classification Report:\n")
                for line in metrics['classification_report'].split('\n'):
                    f.write(f"    {line}\n")
                f.write("\n")
                
                f.write("  Plot Files:\n")
                for plot_name, plot_path in res['plots'].items():
                    f.write(f"    {plot_name}: {plot_path}\n")
                f.write("\n\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write(f"AGGREGATED RESULTS FOR {model_name.upper()}\n")
            f.write("="*100 + "\n\n")
            
            all_y_true = []
            all_y_pred = []
            
            for res in results_list:
                preds_data = np.load(res['predictions_npz'])
                all_y_true.extend(preds_data['y_true'])
                all_y_pred.extend(preds_data['y_pred'])
            
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            
            agg_metrics = compute_comprehensive_metrics(all_y_true, all_y_pred)
            
            f.write("OVERALL METRICS (All Folds Combined):\n")
            f.write(f"  Accuracy: {agg_metrics['accuracy']:.4f}\n")
            f.write(f"  Cohen's Kappa: {agg_metrics['kappa']:.4f}\n\n")
            
            f.write("  F1 Scores:\n")
            f.write(f"    Macro: {agg_metrics['f1_macro']:.4f}\n")
            f.write(f"    Micro: {agg_metrics['f1_micro']:.4f}\n")
            f.write(f"    Weighted: {agg_metrics['f1_weighted']:.4f}\n\n")
            
            f.write("  Precision:\n")
            f.write(f"    Macro: {agg_metrics['precision_macro']:.4f}\n")
            f.write(f"    Micro: {agg_metrics['precision_micro']:.4f}\n")
            f.write(f"    Weighted: {agg_metrics['precision_weighted']:.4f}\n\n")
            
            f.write("  Recall:\n")
            f.write(f"    Macro: {agg_metrics['recall_macro']:.4f}\n")
            f.write(f"    Micro: {agg_metrics['recall_micro']:.4f}\n")
            f.write(f"    Weighted: {agg_metrics['recall_weighted']:.4f}\n\n")
            
            f.write("  Per-Class Metrics:\n")
            for class_name in CLASS_NAMES:
                f.write(f"    {class_name}:\n")
                f.write(f"      Precision: {agg_metrics[f'precision_{class_name}']:.4f}\n")
                f.write(f"      Recall: {agg_metrics[f'recall_{class_name}']:.4f}\n")
                f.write(f"      F1: {agg_metrics[f'f1_{class_name}']:.4f}\n")
            f.write("\n")
            
            f.write("  Overall Confusion Matrix:\n")
            cm = np.array(agg_metrics['confusion_matrix'])
            f.write("    " + " ".join([f"{c:>10s}" for c in CLASS_NAMES]) + "\n")
            for i, row in enumerate(cm):
                f.write(f"    {CLASS_NAMES[i]:>10s} " + " ".join([f"{val:10d}" for val in row]) + "\n")
            f.write("\n")
            
            f.write("  Overall Classification Report:\n")
            for line in agg_metrics['classification_report'].split('\n'):
                f.write(f"    {line}\n")
            f.write("\n")
            
            f.write("MEAN AND STD ACROSS FOLDS:\n")
            metric_keys = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 
                          'precision_macro', 'precision_micro', 'precision_weighted',
                          'recall_macro', 'recall_micro', 'recall_weighted', 'kappa']
            
            for key in metric_keys:
                values = [r['metrics'][key] for r in results_list]
                f.write(f"  {key}:\n")
                f.write(f"    Mean: {np.mean(values):.4f}\n")
                f.write(f"    Std: {np.std(values):.4f}\n")
            f.write("\n")
            
            f.write("  Per-Class Mean and Std:\n")
            for class_name in CLASS_NAMES:
                f.write(f"    {class_name}:\n")
                for metric in ['precision', 'recall', 'f1']:
                    values = [r['metrics'][f'{metric}_{class_name}'] for r in results_list]
                    f.write(f"      {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
            f.write("\n")
    
    print(f"\n✓ Detailed report saved: {report_path}")
    return report_path

# -------------------------
# Main
# -------------------------
def main():
    t0 = time.time()
    print("="*80)
    print("SLEEP STAGE CLASSIFICATION - OFFICIAL MAMBA MODELS (A40 48GB VRAM)")
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
        models = ['mamba_small', 'mamba_base']
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
    print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: {USE_AMP}")
    print(f"Early Stop Mode: {EARLY_STOP_MODE}")
    print(f"Models: {models}")
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
    
    detailed_report_path = write_detailed_report(all_results, models, n_subjects, total_time, timestamp)
    
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
    
    archive_path = Path(f"sleep_classification_official_mamba_{timestamp}.7z")
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
    print(f"\n✓ Detailed report: {detailed_report_path}")
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