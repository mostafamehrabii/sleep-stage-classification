# %%
# %% [markdown]
# 📊 Sleep Stage Classification - Complete Enhanced ML Pipeline
# All original metrics + plots + comprehensive reporting + aggregated analysis

%pip install -q py7zr

# %%
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
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mne
mne.set_log_level('ERROR')

from scipy import signal
from scipy.stats import kurtosis, skew, entropy

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             cohen_kappa_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import xgboost as xgb
import lightgbm as lgb

import gdown
import py7zr

# -------------------------
# CONFIGURATION
# -------------------------
GDRIVE_FILE_URL = "https://drive.google.com/file/d/1SpxCWWAPWkyhfagPbvuSCCgNfEsCPKlg/view?usp=sharing"
DATA_DIR = Path("sleep_edf_data")
OUT_DIR = Path("output_ml")
PLOT_DIR = Path("plots_ml")
CHECKPOINT_DIR = Path("checkpoints_ml")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_SUBJECTS = 20
EPOCH_LEN = 30.0
RANDOM_STATE = 42
EARLY_STOP_MODE = 'hybrid'
PATIENCE = 10

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

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# -------------------------
# Utility Functions
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

# -------------------------
# Feature Extraction
# -------------------------
def get_feature_names(n_channels=2):
    """Get feature names that match the vectorized extraction."""
    feature_names = []
    
    for ch in range(n_channels):
        # Time domain
        feature_names.extend([
            f'ch{ch}_mean', f'ch{ch}_std', f'ch{ch}_var',
            f'ch{ch}_max', f'ch{ch}_min', f'ch{ch}_range', f'ch{ch}_rms',
            f'ch{ch}_p25', f'ch{ch}_p75', f'ch{ch}_iqr',
            f'ch{ch}_zero_crossing'
        ])
        
        # Frequency domain
        for band in ['delta', 'theta', 'alpha', 'beta']:
            feature_names.extend([
                f'ch{ch}_{band}_power',
                f'ch{ch}_{band}_rel_power'
            ])
        
        # Ratios
        feature_names.extend([
            f'ch{ch}_theta_beta_ratio',
            f'ch{ch}_delta_theta_ratio'
        ])
    
    # Cross-channel
    if n_channels > 1:
        feature_names.append('channel_correlation')
        for band in ['delta', 'theta', 'alpha', 'beta']:
            feature_names.append(f'{band}_power_diff')
    
    return feature_names

def extract_features_from_all_epochs(X, sfreq):
    """Extract features for ALL epochs using vectorized operations."""
    n_epochs, n_channels, n_samples = X.shape
    features_dict = {}
    
    for ch in range(n_channels):
        ch_data = X[:, ch, :]
        
        # Time domain features
        features_dict[f'ch{ch}_mean'] = np.mean(ch_data, axis=1)
        features_dict[f'ch{ch}_std'] = np.std(ch_data, axis=1)
        features_dict[f'ch{ch}_var'] = np.var(ch_data, axis=1)
        features_dict[f'ch{ch}_max'] = np.max(ch_data, axis=1)
        features_dict[f'ch{ch}_min'] = np.min(ch_data, axis=1)
        features_dict[f'ch{ch}_range'] = features_dict[f'ch{ch}_max'] - features_dict[f'ch{ch}_min']
        features_dict[f'ch{ch}_rms'] = np.sqrt(np.mean(ch_data**2, axis=1))
        
        # Percentiles
        features_dict[f'ch{ch}_p25'] = np.percentile(ch_data, 25, axis=1)
        features_dict[f'ch{ch}_p75'] = np.percentile(ch_data, 75, axis=1)
        features_dict[f'ch{ch}_iqr'] = features_dict[f'ch{ch}_p75'] - features_dict[f'ch{ch}_p25']
        
        # Zero crossings
        zero_cross = np.sum(np.diff(np.sign(ch_data), axis=1) != 0, axis=1)
        features_dict[f'ch{ch}_zero_crossing'] = zero_cross / n_samples
        
        # FFT-based frequency features
        fft_vals = np.fft.rfft(ch_data, axis=1)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n_samples, 1/sfreq)
        
        # Band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        total_power = np.sum(psd, axis=1, keepdims=True)
        
        for band_name, (low, high) in bands.items():
            band_idx = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[:, band_idx], axis=1)
            features_dict[f'ch{ch}_{band_name}_power'] = band_power
            features_dict[f'ch{ch}_{band_name}_rel_power'] = band_power / (total_power.squeeze() + 1e-10)
        
        # Band ratios
        features_dict[f'ch{ch}_theta_beta_ratio'] = (features_dict[f'ch{ch}_theta_power'] / 
                                                       (features_dict[f'ch{ch}_beta_power'] + 1e-10))
        features_dict[f'ch{ch}_delta_theta_ratio'] = (features_dict[f'ch{ch}_delta_power'] / 
                                                        (features_dict[f'ch{ch}_theta_power'] + 1e-10))
    
    # Cross-channel features
    if n_channels > 1:
        ch0_data = X[:, 0, :]
        ch1_data = X[:, 1, :]
        
        ch0_mean = np.mean(ch0_data, axis=1, keepdims=True)
        ch1_mean = np.mean(ch1_data, axis=1, keepdims=True)
        
        numerator = np.sum((ch0_data - ch0_mean) * (ch1_data - ch1_mean), axis=1)
        denominator = np.sqrt(np.sum((ch0_data - ch0_mean)**2, axis=1) * 
                             np.sum((ch1_data - ch1_mean)**2, axis=1))
        
        features_dict['channel_correlation'] = numerator / (denominator + 1e-10)
        
        # Power differences
        for band in ['delta', 'theta', 'alpha', 'beta']:
            features_dict[f'{band}_power_diff'] = (features_dict[f'ch0_{band}_power'] - 
                                                    features_dict[f'ch1_{band}_power'])
    
    df = pd.DataFrame(features_dict)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    return df.values

def extract_and_cache_all_subject_features(file_pairs, cache_file='cached_features.pkl'):
    """Extract features once for all subjects and cache them."""
    cache_path = CHECKPOINT_DIR / cache_file
    
    if cache_path.exists():
        print(f"✓ Loading cached features from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data
    
    print("\n" + "="*80)
    print("EXTRACTING FEATURES FOR ALL SUBJECTS")
    print("="*80)
    
    all_features = []
    all_labels = []
    subject_ids = []
    sfreq = 100.0
    
    for subj_idx, (psg, hyp) in enumerate(file_pairs):
        print(f"\nSubject {subj_idx+1}/{len(file_pairs)}:")
        X, y, sf = load_and_extract_epochs(psg, hyp)
        if X is None:
            print(f"  ✗ Skipped (no data)")
            continue
        
        if sf is not None:
            sfreq = sf
        
        features = extract_features_from_all_epochs(X, sfreq)
        
        all_features.append(features)
        all_labels.append(y)
        subject_ids.append(np.full(len(y), subj_idx))
        print(f"  ✓ {X.shape[0]} epochs, {features.shape[1]} features")
    
    n_channels = 2
    feature_names = get_feature_names(n_channels=n_channels)
    
    # Print generated features
    print(f"\n{'='*80}")
    print("GENERATED FEATURES")
    print(f"{'='*80}")
    print(f"Total features: {len(feature_names)}\n")
    
    # Group by category
    time_domain = [f for f in feature_names if any(x in f for x in ['mean', 'std', 'var', 'max', 'min', 'range', 'rms', 'p25', 'p75', 'iqr', 'zero_crossing'])]
    freq_domain = [f for f in feature_names if any(x in f for x in ['power', 'ratio'])]
    cross_channel = [f for f in feature_names if any(x in f for x in ['correlation', 'diff'])]
    
    print(f"Time Domain ({len(time_domain)}):")
    for f in time_domain:
        print(f"  - {f}")
    
    print(f"\nFrequency Domain ({len(freq_domain)}):")
    for f in freq_domain:
        print(f"  - {f}")
    
    print(f"\nCross-Channel ({len(cross_channel)}):")
    for f in cross_channel:
        print(f"  - {f}")
    
    cached_data = {
        'features': all_features,
        'labels': all_labels,
        'subject_ids': subject_ids,
        'feature_names': feature_names,
        'sfreq': sfreq
    }
    
    print(f"\n✓ Caching features to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_data, f)
    
    return cached_data

# -------------------------
# Feature Selection
# -------------------------
def select_features_comprehensive(X_train, y_train, X_val, X_test, feature_names, n_features=50):
    """Comprehensive feature selection."""
    selection_stats = {}
    
    # 1. Variance threshold
    var_threshold = VarianceThreshold(threshold=0.01)
    var_threshold.fit(X_train)
    var_mask = var_threshold.get_support()
    selection_stats['variance'] = var_mask
    
    # 2. Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    mi_scores = np.nan_to_num(mi_scores)
    selection_stats['mutual_info'] = mi_scores
    
    # 3. F-statistic (ANOVA)
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores)
    selection_stats['f_statistic'] = f_scores
    
    # 4. Correlation filtering
    corr_matrix = np.corrcoef(X_train.T)
    corr_matrix = np.nan_to_num(corr_matrix)
    
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > 0.95:
                high_corr_pairs.append((i, j))
    
    corr_remove = set()
    for i, j in high_corr_pairs:
        if mi_scores[i] < mi_scores[j]:
            corr_remove.add(i)
        else:
            corr_remove.add(j)
    
    corr_mask = np.ones(len(feature_names), dtype=bool)
    corr_mask[list(corr_remove)] = False
    selection_stats['correlation'] = corr_mask
    
    # 5. Combined selection
    mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
    f_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
    
    combined_score = (mi_norm + f_norm) / 2.0
    combined_score[~var_mask] = 0
    combined_score[~corr_mask] = 0
    
    selection_stats['combined_score'] = combined_score
    
    # Select top N features
    top_indices = np.argsort(combined_score)[::-1][:n_features]
    final_mask = np.zeros(len(feature_names), dtype=bool)
    final_mask[top_indices] = True
    
    selection_stats['final_mask'] = final_mask
    selection_stats['selected_features'] = [feature_names[i] for i in top_indices]
    selection_stats['selected_indices'] = top_indices
    
    # Apply selection
    X_train_selected = X_train[:, final_mask]
    X_val_selected = X_val[:, final_mask]
    X_test_selected = X_test[:, final_mask]
    
    return X_train_selected, X_val_selected, X_test_selected, selection_stats

def plot_feature_selection_analysis(selection_stats, fold_idx, feature_names):
    """Plot comprehensive feature selection analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Mutual Information scores
    ax1 = fig.add_subplot(gs[0, 0])
    mi_scores = selection_stats['mutual_info']
    top_mi_idx = np.argsort(mi_scores)[::-1][:20]
    ax1.barh(range(len(top_mi_idx)), mi_scores[top_mi_idx], color='skyblue')
    ax1.set_yticks(range(len(top_mi_idx)))
    ax1.set_yticklabels([feature_names[i][:25] for i in top_mi_idx], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mutual Information Score')
    ax1.set_title('Top 20 Features by Mutual Information')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. F-statistic scores
    ax2 = fig.add_subplot(gs[0, 1])
    f_scores = selection_stats['f_statistic']
    top_f_idx = np.argsort(f_scores)[::-1][:20]
    ax2.barh(range(len(top_f_idx)), f_scores[top_f_idx], color='lightcoral')
    ax2.set_yticks(range(len(top_f_idx)))
    ax2.set_yticklabels([feature_names[i][:25] for i in top_f_idx], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('F-statistic Score')
    ax2.set_title('Top 20 Features by ANOVA F-test')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Combined scores
    ax3 = fig.add_subplot(gs[0, 2])
    combined = selection_stats['combined_score']
    top_combined_idx = np.argsort(combined)[::-1][:20]
    ax3.barh(range(len(top_combined_idx)), combined[top_combined_idx], color='lightgreen')
    ax3.set_yticks(range(len(top_combined_idx)))
    ax3.set_yticklabels([feature_names[i][:25] for i in top_combined_idx], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('Combined Score')
    ax3.set_title('Top 20 Features by Combined Score')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Feature selection overview
    ax4 = fig.add_subplot(gs[1, :])
    var_mask = selection_stats['variance']
    corr_mask = selection_stats['correlation']
    final_mask = selection_stats['final_mask']
    
    categories = ['All Features', 'After Variance\nThreshold', 
                  'After Correlation\nFiltering', 'Final Selected']
    counts = [len(feature_names), 
              np.sum(var_mask),
              np.sum(var_mask & corr_mask),
              np.sum(final_mask)]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax4.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Number of Features', fontsize=12)
    ax4.set_title('Feature Selection Pipeline', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 5. Score distributions
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(mi_scores[mi_scores > 0], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax5.axvline(np.median(mi_scores[mi_scores > 0]), color='red', linestyle='--', 
                linewidth=2, label='Median')
    ax5.set_xlabel('Mutual Information Score')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of MI Scores')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(f_scores[f_scores > 0], bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
    ax6.axvline(np.median(f_scores[f_scores > 0]), color='red', linestyle='--', 
                linewidth=2, label='Median')
    ax6.set_xlabel('F-statistic Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of F-scores')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. Selected vs Rejected features
    ax7 = fig.add_subplot(gs[2, 2])
    selected_scores = combined[final_mask]
    rejected_scores = combined[~final_mask]
    
    ax7.boxplot([rejected_scores, selected_scores], labels=['Rejected', 'Selected'],
                patch_artist=True,
                boxprops=dict(facecolor='lightgray', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax7.set_ylabel('Combined Score')
    ax7.set_title('Selected vs Rejected Features')
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Feature Selection Analysis - Fold {fold_idx+1}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = PLOT_DIR / f"feature_selection_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def save_feature_selection_details(selection_stats, fold_idx, feature_names):
    """Save detailed feature selection info to text file."""
    txt_path = OUT_DIR / f"feature_selection_details_fold{fold_idx+1}.txt"
    
    with open(txt_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"FEATURE SELECTION DETAILS - FOLD {fold_idx+1}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Original Features: {len(feature_names)}\n")
        f.write(f"After Variance Threshold: {np.sum(selection_stats['variance'])}\n")
        f.write(f"After Correlation Filtering: {np.sum(selection_stats['correlation'])}\n")
        f.write(f"Final Selected: {len(selection_stats['selected_features'])}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write("SELECTED FEATURES\n")
        f.write(f"{'='*80}\n\n")
        
        mi_scores = selection_stats['mutual_info']
        f_scores = selection_stats['f_statistic']
        combined = selection_stats['combined_score']
        
        for i, feat_name in enumerate(selection_stats['selected_features']):
            feat_idx = selection_stats['selected_indices'][i]
            f.write(f"{i+1}. {feat_name}\n")
            f.write(f"   MI Score: {mi_scores[feat_idx]:.6f}\n")
            f.write(f"   F-Score: {f_scores[feat_idx]:.6f}\n")
            f.write(f"   Combined: {combined[feat_idx]:.6f}\n\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("TOP 20 BY MUTUAL INFORMATION\n")
        f.write(f"{'='*80}\n\n")
        
        top_mi = np.argsort(mi_scores)[::-1][:20]
        for i, idx in enumerate(top_mi):
            f.write(f"{i+1}. {feature_names[idx]}: {mi_scores[idx]:.6f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("TOP 20 BY F-STATISTIC\n")
        f.write(f"{'='*80}\n\n")
        
        top_f = np.argsort(f_scores)[::-1][:20]
        for i, idx in enumerate(top_f):
            f.write(f"{i+1}. {feature_names[idx]}: {f_scores[idx]:.6f}\n")
    
    return txt_path

# -------------------------
# Model Training
# -------------------------
def get_model_with_early_stopping(model_name, n_classes):
    """Initialize model with configuration."""
    
    if model_name == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=n_classes,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=PATIENCE,
            eval_metric='mlogloss'
        )
        return model, True
    
    elif model_name == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multiclass',
            num_class=n_classes,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        return model, True
    
    elif model_name == 'randomforest':
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        return model, False
    
    elif model_name == 'extratrees':
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        return model, False

def compute_class_weights_for_samples(y_train, n_classes):
    """Compute sample weights for each training sample."""
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = len(y_train) / (n_classes * class_counts)
    sample_weights = class_weights[y_train]
    return sample_weights

def train_with_early_stopping(model, X_train, y_train, X_val, y_val, model_name, has_early_stopping):
    """Train model with early stopping."""
    
    eval_results = {}
    n_classes = len(CLASS_NAMES)
    
    if has_early_stopping:
        if model_name == 'xgboost':
            train_weights = compute_class_weights_for_samples(y_train, n_classes)
            val_weights = compute_class_weights_for_samples(y_val, n_classes)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                sample_weight=train_weights,
                sample_weight_eval_set=[train_weights, val_weights],
                verbose=False
            )
            eval_results = model.evals_result()
            
        elif model_name == 'lightgbm':
            train_weights = compute_class_weights_for_samples(y_train, n_classes)
            val_weights = compute_class_weights_for_samples(y_val, n_classes)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric='multi_logloss',
                sample_weight=train_weights,
                eval_sample_weight=[train_weights, val_weights],
                callbacks=[lgb.early_stopping(stopping_rounds=PATIENCE, verbose=False)]
            )
            eval_results = {'validation': model.evals_result_['valid_1']['multi_logloss']}
            if 'training' in model.evals_result_:
                eval_results['train'] = model.evals_result_['training']['multi_logloss']
        
        return model, eval_results
    
    else:
        model.fit(X_train, y_train)
        return model, {}

# -------------------------
# Metrics & Plotting
# -------------------------
def compute_comprehensive_metrics(y_true, y_pred):
    """Compute ALL evaluation metrics."""
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

def plot_confusion_matrix(cm, fold_idx, model_type):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_type} Fold {fold_idx+1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = PLOT_DIR / f"confusion_matrix_{model_type}_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def plot_per_class_metrics(metrics, fold_idx, model_type):
    """Plot per-class precision, recall, F1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    classes = CLASS_NAMES
    precision = [metrics[f'precision_{c}'] for c in classes]
    recall = [metrics[f'recall_{c}'] for c in classes]
    f1 = [metrics[f'f1_{c}'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.6
    
    axes[0].bar(x, precision, width, color='skyblue', edgecolor='black')
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Precision per Class', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, recall, width, color='lightcoral', edgecolor='black')
    axes[1].set_ylabel('Score', fontsize=11)
    axes[1].set_title('Recall per Class', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, f1, width, color='lightgreen', edgecolor='black')
    axes[2].set_ylabel('Score', fontsize=11)
    axes[2].set_title('F1 per Class', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(classes)
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Per-Class Metrics - {model_type} Fold {fold_idx+1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = PLOT_DIR / f"per_class_metrics_{model_type}_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def plot_training_curves_ml(eval_results, fold_idx, model_type):
    """Plot training curves for XGBoost/LightGBM."""
    if not eval_results or 'validation' not in eval_results:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = list(range(len(eval_results['validation'])))
    val_metric = eval_results['validation']
    
    axes[0].plot(epochs, val_metric, 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if 'train' in eval_results:
        train_metric = eval_results['train']
        axes[1].plot(epochs, train_metric, 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, val_metric, 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Iteration', fontsize=11)
        axes[1].set_ylabel('Loss', fontsize=11)
        axes[1].set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.suptitle(f'Training Curves - {model_type} Fold {fold_idx+1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = PLOT_DIR / f"training_curves_{model_type}_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def plot_feature_importance(model, model_name, fold_idx, feature_names, top_n=30):
    """Plot comprehensive feature importance analysis."""
    if not hasattr(model, 'feature_importances_'):
        return None, None
    
    importances = model.feature_importances_
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Top N features bar plot
    ax1 = fig.add_subplot(gs[0, :])
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i][:30] for i in indices]
    top_importances = importances[indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importances)))
    bars = ax1.barh(range(len(top_importances)), top_importances, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance Score', fontsize=12)
    ax1.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        ax1.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    # 2. Feature importance distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(importances, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(importances), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(np.median(importances), color='green', linestyle='--', linewidth=2, label='Median')
    ax2.set_xlabel('Importance Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Feature Importances', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative importance
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_importances = np.sort(importances)[::-1]
    cumsum = np.cumsum(sorted_importances)
    cumsum_pct = 100 * cumsum / cumsum[-1]
    
    ax3.plot(range(1, len(cumsum_pct)+1), cumsum_pct, linewidth=2, color='#e74c3c')
    ax3.axhline(90, color='green', linestyle='--', linewidth=2, label='90%')
    ax3.axhline(95, color='orange', linestyle='--', linewidth=2, label='95%')
    ax3.axhline(99, color='red', linestyle='--', linewidth=2, label='99%')
    
    n_90 = np.argmax(cumsum_pct >= 90) + 1
    n_95 = np.argmax(cumsum_pct >= 95) + 1
    n_99 = np.argmax(cumsum_pct >= 99) + 1
    
    ax3.scatter([n_90, n_95, n_99], [90, 95, 99], s=100, c=['green', 'orange', 'red'], 
                zorder=5, edgecolors='black', linewidth=2)
    ax3.text(n_90, 85, f'{n_90} features', ha='center', fontsize=9, fontweight='bold')
    ax3.text(n_95, 91, f'{n_95} features', ha='center', fontsize=9, fontweight='bold')
    ax3.text(n_99, 97, f'{n_99} features', ha='center', fontsize=9, fontweight='bold')
    
    ax3.set_xlabel('Number of Features', fontsize=11)
    ax3.set_ylabel('Cumulative Importance (%)', fontsize=11)
    ax3.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, len(importances)])
    ax3.set_ylim([0, 105])
    
    plt.suptitle(f'Feature Importance Analysis - {model_name} Fold {fold_idx+1}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plot_path = PLOT_DIR / f"feature_importance_{model_name}_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    importance_data = {
        'feature_names': feature_names,
        'importances': importances,
        'top_indices': indices,
        'top_features': top_features,
        'top_importances': top_importances
    }
    
    return str(plot_path), importance_data

def plot_feature_importance_comparison(all_importances, model_names, feature_names, fold_idx, top_n=20):
    """Compare feature importances across multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # 1. Heatmap of top features across models
    all_top_features = set()
    for model_name, importances in all_importances.items():
        if importances is not None:
            top_idx = np.argsort(importances)[::-1][:top_n]
            all_top_features.update(top_idx)
    
    all_top_features = sorted(list(all_top_features))
    top_feature_names = [feature_names[i][:30] for i in all_top_features]
    
    importance_matrix = np.zeros((len(all_top_features), len(model_names)))
    for j, model_name in enumerate(model_names):
        if all_importances.get(model_name) is not None:
            for i, feat_idx in enumerate(all_top_features):
                importance_matrix[i, j] = all_importances[model_name][feat_idx]
    
    importance_matrix_norm = importance_matrix / (importance_matrix.max(axis=1, keepdims=True) + 1e-10)
    
    ax1 = axes[0]
    im = ax1.imshow(importance_matrix_norm, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax1.set_yticks(range(len(top_feature_names)))
    ax1.set_yticklabels(top_feature_names, fontsize=8)
    ax1.set_title(f'Feature Importance Heatmap (Normalized)\nTop Features Across Models', 
                  fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Normalized Importance')
    
    # 2. Ranking consistency
    ax2 = axes[1]
    ranks_matrix = np.zeros((len(feature_names), len(model_names)))
    for j, model_name in enumerate(model_names):
        if all_importances.get(model_name) is not None:
            ranks = np.argsort(np.argsort(all_importances[model_name])[::-1])
            ranks_matrix[:, j] = ranks
    
    avg_rank = ranks_matrix.mean(axis=1)
    std_rank = ranks_matrix.std(axis=1)
    
    consistency_score = avg_rank + std_rank
    top_consistent_idx = np.argsort(consistency_score)[:top_n]
    
    consistent_features = [feature_names[i][:30] for i in top_consistent_idx]
    consistent_ranks = avg_rank[top_consistent_idx]
    consistent_stds = std_rank[top_consistent_idx]
    
    y_pos = np.arange(len(consistent_features))
    ax2.barh(y_pos, consistent_ranks, xerr=consistent_stds, 
             color='skyblue', alpha=0.7, edgecolor='black', error_kw={'linewidth': 2})
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(consistent_features, fontsize=9)
    ax2.invert_yaxis()
    ax2.invert_xaxis()
    ax2.set_xlabel('Average Rank (lower is better)', fontsize=11)
    ax2.set_title(f'Most Consistent Important Features\n(Low Rank ± Low Std)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Feature Importance Comparison Across Models - Fold {fold_idx+1}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = PLOT_DIR / f"feature_importance_comparison_fold{fold_idx+1}.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def save_feature_importance_details(importance_data, model_name, fold_idx):
    """Save feature importance details to text file."""
    txt_path = OUT_DIR / f"feature_importance_{model_name}_fold{fold_idx+1}.txt"
    
    with open(txt_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"FEATURE IMPORTANCE - {model_name.upper()} FOLD {fold_idx+1}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Total Features: {len(importance_data['importances'])}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write("ALL FEATURES RANKED BY IMPORTANCE\n")
        f.write(f"{'='*80}\n\n")
        
        sorted_idx = np.argsort(importance_data['importances'])[::-1]
        for i, idx in enumerate(sorted_idx):
            f.write(f"{i+1}. {importance_data['feature_names'][idx]}: {importance_data['importances'][idx]:.6f}\n")
    
    return txt_path

def save_fold_metrics_details(metrics, model_name, fold_idx):
    """Save complete metrics for fold to text file."""
    txt_path = OUT_DIR / f"metrics_{model_name}_fold{fold_idx+1}.txt"
    
    with open(txt_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"COMPLETE METRICS - {model_name.upper()} FOLD {fold_idx+1}\n")
        f.write(f"{'='*80}\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 40 + "\n")
        metric_keys = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted',
                      'precision_macro', 'precision_micro', 'precision_weighted',
                      'recall_macro', 'recall_micro', 'recall_weighted', 'kappa']
        for key in metric_keys:
            f.write(f"{key}: {metrics[key]:.6f}\n")
        
        # Per-class metrics
        f.write("\n\nPER-CLASS METRICS:\n")
        f.write("-" * 40 + "\n")
        for class_name in CLASS_NAMES:
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics[f'precision_{class_name}']:.6f}\n")
            f.write(f"  Recall: {metrics[f'recall_{class_name}']:.6f}\n")
            f.write(f"  F1-Score: {metrics[f'f1_{class_name}']:.6f}\n")
        
        # Confusion matrix
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        cm = np.array(metrics['confusion_matrix'])
        f.write("       " + "  ".join([f"{c:>8}" for c in CLASS_NAMES]) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{CLASS_NAMES[i]:>8} " + "  ".join([f"{val:>8d}" for val in row]) + "\n")
        
        # Classification report
        f.write("\n\nCLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(metrics['classification_report'])
    
    return txt_path

# -------------------------
# Aggregated Analysis
# -------------------------
def create_aggregated_confusion_matrix(all_results, model_name):
    """Create aggregated confusion matrix across all folds."""
    cm_sum = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    
    for res in all_results[model_name]:
        cm_sum += np.array(res['metrics']['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_sum, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Aggregated Confusion Matrix - {model_name}\n(All Folds Combined)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = PLOT_DIR / f"confusion_matrix_{model_name}_aggregated.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    return str(plot_path), cm_sum

def create_aggregated_per_class_metrics(all_results, model_name):
    """Create aggregated per-class metrics plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    classes = CLASS_NAMES
    
    # Collect all metrics
    precision_all = {c: [] for c in classes}
    recall_all = {c: [] for c in classes}
    f1_all = {c: [] for c in classes}
    
    for res in all_results[model_name]:
        for c in classes:
            precision_all[c].append(res['metrics'][f'precision_{c}'])
            recall_all[c].append(res['metrics'][f'recall_{c}'])
            f1_all[c].append(res['metrics'][f'f1_{c}'])
    
    # Average
    precision_avg = [np.mean(precision_all[c]) for c in classes]
    precision_std = [np.std(precision_all[c]) for c in classes]
    recall_avg = [np.mean(recall_all[c]) for c in classes]
    recall_std = [np.std(recall_all[c]) for c in classes]
    f1_avg = [np.mean(f1_all[c]) for c in classes]
    f1_std = [np.std(f1_all[c]) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.6
    
    axes[0].bar(x, precision_avg, width, yerr=precision_std, color='skyblue', 
                edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Precision per Class (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, recall_avg, width, yerr=recall_std, color='lightcoral', 
                edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    axes[1].set_ylabel('Score', fontsize=11)
    axes[1].set_title('Recall per Class (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, f1_avg, width, yerr=f1_std, color='lightgreen', 
                edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    axes[2].set_ylabel('Score', fontsize=11)
    axes[2].set_title('F1 per Class (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(classes)
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Aggregated Per-Class Metrics - {model_name}\n(Averaged Across All Folds)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = PLOT_DIR / f"per_class_metrics_{model_name}_aggregated.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)

def create_aggregated_feature_importance(all_results, model_name):
    """Create aggregated feature importance across all folds."""
    importance_arrays = []
    feature_names = None
    
    for res in all_results[model_name]:
        if 'importance_data' in res and res['importance_data'] is not None:
            importance_arrays.append(res['importance_data']['importances'])
            if feature_names is None:
                feature_names = res['importance_data']['feature_names']
    
    if len(importance_arrays) == 0:
        return None, None
    
    avg_importance = np.mean(importance_arrays, axis=0)
    std_importance = np.std(importance_arrays, axis=0)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top 30 features
    top_n = 30
    indices = np.argsort(avg_importance)[::-1][:top_n]
    top_features = [feature_names[i][:30] for i in indices]
    top_avg = avg_importance[indices]
    top_std = std_importance[indices]
    
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_avg)))
    bars = ax1.barh(range(len(top_avg)), top_avg, xerr=top_std, color=colors, 
                    edgecolor='black', linewidth=0.5, error_kw={'linewidth': 2})
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Importance (± Std)', fontsize=12)
    ax1.set_title(f'Top {top_n} Features (Averaged Across Folds)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(avg_importance, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(avg_importance), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(np.median(avg_importance), color='green', linestyle='--', linewidth=2, label='Median')
    ax2.set_xlabel('Average Importance', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Average Importance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_avg = np.sort(avg_importance)[::-1]
    cumsum = np.cumsum(sorted_avg)
    cumsum_pct = 100 * cumsum / cumsum[-1]
    
    ax3.plot(range(1, len(cumsum_pct)+1), cumsum_pct, linewidth=2, color='#e74c3c')
    ax3.axhline(90, color='green', linestyle='--', linewidth=2, label='90%')
    ax3.axhline(95, color='orange', linestyle='--', linewidth=2, label='95%')
    
    ax3.set_xlabel('Number of Features', fontsize=11)
    ax3.set_ylabel('Cumulative Importance (%)', fontsize=11)
    ax3.set_title('Cumulative Average Importance', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Aggregated Feature Importance - {model_name}\n(Averaged Across All Folds)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = PLOT_DIR / f"feature_importance_{model_name}_aggregated.png"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    return str(plot_path), {'avg_importance': avg_importance, 'std_importance': std_importance, 
                           'feature_names': feature_names, 'top_indices': indices}

def save_aggregated_details(all_results, models, timestamp):
    """Save comprehensive aggregated details to text file."""
    txt_path = OUT_DIR / f"aggregated_analysis_{timestamp}.txt"
    
    with open(txt_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write("COMPREHENSIVE AGGREGATED ANALYSIS - ALL FOLDS\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total Folds: {len(all_results[models[0]]) if models else 0}\n\n")
        
        for model_name in models:
            f.write(f"\n{'='*80}\n")
            f.write(f"MODEL: {model_name.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            results = all_results[model_name]
            
            # ALL metrics
            metric_keys = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted',
                          'precision_macro', 'precision_micro', 'precision_weighted',
                          'recall_macro', 'recall_micro', 'recall_weighted', 'kappa']
            
            f.write("AGGREGATED METRICS (Mean ± Std):\n")
            f.write("-" * 40 + "\n")
            for key in metric_keys:
                values = [r['metrics'][key] for r in results]
                f.write(f"{key}: {np.mean(values):.6f} ± {np.std(values):.6f}\n")
            
            # Per-class metrics
            f.write("\n\nPER-CLASS METRICS (Mean ± Std):\n")
            f.write("-" * 40 + "\n")
            for class_name in CLASS_NAMES:
                f.write(f"\n{class_name}:\n")
                for metric in ['precision', 'recall', 'f1']:
                    values = [r['metrics'][f'{metric}_{class_name}'] for r in results]
                    f.write(f"  {metric}: {np.mean(values):.6f} ± {np.std(values):.6f}\n")
            
            # Aggregated confusion matrix
            f.write(f"\n\nAGGREGATED CONFUSION MATRIX (Sum of All Folds):\n")
            f.write("-" * 40 + "\n")
            cm_sum = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
            for res in results:
                cm_sum += np.array(res['metrics']['confusion_matrix'])
            
            f.write("         " + "  ".join([f"{c:>10}" for c in CLASS_NAMES]) + "\n")
            for i, row in enumerate(cm_sum):
                f.write(f"{CLASS_NAMES[i]:>10} " + "  ".join([f"{val:>10d}" for val in row]) + "\n")
            
            # Feature importance
            importance_arrays = []
            feature_names = None
            for res in results:
                if 'importance_data' in res and res['importance_data'] is not None:
                    importance_arrays.append(res['importance_data']['importances'])
                    if feature_names is None:
                        feature_names = res['importance_data']['feature_names']
            
            if len(importance_arrays) > 0:
                avg_importance = np.mean(importance_arrays, axis=0)
                std_importance = np.std(importance_arrays, axis=0)
                
                f.write(f"\n\nTOP 30 MOST IMPORTANT FEATURES (Mean ± Std):\n")
                f.write("-" * 40 + "\n")
                
                top_indices = np.argsort(avg_importance)[::-1][:30]
                for i, idx in enumerate(top_indices):
                    f.write(f"{i+1}. {feature_names[idx]}: {avg_importance[idx]:.6f} ± {std_importance[idx]:.6f}\n")
    
    return txt_path

# -------------------------
# Main Training Function
# -------------------------
def train_and_evaluate_fold(train_indices, val_index, test_index, all_features, all_labels, 
                           fold_idx, model_name, feature_names):
    """Train and evaluate a single fold."""
    result = {}
    
    X_train = np.vstack([all_features[i] for i in train_indices])
    y_train = np.hstack([all_labels[i] for i in train_indices])
    
    X_val = all_features[val_index]
    y_val = all_labels[val_index]
    
    X_test = all_features[test_index]
    y_test = all_labels[test_index]
    
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature Selection
    print(f"  Selecting features...")
    n_features_to_select = min(50, X_train_scaled.shape[1] // 2)
    X_train_selected, X_val_selected, X_test_selected, selection_stats = select_features_comprehensive(
        X_train_scaled, y_train, X_val_scaled, X_test_scaled, 
        feature_names, n_features=n_features_to_select
    )
    
    selected_feature_names = selection_stats['selected_features']
    
    # Print selected features (only for first model to avoid spam)
    if model_name == 'xgboost':
        print(f"\n  SELECTED FEATURES ({len(selected_feature_names)}):")
        for i, feat in enumerate(selected_feature_names[:20]):
            print(f"    {i+1}. {feat}")
        if len(selected_feature_names) > 20:
            print(f"    ... and {len(selected_feature_names)-20} more")
        
        # Save feature selection details and plot
        selection_txt = save_feature_selection_details(selection_stats, fold_idx, feature_names)
        selection_plot = plot_feature_selection_analysis(selection_stats, fold_idx, feature_names)
    
    # Get model
    model, has_early_stopping = get_model_with_early_stopping(model_name, len(CLASS_NAMES))
    
    # Train
    print(f"  Training {model_name}...")
    model, eval_results = train_with_early_stopping(
        model, X_train_selected, y_train, X_val_selected, y_val,
        model_name, has_early_stopping
    )
    
    # Predict
    y_pred = model.predict(X_test_selected)
    metrics = compute_comprehensive_metrics(y_test, y_pred)
    
    # Save metrics details
    metrics_txt = save_fold_metrics_details(metrics, model_name, fold_idx)
    
    # Generate all plots
    plot_paths = {}
    plot_paths['confusion_matrix'] = plot_confusion_matrix(
        np.array(metrics['confusion_matrix']), fold_idx, model_name
    )
    plot_paths['per_class_metrics'] = plot_per_class_metrics(metrics, fold_idx, model_name)
    
    # Feature importance
    importance_plot, importance_data = plot_feature_importance(
        model, model_name, fold_idx, selected_feature_names
    )
    if importance_plot:
        plot_paths['feature_importance'] = importance_plot
        
        # Print important features
        print(f"\n  TOP 10 IMPORTANT FEATURES:")
        for i, (feat, imp) in enumerate(zip(importance_data['top_features'][:10], 
                                             importance_data['top_importances'][:10])):
            print(f"    {i+1}. {feat}: {imp:.6f}")
        
        # Save importance details
        importance_txt = save_feature_importance_details(importance_data, model_name, fold_idx)
    
    # Training curves
    if eval_results:
        curves_plot = plot_training_curves_ml(eval_results, fold_idx, model_name)
        if curves_plot:
            plot_paths['training_curves'] = curves_plot
    
    result.update({
        'model': model_name,
        'fold': fold_idx + 1,
        'metrics': metrics,
        'plots': plot_paths,
        'eval_results': eval_results,
        'selected_features': selected_feature_names,
        'importance_data': importance_data if importance_plot else None
    })
    
    return result

# -------------------------
# Main Pipeline
# -------------------------
def main():
    t0 = time.time()
    print("="*80)
    print("SLEEP STAGE CLASSIFICATION - COMPLETE ENHANCED ML PIPELINE")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    download_and_extract_gdrive(GDRIVE_FILE_URL, DATA_DIR)
    file_pairs = discover_valid_file_pairs(DATA_DIR, n_subjects_limit=N_SUBJECTS)
    n_subjects = len(file_pairs)
    
    if n_subjects < 3:
        raise RuntimeError("Need at least 3 valid subjects")
    
    print(f"\n✓ Using {n_subjects} subjects")
    
    # Extract features
    cached_data = extract_and_cache_all_subject_features(file_pairs)
    all_features = cached_data['features']
    all_labels = cached_data['labels']
    feature_names = cached_data['feature_names']
    
    # Setup LOSO splits
    models = ['xgboost', 'lightgbm', 'randomforest', 'extratrees']
    all_results = {m: [] for m in models}
    
    splits = []
    for test_idx in range(n_subjects):
        val_idx = (test_idx + 1) % n_subjects
        train_idxs = [i for i in range(n_subjects) if i != test_idx and i != val_idx]
        splits.append((train_idxs, val_idx, test_idx))
    
    print(f"\n{'='*80}")
    print(f"LOSO CROSS-VALIDATION: {n_subjects} folds")
    print(f"Models: {models}")
    print(f"{'='*80}\n")
    
    # Training loop
    for fold_idx in range(n_subjects):
        train_idxs, val_idx, test_idx = splits[fold_idx]
        
        if test_idx >= len(all_features) or val_idx >= len(all_features):
            continue
        
        print(f"\n{'─'*80}")
        print(f"FOLD {fold_idx+1}/{n_subjects} | Test: Subject {test_idx+1} | Val: Subject {val_idx+1}")
        print(f"{'─'*80}")
        
        # Collect importances for comparison plot
        fold_importances = {}
        fold_selected_features = None
        
        for model_name in models:
            print(f"\n  Training {model_name}...")
            res = train_and_evaluate_fold(
                train_idxs, val_idx, test_idx,
                all_features, all_labels,
                fold_idx, model_name, feature_names
            )
            all_results[model_name].append(res)
            print(f"  ✓ Acc={res['metrics']['accuracy']:.4f}, F1={res['metrics']['f1_macro']:.4f}")
            
            # Collect for comparison
            if res.get('importance_data'):
                fold_importances[model_name] = res['importance_data']['importances']
                if fold_selected_features is None:
                    fold_selected_features = res['importance_data']['feature_names']
        
        # Generate comparison plot
        if len(fold_importances) > 0 and fold_selected_features:
            comparison_plot = plot_feature_importance_comparison(
                fold_importances, models, fold_selected_features, fold_idx
            )
    
    # Aggregated analysis
    print(f"\n{'='*80}")
    print("GENERATING AGGREGATED ANALYSIS")
    print(f"{'='*80}\n")
    
    for model_name in models:
        print(f"  {model_name}...")
        
        cm_plot, cm_sum = create_aggregated_confusion_matrix(all_results, model_name)
        print(f"    ✓ Aggregated confusion matrix")
        
        per_class_plot = create_aggregated_per_class_metrics(all_results, model_name)
        print(f"    ✓ Aggregated per-class metrics")
        
        imp_plot, imp_data = create_aggregated_feature_importance(all_results, model_name)
        if imp_plot:
            print(f"    ✓ Aggregated feature importance")
    
    # Save aggregated text details
    agg_txt = save_aggregated_details(all_results, models, timestamp)
    print(f"\n✓ Aggregated analysis saved: {agg_txt}")
    
    # Model comparison summary
    rows = []
    for m in models:
        lst = all_results[m]
        metrics_list = [r['metrics'] for r in lst]
        keys = ['accuracy','f1_macro','f1_micro','f1_weighted',
                'precision_macro','precision_micro','precision_weighted',
                'recall_macro','recall_micro','recall_weighted','kappa']
        row = {'model': m}
        for k in keys:
            vals = [md.get(k, 0.0) for md in metrics_list]
            row[k+'_mean'] = float(np.mean(vals))
            row[k+'_std'] = float(np.std(vals))
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_csv = OUT_DIR / f"model_comparison_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    # Per-class metrics CSV
    per_class_rows = []
    for m in models:
        lst = all_results[m]
        for class_name in CLASS_NAMES:
            row = {'model': m, 'class': class_name}
            for metric in ['precision', 'recall', 'f1']:
                vals = [r['metrics'][f'{metric}_{class_name}'] for r in lst]
                row[f'{metric}_mean'] = float(np.mean(vals))
                row[f'{metric}_std'] = float(np.std(vals))
            per_class_rows.append(row)
    
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = OUT_DIR / f"per_class_metrics_{timestamp}.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    
    # Compress results (excluding .npz and .pkl)
    print(f"\n{'='*80}")
    print("COMPRESSING OUTPUTS")
    print(f"{'='*80}\n")
    
    archive_path = Path(f"sleep_ml_complete_{timestamp}.7z")
    with py7zr.SevenZipFile(archive_path, 'w') as archive:
        for item in OUT_DIR.rglob('*'):
            if item.is_file() and item.suffix not in ['.npz', '.pkl']:
                archive.write(item, arcname=f"output/{item.relative_to(OUT_DIR)}")
        for item in PLOT_DIR.rglob('*'):
            if item.is_file():
                archive.write(item, arcname=f"plots/{item.relative_to(PLOT_DIR)}")
    
    total_time = time.time() - t0
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(summary_df.to_string(index=False))
    print(f"\n✓ Aggregated analysis: {agg_txt}")
    print(f"\n✓ Summary CSVs:")
    print(f"  - {summary_csv}")
    print(f"  - {per_class_csv}")
    print(f"✓ Compressed archive: {archive_path}")
    print(f"✓ Total time: {total_time/60:.1f} minutes")
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

# %%


# %%



