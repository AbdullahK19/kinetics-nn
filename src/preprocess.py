"""
Preprocessing utilities for normalizing concentration profiles.

This module provides:
- Train/val/test splitting
- Feature standardization (z-score normalization)
- Scaler persistence for inference
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split


class StandardScaler:
    """Standard scaler for z-score normalization."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Fit scaler to training data.

        Args:
            X: Training data of shape (n_samples, seq_len, n_features)

        Returns:
            self
        """
        # Compute mean and std across samples and sequence length
        # Result: (n_features,)
        self.mean = np.mean(X, axis=(0, 1))
        self.std = np.std(X, axis=(0, 1))

        # Avoid division by zero
        self.std = np.where(self.std > 1e-8, self.std, 1.0)

        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Data of shape (n_samples, seq_len, n_features)

        Returns:
            Standardized data
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform standardized data.

        Args:
            X: Standardized data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")

        return X * self.std + self.mean

    def save(self, path: str) -> None:
        """Save scaler parameters to JSON."""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted scaler")

        params = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'fitted': self.fitted
        }

        with open(path, 'w') as f:
            json.dump(params, f, indent=2)

    def load(self, path: str) -> 'StandardScaler':
        """Load scaler parameters from JSON."""
        with open(path, 'r') as f:
            params = json.load(f)

        self.mean = np.array(params['mean'])
        self.std = np.array(params['std'])
        self.fitted = params['fitted']

        return self


def load_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from npz file.

    Args:
        data_path: Path to dataset.npz

    Returns:
        Tuple of (X, y)
    """
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train/val/test sets.

    Args:
        X: Features
        y: Labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=y
    )

    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_test_ratio),
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_dataset(
    data_path: str,
    save_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    normalize: bool = True
) -> None:
    """
    Preprocess and split dataset.

    Args:
        data_path: Path to dataset.npz
        save_dir: Directory to save preprocessed data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        normalize: Whether to apply standardization
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X, y = load_dataset(data_path)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    print("\nSplitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, train_ratio, val_ratio, test_ratio, random_state
    )

    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Val set:   {X_val.shape}, {y_val.shape}")
    print(f"Test set:  {X_test.shape}, {y_test.shape}")

    # Class distribution
    print("\nClass distribution:")
    print(f"  Train: {np.bincount(y_train)}")
    print(f"  Val:   {np.bincount(y_val)}")
    print(f"  Test:  {np.bincount(y_test)}")

    if normalize:
        print("\nNormalizing features...")
        scaler = StandardScaler()

        # Fit on training data only
        scaler.fit(X_train)

        # Transform all sets
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Save scaler
        scaler_path = save_dir / 'scaler.json'
        scaler.save(str(scaler_path))
        print(f"Scaler saved to: {scaler_path}")

        print(f"Feature statistics (after normalization):")
        print(f"  Mean: {np.mean(X_train, axis=(0,1))}")
        print(f"  Std:  {np.std(X_train, axis=(0,1))}")

    # Save preprocessed data
    train_path = save_dir / 'train.npz'
    val_path = save_dir / 'val.npz'
    test_path = save_dir / 'test.npz'

    np.savez_compressed(train_path, X=X_train, y=y_train)
    np.savez_compressed(val_path, X=X_val, y=y_val)
    np.savez_compressed(test_path, X=X_test, y=y_test)

    print(f"\nPreprocessed data saved:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")

    # Save split info
    split_info = {
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_state': random_state,
        'normalize': normalize,
        'train_size': int(len(y_train)),
        'val_size': int(len(y_val)),
        'test_size': int(len(y_test)),
        'num_features': int(X.shape[2]),
        'seq_length': int(X.shape[1])
    }

    info_path = save_dir / 'split_info.json'
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Split info saved to: {info_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/dataset.npz',
        help='Path to dataset.npz'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./data',
        help='Directory to save preprocessed data'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Skip normalization'
    )

    args = parser.parse_args()

    preprocess_dataset(
        data_path=args.data_path,
        save_dir=args.save_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
        normalize=not args.no_normalize
    )
