"""
Training script for mechanism classification model.

This module:
- Loads preprocessed data
- Trains the neural network
- Tracks metrics (accuracy, confusion matrix)
- Saves best model checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F

from model import create_model, count_parameters
from ode_solver import MECHANISM_REGISTRY


class FocalLoss(nn.Module):
    """Focal loss to focus learning on hard-to-classify examples."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def apply_augmentation(x: torch.Tensor) -> torch.Tensor:
    """Apply augmentation transforms to a batch tensor (for TTA)."""
    x = x.clone()
    batch_size = x.shape[0]
    for i in range(batch_size):
        # Gaussian noise on concentration + derivative channels
        if torch.rand(1).item() < 0.5:
            noise_idx = list(range(5)) + list(range(8, min(12, x.shape[2])))
            noise = torch.randn(x.shape[1], len(noise_idx)) * 0.02
            x[i, :, noise_idx] += noise.to(x.device)
        # Scaling
        if torch.rand(1).item() < 0.5:
            scale = 0.8 + 0.4 * torch.rand(1).item()
            x[i, :, :5] *= scale
            if x.shape[2] > 8:
                x[i, :, 8:12] *= scale
    return x


class AugmentedDataset(Dataset):
    """Dataset with online data augmentation for concentration profiles."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = True):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # Gaussian noise on concentration + derivative channels (0-4, 8-11)
            if torch.rand(1).item() < 0.5:
                noise_idx = list(range(5)) + list(range(8, min(12, x.shape[1])))
                noise = torch.randn(x.shape[0], len(noise_idx)) * 0.02
                x[:, noise_idx] += noise

            # Scaling: multiply concentration channels by uniform(0.8, 1.2)
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + 0.4 * torch.rand(1).item()
                x[:, :5] *= scale
                # Also scale derivatives if present
                if x.shape[1] > 8:
                    x[:, 8:12] *= scale

            # Channel dropout: zero out one species channel (A/B/C/D)
            if torch.rand(1).item() < 0.2:
                ch = torch.randint(0, 4, (1,)).item()
                x[:, ch] = 0.0

        return x, y


class Trainer:
    """Trainer for mechanism classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: str,
        num_classes: int = 3
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            save_dir: Directory to save checkpoints
            num_classes: Number of classes
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.max_grad_norm = 1.0
        self.mixup_alpha = 0.2
        self.mixup_prob = 0.3

    def train_epoch(self) -> tuple:
        """Train for one epoch with optional mixup."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Mixup augmentation
            use_mixup = np.random.random() < self.mixup_prob
            if use_mixup:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                idx = torch.randperm(inputs.size(0), device=self.device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[idx]
                outputs = self.model(mixed_inputs)
                loss = lam * self.criterion(outputs, targets) + \
                       (1 - lam) * self.criterion(outputs, targets[idx])
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Metrics (use unmixed targets for accuracy tracking)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self) -> tuple:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

    def validate_tta(self, n_augments: int = 5) -> tuple:
        """Validate with test-time augmentation (average over augmented copies)."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='TTA Validation'):
                inputs = inputs.to(self.device)

                # Original prediction
                probs = F.softmax(self.model(inputs), dim=1)

                # Augmented predictions
                for _ in range(n_augments):
                    aug_inputs = apply_augmentation(inputs)
                    probs = probs + F.softmax(self.model(aug_inputs), dim=1)

                probs /= (n_augments + 1)
                _, predicted = probs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        accuracy = 100. * (all_preds == all_targets).sum() / len(all_targets)
        return accuracy, all_preds, all_targets

    def train(self, num_epochs: int, scheduler=None, patience: int = 25, start_epoch: int = 0) -> None:
        """
        Train the model with early stopping.

        Args:
            num_epochs: Number of epochs to train
            scheduler: Learning rate scheduler (optional)
            patience: Early stopping patience (epochs without improvement)
            start_epoch: Epoch to start from (for resuming)
        """
        print(f"\nStarting training for {num_epochs} epochs (early stopping patience={patience})...")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Model parameters: {count_parameters(self.model):,}")

        patience_counter = 0

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt', epoch, val_acc)
                print(f"✓ New best model saved (val_acc: {val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                    break

            # Learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")

        # Save final model
        self.save_checkpoint('final_model.pt', num_epochs, val_acc)

        # Plot training curves
        self.plot_training_curves()

        # Load best model before computing confusion matrix
        best_ckpt = self.save_dir / 'best_model.pt'
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model (epoch {self.best_epoch}) for evaluation")

        # Compute confusion matrix on best model
        self.compute_confusion_matrix()

        # TTA evaluation
        tta_acc, _, _ = self.validate_tta(n_augments=5)
        print(f"TTA Validation Accuracy (5 augments): {tta_acc:.2f}%")

    def save_checkpoint(self, filename: str, epoch: int, val_acc: float) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, checkpoint_path)

    def plot_training_curves(self) -> None:
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150)
        print(f"Training curves saved to: {plot_path}")
        plt.close()

    def compute_confusion_matrix(self) -> None:
        """Compute and plot confusion matrix on validation set."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)

        # Plot
        labels = [MECHANISM_REGISTRY[i].name
                  if i in MECHANISM_REGISTRY else f'Class {i}'
                  for i in range(self.num_classes)]
        plt.figure(figsize=(max(8, self.num_classes * 2), max(6, self.num_classes * 1.5)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Validation Set)')
        plt.tight_layout()
        cm_path = self.save_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150)
        print(f"Confusion matrix saved to: {cm_path}")
        plt.close()

        # Classification report
        class_names = [MECHANISM_REGISTRY[i].name
                       if i in MECHANISM_REGISTRY else f'Class {i}'
                       for i in range(self.num_classes)]
        report = classification_report(all_targets, all_preds, target_names=class_names,
                                       zero_division=0)
        print("\nClassification Report (Validation Set):")
        print(report)

        # Save report
        report_path = self.save_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)


def load_data(data_dir: str) -> tuple:
    """Load preprocessed train and validation data."""
    data_dir = Path(data_dir)

    # Load train data
    train_data = np.load(data_dir / 'train.npz')
    X_train, y_train = train_data['X'], train_data['y']

    # Load validation data
    val_data = np.load(data_dir / 'val.npz')
    X_val, y_val = val_data['X'], val_data['y']

    return X_train, y_train, X_val, y_val


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = False
) -> tuple:
    """Create PyTorch data loaders."""
    # Create datasets
    if augment:
        train_dataset = AugmentedDataset(X_train, y_train, augment=True)
    else:
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        train_dataset = TensorDataset(X_train_t, y_train_t)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train mechanism classification model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory with preprocessed data')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'rescnn', 'gru', 'lstm'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda/cpu/mps)')
    parser.add_argument('--hidden_channels', type=int, nargs='+', default=None,
                        help='CNN hidden channels (e.g. 64 128 256)')
    parser.add_argument('--fc_hidden', type=int, default=None,
                        help='FC hidden layer size')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'],
                        help='LR scheduler: plateau (ReduceLROnPlateau) or cosine (warm restarts)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing (0.0 = none, 0.1 = typical)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help='Optimizer: adam or adamw')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--class_weights', action='store_true',
                        help='Use inverse-frequency class weights in loss')
    parser.add_argument('--augment', action='store_true',
                        help='Enable online data augmentation')
    parser.add_argument('--focal', action='store_true',
                        help='Use focal loss instead of cross-entropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (default 2.0)')
    parser.add_argument('--blocks_per_stage', type=int, default=1,
                        help='Number of ResBlocks per stage (default 1)')
    parser.add_argument('--swa_epochs', type=int, default=0,
                        help='SWA epochs after main training (0 = disabled)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from best_model.pt checkpoint')
    parser.add_argument('--swa_lr', type=float, default=1e-4,
                        help='SWA learning rate (default 1e-4)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data(args.data_dir)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        augment=args.augment
    )

    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    num_features = X_train.shape[2]
    seq_length = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    extra_kwargs = {}
    if args.hidden_channels is not None:
        extra_kwargs['hidden_channels'] = tuple(args.hidden_channels)
    if args.fc_hidden is not None:
        extra_kwargs['fc_hidden'] = args.fc_hidden
    if args.blocks_per_stage > 1:
        extra_kwargs['blocks_per_stage'] = args.blocks_per_stage

    model = create_model(
        model_type=args.model_type,
        num_features=num_features,
        num_classes=num_classes,
        seq_length=seq_length,
        dropout=args.dropout,
        **extra_kwargs
    )

    # Class weights
    class_weight = None
    if args.class_weights:
        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_weight = len(y_train) / (num_classes * class_counts + 1e-6)
        class_weight = torch.FloatTensor(class_weight).to(device)
        print(f"Class weights (min={class_weight.min():.2f}, max={class_weight.max():.2f})")

    # Loss and optimizer
    if args.focal:
        criterion = FocalLoss(
            alpha=class_weight,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weight,
            label_smoothing=args.label_smoothing
        )

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        num_classes=num_classes
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.save_dir) / 'best_model.pt'
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            trainer.best_val_acc = ckpt['val_acc']
            trainer.history = ckpt.get('history', trainer.history)
            start_epoch = ckpt['epoch'] + 1
            trainer.best_epoch = ckpt['epoch']
            # Advance scheduler to match current epoch
            for _ in range(start_epoch):
                scheduler.step()
            print(f"Resumed from epoch {start_epoch}, best val acc: {trainer.best_val_acc:.2f}%")

    # Train
    trainer.train(num_epochs=args.epochs, scheduler=scheduler, patience=args.patience,
                  start_epoch=start_epoch)

    # SWA phase (optional)
    if args.swa_epochs > 0:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        print(f"\n--- SWA Phase: {args.swa_epochs} epochs at lr={args.swa_lr} ---")

        # Load best model as starting point
        best_ckpt = Path(args.save_dir) / 'best_model.pt'
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])

        swa_model = AveragedModel(model)
        swa_optimizer = optim.SGD(model.parameters(), lr=args.swa_lr)
        swa_scheduler = SWALR(swa_optimizer, swa_lr=args.swa_lr)

        # Replace trainer optimizer temporarily
        orig_optimizer = trainer.optimizer
        trainer.optimizer = swa_optimizer

        for swa_epoch in range(args.swa_epochs):
            print(f"\nSWA Epoch {swa_epoch + 1}/{args.swa_epochs}")
            train_loss, train_acc = trainer.train_epoch()
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"SWA Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

        # Update batch norm statistics
        print("Updating SWA batch norm statistics...")
        update_bn(trainer.train_loader, swa_model, device=device)

        # Evaluate SWA model
        orig_model = trainer.model
        trainer.model = swa_model
        swa_val_loss, swa_val_acc, _, _ = trainer.validate()
        print(f"SWA Val Acc: {swa_val_acc:.2f}% (vs best: {trainer.best_val_acc:.2f}%)")

        # Save SWA model if it's better
        if swa_val_acc > trainer.best_val_acc:
            print("SWA model is better — saving as best model")
            torch.save({
                'epoch': -1,
                'model_state_dict': swa_model.module.state_dict(),
                'optimizer_state_dict': swa_optimizer.state_dict(),
                'val_acc': swa_val_acc,
            }, Path(args.save_dir) / 'best_model.pt')
            trainer.best_val_acc = swa_val_acc

        # Also run TTA on SWA model
        swa_tta_acc, _, _ = trainer.validate_tta(n_augments=5)
        print(f"SWA + TTA Val Acc: {swa_tta_acc:.2f}%")

        trainer.model = orig_model
        trainer.optimizer = orig_optimizer

    # Save training config
    config = vars(args)
    config['best_val_acc'] = trainer.best_val_acc
    config['best_epoch'] = trainer.best_epoch
    config['num_parameters'] = count_parameters(model)
    config['num_classes'] = num_classes
    config['num_features'] = num_features
    config['grid_length'] = seq_length
    if args.hidden_channels is not None:
        config['hidden_channels'] = list(args.hidden_channels)
    if args.fc_hidden is not None:
        config['fc_hidden'] = args.fc_hidden
    if args.blocks_per_stage > 1:
        config['blocks_per_stage'] = args.blocks_per_stage

    config_path = Path(args.save_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nTraining config saved to: {config_path}")


if __name__ == '__main__':
    main()
