"""
Neural network models for mechanism classification.

Implements:
- 1D CNN classifier for concentration-time profiles
- Alternative GRU/LSTM models (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Conv1DClassifier(nn.Module):
    """
    1D CNN classifier for concentration profiles.

    Architecture:
    - Multiple 1D convolutional blocks with batch norm and pooling
    - Global average pooling
    - Fully connected layers
    - Output: class logits
    """

    def __init__(
        self,
        num_features: int = 6,
        num_classes: int = 3,
        seq_length: int = 128,
        hidden_channels: Tuple[int, ...] = (32, 64, 128),
        fc_hidden: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize 1D CNN classifier.

        Args:
            num_features: Number of input features per timestep
            num_classes: Number of output classes
            seq_length: Length of input sequence
            hidden_channels: Tuple of hidden channel sizes for conv layers
            fc_hidden: Hidden size for fully connected layer
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.seq_length = seq_length

        # Convolutional blocks
        layers = []
        in_channels = num_features

        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Convolutional blocks
        x = self.conv_blocks(x)

        # Global average pooling
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(2)  # (batch, channels)

        # Fully connected
        logits = self.fc(x)

        return logits


class GRUClassifier(nn.Module):
    """
    GRU-based classifier for concentration profiles.

    Alternative to CNN-based model.
    """

    def __init__(
        self,
        num_features: int = 6,
        num_classes: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize GRU classifier.

        Args:
            num_features: Number of input features per timestep
            num_classes: Number of output classes
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU layers
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output size
        gru_output_size = hidden_size * (2 if bidirectional else 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # GRU forward
        # output: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        output, h_n = self.gru(x)

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Fully connected
        logits = self.fc(hidden)

        return logits


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for concentration profiles.

    Alternative to CNN and GRU models.
    """

    def __init__(
        self,
        num_features: int = 6,
        num_classes: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM classifier.

        Args:
            num_features: Number of input features per timestep
            num_classes: Number of output classes
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # LSTM forward
        # output: (batch, seq_len, hidden_size * num_directions)
        # (h_n, c_n): (num_layers * num_directions, batch, hidden_size)
        output, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Fully connected
        logits = self.fc(hidden)

        return logits


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1)
        return x * scale


class ResBlock1D(nn.Module):
    """Residual block with two conv layers and SE attention."""

    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class ResConv1DClassifier(nn.Module):
    """
    Residual 1D CNN with Squeeze-and-Excitation attention.

    Architecture:
    - Wide stem convolution (kernel=7)
    - Residual blocks with SE attention at each scale
    - Global average pooling
    - 3-layer FC head
    """

    def __init__(
        self,
        num_features: int = 16,
        num_classes: int = 20,
        seq_length: int = 128,
        hidden_channels: Tuple[int, ...] = (64, 128, 256),
        fc_hidden: int = 256,
        dropout: float = 0.25,
        blocks_per_stage: int = 1
    ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        # Stem: wide convolution to capture broad temporal context
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, hidden_channels[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(inplace=True)
        )

        # Build stages: projection + residual blocks at each scale
        stages = []
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels:
            if out_ch != in_ch:
                # Projection to new channel count + downsample
                stages.append(nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2)
                ))
            else:
                stages.append(nn.MaxPool1d(2))
            # Residual blocks at this scale
            for _ in range(blocks_per_stage):
                stages.append(ResBlock1D(out_ch, dropout=dropout))
            in_ch = out_ch

        self.stages = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 3-layer FC head
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc_hidden // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_pool(x).squeeze(2)
        return self.fc(x)


def create_model(
    model_type: str = 'cnn',
    num_features: int = 6,
    num_classes: int = 3,
    seq_length: int = 128,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('cnn', 'rescnn', 'gru', 'lstm')
        num_features: Number of input features
        num_classes: Number of output classes
        seq_length: Sequence length
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance
    """
    model_type = model_type.lower()

    if model_type == 'cnn':
        return Conv1DClassifier(
            num_features=num_features,
            num_classes=num_classes,
            seq_length=seq_length,
            **kwargs
        )
    elif model_type == 'rescnn':
        return ResConv1DClassifier(
            num_features=num_features,
            num_classes=num_classes,
            seq_length=seq_length,
            **kwargs
        )
    elif model_type == 'gru':
        return GRUClassifier(
            num_features=num_features,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'lstm':
        return LSTMClassifier(
            num_features=num_features,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'cnn', 'rescnn', 'gru', 'lstm'")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    batch_size = 16
    seq_length = 128
    num_features = 6
    num_classes = 6

    # Create dummy input
    x = torch.randn(batch_size, seq_length, num_features)

    # Test CNN
    print("Testing Conv1DClassifier...")
    model_cnn = Conv1DClassifier(num_features, num_classes, seq_length)
    output_cnn = model_cnn(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_cnn.shape}")
    print(f"  Parameters: {count_parameters(model_cnn):,}")

    # Test GRU
    print("\nTesting GRUClassifier...")
    model_gru = GRUClassifier(num_features, num_classes)
    output_gru = model_gru(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_gru.shape}")
    print(f"  Parameters: {count_parameters(model_gru):,}")

    # Test LSTM
    print("\nTesting LSTMClassifier...")
    model_lstm = LSTMClassifier(num_features, num_classes)
    output_lstm = model_lstm(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_lstm.shape}")
    print(f"  Parameters: {count_parameters(model_lstm):,}")

    print("\nAll models work correctly!")
