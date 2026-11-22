import torch
import torch.nn as nn
from typing import Tuple

class EmbeddingNetwork(nn.Module):
    """Embeds sequential feature space X into latent code space H."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        h = self.activation(self.fc(out))
        return h

class RecoveryNetwork(nn.Module):
    """Reconstructs original sequential features X_tilde from latent codes H."""
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Scaled values are normalized in [0, 1]
        self.activation = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        x_tilde = self.activation(self.fc(out))
        return x_tilde

class GeneratorNetwork(nn.Module):
    """Generates synthetic latent representations E from random noise Z."""
    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(z)
        e = self.activation(self.fc(out))
        return e

class SupervisorNetwork(nn.Module):
    """Captures stepwise temporal transitions in the latent code space."""
    def __init__(self, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        s = self.activation(self.fc(out))
        return s

class DiscriminatorNetwork(nn.Module):
    """Discriminates between real and synthetic latent representations."""
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        logits = self.fc(out)
        return logits

class TimeGAN(nn.Module):
    """
    Time-series Generative Adversarial Network (TimeGAN).
    Jointly combines an autoencoder (Embedder + Recovery) to learn latent representations
    with an adversarial framework (Generator + Supervisor + Discriminator) to capture
    both temporal transitions and marginal distributions.
    """
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        num_layers: int = 2,
        noise_dim: int = 7
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        self.embedder = EmbeddingNetwork(input_dim, hidden_dim, num_layers)
        self.recovery = RecoveryNetwork(hidden_dim, input_dim, num_layers)
        self.generator = GeneratorNetwork(noise_dim, hidden_dim, num_layers)
        self.supervisor = SupervisorNetwork(hidden_dim, num_layers=max(1, num_layers - 1))
        self.discriminator = DiscriminatorNetwork(hidden_dim, num_layers)

    def generate(self, n_samples: int, seq_len: int, device: str = "cpu") -> torch.Tensor:
        """
        Samples random noise and generates synthetic sequence in normalized space [0, 1].
        Output shape: [n_samples, seq_len, input_dim]
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, seq_len, self.noise_dim, device=device)
            e = self.generator(z)
            h = self.supervisor(e)
            x_fake = self.recovery(h)
        return x_fake
