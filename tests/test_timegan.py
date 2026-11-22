import pytest
import torch
from src.timegan import TimeGAN, EmbeddingNetwork, RecoveryNetwork, GeneratorNetwork, SupervisorNetwork, DiscriminatorNetwork

def test_timegan_submodules_shapes():
    batch_size = 4
    seq_len = 30
    input_dim = 7
    hidden_dim = 16
    noise_dim = 7

    x = torch.randn(batch_size, seq_len, input_dim)
    z = torch.randn(batch_size, seq_len, noise_dim)

    # Embedder
    embedder = EmbeddingNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
    h = embedder(x)
    assert h.shape == (batch_size, seq_len, hidden_dim)

    # Recovery
    recovery = RecoveryNetwork(hidden_dim=hidden_dim, output_dim=input_dim)
    x_tilde = recovery(h)
    assert x_tilde.shape == (batch_size, seq_len, input_dim)

    # Generator
    generator = GeneratorNetwork(noise_dim=noise_dim, hidden_dim=hidden_dim)
    e = generator(z)
    assert e.shape == (batch_size, seq_len, hidden_dim)

    # Supervisor
    supervisor = SupervisorNetwork(hidden_dim=hidden_dim)
    s = supervisor(e)
    assert s.shape == (batch_size, seq_len, hidden_dim)

    # Discriminator
    discriminator = DiscriminatorNetwork(hidden_dim=hidden_dim)
    d = discriminator(h)
    assert d.shape == (batch_size, seq_len, 1)

def test_timegan_end_to_end_generation():
    model = TimeGAN(input_dim=7, hidden_dim=16, num_layers=2, noise_dim=7)
    fake_data = model.generate(n_samples=5, seq_len=30)
    assert fake_data.shape == (5, 30, 7)
    # Output is sigmoid activated, values must be strictly in [0, 1]
    assert (fake_data >= 0.0).all() and (fake_data <= 1.0).all()
