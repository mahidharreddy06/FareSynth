import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
from src.timegan import TimeGAN
from src.config import TimeGANConfig, ProjectConfig
from src.preprocessor import AirlineDataPreprocessor

class TimeGANTrainer:
    """
    Manages 3-phase training of TimeGAN:
    Phase 1: Autoencoder Pretraining (Embedder + Recovery)
    Phase 2: Supervisor Pretraining (Supervisor on real latent codes)
    Phase 3: Joint Adversarial Training (Generator + Discriminator + Supervisor + Embedder + Recovery)
    """
    def __init__(
        self,
        model: TimeGAN,
        config: Optional[TimeGANConfig] = None,
        device: Optional[str] = None
    ):
        self.config = config or TimeGANConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Optimizers
        self.opt_e = optim.Adam(self.model.embedder.parameters(), lr=self.config.learning_rate)
        self.opt_r = optim.Adam(self.model.recovery.parameters(), lr=self.config.learning_rate)
        self.opt_s = optim.Adam(self.model.supervisor.parameters(), lr=self.config.learning_rate)
        self.opt_g = optim.Adam(
            list(self.model.generator.parameters()) + list(self.model.supervisor.parameters()),
            lr=self.config.learning_rate
        )
        self.opt_d = optim.Adam(self.model.discriminator.parameters(), lr=self.config.learning_rate)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.history = {
            "ae_loss": [],
            "sup_loss": [],
            "g_loss": [],
            "d_loss": []
        }

    def train_autoencoder(self, dataloader: DataLoader, epochs: int):
        """Phase 1: Train Embedder & Recovery to accurately reconstruct sequential data."""
        print(f"--- Phase 1: Autoencoder Pretraining ({epochs} epochs) ---")
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                self.opt_e.zero_grad()
                self.opt_r.zero_grad()

                h = self.model.embedder(batch_x)
                x_tilde = self.model.recovery(h)

                loss = 10.0 * torch.sqrt(self.mse_loss(x_tilde, batch_x) + 1e-8)
                loss.backward()

                self.opt_e.step()
                self.opt_r.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.history["ae_loss"].append(avg_loss)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(f"[Phase 1 Autoencoder] Epoch {epoch:3d}/{epochs:3d} | Reconstruction Loss: {avg_loss:.5f}")

    def train_supervisor(self, dataloader: DataLoader, epochs: int):
        """Phase 2: Pretrain Supervisor on real latent dynamics."""
        print(f"--- Phase 2: Supervisor Pretraining ({epochs} epochs) ---")
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                self.opt_s.zero_grad()

                with torch.no_grad():
                    h = self.model.embedder(batch_x)

                # Supervisor predicts next latent code: h_{1:T-1} -> h_{2:T}
                h_hat_super = self.model.supervisor(h[:, :-1, :])
                loss = self.mse_loss(h_hat_super, h[:, 1:, :])

                loss.backward()
                self.opt_s.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.history["sup_loss"].append(avg_loss)
            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(f"[Phase 2 Supervisor] Epoch {epoch:3d}/{epochs:3d} | Supervised Loss: {avg_loss:.5f}")

    def train_joint(self, dataloader: DataLoader, epochs: int):
        """Phase 3: Joint Adversarial Training across all components."""
        print(f"--- Phase 3: Joint Adversarial Training ({epochs} epochs) ---")
        self.model.train()

        for epoch in range(1, epochs + 1):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                batch_size, seq_len, _ = batch_x.shape

                # -----------------------
                # Train Generator & Embedder (2 steps per batch)
                # -----------------------
                for _ in range(2):
                    self.opt_g.zero_grad()
                    self.opt_e.zero_grad()
                    self.opt_r.zero_grad()

                    # Latent representations of real sequences
                    h_real = self.model.embedder(batch_x)
                    x_tilde = self.model.recovery(h_real)

                    # Sample noise
                    z = torch.randn(batch_size, seq_len, self.model.noise_dim, device=self.device)
                    e_fake = self.model.generator(z)
                    h_fake = self.model.supervisor(e_fake)
                    x_fake = self.model.recovery(h_fake)

                    # Discriminator evaluation
                    y_fake_logits = self.model.discriminator(h_fake)
                    y_super_logits = self.model.discriminator(e_fake)

                    # Adversarial generator losses (wants discriminator to classify as real = 1)
                    loss_g_adv1 = self.bce_loss(y_fake_logits, torch.ones_like(y_fake_logits))
                    loss_g_adv2 = self.bce_loss(y_super_logits, torch.ones_like(y_super_logits))
                    loss_g_adv = loss_g_adv1 + loss_g_adv2

                    # Supervised consistency loss
                    h_super_real = self.model.supervisor(h_real[:, :-1, :])
                    loss_g_s = self.mse_loss(h_super_real, h_real[:, 1:, :])

                    # Moment matching loss for distribution realism
                    mean_real = torch.mean(batch_x, dim=(0, 1))
                    mean_fake = torch.mean(x_fake, dim=(0, 1))
                    var_real = torch.var(batch_x, dim=(0, 1))
                    var_fake = torch.var(x_fake, dim=(0, 1))
                    loss_moments = torch.mean(torch.abs(mean_real - mean_fake)) + torch.mean(torch.abs(var_real - var_fake))

                    # Reconstruction loss
                    loss_recon = 10.0 * torch.sqrt(self.mse_loss(x_tilde, batch_x) + 1e-8)

                    loss_g_total = loss_g_adv + self.config.gamma * loss_g_s + 10.0 * loss_moments + loss_recon

                    loss_g_total.backward()
                    self.opt_g.step()
                    self.opt_e.step()
                    self.opt_r.step()

                # -----------------------
                # Train Discriminator
                # -----------------------
                self.opt_d.zero_grad()

                with torch.no_grad():
                    h_real = self.model.embedder(batch_x)
                    z = torch.randn(batch_size, seq_len, self.model.noise_dim, device=self.device)
                    e_fake = self.model.generator(z)
                    h_fake = self.model.supervisor(e_fake)

                y_real_logits = self.model.discriminator(h_real)
                y_fake_logits = self.model.discriminator(h_fake)
                y_super_logits = self.model.discriminator(e_fake)

                loss_d_real = self.bce_loss(y_real_logits, torch.ones_like(y_real_logits))
                loss_d_fake = self.bce_loss(y_fake_logits, torch.zeros_like(y_fake_logits))
                loss_d_super = self.bce_loss(y_super_logits, torch.zeros_like(y_super_logits))

                loss_d = loss_d_real + loss_d_fake + loss_d_super

                loss_d.backward()
                self.opt_d.step()

                epoch_g_loss += loss_g_total.item()
                epoch_d_loss += loss_d.item()

            avg_g = epoch_g_loss / len(dataloader)
            avg_d = epoch_d_loss / len(dataloader)
            self.history["g_loss"].append(avg_g)
            self.history["d_loss"].append(avg_d)

            if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
                print(f"[Phase 3 Joint] Epoch {epoch:3d}/{epochs:3d} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

    def fit(self, sequences: np.ndarray, ae_epochs: int = 40, sup_epochs: int = 40, joint_epochs: int = 60):
        """Executes full 3-phase training workflow."""
        tensor_data = torch.tensor(sequences, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        print(f"Starting TimeGAN Training on device: {self.device} | Total sequences: {len(sequences)}")
        self.train_autoencoder(dataloader, ae_epochs)
        self.train_supervisor(dataloader, sup_epochs)
        self.train_joint(dataloader, joint_epochs)
        print("TimeGAN Training successfully completed!")

    def save_checkpoint(self, path: Path):
        """Saves PyTorch state dict and model architecture specs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "history": self.history
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cpu") -> "TimeGANTrainer":
        """Loads TimeGAN model checkpoint from file."""
        checkpoint = torch.load(path, map_location=device)
        saved_cfg = checkpoint.get("config", {})
        
        cfg = TimeGANConfig(
            seq_len=saved_cfg.get("seq_len", 30),
            hidden_dim=saved_cfg.get("hidden_dim", 32),
            num_layers=saved_cfg.get("num_layers", 2),
            noise_dim=saved_cfg.get("noise_dim", 7),
            device=device
        )
        
        model = TimeGAN(
            input_dim=7,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            noise_dim=cfg.noise_dim
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        trainer = cls(model=model, config=cfg, device=device)
        trainer.history = checkpoint.get("history", {})
        print(f"TimeGAN successfully loaded from {path}")
        return trainer
