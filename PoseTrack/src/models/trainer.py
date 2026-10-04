import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def temporal_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalises sudden angle accelerations (second derivative)."""
    first_diff  = pred[:, 1:, :]  - pred[:, :-1, :]
    second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]
    return second_diff.pow(2).mean()

class BaseTrainer:
    """
    Standardized Trainer for PoseTrack Sequence Models.
    Handles training loops, validation, early stopping, and checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        checkpoint_path: str = "best_model.pt",
        patience: int = 15,
        clip_grad: float | None = 1.0,
    ):
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.clip_grad = clip_grad
        
        self.model.to(self.device)
        
    def train_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            pred = self.model(xb)
            loss = self.compute_loss(pred, yb, criterion)
            loss.backward()
            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(loader.dataset)
        
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Override this to add custom losses (e.g. smoothness loss)."""
        return criterion(pred, target)
        
    def validate(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                total_loss += loss.item() * xb.size(0)
        return total_loss / len(loader.dataset)
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> dict:
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            
            if scheduler is not None:
                scheduler.step()
                
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
                
            # Early stopping & Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stop at epoch {epoch}.")
                    break
                    
        print(f"  Best val loss: {best_val_loss:.4f} -> saved to {self.checkpoint_path}")
        return history

class GANTrainer(BaseTrainer):
    """
    Standardized GANTrainer for Adversarial Sequence Refinement.
    Inherits from BaseTrainer to reuse pretraining and general structure.
    """
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: str = "cpu",
        checkpoint_path: str = "gan_generator_best.pt",
        patience: int = 15,
        clip_grad: float | None = 1.0,
        lambda_recon: float = 10.0,
        lambda_adv: float = 1.0,
        lambda_smooth: float = 5.0,
    ):
        super().__init__(model=generator, device=device, checkpoint_path=checkpoint_path, patience=patience, clip_grad=clip_grad)
        self.discriminator = discriminator
        self.discriminator.to(device)
        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
        self.lambda_smooth = lambda_smooth
        self.use_smoothness_pretrain = False
        
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        loss = criterion(pred, target)
        if self.use_smoothness_pretrain:
            loss = loss + self.lambda_smooth * temporal_smoothness_loss(pred)
        return loss

    def fit_gan(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pretrain_epochs: int,
        gan_epochs: int,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        mse_criterion: nn.Module,
        bce_criterion: nn.Module,
    ) -> dict:
        # Phase A: Pretrain generator alone using BaseTrainer.fit
        self.use_smoothness_pretrain = True
        pretrain_history = self.fit(
            train_loader,
            val_loader,
            epochs=pretrain_epochs,
            optimizer=opt_g,
            criterion=mse_criterion
        )
        self.use_smoothness_pretrain = False
        
        # Phase B: Adversarial training
        print(f"\nAdversarial training for {gan_epochs} epochs...")
        history = {
            "pretrain_loss": pretrain_history["train_loss"],
            "g_loss": [],
            "d_loss": [],
            "val_mse": []
        }
        best_val = float("inf")
        
        for epoch in range(1, gan_epochs + 1):
            self.model.train()
            self.discriminator.train()
            g_ep, d_ep = 0.0, 0.0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                real_label = torch.ones(xb.size(0), 1, device=self.device)
                fake_label = torch.zeros(xb.size(0), 1, device=self.device)
                
                # --- Update Discriminator ---
                opt_d.zero_grad()
                fake_seq = self.model(xb).detach()
                d_real_loss = bce_criterion(self.discriminator(yb), real_label)
                d_fake_loss = bce_criterion(self.discriminator(fake_seq), fake_label)
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                d_loss.backward()
                opt_d.step()
                
                # --- Update Generator ---
                opt_g.zero_grad()
                fake_seq = self.model(xb)
                adv_loss = bce_criterion(self.discriminator(fake_seq), real_label)
                recon_loss = mse_criterion(fake_seq, yb)
                smooth_loss = temporal_smoothness_loss(fake_seq)
                g_loss = self.lambda_recon * recon_loss + self.lambda_adv * adv_loss + self.lambda_smooth * smooth_loss
                g_loss.backward()
                if self.clip_grad is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
                opt_g.step()
                
                g_ep += g_loss.item() * xb.size(0)
                d_ep += d_loss.item() * xb.size(0)
                
            g_ep /= len(train_loader.dataset)
            d_ep /= len(train_loader.dataset)
            
            # Validation MSE only
            self.model.eval()
            val_mse = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_mse += mse_criterion(pred, yb).item() * xb.size(0)
            val_mse /= len(val_loader.dataset)
            
            history["g_loss"].append(g_ep)
            history["d_loss"].append(d_ep)
            history["val_mse"].append(val_mse)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  GAN epoch {epoch:3d}/{gan_epochs} | G={g_ep:.4f} | D={d_ep:.4f} | val_mse={val_mse:.4f}")
                
            if val_mse < best_val:
                best_val = val_mse
                torch.save(self.model.state_dict(), self.checkpoint_path)
                
        print(f"Best GAN val MSE: {best_val:.4f} -> {self.checkpoint_path}")
        return history
