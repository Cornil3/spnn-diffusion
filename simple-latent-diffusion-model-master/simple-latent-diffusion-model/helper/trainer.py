import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from typing import Callable
from helper.ema import EMA

class Trainer():
    def __init__(self,
                 model: nn.Module,
                 loss_fn: Callable,
                 ema: EMA = None,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 start_epoch = 0,
                 best_loss = float("inf"),
                 accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0):
        self.accelerator = Accelerator(mixed_precision = 'fp16', gradient_accumulation_steps=accumulation_steps)
        self.model = model.to(self.accelerator.device)
        if ema is None:
            self.ema = EMA(self.model).to(self.accelerator.device)            
        else:
            self.ema = ema.to(self.accelerator.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-4)
        self.scheduler = scheduler
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.start_epoch = start_epoch
        self.best_loss = best_loss
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
            
    def train(self, dl : DataLoader, epochs: int, file_name : str, no_label : bool = False):
        self.model.train()
        self.model, self.optimizer, data_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, dl, self.scheduler
            )
        
        for epoch in range(self.start_epoch + 1, epochs + 1):
            epoch_loss = 0.0
            progress_bar = tqdm(data_loader, leave=False, desc=f"Epoch {epoch}/{epochs}", colour="#005500", disable = not self.accelerator.is_local_main_process)
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):  # Context manager for accumulation
                    if no_label:
                        if isinstance(batch, list):
                            x = batch[0].to(self.accelerator.device)
                        else:
                            x = batch.to(self.accelerator.device)
                    else:
                        x, y = batch[0].to(self.accelerator.device), batch[1].to(self.accelerator.device)
                    
                    with self.accelerator.autocast():
                        if no_label:
                            loss = self.loss_fn(x)
                        else:
                            loss = self.loss_fn(x, y=y)

                    # Normalize the loss
                    self.accelerator.backward(loss)

                    # Gradient Clipping:
                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Only step optimizer and scheduler when we have accumulated enough
                    self.optimizer.step()
                    self.ema.update()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=epoch_loss / (min(step + 1, len(data_loader)))) # Correct progress bar update
                
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                epoch_loss = epoch_loss / len(progress_bar)
                self.scheduler.step()
                log_string = f"Loss at epoch {epoch}: {epoch_loss :.4f}"

                # Save the best model
                if self.best_loss > epoch_loss:
                    self.best_loss = epoch_loss
                    torch.save({
                        "model_state_dict": self.accelerator.get_state_dict(self.model),
                        "ema_state_dict": self.ema.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "epoch": epoch,
                        "training_steps": epoch * len(dl),
                        "best_loss": self.best_loss,
                        "batch_size": dl.batch_size,
                        "number_of_batches": len(dl)
                        }, file_name + '.pth')
                    log_string += " --> Best model ever (stored)"
                print(log_string)
