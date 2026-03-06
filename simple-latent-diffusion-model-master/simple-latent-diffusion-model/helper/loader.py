import torch
import torch.nn as nn
from helper.ema import EMA
from transformers import get_cosine_schedule_with_warmup

class Loader():
    def __init__(self, device = None):
        self.device = device
        
    def print_model(self, check_point):
        print("Epoch: " + str(check_point["epoch"]))
        print("Training step: " + str(check_point["training_steps"]))
        print("Best loss: " + str(check_point["best_loss"]))
        print("Batch size: " + str(check_point["batch_size"]))
        print("Number of batches: " + str(check_point["number_of_batches"]))
        
    def model_load(self, file_name : str, model : nn.Module, 
             print_dict : bool = True, is_ema: bool = True):
        check_point = torch.load(file_name + ".pth", map_location=self.device,
                                 weights_only=True)
        if print_dict: self.print_model(check_point)
        if is_ema:
            model = EMA(model)
            model.load_state_dict(check_point['ema_state_dict'])
            model = model.ema_model
        else:
            model.load_state_dict(check_point['model_state_dict'])
        model.eval()
        print("===Model loaded!===")
        return model
        
    def load_for_training(self, file_name: str, model: nn.Module, print_dict: bool = True):
        check_point = torch.load(file_name + ".pth", map_location=self.device,
                                 weights_only=True)
        if print_dict: self.print_model(check_point)
        model.load_state_dict(check_point['model_state_dict'])
        model.train()
        ema = EMA(model)
        ema.load_state_dict(check_point['ema_state_dict'])
        ema.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        scheduler.load_state_dict(check_point["scheduler_state_dict"])
        epoch = check_point["epoch"]
        loss = check_point["best_loss"]
        print("===Model/EMA/Optimizer/Scheduler/Epoch/Loss loaded!===")
        return model, ema, optimizer, scheduler, epoch, loss