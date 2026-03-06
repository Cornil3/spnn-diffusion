from clip.models.ko_clip import KoCLIPWrapper
from helper.trainer import Trainer
from helper.data_generator import DataGenerator
import torch

if __name__ == "__main__":
    clip = KoCLIPWrapper()
    dg = DataGenerator(num_workers = 0, pin_memory = False)
    data_loader = dg.composite('./datasets/images/', './datasets/labels/', is_process=True)
    
    trainer = Trainer(clip, clip.loss, accumulation_steps = 2)
    trainer.train(data_loader, 100, 'composite_clip', no_label = True)