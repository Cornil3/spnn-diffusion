from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize, RandomHorizontalFlip
import os
import torch
import json
from PIL import Image as im
from helper.tokenizer import Tokenizer
from transformers import AutoProcessor

def center_crop_and_resize(img, crop_size, resize_size):
    width, height = img.size

    # 1. Center Crop
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    img_cropped = img.crop((left, top, right, bottom))

    # 2. Resize
    img_resized = img_cropped.resize((resize_size, resize_size), im.Resampling.BICUBIC)

    return img_resized

class UnlabelDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.file_list = os.listdir(path)
        self.transform = transform
        
    def __len__(self) :
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.path + self.file_list[index]
        image = im.open(img_path)
        image = self.transform(image)
        return image
    
class CompositeDataset(Dataset):
    def __init__(self, path, text_path, processor: AutoProcessor = None):
        self.path = path
        self.text_path = text_path
        self.tokenizer = Tokenizer()
        self.processor = processor
        
        self.file_numbers = os.listdir(path)
        self.file_numbers = [ os.path.splitext(filename)[0] for filename in self.file_numbers ]
        
        self.transform = Compose([
                ToTensor(),
                CenterCrop(400),
                Resize(256, antialias=True),
                RandomHorizontalFlip(),
                Lambda(lambda x: (x - 0.5) * 2)
            ])
        
    def __len__(self) :
        return len(self.file_numbers)
    
    def get_text(self, text_path):
        with open(text_path, encoding = 'CP949') as f:
            text = json.load(f)['description']['impression']['description']
        return text

    def __getitem__(self, idx) :
        img_path = self.path + self.file_numbers[idx] + '.png'
        text_path = self.text_path + self.file_numbers[idx] + '.json'
        image = im.open(img_path)
        text = self.get_text(text_path)
        if self.processor is not None:
            image = center_crop_and_resize(image, 400, 256)
            inputs = self.processor(
                text=text,
                images=image, 
                return_tensors="pt", 
                padding='max_length', 
                max_length=77, 
                truncation=True,
                )
            for j in inputs:
                inputs[j] = inputs[j].squeeze(0)
            return inputs
        else:
            image = self.transform(image)
            text = self.tokenizer.tokenize(text)
            for j in text:
                text[j] = text[j].squeeze(0)
            return image, text

class DataGenerator():
    def __init__(self, num_workers: int = 4, pin_memory: bool = True):
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda x: (x - 0.5) * 2)
            ])
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def cifar10(self, path = './datasets', batch_size : int = 64, train : bool = True):
        train_data = CIFAR10(path, download = True, train = train, transform = self.transform)
        dl = DataLoader(train_data, batch_size, shuffle = True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return dl
    
    def celeba(self, path = './datasets', batch_size : int = 16):
        train_data = CelebA(path, transform = Compose([
            ToTensor(),
            CenterCrop(178),
            Resize(128),
            Lambda(lambda x: (x - 0.5) * 2)
            ]))
        dl = DataLoader(train_data, batch_size, shuffle = True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return dl
    
    def composite(self, path, text_path, batch_size : int = 16, is_process: bool = False):
        processor = None
        if is_process:
            model_name = "Bingsu/clip-vit-base-patch32-ko"
            processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        dataset = CompositeDataset(path, text_path, processor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def random_data(self, size, batch_size : int = 4):
        train_data = torch.randn(size)
        return DataLoader(train_data, batch_size)
        