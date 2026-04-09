from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset  # HF datasets, no collision now


class HFCelebAHQ(Dataset):
    def __init__(self, hf_name="korexyz/celeba-hq-256x256", split="validation", image_size=256):
        self.ds = load_dataset(hf_name, split=split)
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"].convert("RGB")
        return self.tf(img), 0