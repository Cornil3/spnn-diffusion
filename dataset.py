import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CelebAHQDataset(Dataset):
    """
    CelebA-HQ dataset with reproducible train/test split.
    split="train" returns the first (n - n_test) images.
    split="test"  returns the last n_test images.
    split="all"   returns everything (original behavior).
    """
    def __init__(self, img_size=256, max_images=None, split="all", n_test=1000):
        from datasets import load_dataset
        print("Loading Ryan-sjtu/celebahq-caption dataset...")
        ds = load_dataset("Ryan-sjtu/celebahq-caption", split="train")

        if max_images is not None:
            ds = ds.select(range(min(max_images, len(ds))))

        n = len(ds)
        n_test = min(n_test, n)
        n_train = n - n_test

        if split == "train":
            ds = ds.select(range(n_train))
        elif split == "test":
            ds = ds.select(range(n_train, n))

        print(f"  Split: {split} — {len(ds)} images (total {n}, n_test={n_test})")
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        return self.transform(img)
