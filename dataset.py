import os
import glob
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


class LAIONAestheticDataset(Dataset):
    """
    LAION-Aesthetics v2 5+ dataset downloaded via img2dataset (webdataset format).
    Reads all .tar shards, extracts .jpg images from them.

    Args:
        data_dir: path to the img2dataset output directory containing .tar shards
        img_size: resize images to this size
        split: "train", "test", or "all"
        n_test: number of images reserved for the test split (last n_test)
        max_images: cap total images (None = use all)
    """
    def __init__(self, data_dir, img_size=256, split="all", n_test=1000, max_images=None):
        import tarfile
        import io

        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # Collect all (tar_path, member_name) pairs for .jpg files
        tar_files = sorted(glob.glob(os.path.join(data_dir, "*.tar")))
        if not tar_files:
            raise FileNotFoundError(f"No .tar shards found in {data_dir}")

        print(f"Loading LAION-Aesthetic from {data_dir} ({len(tar_files)} shards)...")
        self.samples = []  # list of (tar_path, member_name)
        for tf_path in tar_files:
            with tarfile.open(tf_path, "r") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(".jpg"):
                        self.samples.append((tf_path, member.name))
            if max_images and len(self.samples) >= max_images:
                self.samples = self.samples[:max_images]
                break

        n = len(self.samples)
        n_test = min(n_test, n)
        n_train = n - n_test

        if split == "train":
            self.samples = self.samples[:n_train]
        elif split == "test":
            self.samples = self.samples[n_train:]

        print(f"  Split: {split} — {len(self.samples)} images (total {n}, n_test={n_test})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import tarfile
        import io

        tar_path, member_name = self.samples[idx]
        with tarfile.open(tar_path, "r") as tf:
            f = tf.extractfile(member_name)
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        return self.transform(img)
