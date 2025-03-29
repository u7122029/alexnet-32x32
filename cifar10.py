from torchvision.transforms.v2 import Normalize, ToTensor, Compose, ToImage, ToDtype

from paths import TEMP_DIR, DATASETS_PATH

import torch
from torchvision.datasets import CIFAR10

class NewToTensor:
    """
    Here because ToTensor() is deprecated in torchvision.transforms.v2
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return Compose([ToImage(), ToDtype(torch.float32, scale=True)])(x)

def obtain_cifar10_mean_std(download=False):
    train_cifar10 = CIFAR10(DATASETS_PATH / "CIFAR10", True, NewToTensor(), download=download)
    combined = torch.stack([x for x, y in train_cifar10])
    std, mean = torch.std_mean(combined, dim=(0,2,3))
    torch.save({"mean": mean, "std": std}, "temp/cifar10_normalisation_params.pt")

def get_transform_cifar10():
    norm_filepath = TEMP_DIR / "cifar10_normalisation_params.pt"
    if not norm_filepath.exists():
        obtain_cifar10_mean_std(False)
    norm_params = torch.load(norm_filepath)
    mean = norm_params["mean"]
    std = norm_params["std"]

    transform = Compose([NewToTensor(), Normalize(mean, std)])
    return transform

def load_cifar10_datasets():
    transform = get_transform_cifar10()
    train_dset = CIFAR10(DATASETS_PATH / "CIFAR10", True, transform)
    test_dset = CIFAR10(DATASETS_PATH / "CIFAR10", False, transform)
    return train_dset, test_dset

if __name__ == "__main__":
    train_dset, test_dset = load_cifar10_datasets()
    print(train_dset[0])