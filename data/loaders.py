import glob, os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class BiologicalVesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512, pattern="*.tif"):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, pattern)))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.gif" if "DRIVE" in img_dir else "*.png")))
        self.transform = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self): return min(len(self.img_paths), len(self.mask_paths))

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        return self.transform(img), self.transform(mask)
