import torch
from core.unet_base import SpectralUNet
from data.loaders import BiologicalVesselDataset
from torch.utils.data import DataLoader

def train_identity(dataset_name, img_path, mask_path):
    model = SpectralUNet().cuda()
    loader = DataLoader(BiologicalVesselDataset(img_path, mask_path), batch_size=2, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    for epoch in range(100):
        for imgs, masks in loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Dataset: {dataset_name} | Epoch: {epoch} | Beta: {model.spectral_fix.beta.item():.4f}")

if __name__ == "__main__":
    train_identity("DRIVE", "DRIVE_data/training/images", "DRIVE_data/training/1st_manual")
