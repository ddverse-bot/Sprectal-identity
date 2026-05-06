import torch
import numpy as np
from core.unet_base import SpectralUNet
from data.loaders import BiologicalVesselDataset
from utils.metrics import calculate_spectral_fidelity, calculate_dice
from torch.utils.data import DataLoader
import pandas as pd

def run_evaluation(model_path, dataset_path, label_path, device="cuda"):
    # 1. Load Model and Weights
    model = SpectralUNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Setup Data
    dataset = BiologicalVesselDataset(dataset_path, label_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    print(f"Evaluating Spectral Identity (Learned Beta: {model.spectral_fix.beta.item():.4f})...")
    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            # Convert to numpy for metric calculation
            pred_np = pred.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            img_np = img.squeeze().cpu().numpy()
            # 3. Calculate Scientific Metrics
            dice = calculate_dice(mask_np, pred_np)
            fidelity = calculate_spectral_fidelity(img_np, pred_np) 
            results.append({
                "Image_ID": i,
                "Dice_Score": dice,
                "Spectral_Fidelity": fidelity
            })
    # 4. Generate Results Table for the Paper
    df = pd.DataFrame(results)
    print("\n--- Final Evaluation Summary ---")
    print(df.describe().loc[['mean', 'std']])
    df.to_csv("evaluation_results.csv", index=False)

if __name__ == "__main__":
    run_evaluation("weights/best_model.pth", "data/test/images", "data/test/labels")
