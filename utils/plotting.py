import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def plot_structural_identity(image, mask, prediction, beta, dice_score, save_path=None):
    """
    Generates the 3-panel 'Proof Plate' seen in the paper.
    Panel 1: Raw Input
    Panel 2: Spectral Probability Map (Magma)
    Panel 3: Yellow Overlay (Red=Pred, Green=GT, Yellow=Match)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    
    # 1. Raw Biological Input
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Raw Input", fontsize=12)
    axes[0].axis('off')
    
    # 2. Spectral Map
    axes[1].imshow(prediction, cmap='magma')
    axes[1].set_title(f"Spectral Map (β={beta:.3f})", fontsize=12)
    axes[1].axis('off')
    
    # 3. Yellow Structural Overlay
   
    h, w = mask.shape
    overlay = np.zeros((h, w, 3))
    overlay[..., 1] = mask 
    overlay[..., 0] = (prediction > 0.5).astype(float)  
    axes[2].imshow(overlay)
    axes[2].set_title(f"Yellow Alignment (Dice: {dice_score:.4f})", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_convergence(beta_history, loss_history):
    """Plots the learning trajectory of the Harmonic Identity."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Learned Beta (β)', color=color)
    ax1.plot(beta_history, color=color, linewidth=2, label='Beta Evolution')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=4.0, color='r', linestyle='--', label='Harmonic Identity (4.0)')

    ax2 = ax1.twinx()
    color = 'tab:gray'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(loss_history, color=color, alpha=0.5, label='Training Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Convergence to Harmonic Identity")
    fig.tight_layout()
    plt.show()

def apply_adaptive_overlay(background, mask, color=[255, 255, 0]):
    """Utility for creating the blended yellow highlights."""
    bg_color = np.stack([(background * 255).astype(np.uint8)] * 3, axis=-1)
    overlay = bg_color.copy()
    overlay[mask > 0.5] = color
    return cv2.addWeighted(bg_color, 0.6, overlay, 0.4, 0)
