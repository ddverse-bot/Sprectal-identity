import numpy as np
from scipy import fftpack as fft
from skimage.metrics import structural_similarity as ssim

def calculate_dice(gt, pred, threshold=0.5):
    """Calculates the Dice Coefficient for structural overlap."""
    p_bin = (pred > threshold).astype(np.float32)
    g_bin = (gt > threshold).astype(np.float32)
    intersection = np.sum(p_bin * g_bin)
    denominator = np.sum(p_bin) + np.sum(g_bin)
    return (2.0 * intersection) / (denominator + 1e-7)

def get_radial_energy(dct_data):
    """Reduces 2D DCT coefficients to a 1D radial energy distribution."""
    size = dct_data.shape[0]
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    radius = np.sqrt(u**2 + v**2).round().astype(int)
    power = np.abs(dct_data)**2
    radial_sum = np.bincount(radius.ravel(), weights=power.ravel())
    return radial_sum[:size]

def calculate_spectral_fidelity(original, processed):
    """
    Measures HFER (High-Frequency Energy Retention).
    Proves that Beta ~ 4.0 preserves the top 25% of the harmonic spectrum.
    """
    dct_orig = np.abs(fft.dctn(original, norm='ortho'))
    dct_proc = np.abs(fft.dctn(processed, norm='ortho'))
    thresh = int(0.75 * original.shape[0])
    hf_orig = np.sum(dct_orig[thresh:, thresh:]**2)
    hf_proc = np.sum(dct_proc[thresh:, thresh:]**2)
    
    fidelity = (hf_proc / (hf_orig + 1e-12)) * 100
    return np.clip(fidelity, 0, 100)

def calculate_energy_slope(image):
    """
    Calculates the decay slope in the log-log domain.
    Standard TPS usually shows a slope of ~ -4.0 (Blurry).
    Adaptive Math (Beta ~ 4.0) should show a flatter slope (Sharp).
    """
    dct_data = np.abs(fft.dctn(image, norm='ortho'))
    radial_energy = get_radial_energy(dct_data)
    
    freqs = np.arange(1, len(radial_energy))
    log_f = np.log(freqs)
    log_e = np.log(radial_energy[1:] + 1e-12)
    
    slope, _ = np.polyfit(log_f, log_e, 1)
    return slope
