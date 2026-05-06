# Dataset Overview and Processing Pipeline

## 1. Dataset Overview

This suite contains a progression of datasets used to validate the Harmonic Identity ($\beta \approx 4.0$). It spans from high-frequency synthetic patterns to real-world medical imaging.

| Domain | Source | Modality | Purpose |
|--------|--------|----------|--------|
| Synthetic | Procedural Grid | Mathematical | Proof of High-Frequency (HF) Preservation |
| Texture Benchmark | Barbara / Ascent | Digital Image | Stress-test for harmonic texture retention |
| Ophthalmology | CHASE_DB1 / DRIVE | Fundus Photography | Real-world macro-vascular segmentation |
| Cytology | Cells3D (skimage) | Fluorescence | Real-world micro-membrane extraction |

---

## 2. Spectral Characteristics

The validation targets were selected based on their distinct frequency-domain behaviors:

- **Synthetic Grid**  
  Features a perfectly repeating high-frequency impulse. Used to demonstrate that while standard TPS blurs the grid into a uniform gray field, the adaptive formulation preserves the binary intensity contrast (1.0 / 0.0).

- **Barbara Image**  
  Contains dense high-frequency textures (e.g., the shawl region). Used to evaluate resistance to aliasing and over-smoothing in spline-based filtering.

- **Biological Data**  
  Exhibits a natural radial energy decay with slope ($\gamma$) typically in the range $[-3.6, -4.1]$. The proposed method compensates for this power-law decay to preserve structural details.

---

## 3. Technical Specifications

- **Dimensions**  
  Standardized to $512 \times 512$  
  (Synthetic data: $256 \times 256$ for controlled experiments)

- **Normalization**  
  All inputs are scaled to $[0, 1]$ to ensure numerical stability in the DCT domain.

- **Labels**  
  - *Synthetic / Barbara*: Original image used as self-ground-truth  
  - *Biological*: Expert manual annotations (First Human Observer)

---

## 4. Data Processing Pipeline

1. **Direct Transform**  
   Apply 2D Discrete Cosine Transform (DCT) to map spatial features to harmonic frequency components.

2. **Adaptive Filtering**  
   Apply frequency-dependent filtering:
   $$
   H(\omega) = \frac{1}{1 + \lambda(\omega)\omega^4}
   $$

3. **Inverse Transform**  
   Reconstruct the image using inverse DCT (IDCT) to evaluate structural fidelity and preservation.

---
