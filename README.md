# Temporal-Stacking-for-Above-water-Drone-Image-Enhancement
A lightweight tool for enhancing above-water drone imagery using temporal filtering (min / median / percentile) after image alignment.  This method is particularly effective for: - removing sun glint and bottom caustics - improving visibility of submerged features

This method is particularly effective for:
- removing sun glint and bottom caustics
- improving visibility of submerged features

---

## Quick Start

### Option 1 — Run in Google Colab (recommended)

No installation required.
Open the link: https://gist.github.com/EvangelosAlevizos/6a5f32ae1c8c62faaeecddac248eca4b

**Steps:**
1. Upload a ZIP of your drone images
2. Adjust parameters (optional)
3. Run all cells
4. Download processed images

---

### Option 2 — Run locally (Python script)

#### 1. Install dependencies

pip install opencv-python imageio numpy exifread

Install ExifTool (required for metadata copy):

Windows: https://exiftool.org/

Mac:
brew install exiftool

Linux:
sudo apt install exiftool

#### 2. Run the script:
temporal_stacking.py 
  

| Parameter       | Description                                                |
| --------------- | ---------------------------------------------------------- |
| `--input`       | Input folder with JPG images                               |
| `--output`      | Output folder                                              |
| `--chunk`       | Number of images per temporal group                        |
| `--percentile`  | Temporal filter: `0=min`, `1=median`, otherwise percentile |
| `--downscale`   | Alignment scale factor (default: 0.25)                     |
| `--no_metadata` | Disable EXIF metadata copying                              |



Use Cases
Coastal and shallow water mapping
Benthic habitat monitoring
Removing dynamic surface noise
Improving photogrammetric reconstructions

Citation / Attribution

If you use this method in your work, please consider citing:
Alevizos, E. (2026). A temporal stacking tool for removing specular noise and caustics from shallow-water drone imagery (v1.0). Zenodo. https://doi.org/10.5281/zenodo.19813852


Author: Evangelos Alevizos 

Contributions, Suggestions, improvements, and issues are welcome!
