# HTC-DC Net

This repository provides the implementation for **HTC-DC Net**, a network designed for **monocular height estimation (nDSM)** from single-view remote sensing images. For detailed methodology and results, please refer to our [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10294289).

---

## 🛠️ Installation

> ⚠️ **Note:** The provided environment file is for **reference only**. Some packages may require manual installation or configuration.

### Recommended Environment
- `pytorch` 1.7.1  
- `pytorch3d` 0.4.0  
  > ⚙️ PyTorch3D typically requires manual compilation with GPU support. Please follow the official [installation instructions](https://github.com/facebookresearch/pytorch3d).  
- `fvcore` 0.1.5  
- `timm` 0.9.7  
- `scikit-image` 0.21.0  
- `wandb` (for experiment tracking)

---

## 🚀 Usage

### ⚙️ Configuration Files

Two configuration files are needed to launch training:

1. **Data & Logging Configuration**  
   See `configs/configs1.yaml` as an example. You must define:
   - `data_dir`: path to your dataset  
   - `data_split_dirs`: directory containing train/val/test splits  
   - `test_data_split_dirs`: directory with test splits  
   - `patch_size`: patch size used by the transformer module (specific to HTC-DC Net)

2. **Model & Training Configuration**  
   See `configs/htcdc.yaml` as a reference. Most hyperparameters are set as defaults in `htcdc.py`, but you may override them here if desired.

---

### 📂 Data Preparation

Organize your dataset in the following structure under `data_dir`:
```
📂 data_dir
    📂 image # Opitcal satellite images
    📂 mask # Building footprint masks (optional, for computing building metrics)
    📂 ndsm (Ground truth normalized DSMs)
```

Each scene should have the same filename base, e.g., `scene_001`, with different suffixes:
- `_IMG.tif` – optical image  
- `_BLG.tif` – building mask (optional)  
- `_AGL.tif` – nDSM height map


**Example:**
```
scene_001_IMG.tif
scene_001_BLG.tif
scene_001_AGL.tif
```

Define your data splits in `data_split_dirs` as:
```
📂 data_split_dirs
    🖹 train.txt
    🖹 val.txt
    🖹 test.txt
```
Each file lists scene bases (without extensions), e.g.:
```
scene_001
scene_002
...
scene_xxx
```
In `test_data_split_dirs`, only `test.txt` is needed.

---

### 🎯 Training

Start training with:
```bash
python train.py --config configs/configs1.yaml --exp_config configs/htcdc.yaml
```
This will save the configuration and checkpoints under a timestamped directory.

To **resume** training:
```bash
python train.py --exp_config /path/to/saved/config --restore
```
After training, there will be several checkpoint files under the checkpoint directory, `checkpoint_last.pth.tar` for the last epoch, `checkpoint_best_rmse.pth.tar` for the epoch with best validation RMSE, and so on.

---

### 📊 Evaluation
Evaluate a trained model with:
```bash
python test.py --config /path/to/archived/config/under/checkpoint/directory test_checkpoint_file checkpoint_best_rmse.pth.tar
```
Replace checkpoint_best_rmse.pth.tar with any other saved checkpoint as needed. The results with be save as `result_best_rmse.pth.tar`.
