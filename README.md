# Wrist2Finger

This repository implements a deep learning framework to jointly estimate **hand pose (21 joints)** and **finger pressure (5 fingertips)** from multimodal sensor data including EMG and IMU.

## Overview

The model fuses EMG and IMU signals using dual encoders, cross-attention modules, and Transformer decoders to predict:

- **3D Hand Keypoints** (21 joints × 3 coordinates)
- **Pressure Values** (5 fingers: thumb, index, middle, ring, pinky)

We also leverage auxiliary predictions from unimodal branches (IMU-only and EMG-only) to regularize the multitask training.

---

## Architecture Summary

### Inputs
- **EMG**: shape `[batch_size, seq_len=30, emg_dim=6]` (pre-extracted 6 features from raw 1-channel signal)
- **IMU**: shape `[batch_size, seq_len=30, imu_dim=24]`
  - IMU data is collected from two sensors:
    - **WT1**: placed on the **thumb** (embedded in a smart ring worn on the finger)
    - **WT6**: placed on the **wrist** (embedded in a smartwatch-like device)

### EMG Feature Extraction
Raw EMG signals are originally 1 channel (collected from an EMG sensor embedded in the smartwatch on the wrist), and are preprocessed outside this repository into 6 handcrafted features per frame:
- Mean
- Standard Deviation (Std)
- Root Mean Square (RMS)
- Envelope
- Mean Power Spectral Density (Mean_PSD)
- Max Power Spectral Density (Max_PSD)

These features are stored in the CSV files and loaded directly by this code. Normalization is performed per user within this repository.

### IMU Processing
IMU features are constructed by aligning WT1 (thumb) and WT6 (wrist) sensors. Each sensor provides:
- **3D acceleration** (3D)
- **4D quaternion** (converted to 3×3 rotation matrix = 9D)

Final concatenated IMU feature vector per timestep includes:
- WT1 acceleration: 3D
- WT1 rotation matrix (from quaternion): 9D
- WT6 acceleration: 3D
- WT6 rotation matrix (from quaternion): 9D

**Total IMU feature dimensionality: 24**

### Encoders
- **IMU Encoder**
  - `Conv1D`: `(24 → 128)`
  - `Transformer Encoder`: 2 layers, 4 heads, `d_model=128`

- **EMG Encoder**
  - `LSTM`: 2 layers, input=6, hidden=128
  - `Positional Encoding`: sinusoidal

### Cross Attention Modules
- Two symmetric cross-attention layers (`d_model=128`, 4 heads):
  - EMG as query, IMU as key/value
  - IMU as query, EMG as key/value

### Fusion
- Concatenation of both cross-attention outputs: `256 → 128` via `Linear`

### Decoders
- **Pose Decoder**:
  - `TransformerDecoder`: 2 layers, 4 heads, `d_model=128`
  - Final output: `[batch, seq_len, 21×3]`

- **Pressure Decoder**:
  - Fused features + 5 fingertip pose vectors (5×3)
  - 5 individual linear heads predict pressure: `[batch, seq_len, 5]`

### Auxiliary Heads
- Pose & pressure predictions from:
  - IMU-only encoder branch
  - EMG-only encoder branch
- Pressure prediction from cross-attention output

---

## Training Details

| Parameter         | Value                     |
|------------------|---------------------------|
| Batch size       | 256                       |
| Sequence length  | 30 (window), step=5       |
| Optimizer        | Adam                      |
| Learning rate    | 0.001 (train), 0.0001 (fine-tune) |
| LR Scheduler     | StepLR(step=30, γ=0.5)    |
| Epochs           | 100 (train & fine-tune)   |
| Patience         | 50                        |
| Loss Functions   | MSE (pose), SmoothL1 (pressure) |

### Multitask Loss Weights

| Loss Type         | λ (Weight)  |
|------------------|-------------|
| Final Pose       | 1.0         |
| Pose (IMU/EMG)   | 0.3 each    |
| Final Pressure   | 3.0         |
| Pressure (IMU/EMG) | 0.6 each  |
| Cross Attention  | 0.5         |

---

## Data Requirements

The dataset should be organized as follows:

```
UIST/data/
├── EMG_Press/
│   ├── EMG/
│   └── Pressure/
├── IMU/
│   └── IMU/
└── video/
    └── keypoints/
```

Each user should have 20 samples. Each sample includes:
- EMG: `aligned_emg_#.csv`, shape `(T, 6)` (already feature-extracted)
- IMU:
  - `WT1`: IMU on thumb
  - `WT6`: IMU on wrist
  - shape `(T, 7)` for each (3D acceleration + quaternion)
- Pressure: 5 CSVs for each finger (log-transformed, smoothed)
- Keypoints: 21 joints × 3D position, per frame

Sliding window processing is applied during loading:
- Window: 30 frames
- Step: 5 frames

Data is split by user:
- **Train**: full sequences
- **Fine-tune/Test**: first 20% for fine-tuning, rest for testing

---

## Run Training

```bash
python train.py \
  --train_users user1 user2 user3 ... \
  --test_user user4 \
  --log_dir runs/train_run \
  --model_dir models/multitask_model
```

---

## Notes for Replication

- **Feature dimensions are explicitly handled**:
  - EMG: preprocessed to 6 features per frame (external pipeline)
  - IMU: 24D per frame (relative position and quaternion-derived matrices)
- **Quaternion alignment is computed as**:
  - `R_rel = R_wt6^T @ R_wt1`
  - Output reshaped as flat matrix per timestep
- **All hyperparameters are embedded in the script.**
- Normalization parameters are saved in `model_dir/pressure_normalization_params.npz`
- TensorBoard logs include losses, gradients, timing

---

## 7. Logging

- TensorBoard support is included
- Scalars: loss terms, timing, gradients
- Logs are saved under `log_dir` (e.g., `runs/train_run/`)

