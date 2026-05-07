### Setup

Run everything from this folder:

```bash
cd code-submit
```

Create a Python virtual environment named `fastdiff` and install dependencies from `requirements.txt`:

```bash
python3 -m venv fastdiff
source fastdiff/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- You’ll typically want a CUDA-enabled GPU and a CUDA-compatible PyTorch build. If you need a specific CUDA wheel, follow the official PyTorch install selector and then re-run `pip install -r requirements.txt`.

### Folder contents

- `src/`: training code (`python -m src.train`)
- `scripts/`: shell scripts that run training
- `training_section.tex`: clean LaTeX training section for the paper

### Dataset

We use the Kaggle Image Classification 200K dataset:

- `https://www.kaggle.com/datasets/ashishjangra27/image-classification-200k-dataset`

The scripts download the dataset automatically with `kagglehub` when you do not pass `DATA_ROOT`.

### How to run

#### 1) Train the SD1.5 UNet

```bash
EPOCHS=N BATCH_SIZE=2 GPU_ID=1 ./scripts/train_sd15_200k.sh
```

This writes checkpoints under `checkpoints/`.

#### 2) Train the MD teacher UNet backbone (3 epochs)

MD-UFO needs an initialization checkpoint for both the generator and the discriminator. We use the paper-like UNet backbone training script to produce it.

```bash
EPOCHS=3 BATCH_SIZE=2 GPU_ID=1 OUTPUT_DIR=checkpoints ./scripts/train_md_paper_like.sh
```

This writes checkpoints under `checkpoints/`.

#### 3) Run MD-UFO adversarial one-step training

```bash
TEACHER=checkpoints/unet_md_paper_like_epoch2.pt \\
STEPS=5000 BATCH_SIZE=2 GPU_ID=1 ONE_STEP_T=999 \\
./scripts/train_md_ufo_adv.sh
```

This writes checkpoints under `checkpoints/`.

### Optional: use a local dataset folder

If you already downloaded the dataset, set `DATA_ROOT`:

```bash
DATA_ROOT=/path/to/dataset_root EPOCHS=3 BATCH_SIZE=2 GPU_ID=1 ./scripts/train_sd15_200k.sh
```

