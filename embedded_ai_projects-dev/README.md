# README.md — Embedded AI on Jetson (PyTorch) — Real-time Video Demo (Split Screen)

This repository is a **2-day applied GPU project**: train a model on a **public dataset**, evaluate it, then run **real-time inference on Jetson** with a **split-screen demo**:
- **Left:** original camera stream
- **Right:** processed output / result overlay

You will work in **three phases**:
1) Dataset preparation + dataloaders  
2) Train + evaluate + test  
3) Jetson integration + real-time demo (split screen)

---

## Example mini pipeline (this repo)
This repo now includes a tiny, runnable example that mirrors the full workflow.

Example layout:
```
.
├── data/
│   └── dataset_example/
│       ├── train/
│       │   ├── input/   *.ppm
│       │   └── target/  *.ppm
│       ├── eval/
│       │   ├── input/
│       │   └── target/
│       └── test/
│           ├── input/
│           └── target/
├── outputs/
│   └── example_run/
│       ├── model_best.pt
│       ├── model_last.pt
│       ├── loss_curve.png
│       ├── ssim_curve.png
│       ├── train_metrics.json
│       └── test_metrics.json
└── src/
    ├── launch.py
    ├── 01_data_loaders/
    │   └── data_loader_example.py
    ├── 02_model/
    │   └── model_example.py
    └── 03_train_test/
        ├── metrics_example.py
        ├── train_example.py
        └── test_example.py
```

Run the example end-to-end (train + test, saves weights in `outputs/example_run/`):
```bash
python3 src/launch.py
```

Then run the real-time demo (press `q` to quit):
```bash
python3 src/demo.py --weights outputs/example_run/model_best.pt --size 128
```
The demo uses OpenCV (`cv2`), which is usually preinstalled on Jetson; install
`opencv-python` if needed.

The example uses 64x64 image pairs and reports SSIM on the test split. The
sample dataset contains random shape images with Sobel edge targets so you
can see the input/output difference; training stops once train SSIM reaches
0.90. The dataset uses simple PPM images to avoid extra dependencies; install
Pillow if you switch to PNG/JPG.

---

## Choose ONE of the 10 projects

Pick one project from the list below. Each project has a recommended dataset + task type.

1) **Low-light enhancement (paired)**
   - Input: low-light image, Target: normal-light image
   - Dataset: [LOL (paired)](https://daooshee.github.io/BMVC2018website/)
   - Metric: PSNR / SSIM

2) **Image denoising (paired)**
   - Input: noisy image, Target: clean image
   - Dataset: [SIDD (paired)](https://www.eecs.yorku.ca/~kamel/sidd/)
   - Metric: PSNR / SSIM

3) **Super-resolution ×2 (paired via synthetic LR)**
   - Input: low-res (downsampled), Target: high-res
   - Dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (create LR from HR)
   - Metric: PSNR / SSIM

4) **Frame interpolation ×2 (triplet)**
   - Input: frame(t), frame(t+2), Target: frame(t+1)
   - Dataset: [Vimeo-90K Triplet](http://toflow.csail.mit.edu/)
   - Metric: PSNR / SSIM

5) **Person segmentation → background blur/replace (segmentation)**
   - Input: RGB image, Target: person mask
   - Dataset: [COCO](https://cocodataset.org/#download) (person masks) or [Supervisely Person Segmentation](https://supervise.ly/datasets/person-segmentation)
   - Metric: mIoU (and qualitative live demo)

6) **Makeup transfer (image-to-image translation)**
   - Input: face no-makeup, Target: makeup style
   - Dataset: [Makeup Transfer (MT)](https://github.com/wtjiang98/PSGAN) (paired or weakly paired, depending on chosen source)
   - Metric: qualitative + optional identity similarity

7) **Add hats / glasses via Mask→Photo (pix2pix, conditional)**
   - Input: segmentation mask (editable), Target: photo
   - Dataset: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) (includes hat/eyeglasses classes)
   - Metric: qualitative + optional FID on a subset (optional)

8) **Edge→Portrait (pix2pix, conditional)**
   - Input: edge map (Canny), Target: photo
   - Dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset) (generate edges from photos)
   - Metric: qualitative + optional PSNR/SSIM (weak), focus on visual results

9) **Night→Day translation (unpaired; CycleGAN / pix2pix-style)**
   - Input domain: night images, Output domain: day images
   - Dataset: [night2day (CycleGAN)](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/)
   - Metric: qualitative (optionally FID/LPIPS on a subset)

10) **Selfie→Anime / Cartoon (unpaired; CycleGAN)**
   - Input domain: selfies, Output domain: anime
    - Dataset: [selfie2anime (UGATIT)](https://github.com/taki0112/UGATIT) and/or [AnimeGAN2 datasets](https://github.com/TachibanaYoshino/AnimeGANv2)
    - Metric: qualitative

> Teacher note: You (the instructor) will tell teams which dataset package/version to use and provide download instructions for your class environment. Links above are references; verify licenses and mirrors.

---

## Requirements (quick checks)

- Python 3.x: `python3 -V`
- Conda (recommended): `conda -V`
- Git: `git --version`

If any command fails, install the missing tool before running the project scripts.

### Install hints by OS

- macOS (Homebrew):
  - `brew install python git`
  - `brew install --cask miniconda` (or install Miniforge)
- Ubuntu / Jetson (apt):
  - `sudo apt update && sudo apt install -y python3 python3-venv git`
  - Conda: install Miniforge or Miniconda from the official site.
- Windows (PowerShell, Chocolatey):
  - `choco install -y python git miniconda3`
  - Restart the terminal so `python` and `conda` are on PATH.
- Windows (PowerShell, winget):
  - `winget install --id Python.Python.3.11 -e`
  - `winget install --id Git.Git -e`
  - `winget install --id Anaconda.Miniconda3 -e`
  - Restart the terminal so `python` and `conda` are on PATH.
- Windows (GUI installer):
  - Download from `https://www.python.org/downloads/`, `https://git-scm.com/downloads`, and `https://docs.conda.io/en/latest/miniconda.html`.
  - Ensure "Add to PATH" is checked during installation.

---

## Suggested baseline models (student-friendly)

Param counts are approximate for lightweight baselines; adjust if you change channel widths or blocks.

| Project | Suggested models (easy to train) | Params (approx) | Suggested train size | Suggested fine-tune size |
| --- | --- | --- | --- | --- |
| 1) Low-light | Tiny U-Net (base=16), ResNet-9 | ~2M, ~5M | 2k-5k pairs | 200-500 pairs |
| 2) Denoising | DnCNN (17 layers), Tiny U-Net | ~0.56M, ~2M | 2k-5k pairs | 200-500 pairs |
| 3) SR x2 | FSRCNN-x2, ESPCN-x2 | ~0.03M, ~0.05M | 800-1,000 images | 100-200 images |
| 4) Interpolation | U-Net small, SepConv-lite | ~2M, ~1-2M | 50k+ triplets | 2k-5k triplets |
| 5) Segmentation | Fast-SCNN, BiSeNet-lite | ~1.1M, ~2-3M | 5k-20k masks | 300-1k masks |
| 6) Makeup transfer | ResNet-6 generator, U-Net small | ~7-8M, ~2M | 3k-10k pairs | 200-500 pairs |
| 7) Mask→Photo | pix2pix U-Net small, ResNet-6 | ~2M, ~7-8M | 10k-20k pairs | 1k-2k pairs |
| 8) Edge→Portrait | pix2pix U-Net small, ResNet-6 | ~2M, ~7-8M | 20k-50k images | 1k-2k images |
| 9) Night→Day | CycleGAN ResNet-6, Tiny U-Net (paired variant) | ~7-8M, ~2M | 5k-20k per domain | 500-1k per domain |
| 10) Selfie→Anime | CycleGAN ResNet-6, U-Net small | ~7-8M, ~2M | 3k-10k per domain | 300-800 per domain |

---

## Repo layout (expected)
You will implement the following files (minimal set):

```
.
├── data/
├── ignore/
├── report.md
├── README.md
└── src/
    ├── configs/
    │   └── <project_name>.yaml
    ├── datasets/
    │   ├── paired.py           # for 1/2/3 (and deblur if you add it)
    │   ├── triplet.py          # for 4
    │   ├── segmentation.py     # for 5
    │   └── unpaired.py         # for 9/10 (+ possibly 6)
    ├── models/
    │   ├── unet.py
    │   ├── sr_fsrcnn.py        # optional for project 3
    │   ├── seg_bisenet_lite.py # optional for project 5
    │   └── gan_generator.py    # optional for pix2pix/cyclegan
    ├── scripts/
    │   └── export_onnx.py
    ├── train.py
    ├── eval.py
    └── demo_live_split.py
```

---

# Phase 1 — Dataset preparation + dataloaders

## 1.1 Create a dataset folder
Standardize your dataset into one of these formats.

### A) Paired image-to-image (projects 1,2,3)
```
data/<dataset_name>/
  train/
    input/   *.png|*.jpg
    target/  *.png|*.jpg   (same filenames as input)
  val/
    input/
    target/
  test/
    input/
    target/
```

### B) Triplet interpolation (project 4)
```
data/<dataset_name>/
  train/
    frame0/  *.png
    frame1/  *.png   (target middle frame)
    frame2/  *.png
  val/
    frame0/
    frame1/
    frame2/
```

### C) Segmentation (project 5)
```
data/<dataset_name>/
  train/
    images/ *.png|*.jpg
    masks/  *.png         (single-channel, 0 background / 1 person)
  val/
    images/
    masks/
```

### D) Unpaired domain translation (projects 9,10; sometimes 6)
```
data/<dataset_name>/
  trainA/ *.png|*.jpg   (domain A)
  trainB/ *.png|*.jpg   (domain B)
  valA/   *.png|*.jpg
  valB/   *.png|*.jpg
```

---

## 1.2 Dataset download (examples)
Use the official dataset download pages and place the files into `data/<dataset_name>/...`.

Put your team’s download steps in a small text file:
- `data/DATASET_NOTES.txt`: where you got it, what split you used, how you preprocessed.

If you need a CLI download example, put URLs inside a code block like this:
```bash
# Example pattern (replace with the dataset’s official link)
mkdir -p data/lol_raw
# wget <DATASET_URL>
# unzip <dataset.zip> -d data/lol_raw
```

---

## 1.3 Implement dataloaders
Create a dataloader that returns tensors in **[0,1]**, shape **(B,C,H,W)**.

Minimum requirements:
- resize or random crop to a fixed training size (e.g., 256 or 320)
- basic augmentations (flip) are optional
- support `--train` and `--val`

Files to implement:
- `src/datasets/paired.py`
- `src/datasets/triplet.py` (only if project 4)
- `src/datasets/segmentation.py` (only if project 5)
- `src/datasets/unpaired.py` (only if project 9/10; sometimes 6)

---

# Phase 2 — Train, evaluate, test model

## 2.1 Select a baseline model
Use a simple baseline that can train quickly:
- Projects 1/2: **Tiny U-Net** or DnCNN (denoise)
- Project 3: FSRCNN/ESPCN or U-Net (LR→HR)
- Project 4: small interpolation net (U-Net-like)
- Project 5: lightweight segmentation network (BiSeNet-lite / MobileNet-DeepLab)
- Projects 6–10: pix2pix (conditional) or CycleGAN (unpaired)

Put your choice in:
- `src/configs/<project_name>.yaml`

Example config keys (suggested):
```yaml
project: lowlight_unet
data_root: data/lol
train_size: 256
batch_size: 16
epochs: 8
lr: 0.001
model: unet_tiny
loss: l1
mixed_precision: true
```

---

## 2.2 Train
Run:
```bash
python3 src/train.py --config src/configs/<project_name>.yaml
```

Minimum training outputs:
- prints loss each epoch
- saves `src/runs/<exp_name>/best.pt`
- logs one validation metric:
  - PSNR/SSIM for image-to-image
  - mIoU for segmentation
  - PSNR/SSIM for interpolation
  - for GANs: at least save sample grids each epoch

---

## 2.3 Evaluate + test
Run:
```bash
python3 src/eval.py --config src/configs/<project_name>.yaml --weights src/runs/<exp_name>/best.pt
```

You must report:
- one metric (PSNR/SSIM or mIoU)
- plus 8–16 qualitative examples saved to `src/runs/<exp_name>/samples/`

---

## 2.4 (Optional but recommended) Export to ONNX
Export:
```bash
python3 src/scripts/export_onnx.py --weights src/runs/<exp_name>/best.pt --out src/runs/<exp_name>/model.onnx
```

If time allows, you may accelerate inference with TensorRT (bonus).

---

# Phase 3 — Jetson integration: real-time split-screen demo

## 3.1 Run live demo (split screen)
Your live demo must:
- capture frames from the IMX219 CSI camera
- run your model inference
- display a window with **left = original**, **right = processed**
- show FPS (or frame time) on screen

Command:
```bash
python3 src/demo_live_split.py --weights src/runs/<exp_name>/best.pt --size 256
```

### CSI camera capture (Jetson)
Use a GStreamer pipeline (example):
```python
"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
```

If CSI is not available, fall back to:
- `cv2.VideoCapture(0)`

---

## 3.2 Performance checklist (minimum)
Do at least **3** of the following:
- run inference at a smaller resolution (e.g., 256/320) and upscale for display
- enable mixed precision (FP16) in inference
- measure FPS and report it
- reduce post-processing cost (simple ops, avoid heavy CPU work)
- optionally export ONNX and compare speed

---

# What to submit
Each team submits:
- the code in this repo
- `src/runs/<exp_name>/` folder containing:
  - `best.pt`
  - `eval_metrics.json` (or printed metrics pasted into report)
  - sample images
- a short `report.md` (1 page):
  - dataset + preprocessing
  - model + loss
  - metric results
  - Jetson FPS + demo description
  - 2–3 lessons learned (accuracy vs speed)

---

## Tips for a strong demo
- Use a dark room for low-light/denoise projects
- For SR: show a downsampled input on the left, SR output on the right
- For segmentation: right side = background blur or replacement
- Keep the demo stable: fixed resolution, no stutters, clear FPS overlay
