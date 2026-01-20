# Repository Guidelines

This repository is a teaching template for a 2-day embedded AI project. The goal is to guide students from dataset preparation to model training, evaluation, and a Jetson Nano real-time demo. Keep contributions instructional, minimal, and runnable on student hardware.

## Project Structure & Module Organization

- `src/configs/`: per-project YAML configs (e.g., `src/configs/lowlight_unet.yaml`).
- `src/datasets/`: dataloaders by task type (`paired.py`, `triplet.py`, `segmentation.py`, `unpaired.py`).
- `src/models/`: baseline architectures (U-Net, FSRCNN, segmentation nets, GAN generator).
- `src/scripts/`: utilities like `export_onnx.py`.
- `src/train.py`, `src/eval.py`, `src/demo_live_split.py`: main entry points.
- `data/`: standardized dataset layouts; students must add `data/DATASET_NOTES.txt`.
- `src/runs/`: experiment outputs (`best.pt`, metrics, samples).
- `report.md`: 1-page submission with dataset, model, metrics, and Jetson FPS.

## Student Workflow (Instructor-Focused)

1) **Dataset setup**: pick a project and normalize the dataset into the required folder layout in `data/<dataset_name>/`.
2) **Dataloader creation**: implement the matching loader in `src/datasets/` that returns `(B,C,H,W)` tensors in `[0,1]`.
3) **Training**: choose a baseline model and configure it in `src/configs/<project_name>.yaml`.
4) **Evaluation/validation**: run `eval.py` to compute PSNR/SSIM or mIoU and save samples.
5) **Jetson demo**: run `demo_live_split.py` with FPS overlay and split-screen output.

## Build, Test, and Development Commands

- `python3 src/train.py --config src/configs/<project_name>.yaml`: trains and saves `src/runs/<exp_name>/best.pt`.
- `python3 src/eval.py --config src/configs/<project_name>.yaml --weights src/runs/<exp_name>/best.pt`: validates and saves metrics/samples.
- `python3 src/demo_live_split.py --weights src/runs/<exp_name>/best.pt --size 256`: Jetson Nano split-screen demo.
- `python3 src/scripts/export_onnx.py --weights src/runs/<exp_name>/best.pt --out src/runs/<exp_name>/model.onnx`: optional ONNX export.

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8 (`snake_case` functions/files, `CamelCase` classes).
- Configs: `<project_name>.yaml` in `src/configs/`.
- Experiments: `src/runs/<exp_name>/` with `best.pt`, samples, and metrics JSON.

## Testing Guidelines

- No automated test harness is defined yet.
- If adding tests, use `pytest` in `tests/` (e.g., `tests/test_datasets.py`).

## Commit & Pull Request Guidelines

- No existing commit convention. Prefer Conventional Commits (e.g., `feat: add triplet dataloader`).
- PRs should include: dataset name, config used, key metrics, and demo screenshot or sample grid.

## Data & Artifact Handling

- Do not commit large datasets or raw videos.
- Keep only small example outputs for documentation; store the rest in `src/runs/`.
