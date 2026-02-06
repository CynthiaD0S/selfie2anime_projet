#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time split-screen GAN demo.")
    parser.add_argument(
        "--weights",
        default="runs/selfie2anime/model_best.pt",
        help="Path to trained weights (.pt/.pth). Can be a full checkpoint.",
    )
    parser.add_argument(
        "--gen",
        default="resnet6",
        choices=["unet_small", "unet_tiny", "resnet6", "resnet9"],
        help="Generator type to instantiate (must match training).",
    )
    parser.add_argument(
        "--direction",
        default="AB",
        choices=["AB", "BA"],
        help="Which generator to use if checkpoint is CycleGAN: AB=Selfie->Anime, BA=Anime->Selfie",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Resize camera frames to this square size for inference.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or a video path / pipeline string.",
    )
    parser.add_argument(
        "--window",
        default="GAN Demo",
        help="Window title.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay FPS + inference time on the video.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar (useful for webcam).",
    )
    return parser.parse_args()


def prepare_frame(frame_bgr, size):
    # model expects input in [-1,1]
    resized = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(rgb).float() / 255.0   # [0,1]
    x = x * 2.0 - 1.0                           # [-1,1]
    x = x.permute(2, 0, 1).unsqueeze(0)         # (1,C,H,W)
    return x


def tensor_to_bgr(tensor):
    # output in [-1,1] -> [0,255]
    image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = (image + 1.0) / 2.0
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def is_webcam_source(source: str) -> bool:
    return isinstance(source, str) and source.isdigit()


def draw_hud(frame, text, y=30):
    cv2.putText(
        frame,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def extract_generator_state_dict(ckpt, direction: str):
    """
    Supports:
    - plain generator state_dict
    - OneWay checkpoint: {"G_state_dict": ...}
    - CycleGAN checkpoint: {"G_AB_state_dict": ..., "G_BA_state_dict": ...}
    - wrapped: {"state_dict": ...} or {"model": ...}
    """
    if not isinstance(ckpt, dict):
        return ckpt  # might already be a state_dict

    # CycleGAN
    if "G_AB_state_dict" in ckpt or "G_BA_state_dict" in ckpt:
        key = "G_AB_state_dict" if direction == "AB" else "G_BA_state_dict"
        if key not in ckpt:
            raise KeyError(f"Checkpoint has no key {key}. Available keys: {list(ckpt.keys())[:20]} ...")
        return ckpt[key]

    # One-way
    if "G_state_dict" in ckpt:
        return ckpt["G_state_dict"]

    # common wrappers
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if "model" in ckpt:
        return ckpt["model"]

    # maybe it's already a plain state_dict
    any_tensor = any(torch.is_tensor(v) for v in ckpt.values())
    if any_tensor:
        return ckpt

    raise KeyError(f"Could not find generator weights in checkpoint. Keys: {list(ckpt.keys())}")


def main():
    args = parse_args()

    # Load your gan_generator.py
    root = Path(__file__).resolve().parent
    model_path = root / "src" / "models" / "gan_generator.py"
    model_module = load_module("gan_generator", model_path)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate generator
    model = model_module.create_generator(args.gen).to(device)

    # Load weights
    ckpt = torch.load(args.weights, map_location=device)
    state_dict = extract_generator_state_dict(ckpt, args.direction)

    # strict=True is fine if arch matches training (resnet6 vs resnet9 etc.)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded weights: {args.weights}")
    if isinstance(ckpt, dict) and ("G_AB_state_dict" in ckpt or "G_BA_state_dict" in ckpt):
        print(f"Checkpoint type: CycleGAN | using direction: {args.direction}")
    else:
        print("Checkpoint type: One-way / state_dict")
    print(f"Generator: {args.gen}")

    # Open capture
    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera/source: {args.source}")

    webcam = is_webcam_source(args.source)

    total_frames = None
    if (not webcam) and (not args.no_tqdm):
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tf > 0:
            total_frames = tf

    pbar = tqdm(total=total_frames, desc="Video", unit="frame", dynamic_ncols=True) if total_frames else None

    last_time = time.perf_counter()
    fps_smooth = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            x = prepare_frame(frame, args.size).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                y = model(x)
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0

            processed = tensor_to_bgr(y)
            processed = cv2.resize(processed, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            combined = cv2.hconcat([frame, processed])

            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 1.0 / dt
                fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)

            if args.show_fps and fps_smooth is not None:
                draw_hud(combined, f"FPS: {fps_smooth:5.1f} | Inference: {infer_ms:5.1f} ms")

            cv2.imshow(args.window, combined)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    fps=f"{fps_smooth:.1f}" if fps_smooth else "?",
                    infer_ms=f"{infer_ms:.1f}",
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if pbar is not None:
            pbar.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
