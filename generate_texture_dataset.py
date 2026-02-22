#!/usr/bin/env python3
"""Generate a sound texture dataset with frame-level annotations using pyo.

Outputs:
- <dataset_dir>/parameters.json
- <dataset_dir>/<prefix>_0001.wav
- <dataset_dir>/<prefix>_0001.csv
- ...

Each CSV is annotated at fixed fps (default 75).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from typing import Dict

import numpy as np
try:
    from pyo import (
        ButLP,
        Cloud,
        DataTable,
        Disto,
        HannTable,
        Noise,
        Pan,
        PinkNoise,
        Server,
        Sine,
        TableRead,
        TrigEnv,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyo'. On this machine, run with /usr/bin/python3.12 "
        "after installing system package 'python3-pyo'."
    ) from exc


PARAMETERS = {
    "density": {"type": "continuous", "min": 0.0, "max": 1.0},
    "loudness": {"type": "continuous", "min": 0.0, "max": 1.0},
    "brightness": {"type": "continuous", "min": 0.0, "max": 1.0},
    "roughness": {"type": "continuous", "min": 0.0, "max": 1.0},
    "speed": {"type": "continuous", "min": 0.0, "max": 1.0},
}


def _smooth(x: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    if kernel_size <= 1:
        return x
    kernel = np.ones(kernel_size, dtype=np.float64)
    kernel /= kernel.sum()
    padded = np.pad(x, (kernel_size, kernel_size), mode="edge")
    y = np.convolve(padded, kernel, mode="same")
    return y[kernel_size:-kernel_size]


def _random_curve(
    rng: np.random.Generator,
    frames: int,
    vmin: float = 0.0,
    vmax: float = 1.0,
    anchor_frames: int = 75,
) -> np.ndarray:
    anchors = max(3, math.ceil(frames / max(1, anchor_frames)) + 1)
    idx = np.linspace(0, frames - 1, anchors)
    vals = rng.uniform(vmin, vmax, size=anchors)
    curve = np.interp(np.arange(frames), idx, vals)
    curve = _smooth(curve, kernel_size=17)
    return np.clip(curve, vmin, vmax)


def generate_parameter_frames(
    rng: np.random.Generator,
    frames: int,
    fps: int,
) -> Dict[str, np.ndarray]:
    anchor = max(12, int(1.1 * fps))

    density = _random_curve(rng, frames, 0.15, 0.95, anchor)
    loudness = _random_curve(rng, frames, 0.2, 0.9, anchor)
    brightness = _random_curve(rng, frames, 0.1, 1.0, anchor)
    roughness = _random_curve(rng, frames, 0.05, 0.9, anchor)
    speed = _random_curve(rng, frames, 0.1, 1.0, anchor)

    # Mild coupling to keep textures coherent.
    loudness = np.clip(0.7 * loudness + 0.3 * density, 0.0, 1.0)
    brightness = np.clip(0.8 * brightness + 0.2 * speed, 0.0, 1.0)

    return {
        "density": density,
        "loudness": loudness,
        "brightness": brightness,
        "roughness": roughness,
        "speed": speed,
    }


def write_csv(csv_path: Path, param_frames: Dict[str, np.ndarray]) -> None:
    names = list(PARAMETERS.keys())
    n = len(next(iter(param_frames.values())))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for i in range(n):
            writer.writerow([f"{float(param_frames[k][i]):.6f}" for k in names])


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def resample_param_frames(param_frames: Dict[str, np.ndarray], target_frames: int) -> Dict[str, np.ndarray]:
    old_frames = len(next(iter(param_frames.values())))
    if target_frames <= 0:
        target_frames = 1
    if target_frames == old_frames:
        return param_frames

    old_x = np.linspace(0.0, 1.0, old_frames)
    new_x = np.linspace(0.0, 1.0, target_frames)
    out: Dict[str, np.ndarray] = {}
    for k, v in param_frames.items():
        out[k] = np.interp(new_x, old_x, v)
    return out


def render_with_pyo(
    wav_path: Path,
    param_frames: Dict[str, np.ndarray],
    sr: int,
    fps: int,
    channels: int = 2,
) -> None:
    frames = len(next(iter(param_frames.values())))
    dur = frames / float(fps)

    s = Server(sr=sr, nchnls=channels, buffersize=512, duplex=0, audio="offline").boot()
    s.recordOptions(dur=dur, filename=str(wav_path), fileformat=0, sampletype=1)

    tables: Dict[str, DataTable] = {}
    controls = {}
    for name, values in param_frames.items():
        t = DataTable(size=frames)
        t.replace(values.tolist())
        tables[name] = t
        controls[name] = TableRead(t, freq=1.0 / dur, loop=False, interp=2)

    density = controls["density"]
    loudness = controls["loudness"]
    brightness = controls["brightness"]
    roughness = controls["roughness"]
    speed = controls["speed"]

    base = PinkNoise(mul=0.3)
    burst_noise = Noise(mul=0.5)

    trig = Cloud(density=(2.0 + density * 26.0) * (0.3 + speed * 0.7))
    burst_env = TrigEnv(trig, table=HannTable(), dur=0.03 + (1.0 - speed) * 0.15, mul=0.65)

    cutoff = 250.0 + brightness * 7200.0
    motion = Sine(freq=0.03 + speed * 1.2, mul=600.0 + brightness * 1200.0, add=cutoff)

    bed = ButLP(base, freq=motion, mul=0.55)
    bursts = ButLP(burst_noise, freq=cutoff * 1.35 + 300.0, mul=burst_env)

    mix = bed + bursts
    rough = Disto(mix, drive=0.05 + roughness * 0.9, slope=0.85, mul=1.0)

    pan_lfo = Sine(freq=0.015 + speed * 0.17, mul=0.32, add=0.5)
    out = Pan(rough, outs=channels, pan=pan_lfo, spread=0.1, mul=0.05 + loudness * 0.95)
    out.out()

    s.start()
    s.shutdown()


def write_readme(dataset_dir: Path, fps: int, sr: int, total_seconds: float, files: int) -> None:
    text = f"""# Sound Texture Dataset

Generated by `generate_texture_dataset.py` using **Python + pyo**.

## Summary
- Total files: {files}
- Total duration: {total_seconds/60.0:.2f} minutes
- Sample rate: {sr} Hz
- Annotation frame rate: {fps} fps
- Sound type: stochastic noise textures (wind/rain/fire-like)

## Parameters
- `density` (continuous, 0-1): event activity of the burst layer
- `loudness` (continuous, 0-1): overall amplitude
- `brightness` (continuous, 0-1): filter cutoff tendency
- `roughness` (continuous, 0-1): distortion amount
- `speed` (continuous, 0-1): modulation speed and event pace

## Files
Each audio file has a matching CSV:
- `sound_0001.wav`
- `sound_0001.csv`

CSV format:
- Header row: parameter names
- One row per frame at {fps} fps
"""
    (dataset_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a pyo-based sound texture dataset.")
    p.add_argument("--out", type=Path, default=Path("raw"), help="Output dataset directory")
    p.add_argument("--target-minutes", type=float, default=10.0, help="Target total duration in minutes")
    p.add_argument("--file-min-sec", type=float, default=8.0, help="Minimum duration per file (seconds)")
    p.add_argument("--file-max-sec", type=float, default=14.0, help="Maximum duration per file (seconds)")
    p.add_argument("--sr", type=int, default=44100, help="Audio sample rate")
    p.add_argument("--fps", type=int, default=75, help="Annotation frame rate")
    p.add_argument("--prefix", type=str, default="sound", help="Output file prefix")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.file_min_sec <= 0 or args.file_max_sec <= 0:
        raise ValueError("file durations must be positive")
    if args.file_min_sec > args.file_max_sec:
        raise ValueError("--file-min-sec must be <= --file-max-sec")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "parameters.json").write_text(
        json.dumps(PARAMETERS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rng = np.random.default_rng(args.seed)
    target_seconds = args.target_minutes * 60.0

    total = 0.0
    idx = 1

    while total < target_seconds:
        sec = float(rng.uniform(args.file_min_sec, args.file_max_sec))
        remaining = target_seconds - total
        if remaining < args.file_min_sec:
            sec = max(remaining, 1.0)
        elif sec > remaining:
            sec = remaining

        frames = max(1, int(round(sec * args.fps)))
        actual_sec = frames / float(args.fps)
        params = generate_parameter_frames(rng, frames=frames, fps=args.fps)

        stem = f"{args.prefix}_{idx:04d}"
        wav_path = out_dir / f"{stem}.wav"
        csv_path = out_dir / f"{stem}.csv"

        render_with_pyo(wav_path, params, sr=args.sr, fps=args.fps, channels=2)
        rendered_sec = wav_duration_seconds(wav_path)
        rendered_frames = max(1, int(round(rendered_sec * args.fps)))
        aligned_params = resample_param_frames(params, rendered_frames)
        write_csv(csv_path, aligned_params)

        total += actual_sec
        print(f"[{idx:04d}] {wav_path.name}  dur={actual_sec:.2f}s")
        idx += 1

    write_readme(out_dir, fps=args.fps, sr=args.sr, total_seconds=total, files=idx - 1)
    print(f"Done. Files: {idx - 1}, total duration: {total/60.0:.2f} min, out: {out_dir}")


if __name__ == "__main__":
    main()
