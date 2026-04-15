"""Build a curated TFHP subset for talking-head experiments.

Selection goal:
- slightly longer clips
- clear subject
- large, stable frontal face

The script scans TFHP raw videos, samples a few frames per video, computes:
- duration / fps / resolution
- face detection success rate
- face area ratio
- image sharpness (Laplacian variance)

Then it filters and ranks videos, and creates a curated dataset using symlinks
to avoid duplicating the raw media.

Usage:
    /home/zzy/anaconda3/envs/wan22/bin/python \
        /media/zzy/SN5601/wan22/Audio2Avatar/scripts/build_tfhp_dataset.py \
        --src_root /media/zzy/data/TFHP_raw/data \
        --out_root /media/zzy/data/TFHP_long_clear_128 \
        --top_k 128
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def compute_video_stats(video_path: Path, face_detector, num_samples: int = 5) -> dict | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps > 0 else 0.0

    if frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

    sample_indices = np.linspace(0, max(frames - 1, 0), num=min(num_samples, frames), dtype=int)
    sharpness_values = []
    face_area_ratios = []
    face_width_ratios = []
    detected_frames = 0

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            continue

        detected_frames += 1
        best = max(faces, key=lambda box: box[2] * box[3])
        _, _, fw, fh = best
        face_area_ratios.append(float((fw * fh) / (width * height)))
        face_width_ratios.append(float(fw / width))

    cap.release()

    if not sharpness_values:
        return None

    return {
        "video_path": str(video_path),
        "sample_id": video_path.parent.name,
        "clip_name": video_path.name,
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration,
        "detected_frames": detected_frames,
        "sampled_frames": len(sample_indices),
        "detect_rate": detected_frames / max(len(sample_indices), 1),
        "sharpness_mean": float(np.mean(sharpness_values)),
        "sharpness_min": float(np.min(sharpness_values)),
        "face_area_ratio_max": float(max(face_area_ratios) if face_area_ratios else 0.0),
        "face_area_ratio_mean": float(np.mean(face_area_ratios) if face_area_ratios else 0.0),
        "face_width_ratio_max": float(max(face_width_ratios) if face_width_ratios else 0.0),
    }


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def rank_candidates(items: list[dict]) -> list[dict]:
    dur_n = normalize([x["duration"] for x in items])
    face_n = normalize([x["face_area_ratio_max"] for x in items])
    sharp_n = normalize([x["sharpness_mean"] for x in items])
    detect_n = normalize([x["detect_rate"] for x in items])

    ranked = []
    for item, d, f, s, r in zip(items, dur_n, face_n, sharp_n, detect_n):
        score = 0.30 * d + 0.35 * f + 0.20 * s + 0.15 * r
        x = dict(item)
        x["score"] = float(score)
        ranked.append(x)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def symlink_dataset(items: list[dict], out_root: Path) -> None:
    clips_dir = out_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    summary_path = out_root / "summary.json"

    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, item in enumerate(items):
            sample_name = f"{idx:04d}_{item['sample_id']}_{Path(item['clip_name']).stem}"
            sample_dir = clips_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            src = Path(item["video_path"])
            dst = sample_dir / "video.mp4"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)

            meta = {
                "id": sample_name,
                **item,
            }
            with (sample_dir / "meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")

    summary = {
        "num_selected": len(items),
        "duration_mean": float(np.mean([x["duration"] for x in items])) if items else 0.0,
        "duration_min": float(np.min([x["duration"] for x in items])) if items else 0.0,
        "face_area_ratio_mean": float(np.mean([x["face_area_ratio_max"] for x in items])) if items else 0.0,
        "sharpness_mean": float(np.mean([x["sharpness_mean"] for x in items])) if items else 0.0,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Build curated TFHP subset")
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--min_duration", type=float, default=20.0)
    parser.add_argument("--min_detect_rate", type=float, default=0.6)
    parser.add_argument("--min_face_area_ratio", type=float, default=0.045)
    parser.add_argument("--min_sharpness", type=float, default=180.0)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade face detector.")

    video_paths = sorted(src_root.glob("TH_*/*.mp4"))
    print(f"Scanning {len(video_paths)} videos from {src_root}")

    all_stats = []
    for i, video_path in enumerate(video_paths, start=1):
        stats = compute_video_stats(video_path, face_detector, num_samples=args.num_samples)
        if stats is not None:
            all_stats.append(stats)
        if i % 50 == 0:
            print(f"  processed {i}/{len(video_paths)}")

    raw_manifest = out_root / "all_stats.json"
    with raw_manifest.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    filtered = [
        x for x in all_stats
        if x["duration"] >= args.min_duration
        and x["detect_rate"] >= args.min_detect_rate
        and x["face_area_ratio_max"] >= args.min_face_area_ratio
        and x["sharpness_mean"] >= args.min_sharpness
    ]
    print(f"Candidates after filtering: {len(filtered)}")

    ranked = rank_candidates(filtered)
    selected = ranked[:args.top_k]
    print(f"Selected top {len(selected)} videos")

    symlink_dataset(selected, out_root)

    preview = out_root / "top10.json"
    with preview.open("w", encoding="utf-8") as f:
        json.dump(selected[:10], f, indent=2, ensure_ascii=False)

    print(f"Done. Curated dataset written to {out_root}")


if __name__ == "__main__":
    main()
