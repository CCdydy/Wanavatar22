# Audio2Avatar via Wan-S2V + Wan-Animate

Inference-only pipeline: single image + audio -> high-fidelity avatar video.

```
[avatar ref image] + [audio]
         |
    Wan-S2V-14B          (coarse driving video)
         |
   extract drivers       (pose + face)
         |
   Wan-Animate-14B       (final avatar video)
```

## Quick Start

```bash
cd /media/zzy/SN5601/wan22/Audio2Avatar

# Full pipeline (V0 + V1)
python scripts/pipeline.py \
    --ref_image assets/ref/avatar.png \
    --audio assets/audio/speech.wav \
    --output_dir outputs/run_001

# Or run steps individually:
python scripts/run_s2v.py --ref_image assets/ref/avatar.png --audio assets/audio/speech.wav --output outputs/s2v/coarse.mp4
python scripts/extract_drivers.py --coarse_video outputs/s2v/coarse.mp4 --ref_image assets/ref/avatar.png --output_dir outputs/drivers
python scripts/run_animate.py --src_root_path outputs/drivers --output outputs/animate/final.mp4
```

## Evaluation (V0 vs V1)

Compare `outputs/run_001/s2v/coarse_s2v.mp4` (V0) against `outputs/run_001/animate/final_avatar.mp4` (V1) on:

1. Identity stability vs reference
2. Lip-sync rhythm preservation
3. Facial detail quality
4. Overall completion level

Pass if 3/4 criteria show improvement.

## Dataset

```
dataset/
├── sample_0001/
│   ├── ref.png
│   ├── audio.wav
│   ├── prompt.txt
│   └── meta.json
└── sample_0002/
    └── ...
```
