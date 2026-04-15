"""Full Audio2Avatar pipeline: S2V -> Extract Drivers -> Animate.

Runs all 3 steps sequentially for a single sample.
Also produces V0 (direct S2V) as the baseline comparison.

Usage:
    cd /media/zzy/SN5601/wan22/Audio2Avatar
    python scripts/pipeline.py \
        --ref_image assets/ref/avatar.png \
        --audio assets/audio/speech.wav \
        --output_dir outputs/run_001
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description='Audio2Avatar full pipeline')
    parser.add_argument('--ref_image', type=str, required=True)
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/run_001')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--skip_s2v', action='store_true',
                        help='Skip S2V if coarse video already exists')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    s2v_dir = output_dir / 's2v'
    drivers_dir = output_dir / 'drivers'
    animate_dir = output_dir / 'animate'

    coarse_path = str(s2v_dir / 'coarse_s2v.mp4')
    final_path = str(animate_dir / 'final_avatar.mp4')

    timings = {}

    # ── Step 1: S2V (V0 baseline) ──────────────────────────────────
    if args.skip_s2v and os.path.exists(coarse_path):
        print(f"Skipping S2V (coarse video exists: {coarse_path})")
    else:
        print("=" * 60)
        print("Step 1: Wan-S2V -> coarse driving video (V0)")
        print("=" * 60)
        t0 = time.time()

        from run_s2v import run_s2v
        run_s2v(args.ref_image, args.audio, coarse_path,
                prompt=args.prompt, seed=args.seed, num_repeat=None)

        timings['s2v'] = time.time() - t0
        print(f"  S2V took {timings['s2v']:.0f}s")

    # ── Step 2: Extract drivers ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Extract pose + face drivers")
    print("=" * 60)
    t0 = time.time()

    from extract_drivers import extract_drivers
    success = extract_drivers(
        coarse_path, args.ref_image, str(drivers_dir),
        fps=args.fps, replace_flag=False,
    )
    if not success:
        print("ERROR: Driver extraction failed. Aborting.")
        sys.exit(1)

    timings['extract'] = time.time() - t0
    print(f"  Extraction took {timings['extract']:.0f}s")

    # ── Step 3: Animate (V1) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Wan-Animate -> final avatar video (V1)")
    print("=" * 60)
    t0 = time.time()

    from run_animate import run_animate
    run_animate(
        str(drivers_dir), final_path,
        seed=args.seed, clip_len=77,
        sampling_steps=20, guide_scale=1.0,
        fps=args.fps,
    )

    timings['animate'] = time.time() - t0
    print(f"  Animate took {timings['animate']:.0f}s")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  V0 (direct S2V):     {coarse_path}")
    print(f"  V1 (S2V -> Animate): {final_path}")
    print(f"  Drivers:             {drivers_dir}/")
    total = sum(timings.values())
    print(f"  Total time:          {total:.0f}s ({total/60:.1f} min)")
    for step, t in timings.items():
        print(f"    {step}: {t:.0f}s")

    # Save run metadata
    meta = {
        'ref_image': args.ref_image,
        'audio': args.audio,
        'prompt': args.prompt,
        'seed': args.seed,
        'fps': args.fps,
        'coarse_video': coarse_path,
        'final_video': final_path,
        'timings': timings,
    }
    with open(output_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
