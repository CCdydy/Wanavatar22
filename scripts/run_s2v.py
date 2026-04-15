"""Step 1: Generate coarse driving video via Wan-S2V.

Input:  reference image + audio
Output: coarse_s2v.mp4

This is also the V0 baseline (direct S2V output).
"""
import argparse
import os
import sys

WAN_REPO = '/media/zzy/SN5601/wan22/Wan2.2'
sys.path.insert(0, WAN_REPO)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def run_s2v(ref_image, audio, output_path, prompt=None, seed=42,
            num_repeat=None, pose_video=None):
    import torch
    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video

    if prompt is None:
        prompt = ("A frontal portrait of a person speaking naturally, "
                  "subtle head motion, realistic facial expressions, "
                  "stable identity, clean background.")

    cfg = WAN_CONFIGS['s2v-14B']
    ckpt_dir = '/media/zzy/SN5601/wan22/Wan2.2-S2V-14B'

    print(f"Loading Wan-S2V...")
    wan_s2v = wan.WanS2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=0, rank=0,
        t5_cpu=True,
        convert_model_dtype=True,
    )

    print(f"Generating coarse video...")
    print(f"  ref: {ref_image}")
    print(f"  audio: {audio}")
    if pose_video:
        print(f"  pose_video: {pose_video}")
    video = wan_s2v.generate(
        input_prompt=prompt,
        ref_image_path=ref_image,
        audio_path=audio,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        num_repeat=num_repeat,
        pose_video=pose_video,
        max_area=MAX_AREA_CONFIGS.get('832*480', 832 * 480),
        infer_frames=48,
        shift=3.0,
        sampling_steps=40,
        guide_scale=4.5,
        seed=seed,
        offload_model=True,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_video(
        tensor=video[None],
        save_file=output_path,
        fps=cfg.sample_fps,
        nrow=1, normalize=True, value_range=(-1, 1),
    )
    print(f"Saved: {output_path}")

    del wan_s2v, video
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Step 1: Wan-S2V coarse video')
    parser.add_argument('--ref_image', type=str, required=True)
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/s2v/coarse_s2v.mp4')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_repeat', type=int, default=None,
                        help='Number of clips. None=auto match audio length')
    parser.add_argument('--pose_video', type=str, default=None,
                        help='Pose reference video for head motion control')
    args = parser.parse_args()
    run_s2v(args.ref_image, args.audio, args.output, args.prompt, args.seed,
            num_repeat=args.num_repeat, pose_video=args.pose_video)


if __name__ == '__main__':
    main()
