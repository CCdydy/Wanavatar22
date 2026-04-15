"""Step 3: Generate final avatar video via Wan-Animate.

Input:  src_root_path containing src_ref.png, src_pose.mp4, src_face.mp4
        (produced by extract_drivers.py)
Output: final_avatar.mp4

Includes aggressive GPU offloading to fit 14B Animate model on 48GB VRAM.
"""
import argparse
import os
import sys

WAN_REPO = '/media/zzy/SN5601/wan22/Wan2.2'
sys.path.insert(0, WAN_REPO)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def _patch_animate_offload(wan_animate):
    """Patch WanAnimate to offload DiT during VAE encode/decode.

    The original generate() loads DiT to GPU (40GB) and never offloads it,
    leaving no room for VAE encode/decode on a 48GB card. This patch wraps
    the VAE encode/decode to temporarily offload DiT+CLIP to CPU.
    """
    import torch
    device = wan_animate.device

    _orig_vae_encode = wan_animate.vae.encode
    _orig_vae_decode = wan_animate.vae.decode

    def _offload_encode(pixel_values):
        wan_animate.noise_model.cpu()
        torch.cuda.empty_cache()
        result = _orig_vae_encode(pixel_values)
        return result

    def _offload_decode(latents):
        wan_animate.noise_model.cpu()
        torch.cuda.empty_cache()
        result = _orig_vae_decode(latents)
        return result

    wan_animate.vae.encode = _offload_encode
    wan_animate.vae.decode = _offload_decode

    # Also offload DiT before prepare_source (cv2.imread can fail under
    # memory pressure when DiT occupies 40GB GPU memory)
    _orig_prepare = wan_animate.prepare_source

    def _offload_prepare(*args, **kwargs):
        wan_animate.noise_model.cpu()
        torch.cuda.empty_cache()
        return _orig_prepare(*args, **kwargs)

    wan_animate.prepare_source = _offload_prepare

    # Patch DiT forward to auto-reload to GPU when needed for denoising
    _orig_forward = wan_animate.noise_model.forward

    def _auto_load_forward(*args, **kwargs):
        if next(wan_animate.noise_model.parameters()).device.type == 'cpu':
            wan_animate.noise_model.to(device)
            torch.cuda.empty_cache()
        return _orig_forward(*args, **kwargs)

    wan_animate.noise_model.forward = _auto_load_forward


def run_animate(src_root_path, output_path, seed=42, clip_len=77,
                sampling_steps=20, guide_scale=1.0, offload_model=True,
                fps=16):
    import torch
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    cfg = WAN_CONFIGS['animate-14B']
    ckpt_dir = '/media/zzy/SN5601/wan22/Wan2.2-Animate-14B'

    print("Loading Wan-Animate...")
    wan_animate = wan.WanAnimate(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=0, rank=0,
        t5_cpu=True,
        convert_model_dtype=True,
    )

    # Apply aggressive offloading for 48GB cards
    _patch_animate_offload(wan_animate)
    print("  Applied DiT offload patches for VRAM management")

    print(f"Generating final avatar video...")
    print(f"  src_root_path: {src_root_path}")
    print(f"  clip_len: {clip_len}, steps: {sampling_steps}")

    video = wan_animate.generate(
        src_root_path=src_root_path,
        replace_flag=False,
        refert_num=1,
        clip_len=clip_len,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_video(
        tensor=video[None],
        save_file=output_path,
        fps=fps,
        nrow=1, normalize=True, value_range=(-1, 1),
    )
    print(f"Saved: {output_path}")

    del wan_animate, video
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Step 3: Wan-Animate final video')
    parser.add_argument('--src_root_path', type=str, required=True,
                        help='Directory with src_ref.png, src_pose.mp4, src_face.mp4')
    parser.add_argument('--output', type=str,
                        default='outputs/animate/final_avatar.mp4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip_len', type=int, default=77,
                        help='Frames per clip (must be 4n+1)')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--guide_scale', type=float, default=1.0)
    parser.add_argument('--fps', type=int, default=16)
    args = parser.parse_args()

    run_animate(
        args.src_root_path, args.output,
        seed=args.seed, clip_len=args.clip_len,
        sampling_steps=args.steps, guide_scale=args.guide_scale,
        fps=args.fps,
    )


if __name__ == '__main__':
    main()
