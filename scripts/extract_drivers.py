"""Step 2: Extract pose and face drivers from coarse S2V video.

Uses Wan-Animate's official preprocessing pipeline to produce:
  - src_pose.mp4  (pose skeleton on black canvas)
  - src_face.mp4  (face crops, 512x512)
  - src_ref.png   (copy of reference image)

These are the exact inputs Wan-Animate expects in its src_root_path.
"""
import argparse
import os
import sys
import tempfile
from PIL import Image

WAN_REPO = '/media/zzy/SN5601/wan22/Wan2.2'
sys.path.insert(0, WAN_REPO)
sys.path.insert(0, os.path.join(WAN_REPO, 'wan/modules/animate/preprocess'))


def extract_drivers(coarse_video, ref_image, output_dir, fps=16,
                    replace_flag=False):
    """Run Wan-Animate official preprocessing on the coarse video.

    This produces src_pose.mp4, src_face.mp4, src_ref.png in output_dir.
    """
    from process_pipepline import ProcessPipeline

    ckpt_base = '/media/zzy/SN5601/wan22/Wan2.2-Animate-14B/process_checkpoint'

    # Pass the actual .onnx file path, not the directory
    # (the code appends 'end2end.onnx' when given a directory, but the file
    # is actually 'yolov10m.onnx')
    det_path = os.path.join(ckpt_base, 'det', 'yolov10m.onnx')

    print("Loading preprocessing models...")
    pipeline = ProcessPipeline(
        det_checkpoint_path=det_path,
        pose2d_checkpoint_path=os.path.join(ckpt_base, 'pose2d', 'vitpose_h_wholebody.onnx'),
        sam_checkpoint_path=os.path.join(ckpt_base, 'sam2') if replace_flag else None,
        flux_kontext_path=None,
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting drivers from: {coarse_video}")
    print(f"  Reference image: {ref_image}")
    print(f"  Output: {output_dir}")
    print(f"  FPS: {fps}")
    print(f"  Mode: {'replace' if replace_flag else 'animate'}")

    success = pipeline(
        video_path=coarse_video,
        refer_image_path=ref_image,
        output_path=output_dir,
        resolution_area=[1280, 720],
        fps=fps,
        replace_flag=replace_flag,
    )

    # Normalize the copied reference image into a stable image payload.
    # On this machine/filesystem, some PNGs become unreadable to cv2/libpng
    # inside Wan-Animate. Writing JPEG payload while keeping the required
    # filename avoids the libpng path entirely; cv2 detects by file header.
    src_ref_path = os.path.join(output_dir, 'src_ref.png')
    fd, tmp_ref_path = tempfile.mkstemp(suffix='.png', dir=output_dir)
    os.close(fd)
    try:
        with Image.open(ref_image) as img:
            img = img.convert('RGB')
            img.save(tmp_ref_path, format='JPEG', quality=95)
        os.replace(tmp_ref_path, src_ref_path)
    finally:
        if os.path.exists(tmp_ref_path):
            os.unlink(tmp_ref_path)

    # Hard-assert required outputs exist
    required = ['src_pose.mp4', 'src_face.mp4', 'src_ref.png']
    if replace_flag:
        required += ['src_bg.mp4', 'src_mask.mp4']

    missing = [f for f in required if not os.path.exists(os.path.join(output_dir, f))]
    if missing:
        print(f"ERROR: Missing required outputs: {missing}")
        return False

    for f in required:
        path = os.path.join(output_dir, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {f}: {size_mb:.1f} MB")
    print("Driver extraction complete.")

    return success and len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description='Step 2: Extract pose/face drivers')
    parser.add_argument('--coarse_video', type=str, required=True,
                        help='Path to coarse S2V video')
    parser.add_argument('--ref_image', type=str, required=True,
                        help='Reference avatar image')
    parser.add_argument('--output_dir', type=str, default='outputs/drivers',
                        help='Output directory for driver videos')
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--replace', action='store_true', default=False,
                        help='Use replacement mode (extracts bg/mask too)')
    args = parser.parse_args()

    extract_drivers(
        args.coarse_video, args.ref_image, args.output_dir,
        fps=args.fps, replace_flag=args.replace,
    )


if __name__ == '__main__':
    main()
