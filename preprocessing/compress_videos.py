import argparse
import os
import random
import subprocess

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from os import cpu_count

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm


def compress_video(video, root_dir):
    parent_dir = video.split("/")[-2]
    out_dir = os.path.join(root_dir, "compressed", parent_dir)
    os.makedirs(out_dir, exist_ok=True)
    video_name = video.split("/")[-1]
    out_path = os.path.join(out_dir, video_name)
    lvl = random.choice([23, 28, 32])
    command = "ffmpeg -i {} -c:v libx264 -crf {} -threads 1 {}".format(video, lvl, out_path)
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except Exception as e:
        print("Could not process vide", str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts jpegs from video")
    parser.add_argument("--root-dir", help="root directory", default="/mnt/sota/datasets/deepfake")

    args = parser.parse_args()
    videos = [video_path for video_path in glob(os.path.join(args.root_dir, "*/*.mp4"))]
    with Pool(processes=cpu_count() - 2) as p:
        with tqdm(total=len(videos)) as pbar:
            for v in p.imap_unordered(partial(compress_video, root_dir=args.root_dir), videos):
                pbar.update()
