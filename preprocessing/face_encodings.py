import argparse
import os
from functools import partial
from multiprocessing.pool import Pool

from tqdm import tqdm

from preprocessing.utils import get_original_video_paths

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import random

import face_recognition
import numpy as np


def write_face_encodings(video, root_dir):
    video_id, *_ = os.path.splitext(video)
    crops_dir = os.path.join(root_dir, "crops", video_id)
    if not os.path.exists(crops_dir):
        return
    crop_files = [f for f in os.listdir(crops_dir) if f.endswith("jpg")]
    if crop_files:
        crop_files = random.sample(crop_files, min(10, len(crop_files)))
        encodings = []
        for crop_file in crop_files:
            img = face_recognition.load_image_file(os.path.join(crops_dir, crop_file))
            encoding = face_recognition.face_encodings(img, num_jitters=10)
            if encoding:
                encodings.append(encoding[0])
        np.save(os.path.join(crops_dir, "encodings"), encodings)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 10 crops encodings for each video")
    parser.add_argument("--root-dir", help="root directory", default="/home/selim/datasets/deepfake")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    originals = get_original_video_paths(args.root_dir, basename=True)
    with Pool(processes=os.cpu_count() - 4) as p:
        with tqdm(total=len(originals)) as pbar:
            for v in p.imap_unordered(partial(write_face_encodings, root_dir=args.root_dir), originals):
                pbar.update()


if __name__ == '__main__':
    main()
