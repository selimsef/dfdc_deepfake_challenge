import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from skimage.measure import compare_ssim

from functools import partial
from multiprocessing.pool import Pool

from tqdm import tqdm

from preprocessing.utils import get_original_with_fakes

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import numpy as np

cache = {}


def save_diffs(pair, root_dir):
    ori_id, fake_id = pair
    ori_dir = os.path.join(root_dir, "crops", ori_id)
    fake_dir = os.path.join(root_dir, "crops", fake_id)
    diff_dir = os.path.join(root_dir, "diffs", fake_id)
    os.makedirs(diff_dir, exist_ok=True)
    for frame in range(320):
        if frame % 10 != 0:
            continue
        for actor in range(2):
            image_id = "{}_{}.png".format(frame, actor)
            diff_image_id = "{}_{}_diff.png".format(frame, actor)
            ori_path = os.path.join(ori_dir, image_id)
            fake_path = os.path.join(fake_dir, image_id)
            diff_path = os.path.join(diff_dir, diff_image_id)
            if os.path.exists(ori_path) and os.path.exists(fake_path):
                img1 = cv2.imread(ori_path, cv2.IMREAD_COLOR)
                img2 = cv2.imread(fake_path, cv2.IMREAD_COLOR)
                try:
                    d, a = compare_ssim(img1, img2, multichannel=True, full=True)
                    a = 1 - a
                    diff = (a * 255).astype(np.uint8)
                    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(diff_path, diff)
                except:
                    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract image diffs")
    parser.add_argument("--root-dir", help="root directory", default="/mnt/sota/datasets/deepfake")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pairs = get_original_with_fakes(args.root_dir)
    os.makedirs(os.path.join(args.root_dir, "diffs"), exist_ok=True)
    with Pool(processes=os.cpu_count() - 2) as p:
        with tqdm(total=len(pairs)) as pbar:
            func = partial(save_diffs, root_dir=args.root_dir)
            for v in p.imap_unordered(func, pairs):
                pbar.update()


if __name__ == '__main__':
    main()
