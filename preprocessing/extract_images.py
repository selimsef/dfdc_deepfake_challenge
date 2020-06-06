import argparse
import os
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


def extract_video(video, root_dir):
    capture = cv2.VideoCapture(video)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        id = os.path.splitext(os.path.basename(video))[0]
        cv2.imwrite(os.path.join(root_dir, "jpegs", "{}_{}.jpg".format(id, i)), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts jpegs from video")
    parser.add_argument("--root-dir", help="root directory")

    args = parser.parse_args()
    os.makedirs(os.path.join(args.root_dir, "jpegs"), exist_ok=True)
    videos = [video_path for video_path in glob(os.path.join(args.root_dir, "*/*.mp4"))]
    with Pool(processes=cpu_count() - 2) as p:
        with tqdm(total=len(videos)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=args.root_dir), videos):
                pbar.update()
