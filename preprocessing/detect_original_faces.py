import argparse
import json
import os
from os import cpu_count
from typing import Type

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch

from preprocessing import face_detector, VideoDataset
from preprocessing.face_detector import VideoFaceDetector
from preprocessing.utils import get_original_video_paths

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a original videos with face detector")
    parser.add_argument("--root-dir", help="root directory")
    parser.add_argument(
        "--detector", help="choose a detector", default="MTCNN")
    parser.add_argument("--detector-type", help="type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    args = parser.parse_args()
    return args


def process_videos(videos, root_dir, select_detector, detector_cls: Type[VideoFaceDetector]):
    detector = face_detector.__dict__[detector_cls](
        detector=select_detector, device=device)
    dataset = VideoDataset(videos)
    loader = DataLoader(dataset, shuffle=False, num_workers=0,
                        batch_size=1, collate_fn=lambda x: x)
    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        id = os.path.splitext(os.path.basename(video))[0]
        batches = [frames[i:i + detector._batch_size]
                   for i in range(0, len(frames), detector._batch_size)]

        for j, frames in enumerate(batches):
            result.update({int(j * detector._batch_size) + i: b for i,
                           b in zip(indices, detector._detect_faces(frames))})

        out_dir = os.path.join(root_dir, "boxes")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
            json.dump(result, f)


def main():
    args = parse_args()
    originals = get_original_video_paths(args.root_dir)
    process_videos(originals, args.root_dir, args.detector, args.detector_type)


if __name__ == "__main__":
    main()
