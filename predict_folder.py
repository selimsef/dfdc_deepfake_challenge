import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path

import cv2
import torch
import pandas as pd

from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s; %(asctime)s; %(module)s:%(funcName)s:%(lineno)d; %(message)s",
    handlers=handlers)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict test videos")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="weights",
        help="path to directory with checkpoints")
    arg('--models', nargs='+', required=True, help="checkpoint files")
    arg('--test-dir', type=str, required=False, default="./test_videos",
        help="path to directory with videos")
    arg('--webcam', type=bool, default=True, required=False,
        help="whether to open webcam for predicting")
    arg('--output', type=str, required=False,
        help="path to output csv", default="submission.csv")
    args = parser.parse_args()

    if not args.webcam and not Path(args.test_dir).exists():
        raise ValueError(
            "You need to open webcam or provide a valid video directory for predicting.")

    models = []
    model_paths = [os.path.join(args.weights_dir, model)
                   for model in args.models]

    for path in model_paths:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
        logger.info("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(
            {re.sub("^module.", "", k): v.to(torch.float32) for k, v in state_dict.items()}, strict=True)
        model.eval()
        del checkpoint
        models.append(model)

    frames_per_video = 1
    video_reader = VideoReader()

    if args.webcam:
        root_dir = "webcam_outputs"
        root = Path(root_dir)
        root.mkdir(exist_ok=True)

        def video_read_fn(x): return video_reader.read_webcam_frames(
            x, num_frames=frames_per_video)

    else:
        def video_read_fn(x): return video_reader.read_frames(
            x, num_frames=frames_per_video)

    face_extractor = FaceExtractor(
        video_read_fn=video_read_fn,
        mode="webcam" if args.webcam else "video",
        detector_type="retinaface"
    )
    input_size = 380
    strategy = confident_strategy
    stime = time.time()

    if args.webcam:
        test_videos = ["from_webcam"]
        logger.info("Predicting from webcam")
    else:
        test_videos = sorted(
            [x for x in os.listdir(args.test_dir) if x[-4:] == ".mp4"])
        logger.info("Predicting {} videos".format(len(test_videos)))

    while True:
        predictions, drawed_img = predict_on_video_set(face_extractor=face_extractor, input_size=input_size, models=models,
                                                       strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                                       num_workers=6, test_dir=args.test_dir)
        # print(drawed_img)

        cv2.imwrite("test.jpg", drawed_img)
        cv2.imshow("webcam", drawed_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    submission_df = pd.DataFrame(
        {"filename": test_videos, "label": predictions})
    submission_df.to_csv(args.output, index=False)
    logger.info(f"Elapsed: {time.time() - stime}")
