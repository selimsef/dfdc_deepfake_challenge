import json
import os
from glob import glob
from pathlib import Path

import cv2
import dlib
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb


def get_original_video_paths(root_dir, basename=False):
    originals = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)
    print(f"length of originals: {len(originals)}")
    return originals_v if basename else originals


def get_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4]))

    return pairs


def get_originals_and_fakes(root_dir):
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append(k[:-4])
            else:
                originals.append(k[:-4])

    return originals, fakes


def landmark_alignment(image, landmark_path):
    # landmarks = np.load(landmark_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    fa = FaceAligner(
        predictor, desiredFaceWidth=image.shape[1], desiredFaceHeight=image.shape[0])

    # load the input image, resize it, and convert it to grayscale
    # image = imutils.resize(image, width=1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # show the original input image and detect faces in the grayscale
    # image

    rects = detector(gray, 2)
    i = 0
    faceAligned = None
    # if len(rects) == 0:
    #     cv2.imwrite("weird.jpg", image)

    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks

        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)
        cv2.imwrite("after.jpg", faceAligned)
        i += 1
    return faceAligned
