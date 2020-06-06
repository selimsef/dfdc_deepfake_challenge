import math
import os
import random
import sys
import traceback

import cv2
import numpy as np
import pandas as pd
import skimage.draw
from albumentations import ImageCompression, OneOf, GaussianBlur, Blur
from albumentations.augmentations.functional import image_compression, rot90
from albumentations.pytorch.functional import img_to_tensor
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure
from torch.utils.data import Dataset
import dlib

from training.datasets.validation_set import PUBLIC_SET


def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('libs/shape_predictor_68_face_landmarks.dat')


def blackout_convex_hull(img):
    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1
        # if random.random() > 0.5:
        #     img[cropped_img == 0] = 0
        #     #leave only face
        #     return img

        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)
        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        img[cropped_img > 0] = 0
    except Exception as e:
        pass


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_nose(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_mouth(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_landmark(image, landmarks):
    if random.random() > 0.5:
        image = remove_eyes(image, landmarks)
    elif random.random() > 0.5:
        image = remove_mouth(image, landmarks)
    elif random.random() > 0.5:
        image = remove_nose(image, landmarks)
    return image


def change_padding(image, part=5):
    h, w = image.shape[:2]
    # original padding was done with 1/3 from each side, too much
    pad_h = int(((3 / 5) * h) / part)
    pad_w = int(((3 / 5) * w) / part)
    image = image[h // 5 - pad_h:-h // 5 + pad_h, w // 5 - pad_w:-w // 5 + pad_w]
    return image


def blackout_random(image, mask, label):
    binary_mask = mask > 0.4 * 255
    h, w = binary_mask.shape[:2]

    tries = 50
    current_try = 1
    while current_try < tries:
        first = random.random() < 0.5
        if random.random() < 0.5:
            pivot = random.randint(h // 2 - h // 5, h // 2 + h // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:pivot, :] = 0
            else:
                bitmap_msk[pivot:, :] = 0
        else:
            pivot = random.randint(w // 2 - w // 5, w // 2 + w // 5)
            bitmap_msk = np.ones_like(binary_mask)
            if first:
                bitmap_msk[:, :pivot] = 0
            else:
                bitmap_msk[:, pivot:] = 0

        if label < 0.5 and np.count_nonzero(image * np.expand_dims(bitmap_msk, axis=-1)) / 3 > (h * w) / 5 \
                or np.count_nonzero(binary_mask * bitmap_msk) > 40:
            mask *= bitmap_msk
            image *= np.expand_dims(bitmap_msk, axis=-1)
            break
        current_try += 1
    return image


def blend_original(img):
    img = img.copy()
    h, w = img.shape[:2]
    rect = detector(img)
    if len(rect) == 0:
        return img
    else:
        rect = rect[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    outline = landmarks[[*range(17), *range(26, 16, -1)]]
    Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
    raw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    raw_mask[Y, X] = 1
    face = img * np.expand_dims(raw_mask, -1)

    # add warping
    h1 = random.randint(h - h // 2, h + h // 2)
    w1 = random.randint(w - w // 2, w + w // 2)
    while abs(h1 - h) < h // 3 and abs(w1 - w) < w // 3:
        h1 = random.randint(h - h // 2, h + h // 2)
        w1 = random.randint(w - w // 2, w + w // 2)
    face = cv2.resize(face, (w1, h1), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))
    face = cv2.resize(face, (w, h), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))

    raw_mask = binary_erosion(raw_mask, iterations=random.randint(4, 10))
    img[raw_mask, :] = face[raw_mask, :]
    if random.random() < 0.2:
        img = OneOf([GaussianBlur(), Blur()], p=0.5)(image=img)["image"]
    # image compression
    if random.random() < 0.5:
        img = ImageCompression(quality_lower=40, quality_upper=95)(image=img)["image"]
    return img


class DeepFakeClassifierDataset(Dataset):

    def __init__(self,
                 data_path="/mnt/sota/datasets/deepfake",
                 fold=0,
                 label_smoothing=0.01,
                 padding_part=3,
                 hardcore=True,
                 crops_dir="crops",
                 folds_csv="folds.csv",
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 rotation=False,
                 mode="train",
                 reduce_val=True,
                 oversample_real=True,
                 transforms=None
                 ):
        super().__init__()
        self.data_root = data_path
        self.fold = fold
        self.folds_csv = folds_csv
        self.mode = mode
        self.rotation = rotation
        self.padding_part = padding_part
        self.hardcore = hardcore
        self.crops_dir = crops_dir
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.transforms = transforms
        self.df = pd.read_csv(self.folds_csv)
        self.oversample_real = oversample_real
        self.reduce_val = reduce_val

    def __getitem__(self, index: int):

        while True:
            video, img_file, label, ori_video, frame, fold = self.data[index]
            try:
                if self.mode == "train":
                    label = np.clip(label, self.label_smoothing, 1 - self.label_smoothing)
                img_path = os.path.join(self.data_root, self.crops_dir, video, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                diff_path = os.path.join(self.data_root, "diffs", video, img_file[:-4] + "_diff.png")
                try:
                    msk = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
                    if msk is not None:
                        mask = msk
                except:
                    print("not found mask", diff_path)
                    pass
                if self.mode == "train" and self.hardcore and not self.rotation:
                    landmark_path = os.path.join(self.data_root, "landmarks", ori_video, img_file[:-4] + ".npy")
                    if os.path.exists(landmark_path) and random.random() < 0.7:
                        landmarks = np.load(landmark_path)
                        image = remove_landmark(image, landmarks)
                    elif random.random() < 0.2:
                        blackout_convex_hull(image)
                    elif random.random() < 0.1:
                        binary_mask = mask > 0.4 * 255
                        masks = prepare_bit_masks((binary_mask * 1).astype(np.uint8))
                        tries = 6
                        current_try = 1
                        while current_try < tries:
                            bitmap_msk = random.choice(masks)
                            if label < 0.5 or np.count_nonzero(mask * bitmap_msk) > 20:
                                mask *= bitmap_msk
                                image *= np.expand_dims(bitmap_msk, axis=-1)
                                break
                            current_try += 1
                if self.mode == "train" and self.padding_part > 3:
                    image = change_padding(image, self.padding_part)
                valid_label = np.count_nonzero(mask[mask > 20]) > 32 or label < 0.5
                valid_label = 1 if valid_label else 0
                rotation = 0
                if self.transforms:
                    data = self.transforms(image=image, mask=mask)
                    image = data["image"]
                    mask = data["mask"]
                if self.mode == "train" and self.hardcore and self.rotation:
                    # landmark_path = os.path.join(self.data_root, "landmarks", ori_video, img_file[:-4] + ".npy")
                    dropout = 0.8 if label > 0.5 else 0.6
                    if self.rotation:
                        dropout *= 0.7
                    elif random.random() < dropout:
                        blackout_random(image, mask, label)

                #
                # os.makedirs("../images", exist_ok=True)
                # cv2.imwrite(os.path.join("../images", video+ "_" + str(1 if label > 0.5 else 0) + "_"+img_file), image[...,::-1])

                if self.mode == "train" and self.rotation:
                    rotation = random.randint(0, 3)
                    image = rot90(image, rotation)

                image = img_to_tensor(image, self.normalize)
                return {"image": image, "labels": np.array((label,)), "img_name": os.path.join(video, img_file),
                        "valid": valid_label, "rotations": rotation}
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                print("Broken image", os.path.join(self.data_root, self.crops_dir, video, img_file))
                index = random.randint(0, len(self.data) - 1)

    def random_blackout_landmark(self, image, mask, landmarks):
        x, y = random.choice(landmarks)
        first = random.random() > 0.5
        #  crop half face either vertically or horizontally
        if random.random() > 0.5:
            # width
            if first:
                image[:, :x] = 0
                mask[:, :x] = 0
            else:
                image[:, x:] = 0
                mask[:, x:] = 0
        else:
            # height
            if first:
                image[:y, :] = 0
                mask[:y, :] = 0
            else:
                image[y:, :] = 0
                mask[y:, :] = 0

    def reset(self, epoch, seed):
        self.data = self._prepare_data(epoch, seed)

    def __len__(self) -> int:
        return len(self.data)

    def _prepare_data(self, epoch, seed):
        df = self.df
        if self.mode == "train":
            rows = df[df["fold"] != self.fold]
        else:
            rows = df[df["fold"] == self.fold]
        seed = (epoch + 1) * seed
        if self.oversample_real:
            rows = self._oversample(rows, seed)
        if self.mode == "val" and self.reduce_val:
            # every 2nd frame, to speed up validation
            rows = rows[rows["frame"] % 20 == 0]
            # another option is to use public validation set
            #rows = rows[rows["video"].isin(PUBLIC_SET)]

        print(
            "real {} fakes {} mode {}".format(len(rows[rows["label"] == 0]), len(rows[rows["label"] == 1]), self.mode))
        data = rows.values

        np.random.seed(seed)
        np.random.shuffle(data)
        return data

    def _oversample(self, rows: pd.DataFrame, seed):
        real = rows[rows["label"] == 0]
        fakes = rows[rows["label"] == 1]
        num_real = real["video"].count()
        if self.mode == "train":
            fakes = fakes.sample(n=num_real, replace=False, random_state=seed)
        return pd.concat([real, fakes])
