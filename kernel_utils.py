import os
import cv2
import numpy as np
import torch
from PIL import Image
from albumentations.augmentations.functional import image_compression
from facenet_pytorch.models.mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import Normalize
import logging

# Setup logging for debugging purposes
logging.basicConfig(level=logging.INFO)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self, verbose=True, insets=(0, 0)):
        self.verbose = verbose
        self.insets = insets

    def read_frames(self, path, num_frames, jitter=0, seed=None):
        """Reads frames that are evenly spaced throughout the video."""
        assert num_frames > 0
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None

        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
        if jitter > 0:
            np.random.seed(seed)
            jitter_offsets = np.random.randint(-jitter, jitter, len(frame_idxs))
            frame_idxs = np.clip(frame_idxs + jitter_offsets, 0, frame_count - 1)

        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def read_random_frames(self, path, num_frames, seed=None):
        """Picks random frames from the video."""
        assert num_frames > 0
        np.random.seed(seed)
        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None

        frame_idxs = sorted(np.random.choice(np.arange(0, frame_count), num_frames))
        result = self._read_frames_at_indices(path, capture, frame_idxs)

        capture.release()
        return result

    def read_frames_at_indices(self, path, frame_idxs):
        """Reads frames from a video and puts them into a NumPy array."""
        assert len(frame_idxs) > 0
        capture = cv2.VideoCapture(path)
        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                ret = capture.grab()
                if not ret:
                    if self.verbose:
                        logging.error(f"Error grabbing frame {frame_idx} from movie {path}")
                    break

                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        if self.verbose:
                            logging.error(f"Error retrieving frame {frame_idx} from movie {path}")
                        break

                    frame = self._postprocess_frame(frame)
                    frames.append(frame)
                    idxs_read.append(frame_idx)

            if len(frames) > 0:
                return np.stack(frames), idxs_read
            if self.verbose:
                logging.error(f"No frames read from movie {path}")
            return None
        except Exception as e:
            if self.verbose:
                logging.error(f"Exception while reading movie {path}: {str(e)}")
            return None

    def _postprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.insets[0] > 0:
            W = frame.shape[1]
            p = int(W * self.insets[0])
            frame = frame[:, p:-p, :]
        if self.insets[1] > 0:
            H = frame.shape[0]
            q = int(H * self.insets[1])
            frame = frame[q:-q, :, :]
        return frame


class FaceExtractor:
    """Extract faces from video frames using MTCNN detector."""
    
    def __init__(self, video_read_fn, detector=None):
        self.video_read_fn = video_read_fn
        self.detector = detector or MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device="cuda")

    def process_videos(self, input_dir, filenames, video_idxs):
        """Process multiple videos for face extraction."""
        results = []
        for video_idx in video_idxs:
            filename = filenames[video_idx]
            video_path = os.path.join(input_dir, filename)
            result = self.video_read_fn(video_path)
            if result is None:
                logging.warning(f"Failed to read video: {filename}")
                continue

            my_frames, my_idxs = result
            for i, frame in enumerate(my_frames):
                faces = self.extract_faces(frame)
                if faces:
                    frame_dict = {
                        "video_idx": video_idx,
                        "frame_idx": my_idxs[i],
                        "frame_w": frame.shape[1],
                        "frame_h": frame.shape[0],
                        "faces": faces['images'],
                        "scores": faces['scores']
                    }
                    results.append(frame_dict)
        return results

    def extract_faces(self, frame):
        """Detect faces in a single frame."""
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize(size=[s // 2 for s in img.size])

        batch_boxes, probs = self.detector.detect(img, landmarks=False)

        if batch_boxes is None:
            return None

        faces = {'images': [], 'scores': []}
        for bbox, score in zip(batch_boxes, probs):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                w = xmax - xmin
                h = ymax - ymin
                p_h = h // 3
                p_w = w // 3
                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                faces['images'].append(crop)
                faces['scores'].append(score)

        return faces


def confident_strategy(pred, t=0.8):
    """Determine the confidence strategy based on predictions."""
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)


def put_to_center(img, input_size):
    """Resize image to fit in the center of a square canvas."""
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w:start_w + img.shape[1], :] = img
    return image


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    """Resize image isotropically to a given size."""
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


def predict_on_video(face_extractor, video_path, batch_size, input_size, models, strategy=np.mean, apply_compression=False):
    """Run prediction on a video and return the aggregated result."""
    try:
        faces = face_extractor.process_video(video_path)
        if faces:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = put_to_center(resized_face, input_size)
                    if apply_compression:
                        resized_face = image_compression(resized_face, quality=90, image_type=".jpg")
                    if n + 1 < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        continue
            if n > 0:
                x = torch.tensor(x[:n], device="cuda").float()
                x = x.permute((0, 3, 1, 2)) / 255.0
                x = normalize_transform(x)
                preds = []
                for model in models:
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(x.half())  # Half precision for inference
                        preds.append(strategy(y_pred.squeeze().cpu().numpy()))
                return np.mean(preds)
    except Exception as e:
        logging.error(f"Prediction error on video {video_path}: {str(e)}")
    return 0.5


def predict_on_video_set(face_extractor, videos, input_size, num_workers, test_dir, frames_per_video, models, strategy=np.mean, apply_compression=False):
    """Process a set of videos and return predictions."""
    def process_file(i):
        filename = videos[i]
        return predict_on_video(face_extractor=face_extractor, 
                                video_path=os.path.join(test_dir, filename),
                                input_size=input_size,
                                batch_size=frames_per_video,
                                models=models,
                                strategy=strategy,
                                apply_compression=apply_compression)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = list(ex.map(process_file, range(len(videos))))

    return predictions
