from __future__ import print_function
import os
import time
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from preprocessing.retinaface.data import cfg_mnet, cfg_re50
from preprocessing.retinaface.layers.functions.prior_box import PriorBox
from preprocessing.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from preprocessing.retinaface.models.retinaface import RetinaFace
from preprocessing.retinaface.utils.box_utils import decode, decode_landm


# parser = argparse.ArgumentParser(description='Retinaface')

# parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='resnet50',
#                     help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--cpu', action="store_true",
#                     default=True, help='Use cpu inference')
# parser.add_argument('--confidence_threshold', default=0.02,
#                     type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4,
#                     type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true",
#                     default=True, help='show detection results')
# parser.add_argument('--vis_thres', default=0.6, type=float,
#                     help='visualization_threshold')
# args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class FaceDetector():
    def __init__(
        self,
        network,
        weights,
        confidence_threshold=0.02,
        top_k=5000,
        nms_threshold=0.4,
        keep_top_k=750,
        vis_thres=0.6,
        cpu=True
    ):
        self.network = network
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.set_grad_enabled(False)
        self.cfg = None
        if self.network == "mobile0.25":
            self.cfg = cfg_mnet
        if self.network == "resnet50":
            self.cfg = cfg_re50

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, self.weights, cpu)
        self.net.eval()
        cudnn.benchmark = True

        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = self.net.to(self.device)
        self.resize = 1

    def detect(self, img, landmarks):
        img_raw = img
        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        # img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0),
                              prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        out_bboxs = []
        out_confids = []
        out_landms = []

        for i, det in enumerate(dets):
            if det[-1] < self.vis_thres:
                continue
            out_bboxs.append(det[:-1])
            out_confids.append(det[-1])
            out_landms.append(landms[i])
        results = {"bbox": out_bboxs,
                   "landmark": landms, "confid": out_confids}

        dets = np.concatenate((dets, landms), axis=1)
        drawed = self.draw(dets, img_raw)

        return results, drawed

    def draw(self, dets, img_raw):
        for b in dets:
            if b[4] < self.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]),
                          (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

        # save image
        # name = "test.jpg"
        # cv2.imwrite(name, img_raw)
        img_raw = np.asarray(img_raw, dtype=np.uint8)
        return img_raw

# if __name__ == '__main__':
#     # testing begin
#     image_path = "./curve/after.jpg"
#     img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

#     img = np.float32(img_raw)

    # show image
    # if args.save_image:
