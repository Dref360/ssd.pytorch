from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm

from data import BaseTransform, MIO_CLASSES
from data import MIO_CLASSES as labelmap, MIOAnnotationTransform, MIODetection
from ssd import build_ssd

SHOW = False

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_MIO_30000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='.', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.0, type=float,
                    help='Final confidence threshold')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size')

parser.add_argument('--cuda', action='store_true',
                    help='Use cuda to train model')
parser.add_argument('--root', default='/data/mio_tcd_seg', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    num_images = len(testset)
    results = []
    n_batch = int(np.ceil(num_images / args.batch_size))
    for b_idx in tqdm(range(n_batch)):
        idx = range(b_idx*args.batch_size, min(len(testset.ids), (b_idx+1) * args.batch_size))
        imgs = []
        odfs = []
        img_ids = []
        for i in idx:
            imgs.append(testset.pull_image(i))
            odfs.append(testset.pull_odf(i))
            img_ids.append(testset.ids[i][0])

        x = torch.from_numpy(np.array([transform(img)[0][..., (2, 1, 0)] for img in imgs])).permute(0,3,1,2)
        odfs = torch.from_numpy(np.array(odfs).astype(np.float32)).permute(0,3,1,2)
        if cuda:
            x = x.cuda()
            odfs = odfs.cuda()

        y = net(x, odfs)  # forward pass
        detections_v = y.data
        # scale each detection back up to the image
        scales = [torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]) for img in imgs]

        for detections, img, scale, img_id in zip(detections_v, imgs, scales, img_ids):
            for i in range(detections.size(0)):
                j = 0
                while detections[i, j, 0] > thresh:
                    score = detections[i, j, 0].item()
                    label_name = labelmap[i - 1]
                    pt = (detections[i, j, 1:-2] * scale).cpu().numpy()
                    orientation = (detections[i, j, -2]).item() * (2 * np.pi)
                    parked = (detections[i, j, -1]).item()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    coords_int = tuple(map(int, coords))
                    cx, cy = int(np.mean(coords[::2])), int(np.mean(coords[1::2]))
                    results.append([img_id, label_name, score, *coords, orientation, parked])
                    if SHOW:
                        clr = (0, 255, 0) if parked < 0.5 else (0, 0, 255)
                        cv2.putText(img, label_name + '' + str(score)[:8],
                                    (coords_int[0], coords_int[1]),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5, (0, 0, 255))
                        cv2.arrowedLine(img, (cx, cy), (int(cx + 20 * np.cos(orientation)), int(cy + 20 * np.sin(orientation))),
                                        clr, 1, tipLength=2.)

                        cv2.rectangle(img, (coords_int[0], coords_int[1]), (coords_int[2], coords_int[3]), (255, 0, 0))
                    j += 1
            if SHOW:
                cv2.imshow('lol', img)
                cv2.waitKey(500)

    with open('machin.csv', 'w') as f:
        f.writelines([','.join(map(str, k)) + '\n' for k in results])


def test_voc():
    # load net
    num_classes = len(MIO_CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu' if not args.cuda else None))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = MIODetection(args.root, None, MIOAnnotationTransform(), is_train=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    test_voc()
