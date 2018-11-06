from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
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
parser.add_argument('--odf_size', default=None, type=int)
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')

parser.add_argument('--cuda', action='store_true',
                    help='Use cuda to train model')
parser.add_argument('--keep_valid', action='store_true')
parser.add_argument('--root', default='/data/mio_tcd_seg', help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


class TestDataset(Dataset):
    def __init__(self, testset, odf_size, keep_valid, transform):
        self.testset = testset
        self.odf_size = odf_size
        self.keep_valid = keep_valid
        self.transform = transform

    def __len__(self):
        return len(self.testset)

    def __getitem__(self, i):
        odf = self.testset.pull_odf(i, args.odf_size, args.keep_valid)
        if odf is None:
            return None, None, None
        img = self.testset.pull_image(i)
        img_id = self.testset.ids[i][0]
        return img, odf, img_id

    def collate_fn(self, data):
        imgs = []
        odfs = []
        img_ids = []
        shapes = []
        for img, odf, img_id in data:
            if odf is None:
                continue
            shapes.append(img.shape)
            imgs.append(np.array(self.transform(img)[0][..., (2, 1, 0)]))
            odfs.append(np.array(odf).astype(np.float32))
            img_ids.append(img_id)

        return map(np.array,[imgs, odfs, img_ids, shapes])


def test_net(save_folder, net, cuda, testset, transform, thresh):
    testset = TestDataset(testset, args.odf_size, args.keep_valid, transform)
    dataloader = DataLoader(dataset=testset, batch_size=args.batch_size, num_workers=1, collate_fn=testset.collate_fn)

    num_images = len(testset)
    results = []
    n_batch = int(np.ceil(num_images / args.batch_size))
    for imgs, odfs, img_ids, shapes in tqdm(dataloader):
        if len(imgs) == 0:
            continue

        x = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        odfs = torch.from_numpy(odfs).permute(0, 3, 1, 2)
        if cuda:
            x = x.cuda()
            odfs = odfs.cuda()

        y = net(x, odfs)  # forward pass
        detections_v = y.data
        # scale each detection back up to the image
        scales = [torch.Tensor([shape[1], shape[0],
                                shape[1], shape[0]]) for shape in shapes]

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
                        cv2.arrowedLine(img, (cx, cy),
                                        (int(cx + 20 * np.cos(orientation)), int(cy + 20 * np.sin(orientation))),
                                        clr, 1, tipLength=2.)

                        cv2.rectangle(img, (coords_int[0], coords_int[1]), (coords_int[2], coords_int[3]), (255, 0, 0))
                    j += 1
            if SHOW:
                cv2.imshow('lol', img)
                cv2.waitKey(500)

    with open('csvs/test_{}_ODF={}.csv'.format(args.trained_model.split('/')[-1],
                                               '' if args.odf_size is None else args.odf_size), 'w') as f:
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
