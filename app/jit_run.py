import argparse
import pickle
import scipy.ndimage

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from app.transform import BaseTransform, Detect

parser = argparse.ArgumentParser()
parser.add_argument('model', help='Pytorch >= 1.0 Model')
parser.add_argument('video', help='Video file')
parser.add_argument('odf', help='Pickle file with the odf')
parser.add_argument('--priors', default='priors.pkl')
args = parser.parse_args()

ODF_DIM = 20
MIO_CLASSES = [
    "articulated_truck",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "motorized_vehicle",
    "non-motorized_vehicle",
    "pedestrian",
    "pickup_truck",
    "single_unit_truck",
    "work_van"
]
thresh = 0.4
transform = BaseTransform(300, (104, 117, 123))


def process_img(net, priors, detect, img, odf):
    """
    Run the image throught the network and show it.
    :param net: Pytorch Model
    :param priors: Priors for the encoding
    :param detect: Detect object for postprocessing
    :param img: Image in array format.
    :param odf: Array 19x19x20 with the encoded ODF.
    :return: Image with predictions.
    """
    imgs = [img]
    odfs = [odf]

    x = torch.from_numpy(np.array([transform(im)[0][..., (2, 1, 0)] for im in imgs])).permute(0, 3, 1, 2)
    odf = torch.from_numpy(np.array(odfs)).permute(0, 3, 1, 2).float()
    y = net(x, odf)  # forward pass
    detections_v = detect(*y, Variable(torch.from_numpy(priors))).data
    # scale each detection back up to the image
    scales = [torch.Tensor([im.shape[1], im.shape[0],
                            im.shape[1], im.shape[0]]) for im in imgs]

    for detections, img, scale in zip(detections_v, imgs, scales):
        for i in range(detections.size(0)):
            j = 0
            while detections[i, j, 0] > thresh:
                score = detections[i, j, 0].item()
                label_name = MIO_CLASSES[i - 1]
                pt = (detections[i, j, 1:-2] * scale).cpu().numpy()
                orientation = (detections[i, j, -2]).item() * (2 * np.pi)
                parked = (detections[i, j, -1]).item()
                coords = (pt[0], pt[1], pt[2], pt[3])
                coords_int = tuple(map(int, coords))
                cx, cy = int(np.mean(coords[::2])), int(np.mean(coords[1::2]))
                if True:
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
        cv2.imshow('lol', img)
        cv2.waitKey(1)
        yield img


def process_video(video_path, net, priors, detect, odf):
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter('test_odf.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        img = list(process_img(net, priors, detect, frame, odf))[0]
        writer.write(cv2.resize(img, (640, 480)))

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis)[..., np.newaxis])
    return e_x / e_x.sum(axis=axis)[..., np.newaxis]

def resize_odf(odf):
    if odf.shape[-1] != ODF_DIM:
        zoom = ODF_DIM / odf.shape[-1]
        odf = scipy.ndimage.zoom(odf, (1, 1, 1, zoom))
        assert odf.shape[-1] == ODF_DIM
    return odf


def main():
    priors = pickle.load(open(args.priors, 'rb'))
    odf = pickle.load(open(args.odf, 'rb'))
    if len(odf.shape) != 4:
        odf = odf[np.newaxis, ...]
    odf = resize_odf(odf).sum(0)
    odf = softmax(np.maximum(odf, 0.1), axis=-1)
    net = torch.jit.load(args.model)
    net.eval()
    detect = Detect(len(MIO_CLASSES) + 1, 0, 200, 0.01, 0.45, [0.1, 0.2])
    process_video(args.video, net, priors, detect, odf)


if __name__ == '__main__':
    main()
