import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from itertools import product

import cv2
import numpy as np
import pandas as pd

from data import MIO_CLASSES

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import torch
from tqdm import tqdm
from joblib import Parallel, delayed

from layers.box_utils import jaccard

parser = argparse.ArgumentParser()
parser.add_argument('root', help='Dataset root')
parser.add_argument('csv', help='CSV path')
args = parser.parse_args()
print(args.csv)
pjoin = os.path.join

gt_test = pjoin(args.root, 'gt_test.csv')

if True:
    print(subprocess.run(['python', 'localization_evaluation.py', gt_test, args.csv],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).stdout.decode('utf-8'))

gt = json.load(open(pjoin(args.root, 'jsontest.json'), 'r'))
pred = defaultdict(list)
for id, label, conf, xmin, ymin, xmax, ymax, angle, parked in csv.reader(open(args.csv, 'r')):
    d = {'angle': float(angle),
         'parked': float(parked),
         'class': label,
         'conf': float(conf),
         'xmin': float(xmin), 'xmax': float(xmax), 'ymin': float(ymin), 'ymax': float(ymax)}
    pred[id].append(d)


def show():
    gt_clr = (255, 0, 0)
    pred_clr = (0, 255, 0)
    for k, [_, vals] in gt.items():
        img = cv2.imread(pjoin(args.root, 'images', k + '.jpg'))
        for b in vals:
            draw_rect(img, b, gt_clr)
        for b in pred[k]:
            if b['conf'] < 0.1:
                continue
            draw_rect(img, b, pred_clr)
        cv2.imshow('', img)
        cv2.waitKey(1000)


def iou(b1, b2):
    b1_arr = np.array([b1['xmin'], b1['ymin'], b1['xmax'], b1['ymax']]).reshape([1, 4]).astype(np.float)
    b2_arr = np.array([b2['xmin'], b2['ymin'], b2['xmax'], b2['ymax']]).reshape([1, 4]).astype(np.float)
    return jaccard(torch.from_numpy(b1_arr), torch.from_numpy(b2_arr)).numpy()[0].item()


def find_best_iou(b1, vals, min_iou=0.75):
    ious = [(b2, iou(b1, b2)) for b2 in vals]
    if not ious:
        return None
    best = max(ious, key=lambda k: k[1])
    if best[1] < min_iou:
        return None
    else:
        return best[0]


def draw_rect(img, b, clr):
    cv2.rectangle(img, (int(b['xmin']), int(b['ymin'])), (int(b['xmax']), int(b['ymax'])), clr, 1)
    if 'parked' in b:
        prk_clr = (0, 255, 0) if b['parked'] < 0.5 else (0, 0, 255)
    else:
        prk_clr = (0, 255, 255)
    cx, cy = map(int, [np.mean([b['xmin'], b['xmax']]), np.mean([b['ymin'], b['ymax']])])
    cx2 = int(cx + 20 * np.cos(float(b['angle'])))
    cy2 = int(cy + 20 * np.sin(float(b['angle'])))
    cv2.arrowedLine(img, (cx, cy), (cx2, cy2),
                    prk_clr, 1, tipLength=1)


def arccosine_distance(b1, b2):
    return np.arccos(dot_metric(b1, b2))


def cosine_distance(b1, b2):
    return 1. - dot_metric(b1, b2)


def dot_metric(b1, b2):
    a, b = map(float, (b1['angle'], b2['angle']))
    av = np.array([np.cos(a), np.sin(a)])
    bv = np.array([np.cos(b), np.sin(b)])
    dot = np.dot(av, bv)
    return np.clip(dot, -1, 1)


def count_over(threshold):
    def f(b1, b2):
        return int(np.arccos(dot_metric(b1, b2)) > threshold)

    return f


def idle_accuracy(thresh, mag_idle):
    def f(b1, b2):
        gt_idle = float(b2['mag']) < mag_idle
        pred_idle = b1['parked'] > thresh
        return 1 if gt_idle == pred_idle else 0

    return f


def area(b1):
    xmin, ymin, xmax, ymax = map(float, [b1['xmin'], b1['ymin'], b1['xmax'], b1['ymax']])
    return max(1, (xmax - xmin) * (ymax - ymin))


def chain_filter(fs, vals):
    for f in fs:
        vals = filter(f, vals)
    return list(vals)


def generic_test(distance, filters, min_iou, gt_filters):
    score = []
    for k, boxes in pred.items():
        [_, vals] = gt[k]
        vals = chain_filter(gt_filters, vals)
        for b in boxes:
            if any([f(b) for f in filters]):
                continue
            bbox = find_best_iou(b, vals, min_iou=min_iou)
            if bbox is None:
                continue
            dist = distance(b, bbox)
            score.append(dist)
    if not score:
        return None, None
    return np.mean(score), np.std(score)


ALL_METRICS = {"Cosine distance": cosine_distance,
               "ArcCosine distance": arccosine_distance,
               "Dot": dot_metric,
               "More than .3 arccos": count_over(0.3),
               "Parked Acc. at 0.5-2": (False, idle_accuracy(0.5, 2))}


def test_with_filters(min_conf, min_iou, min_mag, size, vhcl_class):
    conf_filter = lambda b: float(b['conf']) < min_conf
    area_filter = lambda b: not (ALL_SIZES[size][0] <= area(b) < ALL_SIZES[size][1])
    filters = [conf_filter, area_filter]

    vhcl_filter = lambda b_gt: (b_gt['class'] == vhcl_class) if vhcl_class is not None else True

    base_record = {'min_conf': min_conf, 'min_iou': min_iou, 'min_mag': min_mag, 'vhcl_class': vhcl_class, 'size': size}
    for name, distance in ALL_METRICS.items():
        use_mag = True
        if isinstance(distance, tuple):
            use_mag, distance = distance
        mag_to_use = -1 if not use_mag else min_mag
        mag_filter = lambda b_gt: (float(b_gt['mag']) > mag_to_use)
        gt_filters = [vhcl_filter, mag_filter]
        mean, std = generic_test(distance=distance, filters=filters, min_iou=min_iou, gt_filters=gt_filters)
        base_record[name] = mean
    return base_record


ALL_SIZES = {'All': [0 ** 2, 1e10 ** 2],
             'small': [0 ** 2, 32 ** 2],
             'medium': [32 ** 2, 96 ** 2],
             'large': [96 ** 2, 1e10 ** 2]}

CONFS = [0.1, 0.3, 0.5, 0.7, 0.9]
IOUS = [0.5, 0.7, 0.9]
MAGS = [2]
SIZES = ['All', 'small', 'medium', 'large']
CLASSES = MIO_CLASSES + [None]
#CLASSES = [None]
dt = (pd.DataFrame.from_records(
    Parallel(n_jobs=4, backend='multiprocessing')(delayed(test_with_filters)(*args) for args in
                                                  tqdm(list(product(CONFS, IOUS, MAGS, SIZES, CLASSES)),
                                                       desc="Computing..."))))
print(dt)
dt.to_csv('pds/{}'.format(args.csv.split('/')[-1]), sep='\t', na_rep='All', index=False)
exit(0)
