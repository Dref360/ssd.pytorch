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


def find_best_iou(b1, vals, keep_parked=False, min_iou=0.75, min_mag=2.):
    best = max([(b2, iou(b1, b2)) for b2 in vals], key=lambda k: k[1])
    if (not keep_parked and float(best[0]['mag']) < min_mag) or best[1] < min_iou:
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


def generic_test(distance, filters, name, min_iou, min_mag):
    score = []
    for k, boxes in pred.items():
        [_, vals] = gt[k]
        for b in boxes:
            if all([f(b) for f in filters]):
                continue
            bbox = find_best_iou(b, vals, keep_parked=False, min_iou=min_iou, min_mag=min_mag)
            if bbox is None:
                continue
            dist = distance(b, bbox)
            score.append(dist)
    return np.mean(score), np.std(score)


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


ALL_METRICS = {"Cosine distance": cosine_distance,
               "ArcCosine distance": arccosine_distance,
               "Dot": dot_metric,
               "More than .3 arccos": count_over(0.3),
               "Parked Acc. at 0.5-2": (False, idle_accuracy(0.5, 2))}


def test_with_filters(min_conf, min_iou, min_mag):
    conf_filter = lambda b: float(b['conf']) < min_conf
    filters = [conf_filter]
    base_record = {'min_conf': min_conf, 'min_iou': min_iou, 'min_mag': min_mag}
    for name, distance in ALL_METRICS.items():
        use_mag = True
        if isinstance(distance, tuple):
            use_mag, distance = distance
        mag_to_use = -1 if not use_mag else min_mag
        mean, std = generic_test(distance=distance, filters=filters, name=name, min_iou=min_iou, min_mag=mag_to_use)
        base_record[name] = mean
    return base_record


CONFS = [0.1, 0.3, 0.5, 0.7, 0.9]
IOUS = [0.5, 0.7, 0.9]
MAGS = [2]
dt = (pd.DataFrame.from_records(Parallel(n_jobs=4)(delayed(test_with_filters)(*args) for args in
                                                   tqdm(list(product(CONFS, IOUS, MAGS)), desc="Computing..."))))
print(dt)
dt.to_clipboard(sep='\t', index=False)
