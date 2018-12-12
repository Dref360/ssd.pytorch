import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import torch

from layers.box_utils import jaccard

parser = argparse.ArgumentParser()
parser.add_argument('root', help='Dataset root')
parser.add_argument('csv', help='CSV path')
args = parser.parse_args()
print(args.csv)
pjoin = os.path.join

gt_test = pjoin(args.root, 'gt_test.csv')

if False:
    print(subprocess.run(['python', 'localization_evaluation.py', gt_test, args.csv],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE).stdout.decode('utf-8'))

gt = json.load(open(pjoin(args.root, 'jsontest.json'), 'r'))
pred = defaultdict(list)
for id, label, conf, xmin, ymin, xmax, ymax, angle, parked in csv.reader(open(args.csv, 'r')):
    if float(conf) < 0.01:
        continue
    d = {'angle': float(angle),
         'parked': float(parked),
         'class': label,
         'conf': float(conf),
         'xmin': float(xmin), 'xmax': float(xmax), 'ymin': float(ymin), 'ymax': float(ymax)}
    pred[id].append(d)


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


def idle_accuracy(thresh, mag_idle):
    def f(b1, b2):
        gt_idle = float(b2['mag']) < mag_idle
        pred_idle = b1['parked'] > thresh
        return 1 if gt_idle == pred_idle else 0

    return f


class IdleMetric():
    def __init__(self, conf, mag):
        self.met = {k: 0 for k in ['tp', 'fp', 'tn', 'fn']}
        self.conf, self.mag = conf, mag

    def __call__(self, b1, b2):
        gt_idle = float(b2['mag']) < self.mag
        pred_idle = b1['parked'] > self.conf
        if gt_idle and pred_idle:
            self.met['tp'] += 1
        elif gt_idle and not pred_idle:
            self.met['fn'] += 1
        elif not gt_idle and not pred_idle:
            self.met['tn'] += 1
        else:
            self.met['fp'] += 1

    def precision(self):
        return self.met['tp'] / (self.met['tp'] + self.met['fp'])

    def recall(self):
        return self.met['tp'] / (self.met['tp'] + self.met['fn'])


def chain_filter(fs, vals):
    for f in fs:
        vals = filter(f, vals)
    return list(vals)


def generic_test(distance, filters, min_iou, gt_filters):
    for k, boxes in pred.items():
        [_, vals] = gt[k]
        for b in boxes:
            bbox = find_best_iou(b, vals, min_iou=min_iou)
            if bbox is None:
                continue
            distance(b, bbox)
    return distance


conf_setup = np.linspace(0, 1, 20)[1:-1]
ALL_METRICS = [IdleMetric(c, 2) for c in conf_setup]


def test_parked():
    base_record = {}
    functors = [generic_test(distance=distance,
                             filters=[],
                             min_iou=0.5, gt_filters=[]) for distance in tqdm(ALL_METRICS)]
    for functor in functors:
        base_record['Parked Rec. at {}'.format(functor.conf)] = functor.recall()
        base_record['Parked Prec. at {}'.format(functor.conf)] = functor.precision()
    return base_record

acc = test_parked()
# CLASSES = [None]
dt = pd.DataFrame(acc, index=[0])
print(dt)
dt.to_csv('machin{}'.format(args.csv.split('/')[-1]), sep='\t', na_rep='All', index=False)
exit(0)
