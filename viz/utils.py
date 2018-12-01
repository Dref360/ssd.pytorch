import os
from itertools import product
from random import shuffle, sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

colors = ['g', 'b', 'm', 'c']
styles = ['-', '--', ':', '-.']
fmts = product(colors, styles)

odf_style = list(product(['y', 'r'], ['-', ':', '-.', '--']))

pjoin = os.path.join
# ArcCosine distance,Cosine distance,Dot,More than .3 arccos,Parked Acc. at 0.5-2
labels = {'arccos_distance': 'ArcCosine distance',
          'cosine_distance': 'Cosine distance',
          '.3_ratio': 'More than .3 arccos',
          'parked_acc': 'Parked Acc. at 0.5-2',
          'dot': 'Dot'}


class PandasHandler():
    def __init__(self, pickle_file, keep_xception=False):
        self.data = pd.read_pickle(pickle_file)
        self.data = self.data.fillna('All')
        for v in labels.values():
            self.data[self.data[v] == 'All'] = 0.
            self.data[v] = pd.to_numeric(self.data[v])
        self.data = self.data[~self.data['filepath'].isin([0.])]

        if not keep_xception:
            self.data = self.data[~self.data['filepath'].isin(['pds/XceptionMeanShiftOut.csv', 'pds/XceptionOut.csv'])]

        self.ious = np.unique(self.data['min_iou'].values)
        self.confs = np.unique(self.data['min_conf'].values)
        self.sizes = np.unique(self.data['size'].values)
        self.vhcl_classes = np.unique(self.data['vhcl_class'].values)
        self.filepaths = np.unique(self.data['filepath'].values)

    def get_with(self, min_conf=None, min_iou=None, vhcl_cls='All', size='All'):
        df = self.data
        if min_conf:
            df = df[df['min_conf'] == min_conf]
        if min_iou:
            df = df[df['min_iou'] == min_iou]
        if isinstance(vhcl_cls, str):
            vhcl_cls = [vhcl_cls]
        df = df[df['vhcl_class'].isin(vhcl_cls)]
        df = df[df['size'] == size]
        return df

    def get_metric(self, df, metric_name):
        assert metric_name in labels
        with_odf = df[df['odf size'] != 'All']
        with_odf = [(i, with_odf[with_odf['odf size'] == i][labels[metric_name]].values[0])
                    for i in np.unique(with_odf['odf size'].values)]

        no_odf = df[df['odf size'] == 'All']
        no_odf = [(i, no_odf[no_odf['filepath'] == i][labels[metric_name]].values[0])
                  for i in np.unique(no_odf['filepath'].values)]

        return with_odf, no_odf

def get_label(lbl):
    lbl = lbl.split('/')[-1].split('.')[0]
    if 'MIOODFOF' in lbl:
        lbl = 'SSD with ODF'
    return lbl


def draw_constant(h, **kwargs):
    plt.axhline(y=h, **kwargs)


def draw_line_from_pds_and_vhcl(label, pds, min_conf, vhcl_cls, size):
    global colors, styles
    shuffle(colors)
    shuffle(styles)
    assert label in labels

    legend = 'conf={}, vhcl={}, size={}'.format(min_conf, vhcl_cls, size)
    df = pds.get_with(min_conf, min_iou=0.5, vhcl_cls=vhcl_cls, size=size)
    with_odf, no_odf = pds.get_metric(df, label)
    x, y = zip(*sorted(with_odf, key=lambda k: k[0]))
    plt.plot(x, y, ''.join(sample(odf_style, 1)[0]), label='With ODF ' + legend)
    for (fp, val), c, s in zip(no_odf, colors, styles):
        draw_constant(val, label=fp.split('/')[-1][:-4] + legend, color=c, linestyle=s)


if __name__ == '__main__':
    fig = plt.figure(121)

    pds = PandasHandler('../analysis490k.pkl')
    label = '.3_ratio'
    conf = 0.5
    draw_line_from_pds_and_vhcl(label, pds, conf, 'All', 'small')
    draw_line_from_pds_and_vhcl(label, pds, conf, 'All', 'large')
    draw_line_from_pds_and_vhcl(label, pds, conf, 'All', 'medium')
    draw_line_from_pds_and_vhcl(label, pds, conf, 'All', 'All')
    """draw_line_from_pds_and_vhcl('.3_ratio', pds, 0.5, 'car', 'All')
    draw_line_from_pds_and_vhcl('.3_ratio', pds, 0.5, 'motorcycle', 'All')
    draw_line_from_pds_and_vhcl('.3_ratio', pds, 0.5, 'bicycle', 'All')
    draw_line_from_pds_and_vhcl('.3_ratio', pds, 0.5, 'articulated_truck', 'All')
    draw_line_from_pds_and_vhcl('.3_ratio', pds, 0.5, 'pedestrian', 'All')"""
    plt.ylabel(labels[label])
    plt.xlabel('Odf size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(right=0.5)
    fig.show()
