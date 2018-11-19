import os
from glob import glob
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

colors = ['r', 'g', 'b', 'y']
styles = ['-', '--', ':', '-.']
fmts = product(colors, styles)

pjoin = os.path.join
# ArcCosine distance,Cosine distance,Dot,More than .3 arccos,Parked Acc. at 0.5-2
labels = {'arccos_distance': 'ArcCosine distance',
          'cosine_distance': 'Cosine distance',
          '.3_ratio': 'More than .3 arccos',
          'parked_acc': 'Parked Acc. at 0.5-2'}


class PandasHandler():
    def __init__(self, csv_file, odf=None):
        self.data = pd.read_csv(csv_file, sep=',')
        if len(self.data.columns) == 1:
            self.data = pd.read_csv(csv_file, sep='\t')
        self.data = self.data.fillna('All')
        self.odf = odf

    def get_with(self, min_conf=None, min_iou=None, vhcl_cls='All'):
        df = self.data
        if min_conf:
            df = df[df['min_conf'] == min_conf]
        if min_iou:
            df = df[df['min_iou'] == min_iou]
        if 'vhcl_class' in df.columns:
            if isinstance(vhcl_cls, str):
                vhcl_cls = [vhcl_cls]
            df = df[df['vhcl_class'].isin(vhcl_cls)]
        return df


def draw_constant(h, **kwargs):
    plt.axhline(y=h, **kwargs)


def draw_line_from_pds_and_vhcl(label, pds, min_conf, vhcl_cls):
    assert label in labels

    legend = 'min_conf={}, vhcl={}'.format(min_conf, vhcl_cls)
    x, y = zip(*sorted([(k, df.get_with(min_conf, 0.9, vhcl_cls)[labels[label]].values[0]) for k, df in pds.items() if
                        isinstance(k, int)], key=lambda k: k[0]))
    gen = [(k, df.get_with(min_conf, 0.9, vhcl_cls)[labels[label]].values[0]) for k, df in pds.items() if
           not isinstance(k, int)]
    plt.plot(x, y, 'y-.', label=legend)
    [draw_constant(yi,color=c, linestyle=s, label=xi + legend) for (xi, yi), c, s in zip(gen,colors, styles)]


if __name__ == '__main__':
    pds = {}
    for pt in glob(pjoin('..', 'pds/*keep_valid.csv')):
        i = int(pt.split('=')[-1][:-len('.keep_valid.csv')])
        pds[i] = PandasHandler(pt, odf=i)

    for pt in ['../pds/XceptionMeanShiftOut.csv', '../pds/XceptionOut.csv', '../pds/SSD_noODF.csv']:
        i = pt.split('/')[-1][:-4]
        pds[i] = PandasHandler(pt, None)

    plt.title('.3_ratio')
    draw_line_from_pds_and_vhcl('.3_ratio', pds, 0.5, 'All')
    draw_line_from_pds_and_vhcl('.3_ratio', pds, 0.1, 'All')

    plt.legend()
    plt.show()
