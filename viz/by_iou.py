from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from viz.utils import PandasHandler, get_label


def filter_on_odf(df):
    df = df[df['odf size'].isin([100, 'All'])]
    return df


pds = PandasHandler('../analysis490k.pkl', keep_xception=True)
dfs = {iou: filter_on_odf(pds.get_with(min_iou=iou, min_conf=0.1)) for iou in pds.ious}

plots = defaultdict(list)
for iou in pds.ious:
    df = dfs[iou]
    for file in  np.unique(df['filepath'].values):
        plots[file].append(df[df['filepath'] == file]['More than .3 arccos'].values[0])

for fp, serie in plots.items():
    plt.plot([0.5, 0.7, 0.9], serie, label=get_label(fp))

plt.xlabel('Minimum IoU')
plt.ylabel('% .3 radians over')
plt.legend()
plt.show()

