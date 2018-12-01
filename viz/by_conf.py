from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from viz.utils import PandasHandler, get_label


def filter_on_odf(df):
    df = df[df['odf size'].isin([100, 'All'])]
    return df


pds = PandasHandler('../analysis490k.pkl', keep_xception=True)
dfs = {conf: filter_on_odf(pds.get_with(min_iou=0.5, min_conf=conf)) for conf in pds.confs}

plots = defaultdict(list)
for iou in pds.confs:
    df = dfs[iou]
    for file in  np.unique(df['filepath'].values):
        plots[file].append(df[df['filepath'] == file]['More than .3 arccos'].values[0])

for fp, serie in plots.items():
    plt.plot(pds.confs, serie, label=get_label(fp))


plt.xlabel('Minimum confidence score')
plt.ylabel('% .3 radians over')
plt.legend()
plt.show()

