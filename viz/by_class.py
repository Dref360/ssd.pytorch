from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from viz.utils import PandasHandler, get_label


def filter_on_odf(df):
    df = df[df['odf size'].isin([100, 'All'])]
    return df


pds = PandasHandler('../analysis490k.pkl', keep_xception=True)
dfs = {vhcl_class: filter_on_odf(pds.get_with(min_iou=0.5, min_conf=0.5,vhcl_cls=vhcl_class)) for vhcl_class in pds.vhcl_classes}

plots = defaultdict(list)
for iou in pds.vhcl_classes:
    df = dfs[iou]
    for file in  np.unique(df['filepath'].values):
        plots[file].append(df[df['filepath'] == file]['More than .3 arccos'].values[0])

n_groups = len(pds.vhcl_classes)

index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.8
for idx, (fp, serie) in enumerate(plots.items()):
    plt.bar(index+bar_width*idx, serie, bar_width, alpha=opacity, label=get_label(fp))

# data to plot

# create plot
plt.xlabel('Vehicle classes')
plt.ylabel('Scores')
plt.title('Effect of class')
plt.xticks(index + bar_width, pds.vhcl_classes, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()