# coding=utf-8
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

result_file = csv.reader(open(sys.argv[1], 'r'))
baseline_result = float(sys.argv[2])  # 80.567 in this protocol

# get results from csv file
all_results = []
header = next(result_file)
for line in result_file:
    all_results.append(line[1:])
all_results = np.array(all_results, dtype="float32") - baseline_result

# visualization
fig, ax = plt.subplots(figsize=(3,4))
ax = sns.heatmap(all_results, cmap="seismic",
                 vmin=-10, vmax=2, center=0, 
                 xticklabels=['ALL', 'Q2', 'Q2P', 'P2Q', 'P2'], 
                 yticklabels=range(1,13))
ax.invert_yaxis()

fig = ax.get_figure()
fig.savefig(sys.argv[3], bbox_inches='tight')