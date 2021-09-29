# https://tech-blog.optim.co.jp/entry/2021/07/07/100000

import motmetrics as mm
import numpy as np

acc = mm.MOTAccumulator(auto_id=True)

acc.update(
    [1, 2],                     # Ground Truth
    [1, 2, 3],                  # Tracker
    [
        [0.1, np.nan, 0.3],     # Ground Truth No.1 vs Tracker
        [0.5,  0.2,   0.3]      # Ground Truth No.2 vs Tracker
    ]
)


a = np.array([
    [0, 0, 1, 2],    # [xmin, ymin, width, height]
    [0, 0, 0.8, 1.5],
])

b = np.array([
    [0, 0, 1, 2],
    [0, 0, 1, 1],
    [0.1, 0.2, 2, 2],
])
mm.distances.iou_matrix(a, b, max_iou=0.5)


mh = mm.metrics.create()

summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches'], name="Name")

strsummary = mm.io.render_summary(
    summary,
    formatters={'mota' : '{:.2%}'.format, 'motp' : '{:.2%}'.format, 'idf1' : '{:.2%}'.format},
    namemap={'mota': 'MOTA', 'motp' : 'MOTP', 'idf1': 'IDF1', 'mostly_tracked': 'MT', 'mostly_lost': 'ML', 'num_false_positives': 'FP', 'num_misses': 'Misses', 'num_switches': 'ID Sw'}
)
print(strsummary)

