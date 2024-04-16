import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, io, plot



# 读取图像
files = ['']
imgs = [io.imread(f) for f in files]

# 加载Cellpose模型
model = models.CellposeModel(model_type='livecell', gpu=True)

# 预测细胞掩膜
diameter = model.diam_labels

# run model on test images
masks, flows, styles = model.eval(imgs,
                                  channels=[2, 1],
                                  diameter=diameter,
                                  flow_threshold=0.4,
                                  cellprob_threshold=0)

nimg = len(imgs)
for idx in range(nimg):
    maski = masks[idx]
    flowi = flows[idx][0]
    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, imgs[idx], maski, flowi)
    plt.tight_layout()
    plt.show()



