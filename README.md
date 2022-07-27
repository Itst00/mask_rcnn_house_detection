# Mask R-CNN for House Detection

This is an implementation of house detection using Mask R-CNN on Python 3, Keras, and TensorFlow. It is modified from [Mask R-CNN for tensorflow2.x](https://github.com/leekunhee/Mask_RCNN) and [a Chinese Mask R-CNN blog](https://blog.csdn.net/l297969586/article/details/79140840/).

Training and testing dataset is [WHU Building Dataset](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html). To run satellite.py, one should change all Image.mode into "P" and apply flood fill for all labels to identify single houses. One might also change palette into labelme form for better visualization.

satellite_pic.py produces the heatmap of house density for a given satellite image.

To download level 19 satellite images, one can use LSV along with FatMap.lrc file. The lrc file is from https://zhuanlan.zhihu.com/p/361042506?utm_medium=social&utm_oi=1246761352125018112&ivk_sa=1024320u.
