# Deep Learning Explorer

Deep learning explorer is a set of tools to quickly see how different deep
learning models work with your data.

Every model is ready to test in an NVIDIA docker environment with Jupyter notebooks.

An NVIDIA GeForce 1080Ti on Ubuntu 16.04 is used for testing, but other cards
and distributions may also work.

Custom data is supported in the COCO format. You can use
[pycococreator](https://github.com/waspinator/pycococreator) to create your
own COCO-style data sets.

Learn how to get started here: https://patrickwasp.com/train-a-mask-r-cnn-model-on-your-own-dataset/

## Currently implemented models

- Mask R-CNN (object detection and segmentation) [[arXiv](https://arxiv.org/abs/1703.06870), [source](https://github.com/matterport/Mask_RCNN)]
- FCN (class segmentation) [[arXiv](https://arxiv.org/abs/1605.06211), [source](https://github.com/aurora95/Keras-FCN)]
