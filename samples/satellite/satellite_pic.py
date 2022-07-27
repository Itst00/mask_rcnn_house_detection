import os
import sys
import random
import math
import re
import time
import numpy as np
import PIL.Image as Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import yaml
import skimage

import warnings
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

width = 512
height = 512

class SatelliteConfig(Config):
    """
    Derives from the base Config class and overrides values specific
    to the toy satellite dataset.
    """
    # Give the configuration a recognizable name
    NAME = "satellite"

    BACKBONE = "resnet50"

    # Train on 1 GPU and 1 images per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = width
    IMAGE_MAX_DIM = height

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8,16,32,64,128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 180

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 3000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200

config = SatelliteConfig()
config.display()

class InferenceConfig(SatelliteConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

inference_config = InferenceConfig()
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    # model_path = model.find_last()
model_path = os.path.join(ROOT_DIR, "mask_rcnn_satellite_0005.h5")

    # Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

def load_image(path):
    import skimage
    """Load the specified image and return a [H,W,3] Numpy array."""
    # Load image
    image = skimage.io.imread(path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image
name = "gecunping.tif"
original_image = load_image(name)
x,y,z = original_image.shape
slice_x = np.arange(0,x+1,width)
slice_y = np.arange(0,y+1,height)
slice_x[-1] = x
slice_y[-1] = y
house_pos = []

for i in range(0,slice_x.shape[0] - 1):
    for j in range(0,slice_y.shape[0] - 1):
        results = model.detect([original_image[slice_x[i]:slice_x[i+1], slice_y[j]: slice_y[j+1], :]], verbose=0)
        gt_masks = results[0]['masks']
        for k in range(gt_masks.shape[2]):
            pos = np.where(gt_masks[:,:,k])
            size = len(pos[0])
            houseX = np.sum(pos[0])
            houseY = np.sum(pos[1])
            house_pos.append([int(round(houseY / size)) + slice_y[j], int(round(houseX / size)) + slice_x[i], size])
print(len(house_pos))

from pyheatmap.heatmap import HeatMap
hm = HeatMap(house_pos)
background = Image.new("RGB", (y, x), color=0)
hit_img = hm.heatmap(base=background, r = 800)
hit_img = cv2.cvtColor(np.asarray(hit_img),cv2.COLOR_RGB2BGR)
image = cv2.imread(name)
overlay = image.copy()
alpha = 0.5
cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1)
image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
print(hit_img.shape, image.shape)
image = cv2.addWeighted(hit_img, alpha, image, 1-alpha, 0)
cv2.imwrite("gecunping_heat.tif", image)