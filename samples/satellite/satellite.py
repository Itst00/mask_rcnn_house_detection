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

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Initialization
dataset_train_root_path = r"..\..\Satellite dataset Ⅱ (East Asia)\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\train"
img_train_folder = dataset_train_root_path+r"\image"
mask_train_folder = dataset_train_root_path+r"\label"
img_train_list = os.listdir(img_train_folder)
count_train = len(img_train_list)

dataset_val_root_path = r"..\..\Satellite dataset Ⅱ (East Asia)\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test"
img_val_folder = dataset_val_root_path+r"\image"
mask_val_folder = dataset_val_root_path+r"\label"
img_val_list = os.listdir(img_val_folder)
count_val = len(img_val_list)

dataset_train_no_root_path = r"..\..\Satellite dataset Ⅱ (East Asia)\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\train_no"
img_train_no_folder = dataset_train_no_root_path+r"\image"
mask_train_no_folder = dataset_train_no_root_path+r"\label"
img_train_no_list = os.listdir(img_train_no_folder)
count_train_no = len(img_train_no_list)

dataset_val_no_root_path = r"..\..\Satellite dataset Ⅱ (East Asia)\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test_no"
img_val_no_folder = dataset_val_no_root_path+r"\image"
mask_val_no_folder = dataset_val_no_root_path+r"\label"
img_val_no_list = os.listdir(img_val_no_folder)
count_val_no = len(img_val_no_list)

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

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
class SatelliteDataset(utils.Dataset):
    # Get the number of objects in one image
    def get_obj_index(self, image):
        n = np.max(image)
        return n
    
    def draw_mask(self, num_obj, mask, image):
        for i in range(image.width):
            for j in range(image.height):
                at_pixel = image.getpixel((i, j))
                if at_pixel != 0:
                    mask[j, i, at_pixel - 1] = 1
        return mask

    def load_shapes(self, count, img_floder, mask_folder, imglist):
        """Load images.
        count: number of images to load.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "house")
        for i in range(count):
            mask_path = mask_folder + "/" + imglist[i]
            path = img_floder + "/" + imglist[i]
            img = Image.open(path)
            self.add_image("shapes", image_id=i, path=path,width=img.width, height=img.height, mask_path=mask_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([img.height, img.width, num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img)
        if num_obj != 0:
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(num_obj - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels_form=[]
        for i in range(num_obj):
            labels_form.append("house")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

# Training dataset & Validation dataset
dataset_train = SatelliteDataset()
dataset_train.load_shapes(count_train, img_train_folder, mask_train_folder, img_train_list)
dataset_train.prepare()

dataset_val = SatelliteDataset()
dataset_val.load_shapes(count_train, img_train_folder, mask_train_folder, img_train_list)
dataset_val.prepare()

dataset_train_no = SatelliteDataset()
dataset_train_no.load_shapes(count_train_no, img_train_no_folder, mask_train_no_folder, img_train_no_list)
dataset_train_no.prepare()

dataset_val_no = SatelliteDataset()
dataset_val_no.load_shapes(count_train_no, img_train_no_folder, mask_train_no_folder, img_train_no_list)
dataset_val_no.prepare()

dataset_test = SatelliteDataset()
dataset_test.load_shapes(count_val, img_val_folder, mask_val_folder, img_val_list)
dataset_test.prepare()

dataset_test_no = SatelliteDataset()
dataset_test_no.load_shapes(count_val_no, img_val_no_folder, mask_val_no_folder, img_val_no_list)
dataset_test_no.prepare()

def inference():
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
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    image_ids = np.random.choice(dataset_test.image_ids, 150)
    APs = []
    for image_id in image_ids:
                # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(dataset_test, inference_config,
                                        image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
                # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
                # Compute AP
        AP = utils.compute_miou(gt_bbox, gt_class_id, gt_mask,
                                    r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    mAP1 = np.mean(APs)
    print("mIoU1: ", mAP1)

    image_ids = np.random.choice(dataset_test_no.image_ids, 150)
    APs = []
    for image_id in image_ids:
                # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(dataset_test_no, inference_config,
                                        image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
                # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
                # Compute AP
        AP = utils.compute_miou(gt_bbox, gt_class_id, gt_mask,
                                    r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    mAP2 = np.mean(APs)
    print("mIoU2: ", mAP2)

mode = sys.argv[1]
if mode == "train":
    """
    image_ids = np.random.choice(dataset_train.image_ids, 1)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    """
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
                            
    # Which weights to start with?
    init_with = "last"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
        
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    """
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')
    inference()
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=2, 
                layers='heads')
    inference()
    model.train(dataset_train_no, dataset_val_no, 
                learning_rate=config.LEARNING_RATE, 
                epochs=3, 
                layers='heads')
    inference()
    # Fine tune all layers
    # Passing layers="all" trains all layers.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=4, 
                layers="all")
    inference()
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=5, 
                layers="all")
    inference()
    """
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=7, 
                layers='heads')
    inference()
    # Save weights
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_satellite.h5")
    model.keras_model.save_weights(model_path)

else:
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
    model_path = os.path.join(ROOT_DIR, "logs\satellite20220726T0057\mask_rcnn_satellite_0005.h5")

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    if mode == "test":
        # Test on an image
        #image_id = random.choice(dataset_train.image_ids)
        image_id = 1
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_train, inference_config, 
                                image_id)

        results = model.detect([original_image], verbose=1)

        r = results[0]
        """
            log("original_image", original_image)
            log("image_meta", image_meta)
            log("gt_class_id", gt_class_id)
            log("gt_bbox", gt_bbox)
            log("gt_mask", gt_mask)
        """
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize=(8, 8))
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_test.class_names, figsize=(8, 8))

        AP = utils.compute_miou(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'])
        print("AP: ", AP)
    else:
        if mode == "inference": 
            # Compute VOC-Style mAP @ IoU=0.5
            # Running on 10 images. Increase for better accuracy.
            image_ids = np.random.choice(dataset_test.image_ids, 150)
            APs = []
            for image_id in image_ids:
                        # Load image and ground truth data
                image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                            modellib.load_image_gt(dataset_test, inference_config,
                                                image_id)
                molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
                        # Run object detection
                results = model.detect([image], verbose=0)
                r = results[0]
                        # Compute AP
                # AP = utils.compute_miou(gt_bbox, gt_class_id, gt_mask,
                #                            r["rois"], r["class_ids"], r["scores"], r['masks'])
                AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'])
                APs.append(AP)
            # print(APs)
            # print("mIoU1: ", np.mean(APs))
            print("AP:", np.mean(APs))
            """
            image_ids = np.random.choice(dataset_test_no.image_ids, 150)
            APs = []
            for image_id in image_ids:
                        # Load image and ground truth data
                image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                            modellib.load_image_gt(dataset_test_no, inference_config,
                                                image_id)
                molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
                        # Run object detection
                results = model.detect([image], verbose=0)
                r = results[0]
                        # Compute AP
                AP = utils.compute_miou(gt_bbox, gt_class_id, gt_mask,
                                            r["rois"], r["class_ids"], r["scores"], r['masks'])
                APs.append(AP)
            # print(APs)
            print("mIoU2: ", np.mean(APs))
"""
        else:
            def load_image(path):
                import skimage
                """Load the specified image and return a [H,W,3] Numpy array.
                """
                # Load image
                image = skimage.io.imread(path)
                # If grayscale. Convert to RGB for consistency.
                if image.ndim != 3:
                    image = skimage.color.gray2rgb(image)
                # If has an alpha channel, remove it for consistency
                if image.shape[-1] == 4:
                    image = image[..., :3]
                return image
            original_image = load_image(mode)

            results = model.detect([original_image], verbose=1)

            r = results[0]
            visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, figsize=(8, 8))