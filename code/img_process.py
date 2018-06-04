import cv2
import os
import sys
import imageio
import random
from scipy import misc
import numpy as np
from glob import glob

data_dir = "../Train"
rgb_dir = data_dir + "/CameraRGB"
seg_dir = data_dir + "/CameraSeg"
seg_out_dir = data_dir + "/CameraSeg2"  # Cityscapes palette folder
# depth_dir = data_dir + "/Cameradepth" # depth folder

rgb_path = os.path.join(rgb_dir, "*.png")
seg_path = os.path.join(seg_dir, "*.png")
seg2_path = os.path.join(seg_out_dir, "*.png")

# depth_path = os.path.join(depth_dir, "*.png")


# http://carla.readthedocs.io/en/latest/cameras_and_sensors/#cameras-and-sensors
def labels_to_seg2(file):
    """Convert an image containing CARLA semantic segmentation labels to Cityscapes palette."""
    classes = {
        0: [0, 0, 0],  # None
        1: [70, 70, 70],  # Buildings
        2: [190, 153, 153],  # Fences
        3: [72, 0, 90],  # Other
        4: [220, 20, 60],  # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],  # RoadLines
        7: [128, 64, 128],  # Roads
        8: [244, 35, 232],  # Sidewalks
        9: [107, 142, 35],  # Vegetation
        10: [0, 0, 255],  # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]  # TrafficSigns
    }

    depth_seg = imageio.imread(file)
    # Build a blank image
    result = np.zeros((depth_seg.shape[0], depth_seg.shape[1], 3))
    # Person
    result[np.where(depth_seg[:, :, 0] == 4)] = [255, 0, 0]
    # Combine road and roadlines
    result[np.where(depth_seg[:, :, 0] == 6)] = [0, 255, 0]
    result[np.where(depth_seg[:, :, 0] == 7)] = [0, 255, 0]
    # Car and remove car hood
    result[np.where(depth_seg[:490, :, 0] == 10)] = [0, 0, 255]
    return result.astype(np.uint8)


# Sem seg
total_num = 0
for i, file in enumerate(glob(seg_path)):
    file_name = os.path.basename(file)
    result = labels_to_seg2(file)

    imageio.imwrite(seg_out_dir + '/' + file_name, result, format='png')
    total_num += 1

print(total_num, "images has been processed.")
