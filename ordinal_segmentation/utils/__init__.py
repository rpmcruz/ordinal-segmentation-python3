import numpy as np
import cv2


def swap_channels(img):
    channels = cv2.split(img)
    return cv2.merge((channels[2], channels[1], channels[0]))
