import cv2
import numpy as np
import os
from skimage.segmentation import slic
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents

class SuperpixelSegmentationFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        segments = slic(image, n_segments=250, compactness=10)
        mask = np.zeros(image.shape[:2], dtype="uint8")

        cc = ConnectedComponents(segments)
        contour_image = image.copy()
        for i in range(cc.size()):
            cc.draw(cc[i], contour_image, (0, 255, 0))

        self.intermediate_images = {
            "Original Image": image,
            "Segments": segments,
            "Contour Image": contour_image
        }
        
        return contour_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
