import cv2
import numpy as np
import os
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents

class DownsamplingFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        scale_factor = self.config.get("scale_factor", 0.5)
        downsampled = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        upsampled = cv2.resize(downsampled, (image.shape[1], image.shape[0]))
        
        gray = cv2.cvtColor(upsampled, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cc = ConnectedComponents(binary)

        contour_image = image.copy()
        for i in range(cc.size()):
            cc.draw(cc[i], contour_image, (0, 255, 0))

        self.intermediate_images = {
            "Original Image": image,
            "Downsampled and Resized": upsampled,
            "Contour Image": contour_image
        }
        
        return contour_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
