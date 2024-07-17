import cv2
import numpy as np
import os
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents

class AdaptiveGaussianFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        ksize = self.config.get("ksize", (5, 5))
        sigma = self.config.get("sigma", 0)
        gaussian_filtered = cv2.GaussianBlur(image, ksize, sigma)
        
        gray = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cc = ConnectedComponents(binary)

        contour_image = image.copy()
        for i in range(cc.size()):
            cc.draw(cc[i], contour_image, (0, 255, 0))

        self.intermediate_images = {
            "Original Image": image,
            "Adaptive Gaussian Filtered": gaussian_filtered,
            "Contour Image": contour_image
        }
        
        return contour_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
