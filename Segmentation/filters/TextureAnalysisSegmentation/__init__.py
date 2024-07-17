import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents

class TextureAnalysisFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        _, binary = cv2.threshold(lbp, 0.5, 1.0, cv2.THRESH_BINARY)

        cc = ConnectedComponents(binary.astype("uint8") * 255)
        contour_image = image.copy()
        for i in range(cc.size()):
            cc.draw(cc[i], contour_image, (0, 255, 0))

        self.intermediate_images = {
            "Original Image": image,
            "LBP": lbp,
            "Binary Image": binary,
            "Contour Image": contour_image
        }
        
        return contour_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
