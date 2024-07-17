import cv2
import numpy as np
import os
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents

class HoughTransformFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        contour_image = image.copy()
        if lines is not None:
            for rho, theta in lines[:,0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(contour_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.intermediate_images = {
            "Original Image": image,
            "Edges": edges,
            "Contour Image": contour_image
        }
        
        return contour_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
