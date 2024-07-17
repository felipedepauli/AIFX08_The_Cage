import cv2
import numpy as np
import os
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents
from scipy.interpolate import splprep, splev

class ColorMaskingFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        low = np.array([0, 0, 0])
        high = np.array([215, 51, 51])
        mask = cv2.inRange(image, low, high)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cc = ConnectedComponents(binary)

        contour_image = self.draw_smooth_contours(image, cc)

        self.intermediate_images = {
            "Original Image": image,
            "Mask": mask,
            "Segmented": segmented,
            "Contour Image": contour_image
        }
        
        return contour_image

    def draw_smooth_contours(self, image, cc):
        smoothed_image = image.copy()
        for i in range(cc.size()):
            region = cc[i]
            cnt = self.extract_contour(region)
            if len(cnt) < 4:
                cv2.polylines(smoothed_image, [cnt], isClosed=True, color=(0, 255, 0), thickness=2)
                continue
            x = cnt[:, 0]
            y = cnt[:, 1]
            tck, u = splprep([x, y], s=0)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)
            x_new = x_new.astype(np.int32)
            y_new = y_new.astype(np.int32)
            smooth_contour = np.stack((x_new, y_new), axis=-1)
            cv2.polylines(smoothed_image, [smooth_contour], isClosed=True, color=(0, 255, 0), thickness=2)
        return smoothed_image

    def extract_contour(self, region):
        points = []
        current = region.start_seg
        while current is not None:
            for col in range(current.start_col, current.end_col + 1):
                points.append([col, current.row])
            current = current.next
        return np.array(points)

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
