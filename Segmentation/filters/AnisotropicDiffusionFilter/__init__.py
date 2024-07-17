import cv2
import os
import numpy as np
from skimage.restoration import denoise_bilateral
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents
from scipy.interpolate import splprep, splev

class AnisotropicDiffusionFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        sigma_color = self.config.get("sigma_color", 0.05)
        sigma_spatial = self.config.get("sigma_spatial", 15)
        
        # Adicionar o argumento `channel_axis`
        diffusion_filtered = denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=-1)
        
        # Converter a imagem para uint8
        diffusion_filtered_uint8 = (diffusion_filtered * 255).astype(np.uint8)

        gray = cv2.cvtColor(diffusion_filtered_uint8, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cc = ConnectedComponents(binary)

        contour_image = self.draw_smooth_contours(image, cc)

        self.intermediate_images = {
            "Original Image": image,
            "Diffusion Filtered": diffusion_filtered_uint8,
            "Contour Image": contour_image
        }
        
        return contour_image

    def draw_smooth_contours(self, image, cc):
        smoothed_image = image.copy()
        for i in range(cc.size()):
            region = cc[i]
            cnt = self.extract_contour(region)
            if cnt is None or len(cnt) < 4:  # Verificação do número de pontos
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
