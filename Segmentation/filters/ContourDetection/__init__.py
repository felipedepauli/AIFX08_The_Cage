import cv2
import numpy as np
import os
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents
from scipy.interpolate import splprep, splev

class ContourDetectionFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        img_resized = cv2.resize(image, (256, 256))
        
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        
        edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
        
        # Usar ConnectedComponents para encontrar os componentes conectados
        cc = ConnectedComponents(edges)
        
        # Desenhar os componentes conectados na imagem segmentada
        mask = np.zeros((256, 256), np.uint8)
        for i in range(cc.size()):
            cc.draw(cc[i], mask, 255)
        
        segmented = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        
        # Encontre contornos na máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Desenhe contornos suavizados
        smoothed_image = self.draw_smooth_contours(segmented, contours)
        
        self.intermediate_images = {
            "Original Image": img_resized,
            "Gray Image": gray,
            "Thresholded Image": thresh,
            "Edges": edges,
            "Mask": mask,
            "Segmented Image": segmented,
            "Smoothed Image": smoothed_image
        }
        return smoothed_image

    def draw_smooth_contours(self, image, contours):
        smoothed_image = image.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:  # Filtrar pequenos contornos
                # Obter pontos do contorno
                cnt = cnt.squeeze()
                if len(cnt) < 3:
                    continue
                x = cnt[:, 0]
                y = cnt[:, 1]
                
                # Interpolação spline dos contornos
                tck, u = splprep([x, y], s=0)
                u_new = np.linspace(u.min(), u.max(), 1000)
                x_new, y_new = splev(u_new, tck, der=0)
                
                # Converter pontos interpolados para inteiros
                x_new = x_new.astype(np.int32)
                y_new = y_new.astype(np.int32)
                
                # Desenhar contornos suavizados
                smooth_contour = np.stack((x_new, y_new), axis=-1)
                cv2.polylines(smoothed_image, [smooth_contour], isClosed=True, color=(0, 255, 0), thickness=2)
        
        return smoothed_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
