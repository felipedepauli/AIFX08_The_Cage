import cv2
import numpy as np
import os
import time
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents

class KMeansFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        # Implementar lógica de segmentação por KMeans
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        K = self.config.get("K", 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        
        self.intermediate_images = {}
        
        for k in range(1, K + 1):
            initial_time = time.time()
            _, labels, centers = cv2.kmeans(Z, k, None, criteria, 30, cv2.KMEANS_PP_CENTERS)
            end_time = time.time()
            elapsed_time = end_time - initial_time
            print(f"Image segmented into {k} clusters in {elapsed_time:.2f} seconds")
            
            centers = np.uint8(centers)
            res = centers[labels.flatten()]
            res2 = res.reshape((image.shape))
            
            self.intermediate_images[f"Segmented_{k}"] = res2

        # Usar ConnectedComponents na imagem segmentada final com K clusters
        gray = cv2.cvtColor(self.intermediate_images[f"Segmented_{K}"], cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        cc = ConnectedComponents(binary)

        # Desenhar os componentes conectados na imagem segmentada
        contour_image = image.copy()
        for i in range(cc.size()):
            cc.draw(cc[i], contour_image, (0, 255, 0))

        self.intermediate_images["Contour Image"] = contour_image

        return contour_image

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
