import cv2
import numpy as np
import os
from skimage.filters import threshold_otsu
from filters.Filter import Filter
from utils.ConnectedComponents import ConnectedComponents

class OtsuThresholdFilter(Filter):
    def __init__(self, config):
        self.config = config

    def filter(self, image):
        # Implementar lógica de limiarização de Otsu
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = threshold_otsu(img_gray)
        img_otsu = img_gray < thresh

        # Convertendo img_otsu para uint8
        img_otsu_uint8 = img_otsu.astype(np.uint8) * 255

        # Filtrando a imagem
        filtered = self.filter_image(image, img_otsu_uint8)

        # Usar ConnectedComponents na imagem filtrada
        _, binary = cv2.threshold(img_otsu_uint8, 1, 255, cv2.THRESH_BINARY)
        cc = ConnectedComponents(binary)

        # Desenhar os componentes conectados na imagem original
        contour_image = image.copy()
        for i in range(cc.size()):
            cc.draw(cc[i], contour_image, (0, 255, 0))

        self.intermediate_images = {
            "Original Image": image,
            "Gray Image": img_gray,
            "Thresholded Image": img_otsu_uint8,
            "Filtered Image": filtered,
            "Contour Image": contour_image
        }
        
        return contour_image

    def filter_image(self, image, mask):
        r = image[:, :, 0] * mask
        g = image[:, :, 1] * mask
        b = image[:, :, 2] * mask

        return np.dstack([r, g, b])

    def save_intermediate_images(self, output_dir):
        for name, img in self.intermediate_images.items():
            path = os.path.join(output_dir, f"{name.replace(' ', '_')}.jpg")
            cv2.imwrite(path, img)
